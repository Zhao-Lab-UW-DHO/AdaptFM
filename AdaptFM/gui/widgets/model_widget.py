"""
model_widget.py
---------------
Base class for Training and Inference floating windows.

Design principles
-----------------
- All subprocess execution goes through QProcess (non-blocking, streams to log).
- tunable_params() is called in a QThread — never blocks the UI.
- GPU index and output directory are inline fields, not mid-run dialogs.
- Subclasses implement _run_workflow() and optionally _on_model_changed().
- The "terminate" button kills the QProcess and, on Unix, its process group.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Callable, Optional
from qtpy.QtCore import Qt, QProcess, QProcessEnvironment, QThread, Signal, QObject, QTimer
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QSpinBox,
    QScrollArea, QFrame, QTextEdit,
    QFileDialog, QSplitter, QProgressBar, QComboBox,
    QMessageBox,
)

from AdaptFM.model.registry import MODEL_REGISTRY
from AdaptFM.model.nnUNetV2Spec import NNUNetV2ModelSpec


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_GREEN  = "#4caf50"
_AMBER  = "#ff9800"
_RED    = "#f44336"
_BLUE   = "#42a5f5"
_BG     = "#2b2b2b"
_BG2    = "#1e1e1e"
_BD     = "#444"
_TEXT   = "#e8e8e8"
_MUTED  = "#888"

# Global dark stylesheet applied to every widget in the window
_WINDOW_STYLE = f"""
    QWidget {{
        background-color: {_BG};
        color: {_TEXT};
    }}
    QScrollArea {{
        background-color: {_BG};
        border: none;
    }}
    QScrollBar:vertical {{
        background: {_BG};
        width: 8px;
    }}
    QScrollBar::handle:vertical {{
        background: #555;
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QSplitter::handle {{
        background: {_BD};
        height: 1px;
    }}
    QLabel {{
        background: transparent;
        color: {_TEXT};
    }}
    QSpinBox {{
        background: {_BG2};
        color: #d4d4d4;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 2px 6px;
    }}
"""


# ---------------------------------------------------------------------------
# Background worker: calls model.tunable_params() off the main thread
# ---------------------------------------------------------------------------

class _ParamLoader(QObject):
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, model):
        super().__init__()
        self._model = model

    def run(self):
        try:
            schema = self._model.tunable_params()
            self.finished.emit(schema)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Base widget
# ---------------------------------------------------------------------------

class ModelWorkflowWidget:
    """
    Shared base for TrainingWidget and InferenceWidget.

    Subclasses must implement:
        _run_workflow()       — kicks off the QProcess / chain
        _on_model_changed()   — optional; called after model + params reset
        _extra_controls()     — optional; QWidget inserted above Run button
    """

    WINDOW_TITLE = "AdaptFM"
    # If True, skip the "Load Parameters" button; params load automatically
    # on model change (used for nnUNet) or not at all (inference).
    SKIP_PARAMS  = False
    # If True, auto-load params on every model change (no manual button click)
    AUTO_LOAD_PARAMS = False

    def __init__(self, dataset_manager):
        self.dataset_manager             = dataset_manager
        self.model                       = None
        self.dataset_dir: Optional[Path] = None
        self.output_dir:  Optional[Path] = None
        self.param_widgets: dict         = {}
        self.registry=MODEL_REGISTRY
        self.registry_title = "MODEL"

        self._process: Optional[QProcess]       = None
        self._param_thread: Optional[QThread]   = None
        self._param_worker: Optional[_ParamLoader] = None
        self._process_chain: list               = []
        self._chain_env: dict                   = {}
        self._chain_on_done: Optional[Callable] = None
        self._param_load_id: int                = 0

        self.widget = QWidget()
        self.widget.setWindowTitle(self.WINDOW_TITLE)
        self.widget.setWindowFlags(self.widget.windowFlags() | Qt.Window)
        
        # Intercept close events to kill threads, but DO NOT delete the C++ object
        self.widget.closeEvent = self._on_close

        self.widget.setStyleSheet(_WINDOW_STYLE)
        self.widget.resize(660, 700)
        self.widget.setMinimumWidth(520)

        self._build()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build(self):
        root = QVBoxLayout(self.widget)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(8)

        # Title
        title_row = QHBoxLayout()
        lbl = QLabel(self.WINDOW_TITLE)
        lbl.setStyleSheet("font-size: 15px; font-weight: 700;")
        title_row.addWidget(lbl)
        title_row.addStretch()
        root.addLayout(title_row)
        root.addWidget(_hline())

        # Splitter: controls top, log bottom
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {_BD}; height: 2px; }}")
        root.addWidget(splitter)

        # ── Controls ────────────────────────────────────────────────── #
        ctrl_outer = QWidget()
        ctrl = QVBoxLayout(ctrl_outer)
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(8)

        # Model
        ctrl.addWidget(_section_label(self.registry_title))
        self._model_combo = QComboBox()
        self._model_combo.addItems(list(self.registry.keys()))
        self._model_combo.setStyleSheet(_combo_style())
        # Connect AFTER build so the QTimer below is the only initial trigger
        ctrl.addWidget(self._model_combo)

        # Dataset
        ctrl.addWidget(_section_label("Dataset"))
        ctrl.addLayout(self._path_row("_dataset_lbl", self._select_dataset_folder))

        # Output
        ctrl.addWidget(_section_label("Output directory"))
        ctrl.addLayout(self._path_row("_output_lbl", self._select_output_folder))

        # GPU
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(_section_label("GPU index"))
        self._gpu_spin = QSpinBox()
        self._gpu_spin.setRange(0, 15)
        self._gpu_spin.setValue(0)
        self._gpu_spin.setFixedWidth(70)
        gpu_row.addWidget(self._gpu_spin)
        gpu_row.addStretch()
        ctrl.addLayout(gpu_row)


        # Parameters
        self._param_title_lbl = _section_label("Parameters")
        ctrl.addWidget(self._param_title_lbl)

        self._param_status_lbl = QLabel("")
        self._param_status_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        ctrl.addWidget(self._param_status_lbl)

        self._param_progress = QProgressBar()
        self._param_progress.setRange(0, 0)
        self._param_progress.setFixedHeight(3)
        self._param_progress.setVisible(False)
        self._param_progress.setStyleSheet(
            f"QProgressBar {{ border: none; background: #333; }}"
            f"QProgressBar::chunk {{ background: {_BLUE}; }}"
        )
        ctrl.addWidget(self._param_progress)

        # Param form in scroll area
        self._params_form = QFormLayout()
        self._params_form.setSpacing(4)
        self._params_form.setLabelAlignment(Qt.AlignLeft)

        self._params_container = QWidget()
        self._params_container.setLayout(self._params_form)

        self._param_scroll = QScrollArea()
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll.setWidget(self._params_container)
        self._param_scroll.setMinimumHeight(180)
        self._param_scroll.setMaximumHeight(380)
        self._param_scroll.setFrameShape(QFrame.NoFrame)
        ctrl.addWidget(self._param_scroll)

        if not self.SKIP_PARAMS:
            self._load_params_btn = QPushButton("Load Parameters")
            self._load_params_btn.setStyleSheet(_secondary_btn_style())
            self._load_params_btn.setEnabled(False)
            self._load_params_btn.clicked.connect(self._load_tunable_params)
            ctrl.addWidget(self._load_params_btn)

        # Subclass extra controls (e.g. checkpoint selector)
        extra = self._extra_controls()
        if extra is not None:
            ctrl.addWidget(extra)

        ctrl.addStretch()
        splitter.addWidget(ctrl_outer)

        # ── Log panel ────────────────────────────────────────────────── #
        log_outer = QWidget()
        log_layout = QVBoxLayout(log_outer)
        log_layout.setContentsMargins(0, 4, 0, 0)
        log_layout.setSpacing(4)

        log_hdr = QHBoxLayout()
        log_hdr.addWidget(_section_label("Output log"))
        log_hdr.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.setStyleSheet(f"color: {_MUTED}; font-size: 10px; border: none; background: transparent;")
        clear_btn.clicked.connect(self._clear_log)
        log_hdr.addWidget(clear_btn)
        log_layout.addLayout(log_hdr)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet(
            f"background: {_BG2}; color: #d4d4d4; border: 1px solid #333; border-radius: 4px;"
        )
        log_layout.addWidget(self._log)
        splitter.addWidget(log_outer)
        splitter.setSizes([400, 260])

        # ── Run / Terminate ──────────────────────────────────────────── #
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setStyleSheet(
            f"QPushButton {{ background: {_GREEN}; color: #000; border-radius: 5px;"
            f" padding: 6px 16px; font-size: 13px; font-weight: 700; }}"
            f"QPushButton:disabled {{ background: #3a5a3a; color: #666; }}"
        )
        self._run_btn.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self._run_btn)

        self._term_btn = QPushButton("■  Terminate")
        self._term_btn.setEnabled(False)
        self._term_btn.setStyleSheet(
            f"QPushButton {{ background: {_RED}; color: #fff; border-radius: 5px;"
            f" padding: 6px 16px; font-size: 13px; font-weight: 700; }}"
            f"QPushButton:disabled {{ background: #5a2a2a; color: #666; }}"
        )
        self._term_btn.clicked.connect(self._on_terminate_clicked)
        btn_row.addWidget(self._term_btn)
        root.addLayout(btn_row)

        self._run_progress = QProgressBar()
        self._run_progress.setRange(0, 0)
        self._run_progress.setFixedHeight(4)
        self._run_progress.setVisible(False)
        self._run_progress.setStyleSheet(
            f"QProgressBar {{ border: none; background: #333; }}"
            f"QProgressBar::chunk {{ background: {_GREEN}; }}"
        )
        root.addWidget(self._run_progress)

        # Connect combo AFTER build, then fire initial selection via event loop
        self._model_combo.currentTextChanged.connect(self._on_model_selected)
        initial = self._model_combo.currentText()
        if initial:
            QTimer.singleShot(0, lambda: self._on_model_selected(initial))

    # ------------------------------------------------------------------
    # Helper row for paths (stores button reference on self)
    # ------------------------------------------------------------------

    def _path_row(self, lbl_attr: str, slot: Callable) -> QHBoxLayout:
        """Helper: labelled path display + Browse button."""
        row = QHBoxLayout()
        lbl = QLabel("No folder selected")
        lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        lbl.setWordWrap(True)
        setattr(self, lbl_attr, lbl)
        row.addWidget(lbl, stretch=1)
        
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.setStyleSheet(_secondary_btn_style())
        btn.clicked.connect(slot)
        
        # Keep python reference alive on self to prevent GC disconnection
        setattr(self, f"{lbl_attr}_btn", btn)
        row.addWidget(btn)
        return row

    # ------------------------------------------------------------------
    # Model selection & thread lifecycle
    # ------------------------------------------------------------------


# ------------------------------------------------------------------
    # Model Selection & Parameter Management
    # ------------------------------------------------------------------
    def _set_param_section_visible(self, visible: bool):
            """Toggle visibility for all widgets in the Parameters panel."""
            if hasattr(self, "_param_title_lbl") and self._param_title_lbl:
                self._param_title_lbl.setVisible(visible)
            if hasattr(self, "_param_status_lbl") and self._param_status_lbl:
                self._param_status_lbl.setVisible(visible)
            if hasattr(self, "_param_scroll") and self._param_scroll:
                self._param_scroll.setVisible(visible)
            if hasattr(self, "_load_params_btn") and self._load_params_btn:
                self._load_params_btn.setVisible(visible)
            if not visible and hasattr(self, "_param_progress") and self._param_progress:
                self._param_progress.setVisible(False)

    def _on_model_selected(self, model_name: str):
        self.model = MODEL_REGISTRY.get(model_name)
        self._clear_params()

        if hasattr(self, "_on_model_changed"):
            self._on_model_changed()

        # Always enable parameters for nnUNetV2 models, even in InferenceWidget
        should_skip = self.SKIP_PARAMS
        if isinstance(self.model, NNUNetV2ModelSpec):
            should_skip = False

        if should_skip or self.model is None:
            self._set_param_section_visible(False)
            return

        # Show the parameters section if the model requires it
        self._set_param_section_visible(True)

        if isinstance(self.model, NNUNetV2ModelSpec) or self.AUTO_LOAD_PARAMS:
            if hasattr(self, "_load_params_btn"):
                self._load_params_btn.setVisible(False)
            self._load_tunable_params()
        else:
            if hasattr(self, "_load_params_btn"):
                self._load_params_btn.setVisible(True)
                self._load_params_btn.setEnabled(True)


    def _load_tunable_params(self, *args):
            if self.model is None:
                return

            self._param_load_id += 1
            current_id = self._param_load_id

            self._clear_params()
            self._param_status_lbl.setText("Loading parameters…")
            self._param_progress.setVisible(True)
            if hasattr(self, "_load_params_btn"):
                self._load_params_btn.setEnabled(False)

            # Safely clean up previous thread reference if it exists or was deleted by Qt
            if self._param_thread is not None:
                try:
                    if self._param_thread.isRunning():
                        self._param_thread.quit()
                        self._param_thread.wait(500)
                except RuntimeError:
                    # C++ object was already garbage collected by Qt
                    pass
                self._param_thread = None

            thread = QThread(self.widget)
            self._param_thread = thread
            self._param_worker = _ParamLoader(self.model)
            self._param_worker.moveToThread(thread)

            thread.started.connect(self._param_worker.run)
            self._param_worker.finished.connect(
                lambda schema: self._on_params_loaded(schema, current_id)
            )
            self._param_worker.error.connect(
                lambda msg: self._on_params_error(msg, current_id)
            )
            self._param_worker.finished.connect(thread.quit)
            self._param_worker.error.connect(thread.quit)

            # Clear python reference when thread finishes to prevent dead wrapper calls
            def _on_thread_finished():
                if getattr(self, "_param_thread", None) is thread:
                    self._param_thread = None

            thread.finished.connect(_on_thread_finished)
            thread.finished.connect(thread.deleteLater)
            thread.start()

    def _on_params_loaded(self, schema: dict, load_id: int):
        self._param_progress.setVisible(False)
        if hasattr(self, "_load_params_btn"):
            self._load_params_btn.setEnabled(True)

        if load_id != self._param_load_id:
            return

        if not schema:
            self._param_status_lbl.setText("No tunable parameters for this model.")
            return

        self._param_status_lbl.setText(f"{len(schema)} parameter(s) loaded.")
        for name, spec in schema.items():
            w = self._make_param_widget(name, spec)
            if w is not None:
                self.param_widgets[name] = w
                self._params_form.addRow(name, w.native)

    def _on_params_error(self, msg: str, load_id: int):
        self._param_progress.setVisible(False)
        if hasattr(self, "_load_params_btn"):
            self._load_params_btn.setEnabled(True)

        if load_id != self._param_load_id:
            return

        self._param_status_lbl.setText("Failed to load parameters.")
        self._log_line(f"[param error] {msg}", color=_RED)

    def _make_param_widget(self, name: str, spec):
        from magicgui.widgets import create_widget

        if not isinstance(spec, dict):
            return create_widget(value=spec, name=name)

        spec = dict(spec)
        annotation = spec.pop("type", None)
        value = spec.pop("default", None)

        if isinstance(annotation, str):
            type_map = {
                "int": int, "float": float, "str": str,
                "bool": bool, "list": list, "dict": dict
            }
            annotation = type_map.get(annotation.lower(), annotation)

        try:
            return create_widget(value=value, annotation=annotation, options=spec)
        except Exception:
            return create_widget(value=value, name=name)

    def _clear_params(self):
        self.param_widgets.clear()

        # 1. Unparent and schedule deletion for all child widgets inside the container
        if hasattr(self, "_params_container") and self._params_container:
            for child in self._params_container.findChildren(QWidget):
                child.setParent(None)
                child.deleteLater()

        # 2. Completely drain the layout without version-dependent Qt enums
        while self._params_form.count() > 0:
            item = self._params_form.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.hide()
                    w.deleteLater()

        self._param_status_lbl.setText("")

    def collect_params(self) -> dict:
        return {k: w.value for k, w in self.param_widgets.items()}

    # ------------------------------------------------------------------
    # Directory selectors (Non-native dialogs + Safe exception handling)
    # ------------------------------------------------------------------

    def _select_dataset_folder(self, *args):
        # Pass parent=None and DontUseNativeDialog so dialog pops up on top
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select dataset folder",
            options=QFileDialog.DontUseNativeDialog
        )
        if folder:
            self.dataset_dir = Path(folder)
            self._dataset_lbl.setText(str(self.dataset_dir))
            self._dataset_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")
            if self.dataset_manager is not None:
                try:
                    self.dataset_manager.load_from_folder(folder)
                except Exception as exc:
                    self._log_line(f"[dataset warning] {exc}", color=_AMBER)

            if isinstance(self.model, NNUNetV2ModelSpec):
                self.param_widgets['Set Name'].value = self.dataset_dir.name
                
    def _select_output_folder(self, *args):
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select output directory",
            options=QFileDialog.DontUseNativeDialog
        )
        if folder:
            self.output_dir = Path(folder)
            self._output_lbl.setText(str(self.output_dir))
            self._output_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")
            
        if isinstance(self.model, NNUNetV2ModelSpec):

            # Path to nnUNet_raw inside the selected output folder
            raw_dir = self.output_dir / "nnUNet_raw"

            # Default Set ID
            next_id = 1

            if raw_dir.exists() and raw_dir.is_dir():
                # Find all subfolders matching nnUNet dataset naming: DatasetXYZ_*
                max_id = 0
                for sub in raw_dir.iterdir():
                    if sub.is_dir() and sub.name.startswith("Dataset"):
                        # Expected format: DatasetXYZ_NAME
                        # Extract the numeric XYZ part
                        parts = sub.name.split("_")
                        if parts:
                            prefix = parts[0]  # "DatasetXYZ"
                            num_str = prefix.replace("Dataset", "")
                            if num_str.isdigit():
                                num = int(num_str)
                                if num > max_id:
                                    max_id = num

                # Increment largest ID
                next_id = max_id + 1 if max_id > 0 else 1

            # Update the magicgui widget
            if "Set ID" in self.param_widgets:
                self.param_widgets["Set ID"].value = next_id

    # ------------------------------------------------------------------
    # Run / terminate
    # ------------------------------------------------------------------

    def _on_run_clicked(self):
        if self.model is None:
            self._log_line("⚠  No model selected.", color=_AMBER)
            return
        if self.output_dir is None:
            self._log_line("⚠  Please select an output directory.", color=_AMBER)
            return
        if self._process and self._process.state() != QProcess.NotRunning:
            self._log_line("⚠  A process is already running.", color=_AMBER)
            return
        try:
            self._run_workflow()
        except Exception as exc:
            self._log_line(f"[error] {exc}", color=_RED)

    def _run_workflow(self):
        raise NotImplementedError

    def _extra_controls(self) -> Optional[QWidget]:
        return None

    def _on_terminate_clicked(self):
        if not (self._process and self._process.state() != QProcess.NotRunning):
            return
        reply = QMessageBox.question(
            self.widget, "Terminate process",
            "Terminate the running process?\nAny unsaved progress will be lost.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._kill_process()

    def _kill_process(self):
            if self._process is None:
                return

            pid = self._process.processId()
            if pid and hasattr(os, "killpg") and hasattr(os, "getpgid") and hasattr(os, "getpid"):
                try:
                    child_pgid = os.getpgid(pid)
                    gui_pgid = os.getpgid(os.getpid())

                    # Only kill the process group if it is distinct from the main GUI process group
                    if child_pgid != gui_pgid:
                        os.killpg(child_pgid, signal.SIGKILL)
                    else:
                        os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

            # Fallback to standard Qt QProcess termination
            self._process.kill()
            self._process.waitForFinished(2000)

            self._log_line("\n■  Process terminated by user.", color=_RED, bold=True)
            self._set_busy(False)
            self._process_chain.clear()

    def _on_close(self, event):
        """Cleanup running threads and processes when the window is closed."""
        # 1. Terminate background parameter loader thread
        if self._param_thread is not None:
            try:
                if self._param_thread.isRunning():
                    self._param_thread.quit()
                    self._param_thread.wait(1000)
            except (RuntimeError, Exception):
                pass
            self._param_thread = None

        # 2. Terminate dataset prep thread if subclass created one
        if hasattr(self, "_prep_thread") and self._prep_thread is not None:
            try:
                if self._prep_thread.isRunning():
                    self._prep_thread.quit()
                    self._prep_thread.wait(1000)
            except (RuntimeError, Exception):
                pass
            self._prep_thread = None

        # 3. Kill active process execution
        if self._process is not None:
            try:
                self._kill_process()
            except Exception:
                pass

        event.accept()

    # ------------------------------------------------------------------
    # QProcess — single step
    # ------------------------------------------------------------------

    def _start_process(
            self,
            program: str,
            args: list,
            label: str = "",
            env_extra: Optional[dict] = None,
            on_done: Optional[Callable[[int], None]] = None,
        ):
        self._set_busy(True)
        self._log_line(f"\n▶  {label or program}", bold=True)

        env = QProcessEnvironment.systemEnvironment()
        if env_extra:
            for k, v in env_extra.items():
                env.insert(str(k), str(v))

        self._process = QProcess(self.widget)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.setProcessEnvironment(env)
        self._process.readyRead.connect(self._on_output)
        self._process.finished.connect(
            lambda code, _: self._on_step_done(code, on_done)
        )
        self._process.start(program, [str(a) for a in args])

    def _on_output(self):
        data = self._process.readAll().data()
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = str(data)
        self._log_append(text)

    def _on_step_done(self, exit_code: int, on_done: Optional[Callable]):
        if exit_code == 0:
            self._log_line("✓  Done (exit 0)", color=_GREEN, bold=True)
        else:
            self._log_line(f"✗  Exited with code {exit_code}", color=_RED, bold=True)
            self._set_busy(False)
            self._process_chain.clear()
            if on_done:
                on_done(exit_code)
            return
        if on_done:
            on_done(exit_code)

    # ------------------------------------------------------------------
    # QProcess — sequential chain
    # ------------------------------------------------------------------

    def _run_chain(
        self,
        steps: list,
        env_extra: Optional[dict] = None,
        on_all_done: Optional[Callable[[int], None]] = None,
    ):
        self._process_chain = list(steps)
        self._chain_env     = env_extra or {}
        self._chain_on_done = on_all_done
        self._run_next_in_chain()

    def _run_next_in_chain(self):
        if not self._process_chain:
            self._set_busy(False)
            if self._chain_on_done:
                self._chain_on_done(0)
            return
        program, args, label = self._process_chain.pop(0)
        self._start_process(
            program, args, label=label,
            env_extra=self._chain_env,
            on_done=self._on_chain_step_done,
        )

    def _on_chain_step_done(self, exit_code: int):
        if exit_code != 0:
            self._set_busy(False)
            self._process_chain.clear()
            if self._chain_on_done:
                self._chain_on_done(exit_code)
            return
        self._run_next_in_chain()

    # ------------------------------------------------------------------
    # Busy state
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy)
        self._term_btn.setEnabled(busy)
        self._run_progress.setVisible(busy)
        self._model_combo.setEnabled(not busy)

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _log_append(self, text: str):
        self._log.moveCursor(QTextCursor.End)
        self._log.insertPlainText(text)
        self._log.moveCursor(QTextCursor.End)

    def _log_line(self, text: str, color: str = "", bold: bool = False):
        self._log.moveCursor(QTextCursor.End)
        if color or bold:
            b_o = "<b>" if bold else ""
            b_c = "</b>" if bold else ""
            c_o = f'<span style="color:{color};">' if color else ""
            c_c = "</span>" if color else ""
            safe = (text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace("\n", "<br>"))
            self._log.insertHtml(f"{b_o}{c_o}{safe}{c_c}{b_c}<br>")
        else:
            self._log.insertPlainText(text + "\n")
        self._log.moveCursor(QTextCursor.End)

    def _clear_log(self):
        self._log.clear()


# ---------------------------------------------------------------------------
# Style helpers (exported so subclasses can use them)
# ---------------------------------------------------------------------------

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "color: #aaa; font-size: 10px; font-weight: 700;"
        " text-transform: uppercase; letter-spacing: 0.5px;"
        " background: transparent;"
    )
    return lbl


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {_BD}; background: {_BD};")
    return line


def _combo_style() -> str:
    return (
        f"QComboBox {{ background: #1e1e1e; color: #d4d4d4;"
        f" border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }}"
        f"QComboBox::drop-down {{ border: none; }}"
        f"QComboBox QAbstractItemView {{ background: #2b2b2b; color: #d4d4d4;"
        f" selection-background-color: #3c3c3c; }}"
    )


def _secondary_btn_style() -> str:
    return (
        "QPushButton { background: #3c3c3c; color: #d4d4d4;"
        " border: 1px solid #555; border-radius: 4px; padding: 4px 10px; }"
        "QPushButton:hover { background: #4a4a4a; }"
        "QPushButton:disabled { color: #555; background: #2a2a2a; }"
    )

