
from __future__ import annotations
import json
import os
import sys
import signal
from pathlib import Path
from typing import Callable, Optional
from qtpy.QtCore import Qt, QProcess, QProcessEnvironment, QThread, QTimer
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QSpinBox,
    QListWidget, QFrame, QTextEdit,
    QFileDialog, QSplitter, QProgressBar, QComboBox,
    QMessageBox,QTableWidget,QTableWidgetItem
)

from AdaptFM.gui.widgets.metrics_widget import MetricRegistry

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
# Base widget
# ---------------------------------------------------------------------------

class BenchmarkWidget:
    """
    Shared base for TrainingWidget and InferenceWidget.

    Subclasses must implement:
        _run_workflow()       — kicks off the QProcess / chain
        _on_model_changed()   — optional; called after model + params reset
        _extra_controls()     — optional; QWidget inserted above Run button
    """

    WINDOW_TITLE = "Benchmarking"


    def __init__(self):
        self.metric                       = None
        self.ground_truth: Optional[Path] = None
        self.output_dir:  Optional[Path] = None
        self.param_widgets: dict         = {}
        self._predictions: dict          = {}
        self.registry=MetricRegistry
        self.registry_title = "Metric"

        self._process: Optional[QProcess]       = None
        self._param_thread: Optional[QThread]   = None
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

        # Metric
        ctrl.addWidget(_section_label(self.registry_title))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(list(self.registry.get_metrics()))
        self._metric_combo.setStyleSheet(_combo_style())
        # Connect AFTER build so the QTimer below is the only initial trigger
        ctrl.addWidget(self._metric_combo)

        # Ground Truth
        ctrl.addWidget(_section_label("Ground Truth"))
        ctrl.addLayout(self._path_row("_dataset_lbl", self._select_ground_truth))

        # Output
        ctrl.addWidget(_section_label("Output directory"))
        ctrl.addLayout(self._path_row("_output_lbl", self._select_output_folder))

        # CPU
        processes_row = QHBoxLayout()
        processes_row.addWidget(_section_label("Number of Processes"))
        self._cpu_spin = QSpinBox()
        self._cpu_spin.setRange(1, 64)
        self._cpu_spin.setValue(0)
        self._cpu_spin.setFixedWidth(70)
        processes_row.addWidget(self._cpu_spin)
        processes_row.addStretch()
        ctrl.addLayout(processes_row)


        self._pred_table = QTableWidget()
        self._pred_table.setColumnCount(2)
        self._pred_table.setHorizontalHeaderLabels(["Display Name", "Folder Path"])
        self._pred_table.horizontalHeader().setStretchLastSection(True)
        self._pred_table.setStyleSheet(_combo_style())
        self._pred_table.setEditTriggers(QTableWidget.AllEditTriggers)



        self._add_btn = QPushButton("Add Prediction Folder")
        self._add_btn.setStyleSheet(_secondary_btn_style())

        self._remove_btn = QPushButton("Remove Selected Predictions")
        self._remove_btn.setStyleSheet(_secondary_btn_style())

        self._add_btn.clicked.connect(self._add_prediction_folder)
        self._remove_btn.clicked.connect(self._remove_prediction_folder)

        ctrl.addWidget(_section_label("Predictions"))
        ctrl.addWidget(self._pred_table)
        self._pred_table.itemChanged.connect(self._on_pred_item_changed)


        pred_btn_row = QHBoxLayout()
        pred_btn_row.addWidget(self._add_btn)
        pred_btn_row.addWidget(self._remove_btn)
        ctrl.addLayout(pred_btn_row)

        # --- end new ---

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
        self._metric_combo.currentTextChanged.connect(self._on_metric_selected)
        initial = self._metric_combo.currentText()
        if initial:
            QTimer.singleShot(0, lambda: self._on_metric_selected())

        self._pred_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {_BG2};
                color: {_TEXT};
                border: 1px solid #444;
                gridline-color: #333;
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QTableWidget::item:hover {{
                background-color: #383838;
            }}
            /* Style cell editor when active */
            QLineEdit {{
                background-color: #111;
                color: #fff;
                border: 1px solid {_BLUE};
                border-radius: 2px;
            }}
        """)

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

    def _on_metric_selected(self):
        self.metric = self._metric_combo.currentText()


    # ------------------------------------------------------------------
    # Directory selectors (Non-native dialogs + Safe exception handling)
    # ------------------------------------------------------------------

    def _select_ground_truth(self):

        if self.metric == 'Compare Counts':
            self.ground_truth, _ = QFileDialog.getOpenFileName(None, "Select Ground Truth File",
                                                  options=QFileDialog.DontUseNativeDialog)
            
        else:
            self.ground_truth = QFileDialog.getExistingDirectory(None, "Select Ground Truth Folder",
                                                    options=QFileDialog.DontUseNativeDialog)
            
        if self.ground_truth:
        
            self._dataset_lbl.setText(str(self.ground_truth))
            self._dataset_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")

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


    
    def _add_prediction_folder(self):
        path = QFileDialog.getExistingDirectory(
            None, "Select Prediction Folder",
            options=QFileDialog.DontUseNativeDialog
        )
        if not path:
            return

        # Prevent duplicates
        if path in self._predictions.values():
            self._log_line(f"⚠ Folder already added: {path}", color=_AMBER)
            return

        # Default display name = folder name
        display_name = os.path.basename(path)

        # Ensure unique display name
        base = display_name
        i = 1
        while display_name in self._predictions:
            display_name = f"{base} ({i})"
            i += 1

        # Store mapping
        self._predictions[display_name] = path

        # Add row to table
        row = self._pred_table.rowCount()
        self._pred_table.insertRow(row)

        name_item = QTableWidgetItem(display_name)
        name_item.setToolTip("✏️ Double-click to edit display name")
        self._pred_table.setItem(row, 0, name_item)
        path_item = QTableWidgetItem(path)
        path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable) # Strip editable flag
        path_item.setToolTip(path)
        self._pred_table.setItem(row, 1, path_item)


    def _remove_prediction_folder(self):
        row = self._pred_table.currentRow()
        if row < 0:
            return

        display_name = self._pred_table.item(row, 0).text()

        # Remove from dict
        if display_name in self._predictions:
            del self._predictions[display_name]

        # Remove from table
        self._pred_table.removeRow(row)

    def _on_pred_item_changed(self, item):
        row = item.row()
        col = item.column()

        # Only handle edits to display name column
        if col != 0:
            return

        # Path cell may not exist yet (Qt fires itemChanged too early)
        path_item = self._pred_table.item(row, 1)
        if path_item is None:
            return  # ignore until row is fully populated

        path = path_item.text()
        new_name = item.text().strip()

        # Find old name
        old_name = None
        for name, p in self._predictions.items():
            if p == path:
                old_name = name
                break

        if old_name is None:
            return

        # Prevent collisions
        if new_name in self._predictions and new_name != old_name:
            new_name = f"{new_name}_copy"
            item.setText(new_name)

        # Update mapping
        self._predictions.pop(old_name)
        self._predictions[new_name] = path

    # ------------------------------------------------------------------
    # Run / terminate
    # ------------------------------------------------------------------

    def _on_run_clicked(self):
        if self.metric is None:
            self._log_line("⚠  No metric selected.", color=_AMBER)
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
        
        if self.ground_truth is None:
            self._log_line("⚠  Please select a ground truth folder or file.", color=_AMBER)
            return

        if self._pred_table.rowCount() == 0:
            self._log_line("⚠  Please select a set of predictions to analyze", color=_AMBER)
            return
        
        predictions_json = json.dumps(self._predictions)
        num_processes = str(self._cpu_spin.value())

        benchmark_cmd = [
            sys.executable, "-m","AdaptFM.gui.run_metric_subprocess",
            "--metric",self.metric,
            "--gt_dir",self.ground_truth,
            "--models_json",predictions_json,
            "--num_processes", num_processes,
            "--output_dir",self.output_dir
        ]

        self._start_process(
            benchmark_cmd[0],
            benchmark_cmd[1:],
            label=f'Benchmarking with [{self.metric}]',
            on_done=self._on_benchmark_done
        )
   

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


    def _on_benchmark_done(self, exit_code: int):
        self._set_busy(False)
        if exit_code == 0:
            self._log_line(
                f"\n✓  Benchmarking complete",
                color=_GREEN, bold=True,
            )
        else:
            self._log_line(
                f"\n✗  Benchmarking failed (exit {exit_code}).",
                color=_RED, bold=True,
            )

    # ------------------------------------------------------------------
    # Busy state
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy)
        self._term_btn.setEnabled(busy)
        self._run_progress.setVisible(busy)
        self._metric_combo.setEnabled(not busy)

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


