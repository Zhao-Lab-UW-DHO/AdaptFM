"""
preprocessing_widget.py
------------------------
Pre-processing / conversion floating window.

Visually restyled to match ModelWorkflowWidget (dark theme, section labels,
splitter with controls on top / output log on bottom). All original
PreprocessingWidget logic (dynamic param widgets from function signature,
worker thread, cancel handling, folder selection) is unchanged.
"""

import inspect
import traceback
from pathlib import Path

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from AdaptFM.preprocessing.registry import PREPROC_REGISTRY

# ---------------------------------------------------------------------------
# Colours (matches ModelWorkflowWidget)
# ---------------------------------------------------------------------------

_GREEN = "#4caf50"
_AMBER = "#ff9800"
_RED = "#f44336"
_BLUE = "#42a5f5"
_BG = "#2b2b2b"
_BG2 = "#1e1e1e"
_BD = "#444"
_TEXT = "#e8e8e8"
_MUTED = "#888"

# Global dark stylesheet applied to the widget (extends ModelWorkflowWidget's
# version with the extra input types PreprocessingWidget's dynamic form uses)
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
    QSpinBox, QDoubleSpinBox {{
        background: {_BG2};
        color: #d4d4d4;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 2px 6px;
    }}
    QLineEdit {{
        background: {_BG2};
        color: #d4d4d4;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 2px 6px;
    }}
    QCheckBox {{
        background: transparent;
        color: {_TEXT};
    }}
"""


# ---------------------------------------------------------------------------
# Style helpers (same as model_widget.py)
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
        "QComboBox { background: #1e1e1e; color: #d4d4d4;"
        " border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView { background: #2b2b2b; color: #d4d4d4;"
        " selection-background-color: #3c3c3c; }"
    )


def _secondary_btn_style() -> str:
    return (
        "QPushButton { background: #3c3c3c; color: #d4d4d4;"
        " border: 1px solid #555; border-radius: 4px; padding: 4px 10px; }"
        "QPushButton:hover { background: #4a4a4a; }"
        "QPushButton:disabled { color: #555; background: #2a2a2a; }"
    )


class CancelledError(Exception):
    pass  # catch worker interrupt


class PreProcWorker(QThread):
    """Background thread to run the conversion without freezing Napari."""

    progress = Signal(int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, func, input_dir, output_dir, extra_kwargs=None):
        super().__init__()
        self.func = func
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.extra_kwargs = extra_kwargs or {}
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _progress_wrapper(self, percent: int):
        if self._is_cancelled:
            raise CancelledError("Processing cancelled by user.")
        self.progress.emit(percent)

    def run(self):
        try:
            self.func(
                self.input_dir,
                self.output_dir,
                progress_callback=self._progress_wrapper,
                **self.extra_kwargs,
            )
        except Exception as e:
            err_msg = f"{e!s}\n\n{traceback.format_exc()}"
            self.error.emit(err_msg)
        finally:
            self.finished.emit()


class PreprocessingWidget(QWidget):
    RESERVED_PARAMS = {"input_dir", "output_dir", "progress_callback"}

    WINDOW_TITLE = "Pre-Processing / Conversion"

    def __init__(self):
        super().__init__()

        self.setWindowTitle(self.WINDOW_TITLE)
        self.setStyleSheet(_WINDOW_STYLE)
        self.resize(660, 700)
        self.setMinimumWidth(520)

        self.folder2process = None
        self.output_folder = None
        self.worker = None
        self.param_widgets = {}
        self._had_error = False

        self._build()

        self._on_pipeline_changed(self.preproc_dropdown.currentText())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build(self):
        root = QVBoxLayout(self)
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
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {_BD}; height: 2px; }}"
        )
        root.addWidget(splitter)

        # ── Controls ────────────────────────────────────────────────── #
        ctrl_outer = QWidget()
        ctrl = QVBoxLayout(ctrl_outer)
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(8)

        # Pipeline
        ctrl.addWidget(_section_label("Pipeline"))
        self.preproc_dropdown = QComboBox()
        self.preproc_dropdown.addItems(PREPROC_REGISTRY.keys())
        self.preproc_dropdown.setStyleSheet(_combo_style())
        self.preproc_dropdown.currentTextChanged.connect(self._on_pipeline_changed)
        ctrl.addWidget(self.preproc_dropdown)

        # Input folder
        ctrl.addWidget(_section_label("Input folder"))
        ctrl.addLayout(
            self._build_dir_row(
                lbl_attr="in_label",
                btn_attr="btn_in",
                initial_text="Input folder: None",
                slot=lambda: self._select_folder(
                    "Select a folder to process", "folder2process"
                ),
            )
        )

        # Output folder
        ctrl.addWidget(_section_label("Output folder"))
        ctrl.addLayout(
            self._build_dir_row(
                lbl_attr="out_label",
                btn_attr="btn_out",
                initial_text="Output folder: None",
                slot=lambda: self._select_folder(
                    "Select an output folder", "output_folder"
                ),
            )
        )

        # Parameters
        ctrl.addWidget(_section_label("Parameters"))
        self.params_form_layout = QFormLayout()
        self.params_form_layout.setSpacing(4)
        self.params_form_layout.setLabelAlignment(Qt.AlignLeft)

        params_container = QWidget()
        params_container.setLayout(self.params_form_layout)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_container)
        params_scroll.setMinimumHeight(120)
        params_scroll.setMaximumHeight(340)
        params_scroll.setFrameShape(QFrame.NoFrame)
        ctrl.addWidget(params_scroll)

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
        clear_btn.setStyleSheet(
            f"color: {_MUTED}; font-size: 10px; border: none; background: transparent;"
        )
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

        # ── Run button ──────────────────────────────────────────────── #
        self._build_run_button(root)

        # ── Progress bar ────────────────────────────────────────────── #
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: #333; }}"
            f"QProgressBar::chunk {{ background: {_GREEN}; }}"
        )
        root.addWidget(self.progress_bar)

    def _build_dir_row(
        self, lbl_attr: str, btn_attr: str, initial_text: str, slot
    ) -> QHBoxLayout:
        """Labelled path display + Browse button (mirrors ModelWorkflowWidget._path_row)."""
        row = QHBoxLayout()
        lbl = QLabel(initial_text)
        lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        lbl.setWordWrap(True)
        setattr(self, lbl_attr, lbl)
        row.addWidget(lbl, stretch=1)

        btn = QPushButton("Browse…")
        btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        btn.setStyleSheet(_secondary_btn_style())
        btn.clicked.connect(slot)
        setattr(self, btn_attr, btn)
        row.addWidget(btn)
        return row

    def _build_run_button(self, root: QVBoxLayout):
        self.btn_run = QPushButton("▶  Run")
        self.btn_run.setStyleSheet(
            f"QPushButton {{ background: {_GREEN}; color: #000; border-radius: 5px;"
            f" padding: 6px 16px; font-size: 13px; font-weight: 700; }}"
            f"QPushButton:disabled {{ background: #3a5a3a; color: #666; }}"
        )
        self.btn_run.clicked.connect(self._on_run_click)
        root.addWidget(self.btn_run)

    # ------------------------------------------------------------------
    # Dynamic parameter form (unchanged logic)
    # ------------------------------------------------------------------

    def _on_pipeline_changed(self, pipeline_name):
        """Dynamically generates inputs based on function signature."""
        # Clear existing form rows
        while self.params_form_layout.rowCount() > 0:
            self.params_form_layout.removeRow(0)
        self.param_widgets.clear()

        if not pipeline_name or pipeline_name not in PREPROC_REGISTRY:
            return

        func = PREPROC_REGISTRY[pipeline_name]
        sig = inspect.signature(func)

        for name, param in sig.parameters.items():
            if name in self.RESERVED_PARAMS:
                continue

            default = (
                param.default if param.default is not inspect.Parameter.empty else None
            )
            is_optional = param.default is None

            # Generate appropriate Qt widget based on type or default value
            if isinstance(default, bool):
                widget = QCheckBox()
                widget.setChecked(default)
            elif isinstance(default, float):
                widget = QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setValue(default)
            elif isinstance(default, int) and not isinstance(default, bool):
                widget = QSpinBox()
                widget.setRange(-1000000, 1000000)
                widget.setValue(default)
            else:
                widget = QLineEdit()
                if default is not None:
                    widget.setText(str(default))
                elif is_optional:
                    widget.setPlaceholderText("optional")

            self.param_widgets[name] = widget
            self.params_form_layout.addRow(name, widget)

    def _get_extra_kwargs(self):
        """Collects current values from all dynamic parameter inputs."""
        kwargs = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                kwargs[name] = widget.isChecked()
            elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                kwargs[name] = widget.value()
            elif isinstance(widget, QLineEdit):
                val_text = widget.text()
                # Attempt light type casting
                try:
                    kwargs[name] = float(val_text) if "." in val_text else int(val_text)
                except ValueError:
                    kwargs[name] = val_text
        return kwargs

    # ------------------------------------------------------------------
    # Folder selection (unchanged logic)
    # ------------------------------------------------------------------

    def _select_folder(self, prompt, attr_name):
        folder = QFileDialog.getExistingDirectory(self, prompt)
        if folder:
            setattr(self, attr_name, Path(folder))
            if attr_name == "folder2process":
                self.in_label.setText(f"Input folder: {folder}")
            elif attr_name == "output_folder":
                self.out_label.setText(f"Output folder: {folder}")

    # ------------------------------------------------------------------
    # Run / cancel (unchanged logic, with output-log calls added)
    # ------------------------------------------------------------------

    def _on_run_click(self):
        if self.worker and self.worker.isRunning():
            self.btn_run.setEnabled(False)
            self.btn_run.setText("Cancelling...")
            self._log_line("…  Cancelling…", color=_AMBER)
            self.worker.cancel()
        else:
            self._run()

    def _run(self):
        if self.folder2process is None or self.output_folder is None:
            QMessageBox.warning(
                self, "Warning", "Select an input and output folder first."
            )
            return

        self.output_folder.mkdir(parents=True, exist_ok=True)
        pipeline_name = self.preproc_dropdown.currentText()
        conversion_func = PREPROC_REGISTRY[pipeline_name]

        extra_kwargs = self._get_extra_kwargs()

        self.progress_bar.setValue(0)
        self._set_ui_enabled(False)

        self._had_error = False
        self._log_line(f"\n▶  {pipeline_name}", bold=True)

        self.worker = PreProcWorker(
            conversion_func,
            self.folder2process,
            self.output_folder,
            extra_kwargs=extra_kwargs,
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(self._processing_finished)
        self.worker.start()

    def _set_ui_enabled(self, enabled: bool):
        self.btn_run.setEnabled(True)  # turns into cancel button
        self.btn_in.setEnabled(enabled)
        self.btn_out.setEnabled(enabled)
        self.preproc_dropdown.setEnabled(enabled)
        for w in self.param_widgets.values():
            w.setEnabled(enabled)

        if not enabled:
            self.btn_run.setText("■  Cancel")
        else:
            self.btn_run.setText("▶  Run")

    def _handle_error(self, err_msg):
        self._had_error = True
        self._log_line(f"[error] {err_msg}", color=_RED)

    def _processing_finished(self):
        self._set_ui_enabled(True)
        self.progress_bar.setValue(100)

        if self.worker is not None and getattr(self.worker, "_is_cancelled", False):
            self._log_line("■  Cancelled by user.", color=_RED, bold=True)
        elif not self._had_error:
            self._log_line("✓  Done", color=_GREEN, bold=True)

        self.worker = None

    # ------------------------------------------------------------------
    # Log (same implementation as ModelWorkflowWidget)
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
            safe = (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            self._log.insertHtml(f"{b_o}{c_o}{safe}{c_c}{b_c}<br>")
        else:
            self._log.insertPlainText(text + "\n")
        self._log.moveCursor(QTextCursor.End)

    def _clear_log(self):
        self._log.clear()
