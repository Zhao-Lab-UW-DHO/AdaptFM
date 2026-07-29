from pathlib import Path
import traceback

import inspect
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QLabel, 
    QFileDialog, QPushButton, QProgressBar, QMessageBox, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QCheckBox, QLineEdit, QSpinBox
)
from qtpy.QtCore import QThread, Signal

from AdaptFM.preprocessing.registry import PREPROC_REGISTRY

class CancelledError(Exception):
    pass # catch worker interrupt

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
            self.func(self.input_dir, self.output_dir, progress_callback=self._progress_wrapper, **self.extra_kwargs)
        except Exception as e:
            err_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(err_msg)
        finally:
            self.finished.emit()


class PreprocessingWidget(QWidget):
    RESERVED_PARAMS = {"input_dir", "output_dir", "progress_callback"}

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pre-Processing / Conversion")
        self.resize(500, 500)

        self.folder2process = None
        self.output_folder = None
        self.worker = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.preproc_dropdown = QComboBox()
        self.preproc_dropdown.addItems(PREPROC_REGISTRY.keys())
        self.preproc_dropdown.currentTextChanged.connect(self._on_pipeline_changed)
        self.layout.addWidget(self.preproc_dropdown)

        self._build_dir_selector()

        self.param_widgets = {}
        self.params_form_layout = QFormLayout()
        self.layout.addLayout(self.params_form_layout)


        self._build_run_button()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self._on_pipeline_changed(self.preproc_dropdown.currentText())

    def _build_dir_selector(self):
        self.btn_in = QPushButton("Select input folder")
        self.btn_in.clicked.connect(
            lambda: self._select_folder("Select a folder to process", "folder2process")
        )
        self.layout.addWidget(self.btn_in)

        self.in_label = QLabel("Input folder: None")
        self.layout.addWidget(self.in_label)

        self.btn_out = QPushButton("Select output folder")
        self.btn_out.clicked.connect(
            lambda: self._select_folder("Select an output folder", "output_folder")
        )
        self.layout.addWidget(self.btn_out)

        self.out_label = QLabel("Output folder: None")
        self.layout.addWidget(self.out_label)

    def _select_folder(self, prompt, attr_name):
        folder = QFileDialog.getExistingDirectory(self, prompt)
        if folder:
            setattr(self, attr_name, Path(folder)) 
            if attr_name == "folder2process":
                self.in_label.setText(f"Input folder: {folder}")
            elif attr_name == "output_folder":
                self.out_label.setText(f"Output folder: {folder}")

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

            default = param.default if param.default is not inspect.Parameter.empty else None
            is_optional = (param.default is None)

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

    def _build_run_button(self):
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self._on_run_click)
        self.layout.addWidget(self.btn_run)

    def _on_run_click(self):
        if self.worker and self.worker.isRunning():
            self.btn_run.setEnabled(False)
            self.btn_run.setText("Cancelling...")
            self.worker.cancel()
        else:
            self._run()

    def _run(self):
        if self.folder2process is None or self.output_folder is None:
            QMessageBox.warning(self, "Warning", "Select an input and output folder first.")
            return
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        pipeline_name = self.preproc_dropdown.currentText()
        conversion_func = PREPROC_REGISTRY[pipeline_name]

        extra_kwargs = self._get_extra_kwargs()

        self.progress_bar.setValue(0)
        self._set_ui_enabled(False)

        self.worker = PreProcWorker(
            conversion_func, 
            self.folder2process, 
            self.output_folder, 
            extra_kwargs=extra_kwargs
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(self._processing_finished)
        self.worker.start()

    def _set_ui_enabled(self, enabled: bool):
        self.btn_run.setEnabled(True) # turns into cancel button
        self.btn_in.setEnabled(enabled)
        self.btn_out.setEnabled(enabled)
        self.preproc_dropdown.setEnabled(enabled)
        for w in self.param_widgets.values():
            w.setEnabled(enabled)
        
        if not enabled:
            self.btn_run.setText("Cancel")
        else:
            self.btn_run.setText("Run")

    def _handle_error(self, err_msg):
        QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{err_msg}")

    def _processing_finished(self):
        self._set_ui_enabled(True)
        self.progress_bar.setValue(100)
        self.worker = None