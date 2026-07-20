from pathlib import Path
import traceback

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QLabel, 
    QFileDialog, QPushButton, QProgressBar, QMessageBox
)
from qtpy.QtCore import QThread, Signal

from AdaptFM.preprocessing.registry import PREPROC_REGISTRY


class PreProcWorker(QThread):
    """Background thread to run the conversion without freezing Napari."""
    progress = Signal(int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, func, input_dir, output_dir):
        super().__init__()
        self.func = func
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        try:
            self.func(self.input_dir, self.output_dir, self.progress.emit)
        except Exception as e:
            err_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(err_msg)
        finally:
            self.finished.emit()


class PreprocessingWidget(QWidget):
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
        self.layout.addWidget(self.preproc_dropdown)

        self._build_dir_selector()
        self._build_run_button()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

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

    def _build_run_button(self):
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self._run)
        self.layout.addWidget(self.btn_run)

    def _run(self):
        if self.folder2process is None or self.output_folder is None:
            QMessageBox.warning(self, "Warning", "Select an input and output folder first.")
            return
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        pipeline_name = self.preproc_dropdown.currentText()
        conversion_func = PREPROC_REGISTRY[pipeline_name]

        self.progress_bar.setValue(0)
        self._set_ui_enabled(False)

        self.worker = PreProcWorker(conversion_func, self.folder2process, self.output_folder)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(self._processing_finished)
        self.worker.start()

    def _set_ui_enabled(self, enabled: bool):
        """Helper to lock/unlock the UI so users don't click 'Run' multiple times."""
        self.btn_run.setEnabled(enabled)
        self.btn_in.setEnabled(enabled)
        self.btn_out.setEnabled(enabled)
        self.preproc_dropdown.setEnabled(enabled)
        
        if not enabled:
            self.btn_run.setText("Processing...")
        else:
            self.btn_run.setText("Run")

    def _handle_error(self, err_msg):
        QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{err_msg}")

    def _processing_finished(self):
        self._set_ui_enabled(True)
        self.progress_bar.setValue(100)

        self.worker = None