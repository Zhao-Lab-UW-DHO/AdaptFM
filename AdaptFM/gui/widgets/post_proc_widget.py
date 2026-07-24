from pathlib import Path

from qtpy.QtWidgets import QWidget, QVBoxLayout,QComboBox,QLabel,QFileDialog,QPushButton,QInputDialog
from AdaptFM.postprocessing.post_proc_registry import POSTPROC_REGISTRY


"""Widget has three parts
1. pick a pipeline
2. pick a folder to process
3. pick an output folder""" 

class PostProcessingWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Post Processing")
        self.resize(500, 500)

        self.folder2process = None
        self.output_folder = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Pipeline dropdown
        self.postproc_dropdown = QComboBox()
        self.postproc_dropdown.addItems(POSTPROC_REGISTRY.keys())
        self.postproc_dropdown.currentTextChanged.connect(self._update_output_visibility)
        self.layout.addWidget(self.postproc_dropdown)

        self._build_dir_selector()
        self._build_run_button()
        self._update_output_visibility(self.postproc_dropdown.currentText())

    # ---------- Directory selector ----------
    def _build_dir_selector(self):
        # Input folder button
        btn_in = QPushButton("Select input folder")
        btn_in.clicked.connect(
            lambda: self._select_folder("Select a folder to process", "folder2process")
        )
        self.layout.addWidget(btn_in)

        # Label to show selected input folder
        self.in_label = QLabel("Input folder: None")
        self.layout.addWidget(self.in_label)

        # --- Wrap output folder widgets in a container so we can hide/show them ---
        self.output_container = QWidget()
        out_layout = QVBoxLayout()
        self.output_container.setLayout(out_layout)

        # Output folder button
        btn_out = QPushButton("Select output folder")
        btn_out.clicked.connect(
            lambda: self._select_folder("Select an output folder", "output_folder")
        )
        out_layout.addWidget(btn_out)

        # Label to show selected output folder
        self.out_label = QLabel("Output folder: None")
        out_layout.addWidget(self.out_label)

        self.layout.addWidget(self.output_container)

    # ---------- Folder selection ----------
    def _select_folder(self, prompt, attr_name):
        folder = QFileDialog.getExistingDirectory(None, prompt)
        if folder:
            setattr(self, attr_name, Path(folder))

            if attr_name == "folder2process":
                self.in_label.setText(f"Input folder: {folder}")
            elif attr_name == "output_folder":
                self.out_label.setText(f"Output folder: {folder}")

        # ---------- Run button ----------
    def _build_run_button(self):
        btn = QPushButton("Run")
        btn.clicked.connect(self._run)
        self.layout.addWidget(btn)

    # ---------- Hide/show output folder ----------
    def _update_output_visibility(self, pipeline_name):
        hide_output = pipeline_name == "Rescale4DL"
        self.output_container.setVisible(not hide_output)
        self.output_folder=None


    def _run(self):
        pipeline_name = self.postproc_dropdown.currentText()
        pipeline = POSTPROC_REGISTRY[pipeline_name]

        # If pipeline is Rescale4dl, skip output folder requirement
        if pipeline_name != "Rescale4DL":
            if self.folder2process is None or self.output_folder is None:
                raise RuntimeError("Select an input and output folder")
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            
            gpu, returned_ok = QInputDialog.getInt(
            None,
            "Select GPU",
            "GPU:",
            value=0,
            min=0,
            max=16,
            step=1,
        )
            pipeline.run_postprocess(self.folder2process, self.output_folder, gpu)

        else:
            if self.folder2process is None:
                raise RuntimeError("Select an input folder")
            
            pipeline.run_postprocess(self.folder2process)
