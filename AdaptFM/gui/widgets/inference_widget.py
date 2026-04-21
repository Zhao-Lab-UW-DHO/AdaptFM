from AdaptFM.gui.widgets.model_widget import ModelWorkflowWidget
from qtpy.QtWidgets import QFileDialog, QPushButton,QInputDialog

from pathlib import Path

class InferenceWidget(ModelWorkflowWidget):
    TAG_LABEL = "Inference tag"

    def __init__(self, dataset_manager):
        self.checkpoint_path = None
        self.skip_tag = None
        super().__init__(dataset_manager)

        self._build_checkpoint_selector()

    def _build_checkpoint_selector(self):
        btn = QPushButton("Select model checkpoint")
        btn.clicked.connect(self._select_checkpoint)
        self.layout.insertWidget(3, btn)

    def _select_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Select checkpoint",
            "/mnt/local/data3/demo_files/checkpoints"
        )
        if path:
            self.checkpoint_path = Path(path)

    def _run(self):
        if not all([self.model, self.dataset_dir]):
            raise RuntimeError("Missing dataset")

        params = self.collect_params()
        output_dir = Path(QFileDialog.getExistingDirectory(
            None, "Select output directory",
            '/mnt/local/data3/Organoids/Data/broad_data_testing'))
        
                # 2. Prompt for GPU
        gpu, _ = QInputDialog.getInt(
            None,
            "Select GPU",
            "GPU index:",
            value=0,
            min=0,
            max=16,   # adjust if you want
            step=1,
        )

        params["gpu"] = gpu

        self.model.run_inference(
            dataset_dir=self.dataset_dir,
            checkpoint=self.checkpoint_path,
            output_dir=output_dir,
            params=params,
        )
    def _on_model_changed(self):
        print('_on model changed running')
        self.checkpoint_path = None
