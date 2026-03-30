from AdaptFM.gui.napari_utils import expand_param_grid
from AdaptFM.gui.widgets.model_widget import ModelWorkflowWidget
from qtpy.QtWidgets import QFileDialog
from pathlib import Path

class TrainingWidget(ModelWorkflowWidget):
    TAG_LABEL = "Training tag"

    def _run(self):
        if self.model is None or self.dataset_dir is None:
            raise RuntimeError("Model and dataset must be selected")

        params = self.collect_params()
        expanded_params = list(expand_param_grid(params))

        run_dir = Path(QFileDialog.getExistingDirectory(
            None, "Select output directory"
        ))

        prepared_dataset = self.model.prepare_dataset(
            self.dataset_manager,
            run_dir / "dataset"
        )

        if self.model.name =='nnUNetv2':

            self.model.run_preprocessing(
                dataset_dir=prepared_dataset,
                params=params,
                run_dir=run_dir / "preprocessing"
                )

         
        for p in expanded_params:
            self.model.run_training(
                dataset_info=prepared_dataset,
                params=p,
                run_dir=run_dir
            )
