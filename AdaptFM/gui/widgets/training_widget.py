from AdaptFM.gui.napari_utils import expand_param_grid
from AdaptFM.gui.widgets.model_widget import ModelWorkflowWidget
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QInputDialog

from pathlib import Path

class TrainingWidget(ModelWorkflowWidget):
    TAG_LABEL = "Training tag"
    
    def __init__(self,dataset_manager):
        super().__init__(dataset_manager)
        self.widget.setWindowTitle("Training")  

    def _run(self):
        if self.model is None or self.dataset_dir is None:
            raise RuntimeError("Model and dataset must be selected")

        params = self.collect_params()

        output_dir = Path(QFileDialog.getExistingDirectory(
            self.widget, "Select output directory"
        ))
        
        gpu, returned_ok = QInputDialog.getInt(
            self.widget,
            "Select GPU",
            "GPU index:",
            value=0,
            min=0,
            max=16,   # adjust if you want
            step=1,
        )

        if not returned_ok:
            return
        
        params["gpu"] = gpu

        prepared_dataset = self.model.prepare_dataset(
            self.dataset_manager,
            output_dir,
            params=params
        )


        if self.model.name =='nnUNetV2':


            preprocess_proc = self.model.run_preprocessing(
                dataset_dir=prepared_dataset,
                params=params,
                output_dir=output_dir
                )
            
        # Block until preprocessing is done
            return_code = preprocess_proc.wait()
            
            if return_code != 0:
                raise RuntimeError(f"Preprocessing failed with return code {return_code}")

        self.model.run_training(
            dataset_info=prepared_dataset,
            params=params,
            run_dir=output_dir
        )
