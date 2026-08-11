# AdaptFM/models/registry.py
from AdaptFM.model.nnUNetV2Spec import NNUNetV2ModelSpec,MerlinNNUNetV2ModelSpec
from AdaptFM.model.fmSpec import FoundationModelSpec, MicroSAMSpec,CellposeSAMSpec,SSVTSpec,Sammed3DSpec,CellSAMSpec,BMEXSpec
from AdaptFM.model.foundation_models.ctfm.ctfm_spec import CTFMSpec
import subprocess
from pathlib import Path
import sys

def _read_prefix(env: str) -> str | None:
    p = Path.home() / ".adaptfm" / f"{env}.prefix"
    return p.read_text().strip() if p.exists() else None


def _conda_prefix(env: str) -> str:
    result = subprocess.run(
        ["conda", "run", "-n", env, "python", "-c", "import sys; print(sys.prefix)"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


MODEL_REGISTRY = {
    "nnUNetv2": NNUNetV2ModelSpec(
        conda_env = _read_prefix("nnUNet_adapt")
    ),
    
    "microSAM": MicroSAMSpec(
        name="microSAM",
        conda_env=_read_prefix("micro-sam_adapt"),
        module_path="micro_sam.training",
        training_wrapper_path = "AdaptFM.model.foundation_models.microSAM.train_wrapper",
        inference_wrapper_path = "AdaptFM.model.foundation_models.microSAM.inference_wrapper",
        training_function = 'train_sam'
    ),

    "CellposeSAM": CellposeSAMSpec(
        name="CellposeSAM",
        conda_env=_read_prefix("cellpose_adapt"),
        module_path="cellpose.train",
        training_wrapper_path ="AdaptFM.model.foundation_models.cellposeSAM.train_wrapper",
        inference_wrapper_path = "AdaptFM.model.foundation_models.cellposeSAM.inference_wrapper",
        training_function = 'train_seg'
    ),


    "SAMMed3D": Sammed3DSpec(
        name="SAM-Med3D",
        conda_env=_read_prefix("sammed3d_adapt"),
        module_path="AdaptFM.model.foundation_models.sammed3d.train_wrapper",
        training_wrapper_path="AdaptFM.model.foundation_models.sammed3d.train_wrapper",
        inference_wrapper_path="AdaptFM.model.foundation_models.sammed3d.inference_wrapper",
        training_function = 'launch_training'
    ),

    "SSVT" : SSVTSpec(
        name = "SSVT",
        conda_env = str(Path(sys.prefix)),
        module_path  ="AdaptFM.model.foundation_models.SSVT.train_wrapper",
        training_wrapper_path="AdaptFM.model.foundation_models.SSVT.train_wrapper",
        inference_wrapper_path="AdaptFM.model.foundation_models.SSVT.inference_wrapper",
        training_function = 'train_SSVT'
    ),
    "CellSAM": CellSAMSpec(
    name = "CellSAM",
    conda_env = _read_prefix("cellsam_adapt"),
    module_path= "",#does not support training/finetuning
    training_wrapper_path = "",
    inference_wrapper_path = "AdaptFM.model.foundation_models.cellSAM.inference_wrapper",
    training_function=""#does not support training/finetuning
    ),
    
    
    "BMEX": BMEXSpec(
    name = "BME-X",
    conda_env = _read_prefix("BME-X_adapt"),
    module_path= "AdaptFM.model.foundation_models.bmex.train_wrapper",
    training_wrapper_path = "AdaptFM.model.foundation_models.bmex.train_wrapper",
    inference_wrapper_path = "AdaptFM.model.foundation_models.bmex.inference_wrapper",
    training_function="launch_training"#does not support training/finetuning
    ),
    
    "CT FM": CTFMSpec(
        name="CTFM",
        conda_env = _read_prefix("CTFM_adapt"),
        module_path = "AdaptFM.model.foundation_models.ctfm.training_utils", 
        training_wrapper_path = "",
        inference_wrapper_path = "AdaptFM.model.foundation_models.ctfm.inference_wrapper",
        training_function="launch_training"

    ),

    "Merlin nnUNet" : MerlinNNUNetV2ModelSpec(
        conda_env = _read_prefix("Merlin_nnUNet_adapt"),
        transform_path = "AdaptFM.model.foundation_models.merlin.transforms"
    )
    
}

