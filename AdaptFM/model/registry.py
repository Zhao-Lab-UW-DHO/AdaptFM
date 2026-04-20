# AdaptFM/models/registry.py
from AdaptFM.model.nnUNetV2Spec import NNUNetV2ModelSpec
from AdaptFM.model.fmSpec import FoundationModelSpec, MicroSAMSpec,CellposeSAMSpec,SSVTSpec,Sammed3DSpec
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
        inference_wrapper_path = "AdaptFM.model.foundation_models.microSAM.inference_wrapper"
    ),

    "CellposeSAM": CellposeSAMSpec(
        name="CellposeSAM",
        conda_env=_read_prefix("cellpose_adapt"),
        module_path="cellpose.train",
        training_wrapper_path ="AdaptFM.model.foundation_models.cellposeSAM.train_wrapper",
        inference_wrapper_path = "AdaptFM.model.foundation_models.cellposeSAM.inference_wrapper"
    ),


    "SAMMed3D": Sammed3DSpec(
        name="SAM-Med3D",
        conda_env=_read_prefix("sammed3d_adapt"),
        module_path="AdaptFM.model.foundation_models.sammed3d.train_wrapper",
        training_wrapper_path="AdaptFM.model.foundation_models.sammed3d.train_wrapper",
        inference_wrapper_path="AdaptFM.model.foundation_models.sammed3d.inference_wrapper"
    ),

    "SSVT" : SSVTSpec(
        name = "SSVT",
        conda_env = str(Path(sys.prefix)),
        module_path  ="AdaptFM.model.foundation_models.SSVT.train_wrapper",
        training_wrapper_path="AdaptFM.model.foundation_models.SSVT.train_wrapper",
        inference_wrapper_path="AdaptFM.model.foundation_models.SSVT.inference_wrapper"
    ),
    
}

