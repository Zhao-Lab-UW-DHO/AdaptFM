# gui_scripts/models/registry.py
from gui_scripts.model.nnUNetV2Spec import NNUNetV2ModelSpec
from gui_scripts.model.fmSpec import FoundationModelSpec, MicroSAMSpec,CellposeSAMSpec

MODEL_REGISTRY = {
    "nnUNetv2": NNUNetV2ModelSpec(),

    "microSAM": MicroSAMSpec(
        name="microSAM",
        conda_env="/mnt/local/data3/conda_envs/micro-sam/",
        module_path="micro_sam.training",
        training_wrapper_path = "micro_sam.train_wrapper",
        inference_wrapper_path = "micro_sam.inference_wrapper"
    ),

    "CellposeSAM": CellposeSAMSpec(
        name="CellposeSAM",
        conda_env="/mnt/local/data3/conda_envs/cellpose/",
        module_path="cellpose.train",
        training_wrapper_path ="cellpose.train_wrapper",
        inference_wrapper_path = "cellpose.inference_wrapper"
    ),

    "SAMMed3D" : NNUNetV2ModelSpec(),

    "SSL ViT-MAE (Organoids)" : NNUNetV2ModelSpec()
}
