# AdaptFM/models/registry.py
from AdaptFM.model.nnUNetV2Spec import NNUNetV2ModelSpec
from AdaptFM.model.fmSpec import FoundationModelSpec, MicroSAMSpec,CellposeSAMSpec,SSVTSpec,Sammed3DSpec

MODEL_REGISTRY = {
    "nnUNetv2": NNUNetV2ModelSpec(
        conda_env = "/path/to/conda_envs/nnunet"
    ),
    
    "microSAM": MicroSAMSpec(
        name="microSAM",
        conda_env="/path/to/conda_envs/microsam",
        module_path="micro_sam.training",
        training_wrapper_path = "AdaptFM.model.foundation_models.microSAM.train_wrapper",
        inference_wrapper_path = "AdaptFM.model.foundation_models.microSAM.inference_wrapper"
    ),

    "CellposeSAM": CellposeSAMSpec(
        name="CellposeSAM",
        conda_env="/path/to/conda_envs/cellpose",
        module_path="cellpose.train",
        training_wrapper_path ="AdaptFM.model.foundation_models.cellposeSAM.train_wrapper",
        inference_wrapper_path = "AdaptFM.model.foundation_models.cellposeSAM.inference_wrapper"
    ),


    "SAMMed3D": Sammed3DSpec(
        name="SAM-Med3D",
        conda_env="/path/to/conda_envs/sammed3d",
        module_path="AdaptFM.model.foundation_models.sammed3d.train_wrapper",
        training_wrapper_path="AdaptFM.model.foundation_models.sammed3d.train_wrapper",
        inference_wrapper_path="AdaptFM.model.foundation_models.sammed3d.inference_wrapper"
    ),

    "SSVT" : SSVTSpec(
        name = "SSVT",
        conda_env ="/path/to/AdaptFM/conda_env",
        module_path  ="AdaptFM.model.foundation_models.SSVT.train_wrapper",
        training_wrapper_path="AdaptFM.model.foundation_models.SSVT.train_wrapper",
        inference_wrapper_path="AdaptFM.model.foundation_models.SSVT.inference_wrapper"
    ),
    
}

SAM_REGISTRY={   "SAM2": {"Checkpoint Path" : '/path/to/SAM3/checkpoints/',
            "Repo Root": 'path/to/SAM3/repo/'},

    "SAM3": {"Checkpoint Path" : '/path/to/SAM2/checkpoints/',
        "Repo Root": 'path/to/SAM2/repo/'}}
