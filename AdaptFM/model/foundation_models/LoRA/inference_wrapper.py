import argparse
from tqdm import tqdm
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import  Dataset
import tifffile
import torch.nn.functional as F



def read_tiff(path: Path):
    """
    Reads a 3D TIFF and extracts spacing from metadata.
    Returns:
        volume: np.ndarray [Z, Y, X]
        spacing: (sz, sy, sx)
    """

    volume = tifffile.imread(path)

    return volume # , (sz, sy, sx)

def save_lora_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    model_env: str,
    is_3d: bool,
    target_size: tuple,
    ckpt_path: str,
    lora_config: dict):

    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
        
    config = {
                "model_env": model_env,
                "is_3d": is_3d,
                "target_size": list(target_size),
                "ckpt_path": ckpt_path,
                "lora_config":lora_config
            }

    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=4)

    trainable_state_dict = {
        k: v.cpu() for k,v in model.state_dict().items() if v.requires_grad
    }

    torch.save(trainable_state_dict, save_dir / "adapter_model.pt")
    print(f"Model saved to {save_dir}")
    return


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad=False
    return model


def load_model_with_architecture(model_env:str,ckpt_path:str,device):

    if 'cellpose' in model_env.lower(): #tested

        from cellpose import models
        model = models.CellposeModel(gpu=True)
        model.net = freeze_parameters(model.net)

        is_3d = False
        expected_channels = 3
        return model.net,is_3d,expected_channels
    
    elif 'micro-sam' in model_env.lower(): #tested
        import types


        class MicroSamAdapter(nn.Module):
            """Wraps Micro-SAM, removes Meta's @torch.no_grad() decorator, and exposes a

            standard (B, C, H, W) -> (B, 1, H, W) PyTorch interface.
            """

            def __init__(self, sam_model: nn.Module):
                super().__init__()
                self.model = sam_model

                # Strip Meta's @torch.no_grad() decorator from SAM's forward method
                if hasattr(self.model.forward, "__wrapped__"):
                    self.model.forward = types.MethodType(
                        self.model.forward.__wrapped__, self.model
                    )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, C, H, W = x.shape

                # 1. Expand single channel to 3 channels for SAM
                x_3ch = x.repeat(1, 3, 1, 1) if C == 1 else x

                # 2. Format batched input dictionary
                batched_input = [
                    {"image": x_3ch[i], "original_size": (H, W)} for i in range(B)
                ]

                # 3. Forward pass through SAM (now with autograd enabled)
                outputs = self.model(batched_input, multimask_output=False)

                # 4. Extract low-res logits
                logits_list = [
                    out.get("low_res_logits", out.get("masks")) for out in outputs
                ]
                logits = torch.cat(logits_list, dim=0)

                if logits.shape[1] > 1:
                    logits = logits[:, :1, ...]

                # 5. Interpolate to original resolution
                if logits.shape[-2:] != (H, W):
                    logits = F.interpolate(
                        logits, size=(H, W), mode="bilinear", align_corners=False
                    )

                return logits
            

        from micro_sam.automatic_segmentation import get_predictor_and_segmenter

        predictor, segmenter = get_predictor_and_segmenter("vit_b_lm", device="cuda")

        # Wrap raw SAM model in adapter
        raw_model = predictor.model
        model = MicroSamAdapter(raw_model)

        # Freeze base parameters (LoRA will inject onto model.model inside)
        for param in model.parameters():
            param.requires_grad = False

        is_3d = False
        expected_channels =3
        return model, is_3d,expected_channels
            
    
    elif 'BME-X' in model_env: #tested
        from DUNet3D import DenseUNet3d
        model = DenseUNet3d()
        model_dict = torch.load(ckpt_path, map_location=("cpu"))
        model.load_state_dict(model_dict["state_dict"])
        model = model.to(device)
        model = freeze_parameters(model)
        is_3d=True
        expected_channels=1
        return model,is_3d,expected_channels

    elif 'sammed3d' in model_env.lower(): #NEEDS TESTING
        model = torch.load(ckpt_path,map_location=device,weights_only=False)
        model= model.to(device)
        is_3d=True
        return model,is_3d
    
    elif 'cellsam' in model_env.lower(): #tested
        from cellSAM import get_model
        model = get_model(model="cellsam_extra")
        model = model.to(device)
        model = freeze_parameters(model)
        is_3d=False
        expected_channels=3
        return model,is_3d,expected_channels

    elif 'ctfm' in model_env.lower(): #tested
        from lighter_zoo import SegResNet
        model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
        old_head = model.up_layers[-1].head
        model.up_layers[-1].head = nn.Conv3d(
            in_channels=old_head.in_channels,
            out_channels=2,
            kernel_size=old_head.kernel_size,
            stride=old_head.stride,
            padding=old_head.padding,
        )
        model = model.to(device)
        model = freeze_parameters(model)
        is_3d=True
        expected_channels=1
        return model, is_3d,expected_channels
    
    elif 'adaptfm' in model_env.lower(): #TESTED- user must enter target size of 4,128,128

        from AdaptFM.SSVT.Vit_class import (
            MAE3DUNetDecoderBig,
            MAE3DSegmentation,
            ViTEncoder3D,
        )
        encoder = ViTEncoder3D(
            patch_size=(2, 16, 16), embed_dim=768, depth=8, num_heads=8
        ).to(device)

        # initialize decoder based on fixed parameters

        dummy_patch = torch.zeros(1, 1, *(4, 128, 128), device=device)
        with torch.no_grad():
            _, patch_grid = encoder(dummy_patch, return_grid=True)

        decoder = MAE3DUNetDecoderBig(
            embed_dim=768, patch_grid=patch_grid
        )  # change to MAE3DUNetDecoder if you want the UNET decoder for more complex segmentation tasks
        model = MAE3DSegmentation(encoder, decoder).to(device)

        # load model checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        model = freeze_parameters(model)

        is_3d=True
        return model,is_3d
    
    elif 'merlin' in model_env.lower(): # TESTED - user must use sizepatch size of 112,128,128
        from typing import Tuple, Union


        import nnunetv2
        from batchgenerators.utilities.file_and_folder_operations import join, load_json
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from nnunetv2.utilities.label_handling.label_handling import (
            determine_num_input_channels,
        )
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


        def load_merlin_model(
            model_training_output_dir: str,
            fold: Union[int, str] = 0,
            checkpoint_name: str = "checkpoint_final.pth",
            device: torch.device = torch.device("cuda"),
        ) -> Tuple[nn.Module, bool]:
            """Loads an nnUNet architecture and weights for this nnunetv2 version, freezes

            all parameters, and returns (model, is_3d).
            """
            # 1. Load plans and dataset metadata
            dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
            plans = load_json(join(model_training_output_dir, "plans.json"))
            plans_manager = PlansManager(plans)

            # 2. Load fold checkpoint with weights_only=False
            ckpt_path = join(
                model_training_output_dir, f"fold_{fold}", checkpoint_name
            )
            checkpoint = torch.load(
                ckpt_path, map_location=torch.device("cpu"), weights_only=False
            )

            trainer_name = checkpoint["trainer_name"]
            configuration_name = checkpoint["init_args"]["configuration"]
            configuration_manager = plans_manager.get_configuration(configuration_name)

            # 3. Locate trainer class using recursive_find_python_class
            num_input_channels = determine_num_input_channels(
                plans_manager, configuration_manager, dataset_json
            )
            trainer_class = recursive_find_python_class(
                join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                trainer_name,
                "nnunetv2.training.nnUNetTrainer",
            )
            if trainer_class is None:
                raise RuntimeError(
                    f"Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer."
                )

            # 4. Build architecture without deep supervision
            num_output_channels = plans_manager.get_label_manager(
                dataset_json
            ).num_segmentation_heads
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision=False,
            )

            # 5. Restore state dict, transfer device, and freeze parameters
            network.load_state_dict(checkpoint["network_weights"])
            network.to(device)

            for param in network.parameters():
                param.requires_grad = False

            # 6. Check if model is 3D based on patch size dimension rank
            is_3d = len(configuration_manager.patch_size) == 3

            return network, is_3d
        
        model, is_3d = load_merlin_model(
            model_training_output_dir=ckpt_path, fold='0', device=device
        )
        expected_channels=1
        return model, is_3d,expected_channels


    elif 'nnunet' in model_env.lower(): #tested

        import inspect
        import warnings
        from typing import Tuple, Union


        from batchgenerators.utilities.file_and_folder_operations import join, load_json
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
        from nnunetv2.utilities.label_handling.label_handling import (
            determine_num_input_channels,
        )
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


        def load_nnunet_model(
            model_training_output_dir: str,
            fold: Union[int, str] = 0,
            checkpoint_name: str = "checkpoint_best.pth",
            device: torch.device = torch.device("cuda"),
        ) -> Tuple[nn.Module, bool]:
            """Loads an nnUNet model architecture and weights from a training folder,

            freezes all parameters, and returns (model, is_3d).
            """
            # 1. Load plans and dataset configuration
            dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
            plans = load_json(join(model_training_output_dir, "plans.json"))
            plans_manager = PlansManager(plans)

            # 2. Load fold checkpoint
            ckpt_path = join(
                model_training_output_dir, f"fold_{fold}", checkpoint_name
            )
            checkpoint = torch.load(
                ckpt_path, map_location=torch.device("cpu"), weights_only=False
            )

            trainer_name = checkpoint["trainer_name"]
            configuration_name = checkpoint["init_args"]["configuration"]
            configuration_manager = plans_manager.get_configuration(configuration_name)

            # 3. Determine channels and trainer class
            num_input_channels = determine_num_input_channels(
                plans_manager, configuration_manager, dataset_json
            )
            trainer_class = recursive_find_trainer_class_by_name(trainer_name)
            num_output_channels = plans_manager.get_label_manager(
                dataset_json
            ).num_segmentation_heads

            # 4. Build network architecture without deep supervision
            sig = inspect.signature(trainer_class.build_network_architecture)
            if "plans_manager" in sig.parameters:
                network = trainer_class.build_network_architecture(
                    plans_manager,
                    configuration_manager,
                    num_input_channels,
                    num_output_channels,
                    enable_deep_supervision=False,
                )
            else:
                network = trainer_class.build_network_architecture(
                    configuration_manager.network_arch_class_name,
                    configuration_manager.network_arch_init_kwargs,
                    configuration_manager.network_arch_init_kwargs_req_import,
                    num_input_channels,
                    num_output_channels,
                    enable_deep_supervision=False,
                )

            # 5. Load weights, move to device, and freeze base parameters
            network.load_state_dict(checkpoint["network_weights"])
            network.to(device)

            for param in network.parameters():
                param.requires_grad = False

            # 6. Check if architecture is 3D based on patch size rank
            is_3d = len(configuration_manager.patch_size) == 3

            return network, is_3d
        
        # `ckpt_path` points to the nnUNet model directory (containing fold_0, dataset.json, plans.json)
        model, is_3d = load_nnunet_model(
            model_training_output_dir=ckpt_path, fold='all', device=device
        )
        expected_channels=1
        return model, is_3d,expected_channels
    

import torch
import torch.nn.functional as F


def process_array(
    arr,
    config,
    device="cuda",
    expected_channels=3,
    dtype=torch.float32,
) -> torch.Tensor:
    """Converts a NumPy array/Tensor into a batch-ready PyTorch tensor [B, C,

    ...] matching the target channels, resolution, device, and dtype.
    """
    is_3d = config['is_3d']
    target_size = config['target_size']
    model_env = config['model_env']
    


    if isinstance(arr, torch.Tensor):
        tensor = arr.to(dtype=dtype)
    else:
        tensor = torch.from_numpy(arr).to(dtype=dtype)

    if not is_3d:
        # 1. Normalize 2D shape to (C, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        elif tensor.ndim == 3:
            if tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            else:
                tensor = tensor.max(dim=0)[0].unsqueeze(
                    0
                )  # Max projection -> (1, H, W)

        # 2. Expand 1-channel grayscale to expected channels (e.g. 3 for ViT / SAM / Cellpose)
        if tensor.shape[0] == 1 and expected_channels == 3:
            tensor = tensor.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
        elif tensor.shape[0] > expected_channels:
            tensor = tensor[:expected_channels, ...]

        # 3. Add Batch Dimension [1, C, H, W] & Interpolate Spatial Dimensions
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(
            tensor, size=target_size, mode="bilinear", align_corners=False
        )

    else:
        # 1. Normalize 3D shape to (C, D, H, W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (D, H, W) -> (1, D, H, W)
        elif tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(3, 0, 1, 2)  # (D, H, W, C) -> (C, D, H, W)

        # 2. Expand channels if 3D model expects 3 input channels
        if tensor.shape[0] == 1 and expected_channels == 3:
            tensor = tensor.repeat(3, 1, 1, 1)

        # 3. Add Batch Dimension [1, C, D, H, W] & Interpolate Volumetric Dimensions
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(
            tensor, size=target_size, mode="trilinear", align_corners=False
        )

    # Do NOT squeeze batch dimension; return batch-ready [B, C, ...] directly on device
    return tensor.to(device=device)


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear layer with trainable low-rank matrices A and B."""

    def __init__(self, original_layer: nn.Linear, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.original_layer = original_layer

        for param in self.original_layer.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        if r > 0:
            dtype = original_layer.weight.dtype
            device = original_layer.weight.device
            self.lora_A = nn.Parameter(
                torch.zeros((r, in_features), dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, r), dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        self.dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

    # Expose weight and bias properties for PyTorch modules like MultiheadAttention
    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_layer(x)
        if self.r > 0:
            lora_A = self.lora_A.to(dtype=x.dtype)
            lora_B = self.lora_B.to(dtype=x.dtype)
            lora_output = (
                self.dropout(x) @ lora_A.T @ lora_B.T
            ) * self.scaling
            return base_output + lora_output
        return base_output


class LoRAConv3d(nn.Module):
    """Wraps an existing nn.Conv3d layer with trainable low-rank 3D convolutions."""

    def __init__(self, original_layer: nn.Conv3d, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        

        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 1.0

        if r > 0:
            dtype = original_layer.weight.dtype
            device = original_layer.weight.device

            self.lora_A = nn.Conv3d(
                original_layer.in_channels,
                r,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                groups=original_layer.groups,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.lora_B = nn.Conv3d(
                r,
                original_layer.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                dtype=dtype,
                device=device,
            )

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        self.dropout = (
            nn.Dropout3d(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_layer(x)
        if self.r > 0:
            lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            return base_output + lora_output
        return base_output


class LoRAConv2d(nn.Module):
    """Wraps an existing nn.Conv2d layer with trainable low-rank 2D convolutions."""

    def __init__(self, original_layer: nn.Conv2d, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.original_layer = original_layer

        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 1.0

        if r > 0:
            dtype = original_layer.weight.dtype
            device = original_layer.weight.device

            self.lora_A = nn.Conv2d(
                original_layer.in_channels,
                r,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                groups=original_layer.groups,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.lora_B = nn.Conv2d(
                r,
                original_layer.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                dtype=dtype,
                device=device,
            )

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        self.dropout = (
            nn.Dropout2d(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_layer(x)
        if self.r > 0:
            lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            return base_output + lora_output
        return base_output


def inject_lora_layers(
    model: nn.Module,
    target_keywords=None,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
):
    """Recursively replaces targeted Linear, Conv2d, or Conv3d layers in the model."""
    for name, child in list(model.named_children()):
        matches_target = target_keywords is None or any(
            kw in name for kw in target_keywords
        )

        if matches_target and isinstance(child, nn.Linear):
            setattr(
                model,
                name,
                LoRALinear(
                    child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                ),
            )
        elif matches_target and isinstance(child, nn.Conv3d):
            setattr(
                model,
                name,
                LoRAConv3d(
                    child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                ),
            )
        elif matches_target and isinstance(child, nn.Conv2d):
            setattr(
                model,
                name,
                LoRAConv2d(
                    child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                ),
            )
        else:
            # Recursively traverse nested children modules
            inject_lora_layers(
                child, target_keywords, r, lora_alpha, lora_dropout
            )



    


def load_lora_for_inference(
    checkpoint_path: str, device: torch.device = torch.device("cuda")
):
    """Reconstructs and loads a fine-tuned LoRA model directly from a saved

    checkpoint directory.
    """
    weights_file = Path(checkpoint_path)
    config_file = weights_file.parent/ "adapter_config.json"

    if not config_file.exists() or not weights_file.exists():
        raise FileNotFoundError(
            f"Missing required checkpoint files in {weights_file.parent}"
        )


    # 1. Read configuration metadata
    with open(config_file, "r") as f:
        config = json.load(f)

    model_env = config["model_env"]
    ckpt_path = config["ckpt_path"]
    lora_cfg = config["lora_config"]


    # 2. Load frozen base model using existing dispatcher
   
    model, _,expected_channels = load_model_with_architecture(
        model_env=model_env, ckpt_path=ckpt_path, device=device
    )

    # 3. Dynamically inject LoRA layers matching saved configuration
    inject_lora_layers(
        model,
        target_keywords=lora_cfg.get("target_modules"),
        r=lora_cfg.get("lora_rr", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
    )

    # 4. Load trained LoRA weights
    lora_state_dict = torch.load(weights_file, map_location=device)
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)

    model.to(device)
    model.eval()

    print(
        f"Successfully loaded '{model_env}' model with LoRA weights from {weights_file.parent}"
    )
    return model,config,expected_channels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir")
    parser.add_argument("--output_path")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()


    test_dir = args.test_dir
    output_path = args.output_path
    checkpoint = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model,config,expected_channels= load_lora_for_inference(checkpoint_path=checkpoint,
                                   device=device)
    

    
    test_paths = sorted(Path(test_dir).glob("*.tiff"))

    model_dtype = next(model.parameters()).dtype

    for path in tqdm(test_paths, desc="Testing"):
        vol = read_tiff(path)
        vol = process_array(arr=vol,config=config,expected_channels=expected_channels)
        vol = vol.to(dtype=model_dtype)

        with torch.no_grad():
            segmentation = model(vol)


        if isinstance(segmentation, (tuple, list)):
            segmentation = segmentation[0]

        # 3. Discretize based on output mode
        if segmentation.shape[1] == 1:
            # Binary thresholding
            discrete_outputs = (segmentation > 0.0).to(torch.uint8)
        else:
            # Multi-class argmax
            discrete_outputs = torch.argmax(segmentation, dim=1).to(torch.int32)

        # 4. Move to CPU, squeeze, and convert to NumPy
        segmentation = (
            discrete_outputs.detach().cpu().float().squeeze().numpy().astype("uint32")
        )


        out_path = Path(output_path) / path.name
        tifffile.imwrite(out_path, segmentation)


if __name__ == "__main__":
    main()
