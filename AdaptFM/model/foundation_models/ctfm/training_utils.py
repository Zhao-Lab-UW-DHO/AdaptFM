# training_utils.py — add DiceScore here so it's importable without
# lighter's project module registration

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from lighter_zoo import SegResNet
from monai.metrics import DiceHelper
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torch import Tensor


class DiceScore(Metric):
    def __init__(self, include_background: bool = False, per_class: bool = False):
        super().__init__()
        reduction = "mean_batch" if per_class else "mean"
        self.metric = DiceHelper(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=False,
            ignore_empty=True,
        )
        self.add_state("dice", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        if target.ndim == 4:
            target = target.unsqueeze(1)
        self.dice.append(self.metric(preds, target))

    def compute(self) -> Tensor:
        return dim_zero_cat(self.dice)


def load_ct_fm_segresnet(out_channels: int) -> nn.Module:
    model = SegResNet.from_pretrained("project-lighter/ct_fm_segresnet")
    old_head = model.up_layers[-1].head
    model.up_layers[-1].head = nn.Conv3d(
        in_channels=old_head.in_channels,
        out_channels=out_channels,
        kernel_size=old_head.kernel_size,
        stride=old_head.stride,
        padding=old_head.padding,
    )
    return model


def build_adamw(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def get_adaptfm_datalist(csv_file: str, split: str) -> list:
    df = pd.read_csv(csv_file)
    df = df[df["split"] == split]
    return df[["id", "image", "label"]].to_dict(orient="records")

def launch_training(
        batch_size: int=2,
        max_epochs:int =300,
        learning_rate:float =0.0002,
        num_workers:int=8,
        dataset_dir:str=str(Path.cwd() / "dataset_dir"),
        cache_dir:str=str(Path.cwd() / "cache_dir"),
        save_dir:str=str(Path.cwd() / "save_dir"),
        
):

    return


