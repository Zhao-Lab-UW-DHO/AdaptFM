import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from AdaptFM.model.fmSpec import FoundationModelSpec


class CTFMSpec(FoundationModelSpec):
    def __init__(
        self,
        name,
        conda_env,
        module_path,
        training_wrapper_path,
        inference_wrapper_path,
        training_function,
    ):
        super().__init__(
            name,
            conda_env,
            module_path,
            training_wrapper_path,
            inference_wrapper_path,
            training_function,
        )

    # -------------------------
    # helper
    # -------------------------
    def nested_set(self, d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    # -------------------------
    # dataset builder
    # -------------------------
    def prepare_dataset(self, dataset_manager, output_dir, params):
        """
        Converts:
            dataset_manager.samples -> CT-FM CSV format
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = list(dataset_manager.samples)

        # IMPORTANT: shuffle for reproducibility
        import random

        random.seed(0)
        random.shuffle(samples)

        n_train = int(0.8 * len(samples))

        rows = []

        for i, sample in enumerate(samples):
            split = "train" if i < n_train else "val"

            rows.append(
                {
                    "id": f"case_{i:05d}",
                    "image": str(Path(dataset_manager.folder) / sample["image"]),
                    "label": str(Path(dataset_manager.folder) / sample["mask"]),
                    "split": split,
                }
            )

        csv_path = output_dir / "dataset.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        return {"csv_path": csv_path, "dataset_dir": dataset_manager.folder}

    # -------------------------
    # override yaml builder
    # -------------------------

    def build_override_yaml(self, params, run_dir, dataset_csv=None, dataset_dir=None):

        override = {}

        def _cast(value):
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    try:
                        return float(value)
                    except ValueError:
                        return value
            return value

        # Cast all params upfront
        params = {k: _cast(v) for k, v in params.items() if v is not None}

        # Derived values used across multiple blocks
        max_epochs = params.get("max_epochs", 300)
        out_channels = params.get("out_channels", 2)
        class_labels = params.get(
            "class_labels", [f"class_{i}" for i in range(1, out_channels)]
        )

        # Raw label values in the source masks, in the order they map to
        # output channels 0..out_channels-1 (index 0 is background).
        # e.g. imageTBAD: background=0, true lumen=1, false lumen=2 -> [0, 1, 2]
        # Defaults to an identity map (assumes labels are already 0..N-1
        # contiguous) — but that's exactly the assumption that broke silently
        # before, so it's worth making this a required/explicit GUI field
        # rather than trusting the default.
        class_indices = params.get("class_indices", list(range(out_channels)))
        if len(class_indices) != out_channels:
            raise ValueError(
                f"class_indices has {len(class_indices)} entries but "
                f"out_channels={out_channels}; they must match 1:1."
            )

        # ── shared label-remap transforms, inserted identically into train/val ──
        def _label_remap_transforms():
            return [
                {
                    "_target_": "monai.transforms.LabelFilterd",
                    "keys": "label",
                    "applied_labels": class_indices,
                },
                {
                    "_target_": "monai.transforms.MapLabelValued",
                    "keys": "label",
                    "orig_labels": class_indices,
                    "target_labels": list(range(out_channels)),
                },
            ]

        # ── shared load/orient/intensity/crop-to-foreground prefix ──
        def _common_prefix_transforms():
            return [
                {
                    "_target_": "monai.transforms.LoadImaged",
                    "keys": ["image", "label"],
                    "reader": "ITKReader",
                    "ensure_channel_first": True,
                },
                {
                    "_target_": "monai.transforms.EnsureTyped",
                    "keys": ["image", "label"],
                },
                {
                    "_target_": "monai.transforms.Orientationd",
                    "keys": ["image", "label"],
                    "axcodes": "@vars#axcodes",
                },
                *_label_remap_transforms(),
                {
                    "_target_": "monai.transforms.ScaleIntensityRanged",
                    "keys": "image",
                    "a_min": "$@vars#intensity_range[0]",
                    "a_max": "$@vars#intensity_range[1]",
                    "b_min": 0,
                    "b_max": 1,
                    "clip": True,
                },
                {
                    "_target_": "monai.transforms.CropForegroundd",
                    "keys": ["image", "label"],
                    "source_key": "image",
                    "margin": 10,
                },
            ]

        # the final key-rename step — MUST be last in every transform list
        def _finalize_transform():
            return {
                "_target_": "monai.transforms.Lambda",
                "func": "$lambda x: {'input': x['image'].as_tensor().contiguous(), 'target': x['label'].as_tensor().contiguous(), 'id': x['id']}",
                "track_meta": False,
            }

        # ── train: patch extraction + augmentation, matching base config ──
        def _build_train_transform_list():
            return [
                *_common_prefix_transforms(),
                {
                    "_target_": "monai.transforms.SpatialPadd",
                    "keys": ["image", "label"],
                    "spatial_size": "@vars#patch_size",
                    "mode": "constant",
                },
                {
                    "_target_": "monai.transforms.RandCropByLabelClassesd",
                    "keys": ["image", "label"],
                    "label_key": "label",
                    "image_key": "image",
                    "ratios": "$[0] + [1]*(@vars#out_channels-1)",
                    "num_classes": "@vars#out_channels",
                    "num_samples": 1,
                    "spatial_size": "@vars#patch_size",
                    "warn": False,
                },
                {
                    "_target_": "monai.transforms.Lambda",
                    "func": "$lambda x: x[0]",  # index single sample
                    "track_meta": False,
                },
                {
                    "_target_": "monai.transforms.RandAffined",
                    "keys": ["image", "label"],
                    "mode": ["bilinear", "nearest"],
                    "prob": 0.2,
                    "rotate_range": [0.26, 0.26, 0.26],
                    "scale_range": [0.2, 0.2, 0.2],
                    "cache_grid": True,
                    "padding_mode": "constant",
                },
                {
                    "_target_": "monai.transforms.RandGaussianSmoothd",
                    "keys": "image",
                    "prob": 0.2,
                    "sigma_x": [0.5, 1.0],
                    "sigma_y": [0.5, 1.0],
                    "sigma_z": [0.5, 1.0],
                },
                {
                    "_target_": "monai.transforms.RandScaleIntensityd",
                    "keys": "image",
                    "factors": 0.3,
                    "prob": 0.5,
                },
                {
                    "_target_": "monai.transforms.RandShiftIntensityd",
                    "keys": "image",
                    "offsets": 0.1,
                    "prob": 0.5,
                },
                {
                    "_target_": "monai.transforms.RandGaussianNoised",
                    "keys": "image",
                    "std": 0.1,
                    "prob": 0.2,
                },
                _finalize_transform(),
            ]

        # ── val: fixed-size crop/pad, no augmentation, matching base config ──
        def _build_val_transform_list():
            return [
                *_common_prefix_transforms(),
                {
                    "_target_": "monai.transforms.RandSpatialCropd",
                    "keys": ["image", "label"],
                    "roi_size": "@vars#val_max_patch_size",
                    "random_size": False,
                },
                {
                    "_target_": "monai.transforms.SpatialPadd",
                    "keys": ["image", "label"],
                    "spatial_size": "@vars#val_max_patch_size",
                    "mode": "constant",
                },
                _finalize_transform(),
            ]

        # ── 1. vars — declared wholesale to prevent partial merge wiping base vars ──
        self.nested_set(
            override,
            ["vars"],
            {
                # User-facing
                "batch_size": params.get("batch_size", 2),
                "init_LR": params.get("learning_rate", 0.0002),
                "num_workers": params.get("num_workers", 8),
                "dataset_dir": str(dataset_dir),
                "cache_dir": params.get("cache_dir", ""),
                "save_dir": str(run_dir),
                "out_channels": out_channels,
                # Required by system — not user-facing, hardcoded to lighter/CT-FM defaults
                "pin_memory": True,
                "in_channels": 1,
                "patch_size": [96, 160, 160],
                "val_max_patch_size": [192, 240, 240],
                "axcodes": "SPL",
                "intensity_range": [-1024, 2048],
                "percentage": 100,
                # Required by base config expressions even though we override save/cache dir
                "name": params.get("name", "ctfm_finetune"),
                "project": params.get("project", "adaptfm"),
                "wandb_group": params.get("wandb_group", "finetune"),
                "group": "v2",
                "format": "lighter",
            },
        )

        # ── 2. trainer — max_epochs + callbacks wholesale ──────────────────────────
        self.nested_set(
            override,
            ["trainer"],
            {
                "_target_": "pytorch_lightning.Trainer",
                "max_epochs": max_epochs,
                "callbacks": [
                    {
                        "_target_": "lighter.callbacks.Freezer",
                        "name_starts_with": [],
                    },
                    {
                        "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
                        "dirpath": "@vars#save_dir",
                        "save_last": False,
                        "monitor": "val/metrics/Macro_Dice/epoch",
                        "mode": "max",
                        "filename": "best",
                        "auto_insert_metric_name": False,
                        "verbose": True,
                        "every_n_epochs": 5,
                    },
                    {
                        "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.ModeFixer",
                    },
                ],
            },
        )

        cwd = Path.cwd()

        ctfm_root = (cwd.parent / "CT-FM").resolve()

        # ── 3. system — declared wholesale to prevent partial merge dropping ────────
        #      optimizer, scheduler, metrics, and inferer from the base config

        self.nested_set(
            override,
            ["system"],
            {
                "_target_": "lighter.System",
                # Model: pretrained CT-FM SegResNet with reinitialized head
                "model": {
                    "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.load_ct_fm_segresnet",
                    "out_channels": out_channels,
                },
                # Loss: plain DiceCELoss — lighter_zoo SegResNet has no deep supervision
                "criterion": {
                    "_target_": "monai.losses.DiceCELoss",
                    "softmax": True,
                    "to_onehot_y": True,
                    "include_background": True,
                    "squared_pred": True,
                    "smooth_nr": 0,
                    "smooth_dr": 1.0e-05,
                },
                # Optimizer: built via factory function to avoid eager .parameters()
                # resolution before system#model is instantiated
                "optimizer": {
                    "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.build_adamw",
                    "model": "@system#model",
                    "lr": "@vars#init_LR",
                    "weight_decay": 1.0e-05,
                },
                # Scheduler
                "scheduler": {
                    "_target_": "monai.optimizers.WarmupCosineSchedule",
                    "optimizer": "@system#optimizer",
                    "warmup_steps": "$@trainer#max_epochs/100",
                    "end_lr": "$@system#optimizer#lr * 0.01",
                    "warmup_multiplier": 0.1,
                    "t_total": "@trainer#max_epochs",
                },
                # Metrics
                "metrics": {
                    "train": {
                        "Macro_Dice": {
                            "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.DiceScore",
                            "include_background": False,
                            "per_class": False,
                        },
                        "Classwise_Dice": {
                            "_target_": "torchmetrics.wrappers.ClasswiseWrapper",
                            "metric": {
                                "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.DiceScore",
                                "include_background": True,
                                "per_class": True,
                            },
                            "labels": class_labels,
                        },
                    },
                    "val": "%#train",
                },
                # Dataloaders: CSV-based, no PersistentDataset (cache_dir still in vars
                # for base config compat but not used here)
                "dataloaders": {
                    "train": {
                        "_target_": "torch.utils.data.DataLoader",
                        "batch_size": "%vars#batch_size",
                        "pin_memory": "%vars#pin_memory",
                        "num_workers": "%vars#num_workers",
                        "dataset": {
                            "_target_": "monai.data.Dataset",
                            "data": {
                                "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.get_adaptfm_datalist",
                                "csv_file": str(dataset_csv),
                                "split": "train",
                            },
                            "transform": {
                                "_target_": "monai.transforms.Compose",
                                "map_items": False,
                                "transforms": _build_train_transform_list(),
                            },
                        },
                    },
                    "val": {
                        "_target_": "torch.utils.data.DataLoader",
                        "batch_size": 1,
                        "pin_memory": "%vars#pin_memory",
                        "num_workers": "%vars#num_workers",
                        "dataset": {
                            "_target_": "monai.data.Dataset",
                            "data": {
                                "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.get_adaptfm_datalist",
                                "csv_file": str(dataset_csv),
                                "split": "val",
                            },
                            "transform": {
                                "_target_": "monai.transforms.Compose",
                                "map_items": False,
                                "transforms": _build_val_transform_list(),
                            },
                        },
                    },
                },
                # Adapters: drop the DS list-unwrapping lambda from base config
                "adapters": {
                    "train": {
                        "batch": {
                            "_target_": "lighter.adapters.BatchAdapter",
                            "input_accessor": "input",
                            "target_accessor": "target",
                            "identifier_accessor": "id",
                        },
                        "metrics": {
                            "_target_": "lighter.adapters.MetricsAdapter",
                            "pred_argument": 0,
                            "target_argument": 1,
                            "pred_transforms": [
                                "$lambda x: torch.softmax(x, 1)",
                            ],
                            "target_transforms": [
                                "$lambda x: x.squeeze(1).long()",
                            ],
                        },
                        "logging": {
                            "_target_": "lighter.adapters.LoggingAdapter",
                            "pred_transforms": [
                                "$lambda x: torch.softmax(x, 1)",
                                "$lambda x: x.argmax(dim=1, keepdim=True)",
                            ],
                        },
                    },
                    "val": "%#train",
                },
                # Inferer: sliding window, same as base config
                "inferer": {
                    "_target_": "monai.inferers.SlidingWindowInfererAdapt",
                    "roi_size": "@vars#patch_size",
                    "sw_batch_size": "%vars#batch_size",
                    "overlap": 0.625,
                    "mode": "gaussian",
                },
            },
        )

        output_file = str(run_dir / "override.yaml")
        with Path(output_file).open("w") as f:
            yaml.safe_dump(override, f, default_flow_style=False, sort_keys=False)

        return output_file

    def training_command(self, dataset_dir, params, run_dir):
        cwd = Path.cwd()

        ctfm_root = (cwd.parent / "CT-FM").resolve()

        return [
            "lighter",
            "fit",
            f"--config={ctfm_root / 'evaluation/totalseg.yaml'},{run_dir / 'override.yaml'}",
        ]

    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        dataset_csv = dataset_info["csv_path"]
        dataset_dir = dataset_info["dataset_dir"]

        self.build_override_yaml(
            params, run_dir, dataset_csv=dataset_csv, dataset_dir=dataset_dir
        )

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        cwd = Path.cwd()
        ctfm_root = (cwd.parent / "CT-FM").resolve()

        subprocess.Popen(
            cmd,
            cwd=ctfm_root,
            stdout=(Path(run_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(run_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )

    def inference_command(self, dataset_dir, checkpoint, output_dir):
        return [
            "python",
            f"{self.inference_wrapper_path}",
            "--test_dir",
            str(dataset_dir),
            "--output_path",
            str(output_dir),
            "--checkpoint",
            str(checkpoint),
        ]

    def run_inference(self, dataset_dir, checkpoint, output_dir, params):

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        inference_cmd = self.inference_command(
            dataset_dir=dataset_dir, checkpoint=checkpoint, output_dir=output_dir
        )

        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=(Path(output_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(output_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )
