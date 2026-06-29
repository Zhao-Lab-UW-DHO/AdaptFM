import json
import tifffile as tiff
import SimpleITK as sitk 
from AdaptFM.model.model_utils import extract_tunable_params,normalize_to_uint8
from AdaptFM.model.model_spec import ModelSpec
import subprocess, json
from pathlib import Path
import shutil
import os
import yaml
import pandas as pd

class FoundationModelSpec(ModelSpec):
    def __init__(self, name, conda_env, module_path,training_wrapper_path=None,inference_wrapper_path=None,training_function=None):
        self.name = name
        self.conda_env = conda_env
        self.module_path = module_path
        self.training_wrapper_path = training_wrapper_path
        self.inference_wrapper_path = inference_wrapper_path
        self.training_function = training_function


    def default_params(self):
        return {}


    def tunable_params(self):
        import textwrap, subprocess, json

        code = f"""
    import importlib, inspect, json, sys, io

    # suppress prints from module import
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    mod = importlib.import_module('{self.module_path}')
    fn = getattr(mod, '{self.training_function}')
    sig = inspect.signature(fn)

    sys.stdout = old_stdout  # restore


    params = {{}}
    for k, v in sig.parameters.items():
        val = str(v.default)  # convert everything to string
        params[k] = {{
            "default": val,
            "type": "str"  # mark everything as string
        }}

    print(json.dumps(params),end='')
    """
        code = textwrap.dedent(code)

        cmd = [
            "conda", "run", "-p", str(self.conda_env),
            "python", "-c", code
        ]

        out = subprocess.check_output(cmd, text=True).strip() 

        return json.loads(out)


    def prepare_dataset(self, dataset_manager, output_dir):
        # FM models usually expect raw images + masks
        return dataset_manager.path



    def training_command(self, dataset_dir, params, run_dir):
        return [
            "python", "-m", self.module_path,
            "--dataset", str(dataset_dir),
            "--out", str(run_dir),
            "--params", json.dumps(params),
        ]
    
    def inference_command(self,model_path,images_dir,output_dir):
        return [
            "python",
            "-m", self.module_path,
            "predict",
            "--model", str(model_path),
            "--images", str(images_dir),
            "--out", str(output_dir),
        ]


class MicroSAMSpec(FoundationModelSpec):
    def __init__(self, name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function):
        super().__init__(name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function)

    def prepare_dataset(self, dataset_manager, output_dir,params):
        """
        MicroSAM requires paired raw + label paths and filtering empty masks.
        """
        import os, numpy as np, tifffile as tiff

        output_dir.mkdir(parents=True, exist_ok=True)

        training_dir = output_dir / "training"
        seg_dir = output_dir / "segmentations"

        training_dir.mkdir(exist_ok=True)
        seg_dir.mkdir(exist_ok=True)

        for s in dataset_manager.samples:
            img = tiff.imread(s["image"])
            img = normalize_to_uint8(img)
            out_img = training_dir / Path(s["image"]).name
            tiff.imwrite(out_img, img)
            mask_name = Path(s["mask"]).name
            shutil.copy(s["mask"], seg_dir / mask_name)


        raw_paths = [os.path.join(training_dir, f) for f in os.listdir(training_dir)]
        label_paths = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir)]

        raw_dict = {os.path.basename(p): p for p in raw_paths}

        label_dict = {}
        for p in label_paths:
            fname = os.path.basename(p)
            if fname.endswith("_seg.tiff"):
                base = fname.replace("_seg.tiff", ".tiff")
                label_dict[base] = p

        common = sorted(set(raw_dict.keys()) & set(label_dict.keys()))

        valid_raw = []
        valid_label = []

        for fname in common:

            valid_raw.append(raw_dict[fname])
            valid_label.append(label_dict[fname])

        return {
            "raw_paths": valid_raw,
            "label_paths": valid_label,
        }

    def training_command(self, dataset_info, params, run_dir):
        """
        MicroSAM training is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """
        return [
            "python",
            "-m", f"{self.training_wrapper_path}",
            "--raw_paths", json.dumps(dataset_info["raw_paths"]),
            "--label_paths", json.dumps(dataset_info["label_paths"]),
            "--params", json.dumps(params),
            "--out", str(run_dir),
        ]
     
            
    
     # -------- Execution --------
    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=open(run_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

    def inference_command(self, dataset_dir, checkpoint, output_dir):
        """
        MicroSAM training is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """

        return [
            "python",
            "-m", f"{self.inference_wrapper_path}",
            "--dataset_dir",str(dataset_dir),
            "--output_path",str(output_dir),
            "--checkpoint", str(checkpoint),

        ]
    
    
    def run_inference(self,dataset_dir,checkpoint,output_dir,params):

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)      

        inference_cmd = self.inference_command(dataset_dir=dataset_dir,
                                               checkpoint=checkpoint,
                                               output_dir=output_dir) 
        
        
        cmd = self._wrap_with_conda(inference_cmd)
        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

               

class CellposeSAMSpec(FoundationModelSpec):
    def __init__(self, name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function):
        super().__init__(name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function)  

    def prepare_dataset(self, dataset_manager, output_dir,params):

        import random
        import tifffile as tiff
        output_dir = Path(output_dir)

        training_dir = output_dir / "training"
        testing_dir  = output_dir / "testing"

        training_dir.mkdir(parents=True, exist_ok=True)
        testing_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # 1. Train / test split
        # -------------------------
        samples = dataset_manager.samples.copy()
        random.shuffle(samples)

        split_idx = int(0.8 * len(samples))
        train_samples = samples[:split_idx]
        test_samples  = samples[split_idx:]

        # -------------------------
        # 2. Helper to write slices
        # -------------------------
        def write_slices(samples, out_dir):
            for s in samples:
                img = tiff.imread(s["image"])    # shape: (z, y, x)
                mask = tiff.imread(s["mask"])    # same shape

                base_name = Path(s["image"]).stem  # no suffix

                for z in range(img.shape[0]):
                    img_out  = out_dir / f"{base_name}_z{z}.tiff"
                    mask_out = out_dir / f"{base_name}_z{z}_seg.tiff"

                    tiff.imwrite(img_out, img[z], compression="zlib")
                    tiff.imwrite(mask_out, mask[z], compression="zlib")

        # -------------------------
        # 3. Write datasets
        # -------------------------
        write_slices(train_samples, training_dir)
        write_slices(test_samples, testing_dir)

        return {
            "train_dir": training_dir,
            "test_dir": testing_dir,
        }
    
    def training_command(self, dataset_info, params, run_dir):
        """
        CellposeSAM training is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """
        return [
            "python",
            "-m", f"{self.training_wrapper_path}",
            "--train_dir", str(dataset_info["train_dir"]),
            "--test_dir", str(dataset_info["test_dir"]),
            "--params", json.dumps(params),
        ]

     # -------- Execution --------
    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        gpu = params['gpu']
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=open(run_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

    def inference_command(self, dataset_dir, checkpoint, output_dir):
        """
        CellposeSAM training is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """

        return [
            "python",
            "-m", f"{self.inference_wrapper_path}",
            "--test_dir",str(dataset_dir),
            "--output_path",str(output_dir),
            "--checkpoint", str(checkpoint),

        ]
    
    
    def run_inference(self,dataset_dir,checkpoint,output_dir,params):

        gpu = params['gpu']
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)      

        inference_cmd = self.inference_command(dataset_dir=dataset_dir,
                                               checkpoint=checkpoint,
                                               output_dir=output_dir) 
        
        
        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )



class SSVTSpec(FoundationModelSpec):
    def __init__(self, name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function):
        super().__init__(name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function)  


    def prepare_dataset(self, dataset_manager, output_dir,params):
        import random

        train_raw_images = output_dir / "train_raw_images"
        train_mask_images  = output_dir / "train_mask_images"

        train_raw_images.mkdir(parents=True, exist_ok=True)
        train_mask_images.mkdir(parents=True, exist_ok=True)

        val_raw_images = output_dir / "val_raw_images"
        val_mask_images  = output_dir / "val_mask_images"

        val_raw_images.mkdir(parents=True, exist_ok=True)
        val_mask_images.mkdir(parents=True, exist_ok=True)

        samples = dataset_manager.samples.copy()
        random.shuffle(samples)

        split_idx = int(0.8 * len(samples))
        train_samples = samples[:split_idx]
        test_samples  = samples[split_idx:]

        def copy_samples(samples,raw_images,mask_images):

            for s in samples:
                base_name = Path(s["image"]).stem  # no suffix

                img_out  = raw_images / f"{base_name}.tiff"
                mask_out = mask_images / f"{base_name}_seg.tiff"

                shutil.copy(s['image'],img_out)
                shutil.copy(s['mask'],mask_out)

            return 
        
        copy_samples(train_samples,train_raw_images,train_mask_images)
        copy_samples(test_samples,val_raw_images,val_mask_images)

        return {

            "train_raw_images":train_raw_images,
            "train_mask_images":train_mask_images,
            "val_raw_images":val_raw_images,
            "val_mask_images":val_mask_images


        }


    def training_command(self, dataset_info, params,run_dir):
            """
            SSVT training is Python API–based, not CLI-based.
            So we call a small wrapper script inside the env.
            """
            return [
                "python",
                "-m", f"{self.training_wrapper_path}",
                "--train_raw_images", str(dataset_info["train_raw_images"]),
                "--train_mask_images", str(dataset_info["train_mask_images"]),
                "--val_raw_images", str(dataset_info["val_raw_images"]),
                "--val_mask_images", str(dataset_info["val_mask_images"]),
                "--output_path",run_dir,
                "--params", json.dumps(params),
            ]

     # -------- Execution --------
    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=open(run_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

    def inference_command(self, dataset_dir, checkpoint, output_dir):
        """
        SSVT training is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """

        return [
            "python",
            "-m", f"{self.inference_wrapper_path}",
            "--test_dir",str(dataset_dir),
            "--output_path",str(output_dir),
            "--checkpoint", str(checkpoint),

        ]
    
    
    def run_inference(self,dataset_dir,checkpoint,output_dir,params):

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)      

        inference_cmd = self.inference_command(dataset_dir=dataset_dir,
                                               checkpoint=checkpoint,
                                               output_dir=output_dir) 
        
        
        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )



class Sammed3DSpec(FoundationModelSpec):
    def __init__(self, name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function):
        super().__init__(name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function)

    #this model does not require a prepare dataset

    def prepare_dataset(self, dataset_manager, output_dir,params):

        imagesTrFolder = os.path.join(dataset_manager.folder,'imagesTr')
        labelsTrFolder = os.path.join(dataset_manager.folder,'labelsTr')
        os.makedirs(imagesTrFolder,exist_ok=True)
        os.makedirs(labelsTrFolder,exist_ok=True)

        tiff_images = [file for file in os.listdir(dataset_manager.folder) if file.endswith(('.tif','.tiff'))]

        for tiff_file in tiff_images:
            tiff_image_path = os.path.join(dataset_manager.folder,tiff_file)
            tiff_image = tiff.imread(tiff_image_path)
            tiff_image = sitk.GetImageFromArray(tiff_image)

            nii_name = tiff_file.replace('_seg', '').replace('.tiff', '.nii.gz')

            if '_seg.tiff' in tiff_file:
                nii_path = os.path.join(labelsTrFolder,nii_name)

            if '_seg.tiff' not in tiff_file:
                nii_path = os.path.join(imagesTrFolder,nii_name)

            sitk.WriteImage(tiff_image,nii_path)
            os.remove(tiff_image_path)        

        return {"dataset_dir": dataset_manager.folder}
    

    def training_command(self, dataset_info, params, run_dir):
        """
        CellposeSAM training is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """
        return [
            "python",
            "-m", f"{self.training_wrapper_path}",
            "--params", json.dumps(params),
            "--output_path",run_dir,
            '--dataset_dir', dataset_info['dataset_dir']

        ]

     # -------- Execution --------
    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=open(run_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

    def inference_command(self, dataset_dir, checkpoint, output_dir):
        """
        SAMMED3D inference is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """

        return [
            "python",
            "-m", f"{self.inference_wrapper_path}",
            "--test_dir",str(dataset_dir),
            "--output_path",str(output_dir),
            "--checkpoint", str(checkpoint),

        ]
    
    
    def run_inference(self,dataset_dir,checkpoint,output_dir,params):

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)      

        inference_cmd = self.inference_command(dataset_dir=dataset_dir,
                                               checkpoint=checkpoint,
                                               output_dir=output_dir) 
        
        
        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )


class CellSAMSpec(FoundationModelSpec):
    def __init__(self, name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function):
        super().__init__(name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function)



    def inference_command(self, dataset_dir, checkpoint, output_dir):
        return [
            "python",
            "-m", f"{self.inference_wrapper_path}",
            "--test_dir",str(dataset_dir),
            "--output_path",str(output_dir),
        ]



    def run_inference(self, dataset_dir, checkpoint, output_dir, params):

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)


        inference_cmd = self.inference_command(dataset_dir=dataset_dir,
                                                checkpoint=checkpoint,
                                                output_dir=output_dir) 

        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )



class CTFMSpec(FoundationModelSpec):

    def __init__(self, name, conda_env, module_path,
                 training_wrapper_path, inference_wrapper_path,
                 training_function):
        super().__init__(
            name,
            conda_env,
            module_path,
            training_wrapper_path,
            inference_wrapper_path,
            training_function
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
    def prepare_dataset(self, dataset_manager, output_dir,params):
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

            rows.append({
                "id": f"case_{i:05d}",
                "image": str(Path(dataset_manager.folder) / sample["image"]),
                "label": str(Path(dataset_manager.folder) / sample["mask"]),
                "split": split,
            })

        csv_path = output_dir / "dataset.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        return csv_path

    # -------------------------
    # override yaml builder
    # -------------------------

    def build_override_yaml(self, params, run_dir, dataset_csv=None):

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
        max_epochs  = params.get("max_epochs", 300)
        out_channels = params.get("out_channels", 2)
        class_labels = params.get(
            "class_labels", [f"class_{i}" for i in range(1, out_channels)]
        )

        # ── 1. vars — declared wholesale to prevent partial merge wiping base vars ──
        self.nested_set(override, ["vars"], {
            # User-facing
            "batch_size":       params.get("batch_size", 2),
            "init_LR":          params.get("learning_rate", 0.0002),
            "num_workers":      params.get("num_workers", 8),
            "dataset_dir":      params.get("dataset_dir", ""),
            "cache_dir":        params.get("cache_dir", ""),
            "save_dir":         params.get("save_dir", ""),
            "out_channels":     out_channels,

            # Required by system — not user-facing, hardcoded to lighter/CT-FM defaults
            "pin_memory":       True,
            "in_channels":      1,
            "patch_size":       [96, 160, 160],
            "val_max_patch_size": [192, 240, 240],
            "axcodes":          "SPL",
            "intensity_range":  [-1024, 2048],
            "percentage":       100,

            # Required by base config expressions even though we override save/cache dir
            "name":             params.get("name", "ctfm_finetune"),
            "project":          params.get("project", "adaptfm"),
            "wandb_group":      params.get("wandb_group", "finetune"),
            "group":            "v2",
            "format":           "lighter",
        })

        # ── 2. trainer — max_epochs + callbacks wholesale ──────────────────────────
        self.nested_set(override, ["trainer"], {
            "max_epochs": max_epochs,
            "callbacks": [
                {
                    # name_starts_with=[] is a no-op — avoids frozen encoder
                    # causing empty parameter list at optimizer init
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
            ],
        })

        cwd = Path.cwd()

        ctfm_root = (cwd.parent / "CT-FM").resolve()

        # ── 3. system — declared wholesale to prevent partial merge dropping ────────
        #      optimizer, scheduler, metrics, and inferer from the base config
        self.nested_set(override, ["system"], {
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
                        "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.get_adaptfm_datalist",
                        "csv_file": str(dataset_csv),
                        "split": "train",
                    },
                },
                "val": {
                    "_target_": "torch.utils.data.DataLoader",
                    "batch_size": 1,
                    "pin_memory": "%vars#pin_memory",
                    "num_workers": "%vars#num_workers",
                    "dataset": {
                        "_target_": "AdaptFM.model.foundation_models.ctfm.training_utils.get_adaptfm_datalist",
                        "csv_file": str(dataset_csv),
                        "split": "val",
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
        })

        output_file = str(run_dir / "override.yaml")
        with open(output_file, "w") as f:
            yaml.safe_dump(override, f, default_flow_style=False, sort_keys=False)

        return output_file


    def training_command(self, dataset_dir, params, run_dir):
        cwd = Path.cwd()

        ctfm_root = (cwd.parent / "CT-FM").resolve()

        return [
            "lighter",
            "fit",
            str(ctfm_root / "evaluation/totalseg.yaml"),
            str(run_dir / "override.yaml")
        ]


    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        self.build_override_yaml(params, run_dir,dataset_csv=dataset_info)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        cwd = Path.cwd()
        ctfm_root = (cwd.parent / "CT-FM").resolve()

        subprocess.Popen(
            cmd,
            cwd=ctfm_root,
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=open(run_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )


    def inference_command(self, dataset_dir, checkpoint, output_dir):
        return [
            "python",
            "-m", f"{self.inference_wrapper_path}",
            "--test_dir",str(dataset_dir),
            "--output_path",str(output_dir),
        ]



    def run_inference(self, dataset_dir, checkpoint, output_dir, params):

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)


        inference_cmd = self.inference_command(dataset_dir=dataset_dir,
                                                checkpoint=checkpoint,
                                                output_dir=output_dir) 

        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )

