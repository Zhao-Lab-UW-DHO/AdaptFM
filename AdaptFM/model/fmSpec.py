import json
import tifffile as tiff
import SimpleITK as sitk 
from AdaptFM.model.model_utils import extract_tunable_params,normalize_to_uint8
from AdaptFM.model.model_spec import ModelSpec
import subprocess, json
from pathlib import Path
import shutil
import os
import random
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

        training_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)

        for s in dataset_manager.samples:
            img = tiff.imread(s["image"])
            img = normalize_to_uint8(img)
            out_img = training_dir / Path(s["image"]).name
            tiff.imwrite(out_img, img)
            mask_name = Path(s["mask"]).name
            shutil.copy(s["mask"], seg_dir / mask_name)


        raw_paths = [str(f) for f in Path(training_dir).iterdir() if f.is_file()]
        label_paths = [str(f) for f in Path(seg_dir).iterdir() if f.is_file()]

        raw_dict = {Path(p).name: p for p in raw_paths}

        label_dict = {}
        for p in label_paths:
            fname = Path(p).name
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

        with (Path(run_dir) / 'params.json').open('w') as f:
            json.dump(params, f, indent=4)

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        subprocess.Popen(
            cmd,
            stdout=(run_dir / "stdout.log").open(mode='w'),
            stderr=(run_dir / "stderr.log").open(mode='w'),
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
            stdout=(output_dir / "stdout.log").open(mode='w'),
            stderr=(output_dir / "stderr.log").open(mode='w'),
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

        with (Path(run_dir) / 'params.json').open('w') as f:
            json.dump(params, f, indent=4)

        gpu = params['gpu']
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        subprocess.Popen(
            cmd,
            stdout=(run_dir / "stdout.log").open(mode='w'),
            stderr=(run_dir / "stderr.log").open(mode='w'),
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
            stdout=(output_dir / "stdout.log").open(mode='w'),
            stderr=(output_dir / "stderr.log").open(mode='w'),
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

        with (Path(run_dir) / 'params.json').open('w') as f:
            json.dump(params, f, indent=4)

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        subprocess.Popen(
            cmd,
            stdout=(run_dir / "stdout.log").open(mode='w'),
            stderr=(run_dir / "stderr.log").open(mode='w'),
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
            stdout=(output_dir / "stdout.log").open(mode='w'),
            stderr=(output_dir / "stderr.log").open(mode='w'),
            start_new_session=True,
            env=env
        )



class Sammed3DSpec(FoundationModelSpec):
    def __init__(self, name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function):
        super().__init__(name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function)

    #this model does not require a prepare dataset

    def prepare_dataset(self, dataset_manager, output_dir,params):

        imagesTrFolder = Path(dataset_manager.folder) / 'imagesTr'
        labelsTrFolder = Path(dataset_manager.folder) / 'labelsTr'

        imagesTrFolder.mkdir(parents=True, exist_ok=True)
        labelsTrFolder.mkdir(parents=True, exist_ok=True)

        tiff_images = [file.name for file in Path(dataset_manager.folder).iterdir() if file.is_file() and file.suffix.lower() in ('.tif', '.tiff')]

        for tiff_file in tiff_images:
            tiff_image_path = Path(dataset_manager.folder) / tiff_file
            tiff_image = tiff.imread(tiff_image_path)
            tiff_image = sitk.GetImageFromArray(tiff_image)

            nii_name = Path(tiff_file).stem.replace('_seg', '') + '.nii.gz'

            if '_seg.tiff' in tiff_file:
                nii_path = Path(labelsTrFolder) / nii_name

            else:
                nii_path = Path(imagesTrFolder) / nii_name

            sitk.WriteImage(tiff_image, str(nii_path))
            Path(tiff_image_path).unlink()

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

        with (Path(run_dir) / 'params.json').open('w') as f:
            json.dump(params, f, indent=4)

        gpu = params.pop("gpu", None)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        training_cmd = self.training_command(dataset_info, params, run_dir)

        cmd = self._wrap_with_conda(training_cmd)

        subprocess.Popen(
            cmd,
            stdout=(run_dir / "stdout.log").open(mode='w'),
            stderr=(run_dir / "stderr.log").open(mode='w'),
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
            stdout=(output_dir / "stdout.log").open(mode='w'),
            stderr=(output_dir / "stderr.log").open(mode='w'),
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


class BMEXSpec(FoundationModelSpec):
    
    def __init__(self, name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function):
        super().__init__(name, conda_env, module_path,training_wrapper_path,inference_wrapper_path,training_function)


    def prepare_dataset(self, dataset_manager, output_dir, params):

        imagesTrFolder = os.path.join(output_dir, 'imagesTr')
        labelsTrFolder = os.path.join(output_dir, 'labelsTr')
        os.makedirs(imagesTrFolder, exist_ok=True)
        os.makedirs(labelsTrFolder, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(dataset_manager.folder)
            if f.endswith((".tif", ".tiff", ".nii.gz"))
        ]
        # base_name -> {"image": ..., "label": ...}
        pairs = {}

        for image_file in image_files:
            input_path = os.path.join(dataset_manager.folder, image_file)

            if image_file.endswith(".nii.gz"):
                nii_name = image_file.replace("_seg.nii.gz", ".nii.gz")
                is_label = "_seg.nii.gz" in image_file
            else:
                nii_name = (
                    image_file
                    .replace("_seg", "")
                    .replace(".tiff", ".nii.gz")
                    .replace(".tif", ".nii.gz")
                )
                is_label = "_seg.tif" in image_file or "_seg.tiff" in image_file
            base_name = nii_name  # same root for image/label since '_seg' was stripped

            if is_label:
                nii_path = os.path.join(labelsTrFolder, nii_name)
                rel_path = os.path.join('labelsTr', nii_name)
                pairs.setdefault(base_name, {})['label'] = rel_path
            else:
                nii_path = os.path.join(imagesTrFolder, nii_name)
                rel_path = os.path.join('imagesTr', nii_name)
                pairs.setdefault(base_name, {})['image'] = rel_path

            if image_file.endswith(".nii.gz"):
                shutil.copy2(input_path, nii_path)
            else:
                img = tiff.imread(input_path)
                img = sitk.GetImageFromArray(img)
                sitk.WriteImage(img, nii_path)
                
        # only keep complete image/label pairs
        complete_pairs = [
            {"image": entry["image"], "label": entry["label"]}
            for entry in pairs.values()
            if "image" in entry and "label" in entry
        ]

        # shuffle reproducibly, then split 80/20
        seed = params.get("seed", 42) if params else 42
        rng = random.Random(seed)
        shuffled_pairs = complete_pairs.copy()
        rng.shuffle(shuffled_pairs)

        split_idx = int(round(len(shuffled_pairs) * 0.8))
        training_pairs = shuffled_pairs[:split_idx]
        validation_pairs = shuffled_pairs[split_idx:]

        dataset_dict = {
            "training": training_pairs,
            "validation": validation_pairs
        }

        dataset_dict_path = os.path.join(output_dir, "json_list.json")

        with open(dataset_dict_path, "w") as file:
            json.dump(dataset_dict, file, indent=4)

        dataset_dir = Path(dataset_manager.folder).parent

        return {"dataset_dir": dataset_dir}
    

    def training_command(self, dataset_info, params, run_dir):
        """
        CellposeSAM training is Python API–based, not CLI-based.
        So we call a small wrapper script inside the env.
        """
        return [
            "python",
            "-m", f"{self.training_wrapper_path}",
            "--params", json.dumps(params),
            "--output_dir",run_dir,
            '--data_dir', dataset_info['dataset_dir']

        ]

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
        return [
            "python",
            "-m", f"{self.inference_wrapper_path}",
            "--test_dir",str(dataset_dir),
            "--checkpoint",str(checkpoint),
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
