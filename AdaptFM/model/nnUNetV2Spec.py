import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from AdaptFM.model.model_spec import ModelSpec


class NNUNetV2ModelSpec(ModelSpec):
    def __init__(self, conda_env, supports_training=True):
        super().__init__()
        self.conda_env = conda_env
        self.name = "nnUNetV2"
        self.supports_training = supports_training

    def default_params(self):
        return {
            "config": "3d_fullres",
            "fold": "all",
        }

    def tunable_params(self, **kwargs):
        # kwargs intentionally ignored
        return {
            "Set Name": {"type": str, "default": ""},
            "Set ID": {"type": str, "default": "1"},
            "config": {
                "type": str,
                "choices": ["3d_lowres", "3d_fullres", "3d_cascade_fullres"],
                "default": "3d_fullres",
            },
            "fold": {
                "type": str,
                "default": "all",
            },
        }

    def prepare_dataset(self, dataset_manager, output_dir, params):

        prepared_dataset_folder = dataset_manager.export_for_framework(
            framework="nnunet", out_folder=output_dir, params=params
        )
        return prepared_dataset_folder

    def preprocessing_command(self, params):
        set_id = params["Set ID"]

        return ["nnUNetv2_plan_and_preprocess", "-d", str(set_id)]

    def run_preprocessing(self, dataset_dir, params, output_dir):

        output_dir.mkdir(parents=True, exist_ok=True)
        gpu = params["gpu"]

        env = os.environ.copy()
        env["nnUNet_raw"] = str(Path(dataset_dir) / "nnUNet_raw")
        env["nnUNet_preprocessed"] = str(Path(dataset_dir) / "nnUNet_preprocessed")
        env["nnUNet_results"] = str(Path(dataset_dir) / "nnUNet_results")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        cmd = self._wrap_with_conda(self.preprocessing_command(params))

        process = subprocess.Popen(
            cmd,
            stdout=(output_dir / "stdout.log").open(mode="w"),
            stderr=(output_dir / "stderr.log").open(mode="w"),
            start_new_session=True,
            env=env,
        )

        return process

    def training_command(self, params):
        return [
            "nnUNetv2_train",
            params["Set ID"],
            params["config"],
            params["fold"],
            "-p",
            "nnUNetPlans",
            "-tr",
            "nnUNetTrainer",
            "--npz",
        ]

    # -------- Execution --------
    def run_training(self, dataset_info, params, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)

        gpu = params["gpu"]
        env = os.environ.copy()
        env["nnUNet_raw"] = str(Path(dataset_info) / "nnUNet_raw")
        env["nnUNet_preprocessed"] = str(Path(dataset_info) / "nnUNet_preprocessed")
        env["nnUNet_results"] = str(Path(dataset_info) / "nnUNet_results")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        with (Path(run_dir) / "params.json").open("w") as f:
            json.dump(params, f, indent=4)

        training_commnad = self.training_command(params)

        cmd = self._wrap_with_conda(training_commnad)
        env["TORCHDYNAMO_DISABLE"] = "1"

        subprocess.Popen(
            cmd,
            stdout=(run_dir / "stdout.log").open(mode="w"),
            stderr=(run_dir / "stderr.log").open(mode="w"),
            start_new_session=True,
            env=env,
        )

    def find_nnUNet_base(self, start_dir):
        """
        Walk upward from start_dir until we find a directory containing
        nnUNet_raw, nnUNet_preprocessed, and nnUNet_results.
        Return that directory or None.
        """
        required_dirs = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
        current = Path(start_dir).resolve()

        for p in (current, *current.parents):
            if all((p / folder).is_dir() for folder in required_dirs):
                return str(p)
        return None

    def parse_dataset_name(self, path):
        """
        Given a path inside a nnUNet dataset folder, return (dataset_id, dataset_name).
        Example: /.../Dataset001_TEST/imagesTs → ('001', 'TEST')
        """
        p = Path(path).resolve()
        path = str(p)
        parts = list(p.parts)

        # Find the folder that matches the nnUNet dataset naming pattern
        for p in reversed(parts):
            m = re.match(r"Dataset(\d{3})_(.+)", p)
            if m:
                dataset_id = m.group(1)
                dataset_name = m.group(2)
                return dataset_id, dataset_name

        raise ValueError(f"No nnUNet dataset folder found in path: {path}")

    def dataset_is_inside_raw(self, dataset_dir, nnunet_base) -> bool:
        """
        Check that dataset_dir is somewhere inside nnUNet_raw.
        """
        raw_dir = (Path(nnunet_base) / "nnUNet_raw").resolve()
        dataset_path = Path(dataset_dir).resolve()
        return dataset_path.is_relative_to(raw_dir)

    def run_inference(self, dataset_dir, checkpoint, output_dir, params):

        gpu = params["gpu"]
        env = os.environ.copy()

        # 1. Find nearest nnUNet base directory
        nnunet_base = self.find_nnUNet_base(dataset_dir)
        if nnunet_base is None:
            raise RuntimeError(
                f"No nnUNet directory structure found in parent folders of {dataset_dir}"
            )

        # 2. Verify dataset_dir is inside nnUNet_raw
        if not self.dataset_is_inside_raw(dataset_dir, nnunet_base):
            raise RuntimeError(
                f"dataset_dir={dataset_dir} is not inside nnUNet_raw under {nnunet_base}"
            )

        params["Set ID"], params["Set Name"] = self.parse_dataset_name(dataset_dir)

        # 3. Set environment variables
        env["nnUNet_raw"] = str(Path(nnunet_base) / "nnUNet_raw")
        env["nnUNet_preprocessed"] = str(Path(nnunet_base) / "nnUNet_preprocessed")
        env["nnUNet_results"] = str(Path(nnunet_base) / "nnUNet_results")

        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        inference_cmd = self.inference_command(
            params=params,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            checkpoint=checkpoint,
        )

        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=(output_dir / "stdout.log").open(mode="w"),
            stderr=(output_dir / "stderr.log").open(mode="w"),
            start_new_session=True,
            env=env,
        )

    def inference_command(self, params, dataset_dir, output_dir, checkpoint):

        ckpt_name = Path(checkpoint).name

        return [
            "nnUNetv2_predict",
            "-i",
            dataset_dir,
            "-o",
            output_dir,
            "-d",
            params["Set ID"],
            "-c",
            params["config"],
            "-f",
            params["fold"],
            "-p",
            "nnUNetPlans",  # use as default plans
            "-tr",
            "nnUNetTrainer",  # default trainer
            "-chk",
            ckpt_name,  # use the best checkpoint by default
        ]


class MerlinNNUNetV2ModelSpec(NNUNetV2ModelSpec):
    def __init__(self, conda_env, transform_path, supports_training):
        super().__init__(conda_env, supports_training)
        self.name = "Merlin nnUNet"
        self.transform_path = transform_path

    def ensure_checkpoint_matches_dataset(
        self, checkpoint: str, output_dir: Path, params: dict
    ):
        setID = params["Set ID"]
        setName = params["Set Name"]

        formatted_set_id = f"{int(setID):03d}"
        setName_and_ID = f"Dataset{formatted_set_id}_{setName}"

        # 1. Check if checkpoint contains the dataset tag
        if setName_and_ID in str(checkpoint):
            return  # nothing to do

        # 2. Create nnUNet folder structure
        raw_folder = output_dir / "nnUNet_raw" / setName_and_ID
        preprocessed_folder = output_dir / "nnUNet_preprocessed" / setName_and_ID
        results_folder = output_dir / "nnUNet_results" / setName_and_ID

        raw_folder.mkdir(parents=True, exist_ok=True)
        preprocessed_folder.mkdir(parents=True, exist_ok=True)
        results_folder.mkdir(parents=True, exist_ok=True)

        # 3. Locate the nnUNetTrainerMerlin folder inside the checkpoint path
        checkpoint_path = Path(checkpoint).resolve()

        # Search upward for the folder named nnUNetTrainerMerlin__nnUNetPlans__3d_fullres
        trainer_folder = None
        for parent in checkpoint_path.parents:
            candidate = parent / "nnUNetTrainerMerlin__nnUNetPlans__3d_fullres"
            if candidate.exists() and candidate.is_dir():
                trainer_folder = candidate
                break

        if trainer_folder is None:
            raise RuntimeError(
                "Could not find nnUNetTrainerMerlin__nnUNetPlans__3d_fullres folder "
                "in checkpoint path parents."
            )

        # 4. Move trainer folder into nnUNet_results/DatasetXXX_Name
        destination = results_folder / trainer_folder.name
        if destination.exists():
            # do not overwrite
            return

        shutil.copytree(str(trainer_folder), str(destination))

    def run_preprocessing(self, dataset_dir, params, output_dir):

        output_dir.mkdir(parents=True, exist_ok=True)
        gpu = params["gpu"]

        env = os.environ.copy()
        env["nnUNet_raw"] = str(Path(dataset_dir) / "nnUNet_raw")
        env["nnUNet_preprocessed"] = str(Path(dataset_dir) / "nnUNet_preprocessed")
        env["nnUNet_results"] = str(Path(dataset_dir) / "nnUNet_results")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        setID = params["Set ID"]
        setName = params["Set Name"]

        folder2transform = (
            dataset_dir / "nnUNet_raw" / f"Dataset{setID:03}_{setName}" / "imagesTr"
        )

        transform_cmd = self._wrap_with_conda(
            self.transform_command(folder2transform, params)
        )

        transform_proc = subprocess.Popen(
            transform_cmd,
            stdout=(Path(output_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(output_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )

        # Block until preprocessing is done
        return_code = transform_proc.wait()

        if return_code != 0:
            raise RuntimeError(f"Transforms failed with return code {return_code}")

        cmd = self._wrap_with_conda(self.preprocessing_command(params))

        process = subprocess.Popen(
            cmd,
            stdout=(Path(output_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(output_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )

        return process

    def training_command(self, params):
        return [
            "nnUNetv2_train",
            params["Set ID"],
            params["config"],
            params["fold"],
            "-p",
            "nnUNetPlans",
            "-tr",
            "nnUNetTrainerMerlin",  # training command is identical just uses a different trainer
            "--npz",
        ]

    def transform_command(self, folder2transform, params):

        return [
            "python",
            f"{self.transform_path}",
            "--params",
            json.dumps(params),
            "--folder2transform",
            str(folder2transform),
        ]

    def inference_command(self, params, imagesTs, output_dir, checkpoint):

        ckpt_name = Path(checkpoint).name

        return [
            "nnUNetv2_predict",
            "-i",
            imagesTs,
            "-o",
            output_dir,
            "-d",
            params["Set ID"],
            "-c",
            params["config"],
            "-f",
            params["fold"],
            "-p",
            "nnUNetPlans",  # use as default plans
            "-tr",
            "nnUNetTrainerMerlin",  # default trainer
            "-chk",
            ckpt_name,  # use the best checkpoint by default
        ]

    def run_inference(self, dataset_dir, checkpoint, output_dir, params):

        gpu = params["gpu"]
        env = os.environ.copy()

        env["nnUNet_raw"] = str(Path(output_dir) / "nnUNet_raw")
        env["nnUNet_preprocessed"] = str(Path(output_dir) / "nnUNet_preprocessed")
        env["nnUNet_results"] = str(Path(output_dir) / "nnUNet_results")

        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # ensure that checkpoint is in the set ID,setName results folder. if not, copy it there
        self.ensure_checkpoint_matches_dataset(checkpoint, output_dir, params)

        folder2transform = dataset_dir

        transform_cmd = self._wrap_with_conda(
            self.transform_command(folder2transform, params)
        )

        transform_proc = subprocess.Popen(
            transform_cmd,
            stdout=(Path(output_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(output_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )

        # Block until transforms are done
        return_code = transform_proc.wait()

        if return_code != 0:
            raise RuntimeError(f"Transforms failed with return code {return_code}")

        original_name = folder2transform.name
        transformed_name = f"{original_name}_transformed"

        transformed_folder = folder2transform.parent / transformed_name

        inference_cmd = self.inference_command(
            params=params,
            imagesTs=str(transformed_folder),
            output_dir=output_dir,
            checkpoint=checkpoint,
        )

        cmd = self._wrap_with_conda(inference_cmd)

        subprocess.Popen(
            cmd,
            stdout=(Path(output_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(output_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )
