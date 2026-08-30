import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class PostProcess(ABC):
    name: str = "BasePostProcess"

    def __init__(self, name, conda_env, module_path):
        self.name = name
        self.conda_env = conda_env
        self.module_path = module_path

    @abstractmethod
    def run_postprocess(self, input_dir, output_dir, gpu):
        """run post processing on a separate thread that communicates back to the main"""

    def _wrap_with_conda(self, cmd: list[str]) -> list[str]:
        if self.conda_env is None:
            raise RuntimeError(
                "Could not find conda environment for this post processing option.\n Install with the environment manager\n"
            )
        return ["conda", "run", "-p", self.conda_env, "--no-capture-output", *cmd]


class USegment3DSpec(PostProcess):
    def __init__(self, name, conda_env, module_path):
        super().__init__(name, conda_env, module_path)
        self.name = name
        self.conda_env = conda_env
        self.module_path = module_path

    def run_postprocess(self, input_dir, output_dir, gpu):

        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        post_process_cmd = [
            "python",
            f"{self.module_path}",
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
        ]

        cmd = self._wrap_with_conda(post_process_cmd)

        subprocess.Popen(
            cmd,
            stdout=(Path(output_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(output_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )


class Conv2BinarySpec(PostProcess):
    def __init__(self, name, conda_env, module_path):
        super().__init__(name, conda_env, module_path)
        self.name = name
        self.conda_env = conda_env
        self.module_path = module_path

    def run_postprocess(self, input_dir, output_dir,gpu):
        
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        post_process_cmd = [
            "python",
            f"{self.module_path}",
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
        ]

        cmd = self._wrap_with_conda(post_process_cmd)

        subprocess.Popen(
            cmd,
            stdout=(Path(output_dir) / "stdout.log").open("w", encoding="utf-8"),
            stderr=(Path(output_dir) / "stderr.log").open("w", encoding="utf-8"),
            start_new_session=True,
            env=env,
        )