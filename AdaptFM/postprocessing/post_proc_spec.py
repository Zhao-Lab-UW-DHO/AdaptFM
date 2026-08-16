from abc import ABC, abstractmethod
import subprocess
import os

class PostProcess(ABC):
    name: str = "BasePostProcess"

    def __init__(self, name, conda_env,module_path):
        self.name = name
        self.conda_env = conda_env
        self.module_path = module_path

    @abstractmethod
    def run_postprocess(self, input_dir, output_dir,gpu):
        """run post processing on a separate thread that communicates back to the main"""
        pass


    def _wrap_with_conda(self, cmd: list[str]) -> list[str]:
        if self.conda_env is None:
            raise RuntimeError(
                "Could not find conda environment for this post processing option.\n Install with the environment manager\n"
            )
        return [
            "conda", "run", "-p", self.conda_env,
            "--no-capture-output",
            *cmd
        ]
    


class USegment3DSpec(PostProcess):

    def __init__(self,name,conda_env,module_path):
        super().__init__(name,conda_env,module_path)
        self.name=name
        self.conda_env= conda_env
        self.module_path=module_path

    def run_postprocess(self, input_dir, output_dir,gpu):

        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        post_process_cmd =[
            "python",
            "-m", f"{self.module_path}",
            "--input_dir",input_dir,
            '--output_dir', output_dir
        ]

        cmd = self._wrap_with_conda(post_process_cmd)

        subprocess.Popen(
            cmd,
            stdout=open(output_dir / "stdout.log", "w"),
            stderr=open(output_dir / "stderr.log", "w"),
            start_new_session=True,
            env=env
        )




class ThreeDCellComposerSpec(PostProcess):

    def __init__(self,name,conda_env,module_path):
        super().__init__(name,conda_env,module_path)
        self.name=name
        self.conda_env= conda_env
        self.module_path=module_path

    def run_postprocess(self, input_dir, output_dir):
        return super().run_postprocess(input_dir, output_dir)




