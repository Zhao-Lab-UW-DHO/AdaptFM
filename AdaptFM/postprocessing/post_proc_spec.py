from abc import ABC, abstractmethod



class PostProcess(ABC):
    name: str = "BasePostProcess"

    def __init__(self, name, conda_env,module_path):
        self.name = name
        self.conda_env = conda_env
        self.module_path = module_path

    @abstractmethod
    def run_postprocess(self, input_dir, output_dir):
        """run post processing on a separate thread that communicates back to the main"""
        pass


    def _wrap_with_conda(self, cmd: list[str]) -> list[str]:
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

    def run_postprocess(self, input_dir, output_dir):
        return super().run_postprocess(input_dir, output_dir)


class ThreeDCellComposerSpec(PostProcess):

    def __init__(self,name,conda_env,module_path):
        super().__init__(name,conda_env,module_path)
        self.name=name
        self.conda_env= conda_env
        self.module_path=module_path

    def run_postprocess(self, input_dir, output_dir):
        return super().run_postprocess(input_dir, output_dir)




