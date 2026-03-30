# gui_scripts/models/base.py
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import json
import os

class ModelSpec(ABC):
    name: str
    conda_env: str

    supports_training: bool = True
    supports_inference: bool = True

    # -------- Parameters --------
    @abstractmethod
    def default_params(self) -> dict:
        """Concrete default params (single values)."""

    @abstractmethod
    def tunable_params(self, **kwargs) -> dict:
        """
        Parameter schema for GUI + grid search.
        kwargs allows training_tag, config, etc.
        """

    # -------- Dataset adapter --------
    @abstractmethod
    def prepare_dataset(self, dataset_manager, output_dir: Path) -> Path:
        """Return prepared dataset directory."""
        pass

    # -------- Commands --------
    @abstractmethod
    def training_command(
        self,
        dataset_dir: Path,
        params: dict,
        run_dir: Path,
    ) -> list[str]:
        pass

    @abstractmethod
    def inference_command(
        self,
        model_path: Path,
        images_dir: Path,
        output_dir: Path,
    ) -> list[str]:
        pass

   

    def _wrap_with_conda(self, cmd: list[str]) -> list[str]:
        return [
            "conda", "run", "-p", self.conda_env,
            "--no-capture-output",
            *cmd
        ]
    
    def _wrap_with_python_env(self, cmd: list[str]) -> list[str]:
        python = Path(self.python_env) / "bin" / "python"
        return [str(python), *cmd]
    
    def execution_params(self):
        return {
            "gpu": {
                "type": int,
                "default": 0,
                "min": 0,
                "label": "GPU index",
            }
        }
