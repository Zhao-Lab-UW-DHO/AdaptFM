import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

# def conda_env_exists(env_name: str) -> bool:
#     """Checks if a specific conda environment exists on the system."""
#     try:
#         # Run 'conda env list' and capture the output
#         result = subprocess.run(
#             ["conda", "env", "list"],
#             capture_output=True,
#             text=True,
#             check=True
#         )
#         environments = [line.split()[0] for line in result.stdout.splitlines() if line and not line.startswith('#')]
#         # env looks like #['3dcellcompose_adapt', 'AdaptFM', 'BME-X_adapt', 'CTFM_adapt', 'Merlin_nnUNet_adapt', 'cellpose_adapt', 'cellsam_adapt', 'micro-sam_adapt','nnUNet_adapt', 'sammed3d_adapt']
#         return env_name in environments
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         return False


# MICROSAM_ENV_EXISTS = conda_env_exists("micro-sam_adapt")
# @unittest.skipUnless(MICROSAM_ENV_EXISTS, f"Could not find MicroSAM conda env")
class TestMicroSAM(unittest.TestCase):
    def setUp(self):

        self.test_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.test_dir.name)

        self.input_dir = self.root_path / "input_data"
        self.output_dir = self.root_path / "output_data"

        self.input_dir.mkdir()
        self.output_dir.mkdir()

        self.mock_files = ["sample_01.tif", "sample_02.tif"]
        self.generated_inputs = []

        for filename in self.mock_files:
            file_path = self.input_dir / filename
            mock_data = np.random.randint(0, 255, size=(5, 25, 25), dtype=np.uint8)
            tifffile.imwrite(str(file_path), mock_data)
            self.generated_inputs.append(file_path)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_will_run(self):
        print("OK")
        # from AdaptFM.model.registry import MODEL_REGISTRY
        # msam_spec = MODEL_REGISTRY['microSAM']
        # print(msam_spec.conda_env)

        from AdaptFM.model.foundation_models.microSAM.inference_wrapper import (
            run_microsam_inference,
        )

        run_microsam_inference(f"{self.input_dir}", f"{self.output_dir}", None)
        expected_seg = self.output_dir / "sample_01.tif"
        self.assertTrue(expected_seg.exists(), "Segmentation was not generated.")
        print(list(self.output_dir.glob("*")))

        # the inference wrapper passes: params collected based on the model, and always has the gpu index
        # dataset_dir, checkpoint_path, output_dir

        # specifically the microsam spec run inference
        # gets the GPU, copies system environment and sets the CUDA_VISIBLE_DEVICES to the gpu chosen
        # loads the wrapped inference command:   python -m AdaptFM.model.foundation_models.microSAM.inference_wrapper with args: datadir, outputdir, checkpointdir
        # actual file is <repo-root>/AdaptFM/model/foundation_models/microSAM/inference_wrapper.py
        # and launches the command wrapped with conda  "conda", "run", "-p", self.conda_env --no-capture-output <python -m ... >
        # where self.conda_env is defined in the registry to be the absolute path of the conda environment (the prefix of the python binary)
