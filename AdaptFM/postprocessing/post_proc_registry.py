import sys
from pathlib import Path

from AdaptFM.postprocessing.post_proc_spec import USegment3DSpec, Conv2BinarySpec


def _read_prefix(env: str) -> str | None:
    p = Path.home() / ".adaptfm" / f"{env}.prefix"
    return p.read_text().strip() if p.exists() else None


ADAPTFM_POSTPROC_PATH = Path(__file__).resolve().parent


POSTPROC_REGISTRY = {
    "USegment3D": USegment3DSpec(
        name="USegment3D",
        conda_env=_read_prefix("usegment3d_adapt"),
        module_path=str(ADAPTFM_POSTPROC_PATH / "pipelines" / "useg3d.py"),
    ),

    "Convert uint8 TIFF data to binary mask (0|1)": Conv2BinarySpec(
        name="Convert uint8 TIFF data to binary mask (0|1)",
        conda_env=str(Path(sys.prefix)),
        module_path=str(ADAPTFM_POSTPROC_PATH / "pipelines" / "conv2binary.py")
    )

}
