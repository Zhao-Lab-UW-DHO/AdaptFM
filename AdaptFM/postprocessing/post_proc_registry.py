from pathlib import Path
from AdaptFM.postprocessing.post_proc_spec import USegment3DSpec,ThreeDCellComposerSpec


def _read_prefix(env: str) -> str | None:
    p = Path.home() / ".adaptfm" / f"{env}.prefix"
    return p.read_text().strip() if p.exists() else None

ADAPTFM_POSTPROC_PATH = Path(__file__).resolve().parent


POSTPROC_REGISTRY={
    "USegment3D": USegment3DSpec(
        name= "USegment3D",
        conda_env = _read_prefix("usegment3d_adapt"),
        module_path =  str(ADAPTFM_POSTPROC_PATH / "pipelines" / "useg3d.py")
    )
    }
    
