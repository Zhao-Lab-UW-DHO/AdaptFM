import inspect
import numpy as np
from pathlib import Path
import subprocess, sys

def extract_tunable_params(func):
    sig = inspect.signature(func)
    params = {}

    for name, p in sig.parameters.items():
        if p.default is inspect.Parameter.empty:
            continue

        params[name] = {
            "type": type(p.default),
            "default": p.default,
        }

    return params


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    minv = img.min()
    maxv = img.max()

    if maxv == minv:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - minv) / (maxv - minv)
    img = (img * 255.0).clip(0, 255)

    return img.astype(np.uint8)

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).parent.parent  # AdaptFM repo root


def install_sam2():
    sam2_dir = REPO_ROOT / "segmentation" / "sam2" / "repo"
    sam2_rel_path = "segmentation/sam2/repo"  # IMPORTANT: relative git submodule path

    # Ensure we are inside a git repo
    if not (REPO_ROOT / ".git").exists():
        raise RuntimeError(f"{REPO_ROOT} is not a git repository root")

    # Check whether SAM2 is already installed
    if not (sam2_dir / "setup.py").exists() and not (sam2_dir / "pyproject.toml").exists():
        print("Initializing SAM2 submodule...")

        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive", sam2_rel_path],
            cwd=REPO_ROOT,
            check=True,
        )

    print("Installing SAM2...")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)],
        check=True,
    )

    print("Downloading checkpoints...")

    ckpt_dir = sam2_dir / "checkpoints"
    subprocess.run(
        ["bash", "download_ckpts.sh"],
        cwd=ckpt_dir,
        check=True,
    )

    print(f"Done. SAM2 repo at: {sam2_dir}")




