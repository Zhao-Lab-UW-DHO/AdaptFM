import inspect
import numpy as np
from pathlib import Path
import subprocess, sys
import site
import importlib

def _refresh_sam_pkg_after_install(pgk_name):
    importlib.reload(site)
    importlib.invalidate_caches()
    importlib.import_module(pgk_name)
    if pgk_name in sys.modules:
        importlib.reload(sys.modules[pgk_name])


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

REPO_ROOT = Path(__file__).parent.parent


def install_sam2():
    sam2_dir = REPO_ROOT / "segmentation" / "sam2"

    # Step 1 — clone if missing
    if not sam2_dir.exists():
        print("Cloning SAM2 repository...")

        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/sam2.git",
                str(sam2_dir),
            ],
            check=True,
        )

    # Step 2 — install in editable mode
    print("Installing SAM2...")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)],
        check=True,
    )

    # Step 3 — optional checkpoints
    ckpt_dir = sam2_dir / "checkpoints"
    ckpt_script = ckpt_dir / "download_ckpts.sh"

    if ckpt_script.exists():
        print("Downloading checkpoints...")
        subprocess.run(
            ["bash", str(ckpt_script)],
            cwd=ckpt_dir,
            check=True,
        )

    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=sam2_dir,
        check=True,
    )
    
    _refresh_sam_pkg_after_install("sam2")
    print(f"Done. SAM2 installed at: {sam2_dir}")
    



def install_sam3():
    sam3_dir = REPO_ROOT / "segmentation" / "sam3"

    # Step 1 — clone if missing
    if not sam3_dir.exists():
        print("Cloning SAM3 repository...")

        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/sam3.git",
                str(sam3_dir),
            ],
            check=True,
        )

    # Step 2 — install in editable mode
    print("Installing SAM3...")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(sam3_dir)],
        check=True,
    )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "einops"],
        check=True,
    )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pycocotools"],
        check=True,
    )

    _refresh_sam_pkg_after_install("sam3")
    print(f"Done. SAM3 installed at: {sam3_dir}")
    


