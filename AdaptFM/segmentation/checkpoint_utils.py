import urllib.request
from pathlib import Path
import os

SAM2_CHECKPOINTS = {
    "sam2_hiera_tiny.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",

    "sam2_hiera_small.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",

    "sam2_hiera_base.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",

    "sam2_hiera_large.pt":
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

def check_sam2_installed():

    try:
        import sam2

    except ImportError:

        raise RuntimeError(
            "SAM2 is not installed.\n"
            "Install with:\n"
            "pip install AdaptFM[sam2]"
        )

def check_sam3_installed():

    try:
        import sam3

    except ImportError:

        raise RuntimeError(
            "SAM3 is not installed.\n"
            "Install with:\n"
            "pip install AdaptFM[sam3]"
        )


def get_checkpoint_dir():
    base = Path(
        os.environ.get(
            "ADAPTFM_CHECKPOINT_DIR",
            Path.home() / ".cache" / "adaptfm"
        )
    )

    base.mkdir(parents=True, exist_ok=True)

    return base

def download_file(url, dest):
    print(f"Downloading {dest.name}...")

    urllib.request.urlretrieve(url, dest)


def ensure_sam2_checkpoints():
    ckpt_dir = get_checkpoint_dir() / "sam2"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for name, url in SAM2_CHECKPOINTS.items():

        dest = ckpt_dir / name

        if not dest.exists():

            download_file(url, dest)

    return ckpt_dir


def get_sam3_checkpoint_dir():
    path = get_checkpoint_root() / "sam3"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sam3_missing_message(expected_path):

    return f"""
SAM3 checkpoint not found.

SAM3 requires manual download from Hugging Face.

Steps:

1. Request access to the SAM3 model on Hugging Face

2. Download the checkpoint

3. Place it here:

    {expected_path}

Or set a custom directory:

    export ADAPTFM_CHECKPOINT_DIR=/path/to/models

Then rerun your code.
"""


def ensure_sam3_checkpoint():

    ckpt_dir = get_sam3_checkpoint_dir()

    pt_files = list(ckpt_dir.glob("*.pt"))
    
    if not pt_files:
        raise RuntimeError(sam3_missing_message(ckpt_dir))

    return expected
