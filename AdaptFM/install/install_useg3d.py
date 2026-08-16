"""
install-useg3d: creates the useg3d conda environment, installs useg3d,
then swaps in the user's custom PyTorch build.

Usage (after `pip install -e .`):
    install-useg3d

useg3d pulls in a default PyTorch (often CUDA 13 / latest) that most users
don't have.  This script removes it immediately after install and replaces it
with the build specified in ~/.adaptfm/pytorch_cmd.txt.
Run `adaptfm-set-pytorch` first if that file does not exist yet.
"""

import subprocess
import sys
from pathlib import Path
import shlex
from AdaptFM.model.registry import _conda_prefix

ENV_NAME = "usegment3d_adapt"
PYTHON_VERSION = "3.9"
PYTORCH_CMD_FILE = Path.home() / ".adaptfm" / "pytorch_cmd.txt"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  + {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def _conda_run(env: str, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command inside a conda environment."""
    full_cmd = ["conda", "run", "-n", env, "--no-capture-output"] + cmd
    return _run(full_cmd, check=check)


def _read_pytorch_cmd() -> list[str]:
    if not PYTORCH_CMD_FILE.exists():
        print("ERROR: PyTorch install command not configured.")
        print("  Run `adaptfm-set-pytorch` first to save your pytorch pip command.")
        sys.exit(1)
    raw = PYTORCH_CMD_FILE.read_text().strip()
    if not raw:
        print(f"ERROR: {PYTORCH_CMD_FILE} is empty. Run `adaptfm-set-pytorch` again.")
        sys.exit(1)
    return shlex.split(raw)




def main() -> None:
    print(f"\n=== Installing u-Segment3D environment: {ENV_NAME} ===\n")

    # Check that conda is available
    result = subprocess.run(
        ["conda", "info", "--json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print("ERROR: conda not found. Please install Miniconda or Anaconda first.")
        sys.exit(1)

    # Check whether the environment already exists
    env_check = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    if ENV_NAME in env_check.stdout:
        print(f"Environment '{ENV_NAME}' already exists — skipping creation.")
        print("To reinstall from scratch, run:  conda env remove -n usegment3d_adapt")
        sys.exit(0)

    # Read user's custom PyTorch command
    pytorch_cmd = _read_pytorch_cmd()
    print(f"PyTorch command: {' '.join(pytorch_cmd)}\n")

    # Create bare environment
    _run([
        "conda", "create",
        "-n", ENV_NAME,
        f"python={PYTHON_VERSION}",
        "-y",
    ])

    prefix = _conda_prefix(ENV_NAME)
    config_path = Path.home() / ".adaptfm" / f"{ENV_NAME}.prefix"
    config_path.write_text(prefix + "\n")
    print(f"  Wrote env prefix to {config_path}")   

    # Install u-Segment3D (this drags in a default torch/torchvision)
    print("\n--- Installing u-Segment3D ---")
    _conda_run(ENV_NAME, ["python", "-m", "pip", "install", "u-Segment3D"])

    # Remove the default torch/torchvision that u-Segment3D bundled
    print("\n--- Removing default torch/torchvision ---")
    _conda_run(
        ENV_NAME,
        ["pip", "uninstall", "-y", "torch", "torchvision"],
        check=False,   # OK if they were never installed under these names
    )

    # Install the user's preferred PyTorch build
    print("\n--- Installing user-specified PyTorch ---")
    _conda_run(ENV_NAME, pytorch_cmd)

    print(f"\n✓ u-Segment3D environment '{ENV_NAME}' created successfully.")
    print(f"  Activate with:  conda activate {ENV_NAME}\n")


if __name__ == "__main__":
    main()
