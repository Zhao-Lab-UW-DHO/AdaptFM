"""
install-sammed3d: creates the SAM-Med3D conda environment, installs all
dependencies, clones the SAM-Med3D repo, and writes conda activate/deactivate
hooks that put both SAM-Med3D and AdaptFM on PYTHONPATH automatically.

Usage (after `pip install -e .`):
    install-sammed3d [--sammed3d-root PATH] [--adaptfm-root PATH]

If --sammed3d-root / --adaptfm-root are omitted the script will prompt
interactively.  The SAM-Med3D repo is cloned into --sammed3d-root if that
directory does not already exist.
"""

import argparse
import subprocess
import sys
import shlex
from pathlib import Path
from AdaptFM.model.registry import _conda_prefix


ENV_NAME = "sammed3d_adapt"
PYTHON_VERSION = "3.10"
SAMMED3D_REPO = "https://github.com/uni-medical/SAM-Med3D.git"

UV_PACKAGES = [
    "torch==2.6.0",
    "torchvision==0.21.0",
    "torchaudio==2.6.0",
]

EXTRA_PACKAGES = [
    "torchio",
    "opencv-python-headless",
    "matplotlib",
    "prefetch_generator",
    "monai",
    "edt",
    "surface-distance",
    "medim",
    "numpy",
    "SimpleITK",
    "requests",
]

ACTIVATE_SCRIPT = """\
export SAMMED3D_ROOT={sammed3d_root}
export ADAPTFM_ROOT={adaptfm_root}
export PYTHONPATH=$SAMMED3D_ROOT:$ADAPTFM_ROOT:$PYTHONPATH
"""

DEACTIVATE_SCRIPT = """\
unset SAMMED3D_ROOT
unset ADAPTFM_ROOT
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\\n' | grep -v "AdaptFM\\|SAM-Med3D" | tr '\\n' ':' | sed 's/:$//')
"""


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  + {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def _conda_run(env: str, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    full_cmd = ["conda", "run", "-n", env, "--no-capture-output"] + cmd
    return _run(full_cmd, check=check)





def _write_conda_hooks(env: str, sammed3d_root: Path, adaptfm_root: Path) -> None:
    prefix = _conda_prefix(env)

    activate_dir = prefix / "etc" / "conda" / "activate.d"
    deactivate_dir = prefix / "etc" / "conda" / "deactivate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    deactivate_dir.mkdir(parents=True, exist_ok=True)

    activate_file = activate_dir / "sam_paths.sh"
    deactivate_file = deactivate_dir / "sam_paths.sh"

    activate_file.write_text(
        ACTIVATE_SCRIPT.format(
            sammed3d_root=sammed3d_root,
            adaptfm_root=adaptfm_root,
        )
    )
    deactivate_file.write_text(DEACTIVATE_SCRIPT)

    print(f"  Wrote activate hook:   {activate_file}")
    print(f"  Wrote deactivate hook: {deactivate_file}")


def _prompt_path(name: str, default: Path | None = None) -> Path:
    prompt = f"Enter path for {name}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    while True:
        raw = input(prompt).strip()
        if not raw and default:
            return default
        if raw:
            return Path(raw).expanduser().resolve()
        print("  Path cannot be empty.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the SAM-Med3D conda environment."
    )
    parser.add_argument(
        "--sammed3d-root",
        type=Path,
        default=None,
        help="Directory where SAM-Med3D lives (or will be cloned into).",
    )
    parser.add_argument(
        "--adaptfm-root",
        type=Path,
        default=None,
        help="Root directory of the AdaptFM repository.",
    )
    args = parser.parse_args()

    print(f"\n=== Installing SAM-Med3D environment: {ENV_NAME} ===\n")

    # Check conda
    result = subprocess.run(
        ["conda", "info", "--json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print("ERROR: conda not found. Please install Miniconda or Anaconda first.")
        sys.exit(1)

    # Resolve paths (CLI args take priority, then interactive prompt)
    sammed3d_root: Path = args.sammed3d_root or _prompt_path(
        "SAM-Med3D root (will clone repo here if absent)"
    )
    adaptfm_root: Path = args.adaptfm_root or _prompt_path(
        "AdaptFM root",
        default=Path.cwd(),
    )
    sammed3d_root = sammed3d_root.expanduser().resolve()
    adaptfm_root = adaptfm_root.expanduser().resolve()

    print(f"\n  SAMMED3D_ROOT = {sammed3d_root}")
    print(f"  ADAPTFM_ROOT  = {adaptfm_root}\n")

    # Check whether the environment already exists
    env_check = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
        check=True,
    )
    if ENV_NAME in env_check.stdout:
        print(f"Environment '{ENV_NAME}' already exists — skipping creation.")
        print("To reinstall from scratch, run:  conda env remove -n sammed3d_adapt")
        # Still (re-)write the hooks in case the paths changed
        print("\nUpdating conda activation hooks …")
        _write_conda_hooks(ENV_NAME, sammed3d_root, adaptfm_root)
        sys.exit(0)

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

    # Install uv inside the environment, then use it to install torch + extras
    print("\n--- Installing uv ---")
    _conda_run(ENV_NAME, ["pip", "install", "uv"])

    print("\n--- Installing PyTorch via uv ---")
    _conda_run(ENV_NAME, ["uv", "pip", "install"] + UV_PACKAGES)

    print("\n--- Installing extra dependencies via uv ---")
    _conda_run(ENV_NAME, ["uv", "pip", "install"] + EXTRA_PACKAGES)

    # Clone SAM-Med3D if the directory doesn't already contain the repo
    print("\n--- Cloning SAM-Med3D ---")
    if (sammed3d_root / ".git").exists():
        print(f"  SAM-Med3D already cloned at {sammed3d_root} — skipping.")
    else:
        sammed3d_root.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", SAMMED3D_REPO, str(sammed3d_root)])

    # Write conda activate/deactivate hooks
    print("\n--- Writing conda environment hooks ---")
    _write_conda_hooks(ENV_NAME, sammed3d_root, adaptfm_root)

    print(f"\n✓ SAM-Med3D environment '{ENV_NAME}' created successfully.")
    print(f"  Activate with:  conda activate {ENV_NAME}")
    print("  PYTHONPATH will be set automatically on activation.\n")


if __name__ == "__main__":
    main()
