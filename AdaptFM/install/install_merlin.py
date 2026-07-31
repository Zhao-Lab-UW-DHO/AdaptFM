"""
install-nnunet: creates the Merlin_nnUNet conda environment and installs Merlin_nnUNet + PyTorch.

Usage (after `pip install -e .`):
    install-nnunet

The user's custom PyTorch pip command is read from ~/.adaptfm/pytorch_cmd.txt.
Run `adaptfm-set-pytorch` first if that file does not exist yet.
"""

import subprocess
import sys
from pathlib import Path
import shlex
from AdaptFM.model.registry import _conda_prefix


ENV_NAME = "Merlin_nnUNet_adapt"
PYTHON_VERSION = "3.10"
PYTORCH_CMD_FILE = Path.home() / ".adaptfm" / "pytorch_cmd.txt"
MERLIN_REPO = "https://github.com/ashwinkumargb/Merlin-nnUNet.git"


ACTIVATE_SCRIPT = """\
export MERLIN_ROOT={merlin_root}
export ADAPTFM_ROOT={adaptfm_root}
export PYTHONPATH=$MERLIN_ROOT:$ADAPTFM_ROOT:$PYTHONPATH
"""

DEACTIVATE_SCRIPT = """\
unset MERLIN_ROOT
unset ADAPTFM_ROOT
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\\n' | grep -v "AdaptFM\\|MERLIN_ROOT" | tr '\\n' ':' | sed 's/:$//')
"""


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


def install_merlin_nnunet(repo_dir: Path, env_name: str):
    repo_dir = repo_dir.resolve()

    print("\n--- Running pip install -e . ---")
    result1 = subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install", "-e", str(repo_dir)],
        check=True
    )
    if result1.returncode == 0:
        print("✓ pip install -e . completed successfully.")
    else:
        print("✗ pip install -e . failed.")
        sys.exit(1)

    print("\n--- Running download_weights.py ---")
    download_script = repo_dir / "download_weights.py"
    result2 = subprocess.run(
        ["conda", "run", "-n", env_name, "python", str(download_script)],
        check=True
    )
    if result2.returncode == 0:
        print("✓ download_weights.py executed successfully.")
    else:
        print("✗ download_weights.py failed.")
        sys.exit(1)


def _write_conda_hooks(env: str, merlin_root: Path, adaptfm_root: Path) -> None:
    prefix = Path(_conda_prefix(env))

    activate_dir = prefix / "etc" / "conda" / "activate.d"
    deactivate_dir = prefix / "etc" / "conda" / "deactivate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    deactivate_dir.mkdir(parents=True, exist_ok=True)

    activate_file = activate_dir / "merlin_paths.sh"
    deactivate_file = deactivate_dir / "merlin_paths.sh"

    activate_file.write_text(
        ACTIVATE_SCRIPT.format(
            merlin_root=merlin_root,
            adaptfm_root=adaptfm_root,
        )
    )
    deactivate_file.write_text(DEACTIVATE_SCRIPT)

    print(f"  Wrote activate hook:   {activate_file}")
    print(f"  Wrote deactivate hook: {deactivate_file}")


def main() -> None:
    print(f"\n=== Installing Merlin_nnUNet environment: {ENV_NAME} ===\n")

    # Check that conda is available
    result = subprocess.run(
        ["conda", "info", "--json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print("ERROR: conda not found. Please install Miniconda or Anaconda first.")
        sys.exit(1)

    cwd = Path.cwd()

    merlin_root = (cwd.parent / "Merlin-nnUNet")
    adaptfm_root: Path =cwd
    merlin_root = merlin_root.expanduser().resolve()
    adaptfm_root = adaptfm_root.expanduser().resolve()

    print(f"\n  MERLIN_ROOT = {merlin_root}")
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
        print("To reinstall from scratch, run:  conda env remove -n Merlin_nnUNet_adapt")
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

    # Install PyTorch (user-specified build)
    print("\n--- Installing PyTorch ---")
    _conda_run(ENV_NAME, pytorch_cmd)

    # Install merlin-vlm
    print("\n--- Installing merlin-vlm ---")
    _conda_run(ENV_NAME, ["pip", "install", "merlin-vlm"])
    
        # Clone Merlin-nnUNet if the directory doesn't already contain the repo
    print("\n--- Cloning Merlin-nnUNet ---")
    if (merlin_root / ".git").exists():
        print(f"  Merlin-nnUNet already cloned at {merlin_root} — skipping.")
    else:
        merlin_root.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", MERLIN_REPO, str(merlin_root)])


    print("\n--- Writing conda environment hooks ---")
    _write_conda_hooks(ENV_NAME, merlin_root, adaptfm_root)
    
    install_merlin_nnunet(merlin_root, env_name="Merlin_nnUNet_adapt")


    print(f"\n✓ nnUNet environment '{ENV_NAME}' created successfully.")
    print(f"  Activate with:  conda activate {ENV_NAME}\n")


if __name__ == "__main__":
    main()
