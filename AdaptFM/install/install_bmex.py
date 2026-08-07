"""
install_BME-X: creates the BME-X conda environment, installs BME-X,
then swaps in the user's custom PyTorch build.

Usage (after `pip install -e .`):
    install-BME-X


"""

import subprocess
import sys
from pathlib import Path
import shlex
from AdaptFM.model.registry import _conda_prefix

ENV_NAME = "BME-X_adapt"
PYTHON_VERSION = "3.10" #guessting that this wroks
PYTORCH_CMD_FILE = Path.home() / ".adaptfm" / "pytorch_cmd.txt"
BMEX_REPO = "https://github.com/DBC-Lab/Brain_MRI_Enhancement"

ACTIVATE_SCRIPT = """\
export BMEX_ROOT={bmex_root}
export ADAPTFM_ROOT={adaptfm_root}
export PYTHONPATH=$BMEX_ROOT:$ADAPTFM_ROOT:$PYTHONPATH
"""

DEACTIVATE_SCRIPT = """\
unset BMEX_ROOT
unset ADAPTFM_ROOT
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\\n' | grep -v "AdaptFM\\|Brain_MRI_Enhancement" | tr '\\n' ':' | sed 's/:$//')
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




def _write_conda_hooks(env: str, bmex_root: Path, adaptfm_root: Path) -> None:
    prefix = Path(_conda_prefix(env))

    activate_dir = prefix / "etc" / "conda" / "activate.d"
    deactivate_dir = prefix / "etc" / "conda" / "deactivate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    deactivate_dir.mkdir(parents=True, exist_ok=True)

    activate_file = activate_dir / "sam_paths.sh"
    deactivate_file = deactivate_dir / "sam_paths.sh"

    activate_file.write_text(
        ACTIVATE_SCRIPT.format(
            bmex_root=bmex_root,
            adaptfm_root=adaptfm_root,
        )
    )
    deactivate_file.write_text(DEACTIVATE_SCRIPT)

    print(f"  Wrote activate hook:   {activate_file}")
    print(f"  Wrote deactivate hook: {deactivate_file}")


def main() -> None:
    print(f"\n=== Installing BMEX environment: {ENV_NAME} ===\n")

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
        print("To reinstall from scratch, run:  conda env remove -n bmex_adapt")
        sys.exit(0)

    cwd = Path.cwd()

    bmex_root = (cwd.parent / "Brain_MRI_Enhancement")
    adaptfm_root: Path =cwd
    bmex_root = bmex_root.expanduser().resolve()
    adaptfm_root = adaptfm_root.expanduser().resolve()

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



    print("\n--- Cloning BME-X ---")
    if (bmex_root / ".git").exists():
        print(f" BME-X already cloned at {bmex_root} — skipping.")
    else:
        bmex_root.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", BMEX_REPO, str(bmex_root)])

    print("\n--- Installing dependencies ---")
    _conda_run(ENV_NAME, ["python", "-m", "pip", "install", "-r", str(bmex_root / "requirements.txt")])

    
        # Remove the default torch/torchvision that cellpose bundled
    print("\n--- Removing default torch/torchvision ---")
    _conda_run(
        ENV_NAME,
        ["pip", "uninstall", "-y", "torch", "torchvision"],
        check=False,   # OK if they were never installed under these names
    )

    # Install the user's preferred PyTorch build
    print("\n--- Installing user-specified PyTorch ---")
    _conda_run(ENV_NAME, pytorch_cmd)


    # Write conda activate/deactivate hooks
    print("\n--- Writing conda environment hooks ---")
    _write_conda_hooks(ENV_NAME, bmex_root, adaptfm_root)

    print(f"\n✓ BME-X_adapt environment '{ENV_NAME}' created successfully.")
    print(f"  Activate with:  conda activate {ENV_NAME}\n")


if __name__ == "__main__":
    main()
