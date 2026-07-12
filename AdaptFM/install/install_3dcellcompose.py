"""
install-3dcellcompose: creates the 3dcellcompose conda environment, installs ThreeDCellComposer,
then swaps in the user's custom PyTorch build.

Usage (after `pip install -e .`):
    install-3dcellcompose

"""

import subprocess
import sys
from pathlib import Path
import shlex
from AdaptFM.model.registry import _conda_prefix

ENV_NAME = "3dcellcompose_adapt"
PYTHON_VERSION = "3.10"
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

NUMPY_PIN = "1.23.5"  # exact pin, not a range — avoids any future patch-level ABI drift


def main() -> None:
    print(f"\n=== Installing 3dcellcompose_adapt environment: {ENV_NAME} ===\n")

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
        print("To reinstall from scratch, run:  conda env remove -n 3dcellcompose_adapt")
        sys.exit(0)

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

    # --- Write a pip constraints file pinning numpy exactly ---
    # Every pip install below passes -c constraints.txt so nothing (deepcell's,
    # scikit-image's, or ThreeDCellComposer's loose numpy requirement) can
    # silently resolve to a different numpy than the one extensions get built against.
    constraints_path = Path.home() / ".adaptfm" / f"{ENV_NAME}.constraints.txt"
    constraints_path.write_text(f"numpy=={NUMPY_PIN}\n")
    print(f"  Wrote numpy constraint ({NUMPY_PIN}) to {constraints_path}")

    # --- Finalize numpy + core scientific stack FIRST, before any compiled
    # extension gets built against it. This is the step that used to run
    # AFTER `deepcell`, which is what caused the ABI mismatch. ---
    print("\n--- Installing core deps (numpy pinned) ---")
    _run([
        "conda", "run", "-n", ENV_NAME, "--no-capture-output",
        "conda", "install", "-c", "conda-forge",
        f"numpy={NUMPY_PIN}", "scipy", "pandas", "scikit-image", "-y",
    ])

    # Sanity-check numpy didn't drift during environment solve

    _conda_run(
        ENV_NAME,
        ["python", "-m", "pip", "install", "cython", "-c", str(constraints_path)],
    )

    # --- Build compiled extensions LAST, now that numpy is final and won't
    # move again. --no-cache-dir avoids pip silently reusing a stale wheel
    # built against a different numpy from a previous run. ---
    print("\n--- Installing deepcell (building deepcell-toolbox against pinned numpy) ---")
    _conda_run(
        ENV_NAME,
        [
            "python", "-m", "pip", "install", "deepcell",
            "--no-build-isolation", "--no-cache-dir",
            "-c", str(constraints_path),
        ],
    )

    print("\n--- Installing ThreeDCellComposer (pinned, avoids skimage versions) ---")
    _conda_run(
        ENV_NAME,
        [
            "python", "-m", "pip", "install", "ThreeDCellComposer==1.5.5",
            "--no-build-isolation", "--no-cache-dir",
            "-c", str(constraints_path),
        ],
    )

    # Re-check numpy wasn't silently bumped by either of the above installs

    # --- Smoke test: actually import the extension that broke last time.
    # Fail the install script itself rather than finding out at job runtime. ---
    print("\n--- Verifying deepcell_toolbox extension imports cleanly ---")
    smoke_test = _conda_run(
        ENV_NAME,
        [
            "python", "-c",
            "from deepcell_toolbox.compute_overlap import compute_overlap; "
            "import numpy; print('numpy', numpy.__version__, 'OK')",
        ],
        check=False,
    )
    if smoke_test.returncode != 0:
        print("ERROR: deepcell_toolbox failed to import after install. "
              "numpy ABI mismatch likely reintroduced — see stderr above.")
        sys.exit(1)

    print(f"\n✓ 3dcellcompose environment '{ENV_NAME}' created successfully.")
    print(f"  Activate with:  conda activate {ENV_NAME}\n")
