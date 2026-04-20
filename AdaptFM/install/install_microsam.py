"""
install-microsam: creates the micro-sam conda environment.

Usage (after `pip install -e .`):
    install-microsam
"""

import subprocess
import sys


ENV_NAME = "micro-sam_adapt"
PYTHON_VERSION = "3.10"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  + {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def main() -> None:
    print(f"\n=== Installing micro-sam environment: {ENV_NAME} ===\n")

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
        print("To reinstall from scratch, run:  conda env remove -n micro-sam_adapt")
        sys.exit(0)

    # Create the environment (micro_sam is available on conda-forge)
    _run([
        "conda", "create",
        "-c", "conda-forge",
        "-n", ENV_NAME,
        f"python={PYTHON_VERSION}",
        "micro_sam",
        "-y",
    ])

    print(f"\n✓ micro-sam environment '{ENV_NAME}' created successfully.")
    print(f"  Activate with:  conda activate {ENV_NAME}\n")


if __name__ == "__main__":
    main()
