"""
adaptfm-set-pytorch: saves the user's custom PyTorch pip install command so
that install-nnunet and install-cellposesam can call it automatically.

Usage (after `pip install -e .`):
    adaptfm-set-pytorch

You will be prompted to paste the pip install command from pytorch.org, e.g.:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

The command is saved to ~/.adaptfm/pytorch_cmd.txt and can be updated at any
time by running this command again.
"""

import shlex
import subprocess
import sys
from pathlib import Path


ADAPTFM_ENV = "AdaptFM"


CONFIG_DIR = Path.home() / ".adaptfm"
PYTORCH_CMD_FILE = CONFIG_DIR / "pytorch_cmd.txt"

EXAMPLE = (
    "  pip install torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu118"
)


def main() -> None:
    print("\n=== AdaptFM — configure PyTorch install command ===\n")
    print("Paste the pip install command for your PyTorch build.")
    print("You can find the right command at: https://pytorch.org/get-started/locally/")
    print(f"Example:\n{EXAMPLE}\n")

    if PYTORCH_CMD_FILE.exists():
        existing = PYTORCH_CMD_FILE.read_text().strip()
        print(f"Current command: {existing}")
        overwrite = input("Overwrite? [y/N]: ").strip().lower()
        if overwrite != "y":
            print(f"Keeping existing command. Run conda run -n {ADAPTFM_ENV} {existing} to reinstall torch")
            sys.exit(0)
        print()

    while True:
        raw = input("PyTorch pip command: ").strip()

        # Allow the user to paste the full line including a leading "pip"
        # or just the arguments portion; normalise to a full pip invocation.
        if not raw:
            print("  Command cannot be empty. Try again.")
            continue

        # Strip a leading "pip install" if the user pasted the whole line,
        # then rebuild as a canonical "pip install …" string so _conda_run
        # can split it consistently.
        if raw.startswith("pip3 install "):
            canonical = raw
        elif raw.startswith("install "):
            canonical = "pip3 " + raw
        else:
            # Bare arguments (e.g. "torch torchvision …")
            canonical = "pip3 install " + raw

        print(f"\n  Will save: {canonical}")
        confirm = input("  Confirm? [Y/n]: ").strip().lower()
        if confirm in ("", "y"):
            break
        print("  Let's try again.\n")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PYTORCH_CMD_FILE.write_text(canonical + "\n")

    print(f"\n✓ Saved to {PYTORCH_CMD_FILE}")

    # Install PyTorch into the main adaptfm conda environment
    print(f"\n--- Installing PyTorch into '{ADAPTFM_ENV}' environment ---")
    cmd = ["conda", "run", "-n", ADAPTFM_ENV, "--no-capture-output"] + shlex.split(canonical)
    print(f"  + {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"\nWARNING: PyTorch install into '{ADAPTFM_ENV}' failed (exit code {result.returncode})."
            f"\n  You can retry manually:  conda run -n {ADAPTFM_ENV} {canonical}"
        )
    else:
        print(f"\n✓ PyTorch installed into '{ADAPTFM_ENV}' successfully.")

    print("  This command will also be used by install-nnunet and install-cellposesam.\n")


if __name__ == "__main__":
    main()
