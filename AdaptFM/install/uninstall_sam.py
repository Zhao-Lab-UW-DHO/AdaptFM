# AdaptFM/install/sam_entrypoints.py  (replaces the earlier uninstall_sam2/uninstall_sam3 stubs)
import shutil
import subprocess
import sys
from pathlib import Path


_VALID = {"sam2", "sam3"}


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  + {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def uninstall_sam(sam_version: str) -> None:
    """
    Uninstall SAM2 or SAM3. Unlike the conda-env uninstaller, there is no
    env to remove and no .prefix/activate.d bookkeeping -- SAM2/SAM3 are
    pip-installed in editable mode straight into whatever interpreter is
    running this, from a repo cloned at REPO_ROOT/segmentation/<version>.
    So this only needs to: pip-uninstall the package, then remove that
    cloned directory.
    """
    if sam_version not in _VALID:
        raise ValueError(f"sam_version must be one of {_VALID}, got {sam_version!r}")

    print(f"\n=== Uninstalling {sam_version.upper()} ===\n")

    # 1. pip-uninstall from the *current* interpreter -- there's no other
    #    env to target, this is the env the GUI is already running in.
    print(f"--- Uninstalling '{sam_version}' package ---")
    _run([sys.executable, "-m", "pip", "uninstall", "-y", sam_version], check=False)

    REPO_ROOT = Path(__file__).parent.parent


    # 2. Remove the cloned repo, keeping your two safety checks: don't
    #    delete cwd, and only delete something that's actually a git clone.
    repo_dir = (REPO_ROOT / "segmentation" / sam_version).resolve()
    if not repo_dir.exists():
        print(f"  No cloned repo found at {repo_dir}. Skipping.")
    elif repo_dir == Path.cwd().resolve():
        print(f"  Skipping {repo_dir} (it is your current working directory).")
    elif (repo_dir / ".git").exists():
        print(f"  Removing repository: {repo_dir}")
        shutil.rmtree(repo_dir)
    else:
        print(f"  Warning: {repo_dir} exists but has no .git -- doesn't look "
              f"like a clone. Leaving it in place; remove manually if needed.")

    print(f"\n✓ Uninstallation of '{sam_version}' complete.\n")


# --- entry points wired into SamCard / pyproject.toml (unchanged names) ---


def uninstall_sam2():
    uninstall_sam("sam2")

def uninstall_sam3():
    uninstall_sam("sam3")