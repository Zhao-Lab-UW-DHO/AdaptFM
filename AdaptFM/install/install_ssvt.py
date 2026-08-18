from pathlib import Path
import subprocess
import shutil

REPO_ROOT = Path(__file__).parent.parent

def install_ssvt():

    print("Installing Huggingface...")

    subprocess.run(
    ["python", "-m", "pip", "install", "huggingface_hub"],
    check=True,
    )

    from huggingface_hub import snapshot_download

    ssvt_dir = REPO_ROOT / "SSVT" / "checkpoint"
    ssvt_dir.mkdir(parents=True,exist_ok=True)

    snapshot_download(
        repo_id='hbakhtiar/SSVT_Organoids',
        repo_type='model',
        allow_patterns='SSVT.pth',  # Only download the checkpoint
        local_dir=ssvt_dir  # Add destination folder
    )

    print(f"SSVT downloaded in {ssvt_dir}")

def uninstall_ssvt():

    print("Removing SSVT...")

    ssvt_dir = REPO_ROOT / "SSVT" / "checkpoint"

    if not ssvt_dir.exists():
        print("Checkpoint file does not exist. Did you move the folder?")
        return

    try:
        shutil.rmtree(ssvt_dir)
        print(f"Removed: {ssvt_dir}")
    except Exception as e:
        print(f"Failed to remove {ssvt_dir}: {e}")

    