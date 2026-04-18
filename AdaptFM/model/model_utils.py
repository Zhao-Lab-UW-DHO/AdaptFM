import inspect
import numpy as np

def extract_tunable_params(func):
    sig = inspect.signature(func)
    params = {}

    for name, p in sig.parameters.items():
        if p.default is inspect.Parameter.empty:
            continue

        params[name] = {
            "type": type(p.default),
            "default": p.default,
        }

    return params


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    minv = img.min()
    maxv = img.max()

    if maxv == minv:
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - minv) / (maxv - minv)
    img = (img * 255.0).clip(0, 255)

    return img.astype(np.uint8)

REPO_ROOT = Path(__file__).parent.parent  # adjust to your repo root

def install_sam2():
    sam2_dir = REPO_ROOT / "segmentation" / "sam2" / "repo"
    if not (sam2_dir / "setup.py").exists() and not (sam2_dir / "pyproject.toml").exists():
        print("Initializing SAM2 submodule...")
        subprocess.run(["git", "submodule", "update", "--init", str(sam2_dir)], check=True)
    print("Installing SAM2...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)], check=True)
    print("Downloading checkpoints...")
    subprocess.run(["bash", "download_ckpts.sh"], cwd=sam2_dir / "checkpoints", check=True)
    print(f"Done. SAM2 repo at: {sam2_dir}")





