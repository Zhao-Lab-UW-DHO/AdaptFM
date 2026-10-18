import json
from pathlib import Path

def load_saved_lora(ckpt_dir: str,
                    device: str):

    ckpt_dir = Path(ckpt_dir)
    config_file = ckpt_dir / "adapter_config.json"
    weights_file = ckpt_dir / "adapter_model.pt"

    if not config_file.exists() or not weight_file.exists():
        raise FileNotFoundError(
            f"Missing required checkpoint files in {ckpt_dir}")
    
    with open(config_file, "r") as f:
        config = json.load(f)

    model_env = config["model_env]
    is_3D = config["is_3D"]
    lora_cfg = config["lora_config"]
    
