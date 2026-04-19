def launch_training(
    task_name: str = "union_train",
    click_type: str = "random",
    multi_click: bool = False,
    model_type: str = "vit_b_ori",
    checkpoint: str = "ckpt/sam_med3d.pth",
    device: str = "cuda",
    work_dir: str = "work_dir",

    # train
    num_workers: int = 24,
    gpu_ids: list[int] = [0, 1],
    multi_gpu: bool = False,
    resume: bool = False,
    allow_partial_weight: bool = False,

    # lr_scheduler
    lr_scheduler: str = "multisteplr",
    step_size: list[int] = [120, 180],
    gamma: float = 0.1,
    num_epochs: int = 200,
    img_size: int = 128,
    batch_size: int = 12,
    accumulation_steps: int = 20,
    lr: float = 8e-4,
    weight_decay: float = 0.1,
    port: int = 12361,
):
    return
  
