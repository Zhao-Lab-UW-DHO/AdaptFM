# set up environment
import argparse
import datetime
import logging
import os
import random
from contextlib import nullcontext
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from torch.backends import cudnn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sammed3d.segment_anything.build_sam3D import sam_model_registry3D
from sammed3d.utils.click_method import get_next_click3D_torch_2,get_next_click3D_multiclass
from sammed3d.utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from sammed3d.utils.data_paths import img_datas


join = os.path.join


def initialize_globals(args):
    global device, logger, LOG_OUT_DIR, click_methods, MODEL_SAVE_PATH

    # Device selection (CUDA_VISIBLE_DEVICES already set before torch import)
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available() and args.gpu_ids:
        device = torch.device("cuda:0")  # first visible GPU
    else:
        device = torch.device("cpu")

    args.device = device

    logger = logging.getLogger(__name__)
    LOG_OUT_DIR = join(args.work_dir, args.task_name)
    os.makedirs(LOG_OUT_DIR, exist_ok=True)

    click_methods = {"random": get_next_click3D_torch_2,
                     'multiclass': get_next_click3D_multiclass}

    MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2023)


def main(args):
    initialize_globals(args)
    device_config(args)

    mp.set_sharing_strategy('file_system')

    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Ensure model is on the correct device
        model.to(args.device)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()



def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='label',
                          target_shape=(args.img_size, args.img_size,
                                        args.img_size)),  # crop only object region
            tio.RandomFlip(axes=(0, 1, 2)),
        ]),
        threshold=1000)
    
    
    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # train_dataloader = tio.SubjectsLoader(
    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader


class BaseTrainer:

    def __init__(self, model, dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if (args.resume):
            self.init_checkpoint(
                join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_loss_fn(self):
        #self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.seg_loss = DiceCELoss(
            sigmoid=False,
            softmax=True,
            to_onehot_y=True,      # converts integer gt to one-hot internally
            squared_pred=True,
            reduction='mean'
        )

    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW(
            [
                {
                    'params': sam_model.image_encoder.parameters()
                },  # , 'lr': self.args.lr * 0.1},
                {
                    'params': sam_model.prompt_encoder.parameters(),
                    'lr': self.args.lr * 0.1
                },
                {
                    'params': sam_model.mask_decoder.parameters(),
                    'lr': self.args.lr * 0.1
                },
            ],
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)

        if last_ckpt:
            if (self.args.allow_partial_weight):
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "dices": self.dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                "used_datas": img_datas,
            }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks,
                                   size=gt3D.shape[-3:],
                                   mode='trilinear',
                                   align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        # multiclass returns lists with variable length — handle both cases
        if len(batch_points) == 0:
            return None, None

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la

        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device).float()
        low_res_masks = F.interpolate(
            prev_masks.float(),
            size=(args.img_size // 4, args.img_size // 4, args.img_size // 4)
        )
        random_insert = np.random.randint(2, 9)

        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1 or points_input is None:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model, image_embedding, gt3D, low_res_masks, points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(
                    sam_model, image_embedding, gt3D, low_res_masks,
                    points=[points_input, labels_input])

            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss

        return prev_masks, return_loss

    def get_dice_score(self, prev_masks, gt3D):
        """
        prev_masks: (B, C, D, H, W) logits
        gt3D:       (B, 1, D, H, W) integer labels
        """
        num_classes = prev_masks.shape[1]
        pred_labels = prev_masks.argmax(dim=1)  # (B, D, H, W)
        true_labels = gt3D.squeeze(1)           # (B, D, H, W)

        dice_list = []
        for c in range(1, num_classes):  # skip background class 0
            for i in range(gt3D.shape[0]):
                pred_c = (pred_labels[i] == c)
                gt_c   = (true_labels[i] == c)

                intersection = (pred_c & gt_c).sum().item()
                union        = pred_c.sum().item() + gt_c.sum().item()

                if union == 0:
                    continue  # skip if class absent in both
                dice_list.append(2 * intersection / union)

        return np.nanmean(dice_list) if dice_list else 0.0

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        for step, data3D in enumerate(tbar):
            try:
                image3D, gt3D = data3D["image"], data3D["label"]
            except Exception as e:
                print(f"Error processing batch at step {step}: {e}")
            # import pdb; pdb.set_trace()
            my_context = self.model.no_sync if self.args.rank != - \
                1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1))
                image3D = image3D.unsqueeze(dim=1)
                image3D = image3D.to(device)

                # gt3D stays as long integer labels — do NOT one-hot encode here,
                # DiceCELoss with to_onehot_y=True handles that internally
                gt3D = gt3D.to(device).type(torch.long)
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss = self.interaction(sam_model,
                                                        image_embedding,
                                                        gt3D,
                                                        num_clicks=11)

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks, gt3D)
                cur_loss = loss.item()

                loss /= self.args.accumulation_steps

                self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(epoch,
                                                 sam_model.state_dict(),
                                                 describe=f'{epoch}_step_dice:{print_dice}_best')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss

        epoch_loss /= step + 1
        epoch_dice /= step + 1

        return epoch_loss, epoch_iou, epoch_dice, pred_list

    def eval_epoch(self, epoch, num_clicks):
        return 0

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def train(self):
        use_cuda = torch.cuda.is_available() and args.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                # save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe='latest')

                # save train loss best checkpoint
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, state_dict, describe='loss_best')

                # save train dice best checkpoint
                if epoch_dice > self.best_dice:
                    self.best_dice = epoch_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    """
    Optional additional config for multi-GPU.
    """
    if args.multi_gpu:
        args.nodes = 1
        args.ngpus_per_node = len(args.gpu_ids)
        args.world_size = args.nodes * args.ngpus_per_node


def main_worker(rank, args):
    setup(rank, args.world_size,args)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO if rank in [-1, 0] else logging.WARN,
                        filemode='w',
                        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size,args):
    # initialize the process group
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://127.0.0.1:{args.port}',
                            world_size=world_size,
                            rank=rank)


def cleanup():
    dist.destroy_process_group()

#dummy function with parameters and their defaults

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

if __name__ == '__main__':

    print(os.getpid())

    parser = argparse.ArgumentParser()
    parser.add_argument('--params')
    args = parser.parse_args()

    params = json.loads(args.params)

    for key, value in params.items():

        # --- FIX: robust gpu_ids parsing ---
        if key == "gpu_ids":
            # Case 1: already a list
            if isinstance(value, list):
                value = [int(v) for v in value]

            # Case 2: string like "[6]" or "[3,4]"
            elif isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                value = [int(v) for v in value.strip("[]").split(",")]

            # Case 3: single int as string "6"
            elif isinstance(value, str) and value.isdigit():
                value = [int(value)]

            # Case 4: single int
            elif isinstance(value, int):
                value = [value]

            else:
                raise ValueError(f"Unrecognized gpu_ids format: {value}")

            setattr(args, key, value)
            continue
        # -----------------------------------

        # Convert booleans
        if isinstance(value, str) and value.lower() in ['true', 'false']:
            value = value.lower() == 'true'

        # Convert numbers based on type hints
        elif isinstance(value, str) and key in launch_training.__annotations__:
            expected_type = launch_training.__annotations__[key]
            if expected_type == int:
                value = int(value)
            elif expected_type == float:
                value = float(value)

        setattr(args, key, value)

    # Set CUDA devices BEFORE importing torch
    if hasattr(args, "gpu_ids") and args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in args.gpu_ids)

    main(args)
