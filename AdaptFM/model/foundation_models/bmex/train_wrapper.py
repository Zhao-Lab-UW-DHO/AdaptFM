import argparse
from argparse import Namespace
import json
import torch
from DUNet3D import DenseUNet3d
from tensorboardX import SummaryWriter
import time
import numpy as np
import SimpleITK as sitk
import os

import torch.nn as nn


import math
from monai import transforms, data
from monai.data import load_decathlon_datalist

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0,high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Resized(
                    keys=["image", "label"],
                    spatial_size=(32, 32, 32),
                    mode=("trilinear", "nearest"),
                ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Resized(
                keys=["image", "label"],
                spatial_size=(32, 32, 32),
                mode=("trilinear", "nearest"),
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.ToTensord(keys=["image"]),
            transforms.Resized(
            keys=["image", "label"],
            spatial_size=(32, 32, 32),
            mode=("trilinear", "nearest"),
            ),
            
        ]
    )


    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "testing",
                                            base_dir=data_dir)
        #print(test_files)
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(test_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     sampler=test_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json,
                                           True,
                                           "training",
                                           base_dir=data_dir)

        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist,
                transform=train_transform,
                cache_num=1,
                cache_rate=1.0,
                num_workers=args.workers,
            )

        train_sampler = Sampler(train_ds) if args.distributed else None
        
        train_loader = data.DataLoader(train_ds,
                                       batch_size=args.batch_size,
                                       shuffle=(train_sampler is None),
                                       num_workers=args.workers,
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       persistent_workers=True)
        val_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        loader = [train_loader, val_loader]

    return loader



def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def train_epoch(model,
                loader,
                optimizer,
                scheduler,
                epoch,
                max_epochs,
                batch_size=8,
                logdir=""):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    loss_func = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step_num = 0
    for idx, batch_data in enumerate(loader):
        data, label = batch_data['image'], batch_data['label']

        data = data / 10000.0

        data = data.to(device).float()
        label = label.to(device).long()

        for param in model.parameters(): param.grad = None
        logits = model(data)

        loss = loss_func(logits, label.squeeze(1))  # squeeze channel dim, keep batch dim

        loss.backward()
        optimizer.step()

        run_loss.update(loss.item(), n=data.shape[0])

        print('Epoch {}/{} {}/{}'.format(epoch, max_epochs, idx, len(loader)),
              'loss: {:.4f}'.format(run_loss.avg),
              'time {:.2f}s'.format(time.time() - start_time))

        start_time = time.time()
        step_num = step_num + 1
        modelname = 'step-model-all.pt'

        if (step_num + 1) % 1000 == 0:
            save_checkpoint(model, (epoch), logdir, filename=modelname,
                            best_acc=0,
                            optimizer=optimizer,
                            scheduler=scheduler)

    return run_loss.avg

def save_checkpoint(model,
                    epoch,
                    logdir,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict()
    save_dict = {
        'epoch': epoch,
        'best_acc': best_acc,
        'state_dict': state_dict
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename = os.path.join(logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params')
    parser.add_argument("--data_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = json.loads(args.params)
    for key, value in params.items():

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
    
    checkpoint= args.chk_path
    max_epochs = args.max_epochs

    json_list = os.path.join(args.output_dir,'json_list.json')
    
        
    model1 = DenseUNet3d()
    model1_dict = torch.load(checkpoint, map_location=('cpu'))
    model1.load_state_dict(model1_dict['state_dict'])
    model1 = model1.to(device)

    writer = SummaryWriter(log_dir=args.output_dir)

    val_acc_max = 0.

    optimizer = torch.optim.AdamW(
    model1.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_epochs,
    eta_min=1e-6,
    )

    loader_args = Namespace(
        data_dir=args.output_dir,
        json_list=json_list,
        batch_size=args.batch_size,
        workers=args.num_workers,
        test_mode=False,
        distributed=False,
        use_normal_dataset=True,
    )
    
    train_loader,_ = get_loader(loader_args)

    for epoch in range(0,max_epochs):
        epoch_time = time.time()

        
        print(time.ctime(), 'Epoch:', epoch)
        train_loss = train_epoch(model1,
                                 train_loader,
                                 optimizer,
                                 scheduler,
                                 epoch=epoch,
                                 max_epochs=max_epochs,
                                 batch_size=args.batch_size,
                                 logdir=args.output_dir)
        
    
        print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
              'time {:.2f}s'.format(time.time() - epoch_time), 'lr: {:,.4f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        modelname = 'epoch' + str(epoch + 0) + 'model-all.pt'

        if (epoch + 1) % 1 == 0:
            save_checkpoint(model1, (epoch), args.output_dir, filename=modelname,
                            best_acc=0,
                            optimizer=optimizer,
                            scheduler=scheduler)


#keeping track of things that need to be included in 'params'
#checkpoint path
#max_epochs
#batch size
#num_workers


def launch_training(chk_path:str ="",
                    max_epochs :int =100,
                    batch_size:int=8,
                    num_workers:int=1,
                    learning_rate:float=1e-4,
                    weight_decay:float=1e-5):
    
    return



if __name__ =="__main__":
    main()












