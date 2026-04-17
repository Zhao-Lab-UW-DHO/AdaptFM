import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
import json
from tqdm import tqdm
from AdaptFM.SSVT.Vit_class import ViTEncoder3D, MAE3DSegmentation,MAE3DLinearProbeDecoder
from AdaptFM.SSVT.utils import read_tiff,sliding_window_inference,OrganoidPatchDatasetGPU
from pathlib import Path
# -----------------------------
# Dice Loss
# -----------------------------
def dice_loss(pred, target, eps=1e-6):
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def as_int(x):
    return None if x is None else int(x)

def as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ("1", "true", "yes", "y")

# -----------------------------
# Load pretrained encoder + decoder
# -----------------------------

# create dummy function just to read parameters
def train_SSVT(ckpt_path:str = None,
               samples_per_volume =48,
               batch_size = 8,
               num_workers=16,
               lr = 1e-5,
               weight_decay = 1e-4,
               unfreeze_epoch=100,
               total_epochs=500):
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params')
    parser.add_argument("--train_raw_images")
    parser.add_argument("--train_mask_images")
    parser.add_argument("--val_raw_images")
    parser.add_argument("--val_mask_images")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    params = json.loads(args.params)


    ckpt_path = params['ckpt_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ViTEncoder3D(patch_size=(2,16,16), embed_dim=768, depth=8, num_heads=8)
    state_dict = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state_dict)

    encoder.to(device)  # move model to GPU
    encoder.eval()

    # Create dummy input on same device
    dummy_patch = torch.zeros(1, 1, *(4, 128, 128), device=device)
    with torch.no_grad():
        _, patch_grid = encoder(dummy_patch, return_grid=True)
    decoder = MAE3DLinearProbeDecoder(embed_dim=768, patch_grid=patch_grid)
    # Initially freeze encoder
    freeze_encoder = True
    model = MAE3DSegmentation(encoder, decoder, freeze_encoder=freeze_encoder).to(device)

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    image_paths = sorted(
        [os.path.join(args.train_raw_images, f) for f in os.listdir(args.train_raw_images)],
        key=lambda p: os.path.basename(p).replace('.tiff', '')
    )
    mask_paths = sorted(
        [os.path.join(args.train_mask_images, f) for f in os.listdir(args.train_mask_images)],
        key=lambda p: os.path.basename(p).replace('_seg.tiff', '')
    )

    dataset = OrganoidPatchDatasetGPU(
        image_paths=image_paths,
        mask_paths=mask_paths,
        patch_size=(4,128,128),
        samples_per_volume=as_int(params['samples_per_volume']),
        augment=True,
        min_fg_fraction=0.01
    )

    print(args.train_mask_images)
    print(args.train_mask_images)

    loader = DataLoader(
        dataset,
        batch_size=as_int(params['batch_size']),
        shuffle=True,
        num_workers=as_int(params['num_workers']),
        pin_memory=True,
        persistent_workers=True
    )

    # -----------------------------
    # Optimizer (initially decoder only)
    # -----------------------------
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=float(params['lr']), weight_decay=float(params['weight_decay']))

    # -----------------------------
    # Training Loop with conditional unfreeze
    # -----------------------------
    bce = torch.nn.BCEWithLogitsLoss()

    for epoch in range(0,as_int(params['unfreeze_epoch'])):
        model.train()
        running_loss, running_dice = 0.0, 0.0
        for batch in tqdm(loader):
            images = batch['image'].cuda(non_blocking=True)  
            masks = batch['mask'].cuda(non_blocking=True)            


            optimizer.zero_grad()
            logits = model(images)
            probs = torch.sigmoid(logits)

            loss_dice = dice_loss(probs, masks)
            loss_bce  = bce(logits, masks)
            loss = 0.5 * loss_dice + 0.5 * loss_bce

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += 1 - dice_loss(probs, masks)

        print(f"[Decoder warmup] Epoch {epoch+1} Loss: {running_loss/len(loader):.4f}, "
            f"Voxel Dice: {running_dice/len(loader):.4f}")


    # -----------------------------
    # Phase 2: Unfreeze encoder
    # -----------------------------
    print("Unfreezing encoder...")
    for param in model.encoder.parameters():
        param.requires_grad = True

    # Separate LRs for encoder vs decoder
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 3e-5},  # smaller LR
        {'params': model.decoder.parameters(), 'lr': 3e-4}   # keep decoder LR
    ], weight_decay=1e-4)

    # Fine-tuning loop
    for epoch in range(as_int(params['unfreeze_epoch']),as_int(params['total_epochs'])):
        model.train()
        running_loss, running_dice = 0.0, 0.0
        for batch in tqdm(loader):
            images = batch['image'].cuda(non_blocking=True)  
            masks = batch['mask'].cuda(non_blocking=True)      


            optimizer.zero_grad()
            logits = model(images)
            probs = torch.sigmoid(logits)

            loss_dice = dice_loss(probs, masks)
            loss_bce  = bce(logits, masks)
            loss = 0.5 * loss_dice + 0.5 * loss_bce

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += 1 - dice_loss(probs, masks)

        print(f"[Fine-tuning] Epoch {epoch+1} Loss: {running_loss/len(loader):.4f}, "
            f"Voxel Dice: {running_dice/len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_path,"final_model.pt"))

     # ---- Iterate over validation volumes ----


    val_image_paths = sorted(
        [os.path.join(args.val_raw_images, f) for f in os.listdir(args.val_raw_images)],
        key=lambda p: os.path.basename(p).replace('.tiff', '')
    )
    mask_image_paths = sorted(
        [os.path.join(args.val_mask_images, f) for f in os.listdir(args.val_mask_images)],
        key=lambda p: os.path.basename(p).replace('_seg.tiff', '')
    )
    
    total_dice = 0
    for image_path, mask_path in tqdm(zip(val_image_paths, mask_image_paths), desc="Testing", total=len(val_image_paths)):
        test_vol = read_tiff(image_path)
        test_mask = read_tiff(mask_path)
        
        seg_np, _ = sliding_window_inference(
            model,
            test_vol,
            patch_size=(4,128,128),
            stride=(16,32,32),
            device=device,
            threshold=0.5
        )

        dice_scores = dice_loss(seg_np,test_mask)
        total_dice += dice_scores


    avg_dice = total_dice / len(val_image_paths)
    print(f"Average Dice Loss: {avg_dice:.4f}")




if __name__=='__main__':
    main()


