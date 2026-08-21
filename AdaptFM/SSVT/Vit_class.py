# 3D ViT Masked Autoencoder (MAE) – Pretraining Skeleton
# ----------------------------------------------------


import math

import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: (gz, gy, gx) in number of patches
    Returns: [N, embed_dim]
    """
    gz, gy, gx = grid_size

    z = torch.arange(gz)
    y = torch.arange(gy)
    x = torch.arange(gx)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    coords = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3).float()

    assert embed_dim % 6 == 0, "embed_dim must be divisible by 6"

    dim_each = embed_dim // 3
    div_term = torch.exp(torch.arange(0, dim_each, 2) * (-math.log(10000.0) / dim_each))

    pe = []
    for i in range(3):
        pos = coords[:, i].unsqueeze(1)
        pe.append(torch.sin(pos * div_term))
        pe.append(torch.cos(pos * div_term))

    return torch.cat(pe, dim=1)


# ----------------------------------------------------
# Patch Embedding
# ----------------------------------------------------


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size: tuple[int, int, int], embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 1, Z, Y, X]
        x = self.proj(x)
        B, C, Z, Y, X = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x, (Z, Y, X)


# ----------------------------------------------------
# Transformer Blocks
# ----------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------------------------------
# ViT Encoder
# ----------------------------------------------------

from torch import nn


class ViTEncoder3D(nn.Module):
    def __init__(self, patch_size=(4, 16, 16), embed_dim=132, depth=4, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(patch_size, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # initialize dummy parameter with 1 token
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_grid=False):
        x, grid_size = self.patch_embed(x)  # [B, N, C]
        B, N, C = x.shape

        # interpolate for forward pass only, do NOT overwrite self.pos_embed
        if self.pos_embed.shape[1] != N:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if return_grid:
            return x, grid_size
        return x


# ----------------------------------------------------
# MAE Decoder
# ----------------------------------------------------


class MAEDecoder3D(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        decoder_dim=512,
        depth=4,
        num_heads=8,
        patch_volume=4 * 16 * 16,
    ):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_volume)

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)


# ----------------------------------------------------
# Masking
# ----------------------------------------------------
def cuboid_mask(grid_size, mask_ratio):
    """
    Contiguous 3D cuboid masking in patch space.

    grid_size: (gz, gy, gx)
    returns: mask [N] where True = masked
    """
    gz, gy, gx = grid_size
    N = gz * gy * gx
    num_mask = int(mask_ratio * N)

    mask = torch.zeros((gz, gy, gx), dtype=torch.bool)

    masked = 0
    while masked < num_mask:
        # cuboid size (biased small–medium)
        cz = torch.randint(1, max(2, gz // 2), (1,)).item()
        cy = torch.randint(1, max(2, gy // 2), (1,)).item()
        cx = torch.randint(1, max(2, gx // 2), (1,)).item()

        z0 = torch.randint(0, gz - cz + 1, (1,)).item()
        y0 = torch.randint(0, gy - cy + 1, (1,)).item()
        x0 = torch.randint(0, gx - cx + 1, (1,)).item()

        region = mask[z0 : z0 + cz, y0 : y0 + cy, x0 : x0 + cx]
        newly_masked = (~region).sum().item()
        region[:] = True
        masked += newly_masked

    return mask.flatten()


# ----------------------------------------------------
# Full MAE Model
# ----------------------------------------------------


class MaskedAutoencoder3D(nn.Module):
    def __init__(self, encoder: ViTEncoder3D, decoder: MAEDecoder3D):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask_ratio=0.75):
        B = x.size(0)

        # Get the patch grid for masking
        _, grid = self.encoder.patch_embed(x)

        # Encode input
        tokens, _ = self.encoder(x, return_grid=True)
        N = tokens.size(1)

        # Generate cuboid mask
        mask = cuboid_mask(grid, mask_ratio).to(x.device)

        # Separate visible and masked tokens
        visible = tokens[:, ~mask]

        # Prepare mask tokens
        mask_tokens = self.decoder.mask_token.repeat(B, mask.sum(), 1)
        mask_tokens = torch.zeros(
            B, mask.sum(), tokens.size(-1), device=x.device, dtype=tokens.dtype
        )

        # Combine visible and mask tokens into full token sequence
        full_tokens = torch.zeros(
            B,
            N,
            tokens.size(-1),  # still encoder embed_dim
            device=x.device,
            dtype=tokens.dtype,
        )
        full_tokens[:, ~mask] = visible
        full_tokens[:, mask] = mask_tokens

        # Pass through decoder (decoder will handle projection internally)
        preds = self.decoder(full_tokens)

        return preds, mask, grid

    # ----------------------------------------------------
    # Losses
    # ----------------------------------------------------

    def reconstruction_loss(self, pred, target):
        return F.l1_loss(pred, target)

    def multiscale_reconstruction_loss(
        self,
        preds,
        target,
        grid,
        patch_size,
    ):
        """
        preds: [B, N, patch_volume]
        target: [B, 1, Z, Y, X]
        """
        B = preds.size(0)
        pz, py, px = patch_size
        gz, gy, gx = grid

        # patches → volume
        recon = preds.view(B, gz, gy, gx, pz, py, px)
        recon = recon.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        recon = recon.view(B, 1, gz * pz, gy * py, gx * px)

        # full resolution
        loss_full = F.l1_loss(recon, target)

        # 2× downsample
        loss_2x = F.l1_loss(F.avg_pool3d(recon, 2), F.avg_pool3d(target, 2))

        # 4× downsample
        loss_4x = F.l1_loss(F.avg_pool3d(recon, 4), F.avg_pool3d(target, 4))

        return loss_full + 0.5 * loss_2x + 0.25 * loss_4x


# ----------------------------------------------------
# NOTE:
# - Dataset, augmentation pipeline, and training loop intentionally omitted
# - This file defines ONLY model + masking abstractions
# - Easy to extend with contrastive heads or extra reconstruction targets
# ----------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.float().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.float(), alpha=1 - self.decay
                )

    @torch.no_grad()
    def apply_shadow(self, model):
        """Copy EMA weights into the model, backing up original weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        """Restore original model weights after applying EMA."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


import torch
from torch import nn


# -------------------------
# Segmentation model
# -------------------------
class ViTSegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder

        self.head = nn.Sequential(
            nn.Conv3d(encoder.embed_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        tokens, grid_size = self.encoder(x, return_grid=True)
        feats = tokens_to_volume(tokens, grid_size)
        logits = self.head(feats)
        return logits


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3, 4))
        den = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4)) + self.eps
        return 1 - (num / den).mean()


def tokens_to_volume(tokens, grid_size):
    """
    tokens: [B, N, C]
    grid_size: (Z', Y', X')
    returns: [B, C, Z', Y', X']
    """
    B, N, C = tokens.shape
    Z, Y, X = grid_size
    assert N == Z * Y * X, "Token count does not match grid size"

    return tokens.transpose(1, 2).reshape(B, C, Z, Y, X)


import torch
from torch import nn


class MAE3DUNetDecoder(nn.Module):
    """
    ViT patch-based 3D decoder for voxel-level segmentation.
    Automatically upsamples to match input patch size.
    """

    def __init__(self, embed_dim, patch_grid):
        super().__init__()
        self.Dp, self.Hp, self.Wp = patch_grid

        # Contracting path (refine low-res patch features)
        self.conv1 = nn.Conv3d(embed_dim, 256, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.act2 = nn.GELU()

        # Upsampling path (restores resolution)
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.act3 = nn.GELU()
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.act4 = nn.GELU()

        # Final voxel-level classifier
        self.final = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, tokens, target_patch_size=None):
        """
        tokens: (B, N, C) ViT token embeddings
        target_patch_size: (D, H, W) output voxel resolution for this patch
        """
        B, N, C = tokens.shape

        # reshape tokens into 3D grid
        x = tokens.transpose(1, 2).reshape(B, C, self.Dp, self.Hp, self.Wp)

        # Contracting
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))

        # Upsampling
        x = self.act3(self.up1(x))
        x = self.act4(self.up2(x))

        # Final voxel-level logits
        logits = self.final(x)

        # Interpolate to match the input patch size if necessary
        if target_patch_size is not None and logits.shape[2:] != target_patch_size:
            logits = F.interpolate(
                logits, size=target_patch_size, mode="trilinear", align_corners=False
            )

        return logits


def dice_loss(pred, target, eps=1e-6):
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class FeatureAdapter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.InstanceNorm3d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class ZAdapter(nn.Module):
    def __init__(self, embed_dim, k=3):
        super().__init__()
        self.z_conv = nn.Conv3d(
            embed_dim,
            embed_dim,
            kernel_size=(k, 1, 1),
            padding=(k // 2, 0, 0),
            groups=embed_dim,  # depthwise
        )
        self.pointwise = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.InstanceNorm3d(embed_dim)

    def forward(self, x):
        x = self.z_conv(x)  # depthwise mixing along Z
        x = self.pointwise(x)  # channel mixing
        return self.norm(x)


import torch
from torch import nn


class MAE3DSegmentation(nn.Module):
    """
    Combines ViT 3D encoder with MAE 3D U-Net decoder.
    Automatically handles patch reshaping and voxel-aligned decoding.
    """

    def __init__(self, encoder, decoder, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x, target_patch_size=None):
        if target_patch_size is None:
            target_patch_size = x.shape[2:]  # (D,H,W)

        tokens, _ = self.encoder(x, return_grid=True)
        logits = self.decoder(tokens, target_patch_size=target_patch_size)
        return logits


class MAE3DSegmentationZAdapt(nn.Module):
    def __init__(self, encoder, decoder, freeze_encoder=False, use_adapter=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Add adapter
        self.adapter = ZAdapter(encoder.embed_dim) if use_adapter else nn.Identity()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x, target_patch_size=None):
        if target_patch_size is None:
            target_patch_size = x.shape[2:]  # (D,H,W)

        # tokens: [B, N, C], grid_size: (Dp, Hp, Wp)
        tokens, grid_size = self.encoder(x, return_grid=True)
        B, N, C = tokens.shape
        Dp, Hp, Wp = grid_size

        # reshape tokens → voxel grid
        feats = tokens.transpose(1, 2).reshape(B, C, Dp, Hp, Wp)

        # apply adapter in voxel space
        feats = self.adapter(feats)

        # flatten back to tokens
        tokens = feats.reshape(B, C, -1).transpose(1, 2)

        # decode to full-resolution logits
        logits = self.decoder(tokens, target_patch_size=target_patch_size)
        return logits


import torch
from torch import nn


class MAE3DUNetDecoderBig(nn.Module):
    """
    Beefed-up ViT patch-based 3D decoder for voxel-level segmentation.
    Still respects SSL pretraining contribution: decoder < encoder params.
    """

    def __init__(self, embed_dim, patch_grid):
        super().__init__()
        self.Dp, self.Hp, self.Wp = patch_grid

        # ---------------------------
        # Contracting path (refine low-res patch features)
        # ---------------------------
        self.conv1 = nn.Conv3d(embed_dim, 512, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.act3 = nn.GELU()

        # ---------------------------
        # Upsampling path (restore resolution)
        # ---------------------------
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.act4 = nn.GELU()
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.act5 = nn.GELU()
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.act6 = nn.GELU()

        # ---------------------------
        # Final voxel-level classifier
        # ---------------------------
        self.final = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, tokens, target_patch_size=None):
        """
        tokens: (B, N, C) ViT token embeddings
        target_patch_size: (D, H, W) output voxel resolution for this patch
        """
        B, N, C = tokens.shape

        # reshape tokens into 3D grid
        x = tokens.transpose(1, 2).reshape(B, C, self.Dp, self.Hp, self.Wp)

        # ---------------------------
        # Contracting path
        # ---------------------------
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))

        # ---------------------------
        # Upsampling path
        # ---------------------------
        x = self.act4(self.up1(x))
        x = self.act5(self.up2(x))
        x = self.act6(self.up3(x))

        # ---------------------------
        # Final logits
        # ---------------------------
        logits = self.final(x)

        # Interpolate to match input patch size if necessary
        if target_patch_size is not None and logits.shape[2:] != target_patch_size:
            logits = F.interpolate(
                logits, size=target_patch_size, mode="trilinear", align_corners=False
            )

        return logits


import torch
from torch import nn


class MAE3DLinearProbeDecoder(nn.Module):
    """
    Minimal linear probe for 3D ViT MAE features.

    Purpose:
    Evaluate how much segmentation information is already
    present in the pretrained representation.

    Structure:
    tokens -> reshape -> 1x1x1 conv -> upsample to voxel space
    """

    def __init__(self, embed_dim, patch_grid):
        super().__init__()

        self.Dp, self.Hp, self.Wp = patch_grid

        # Linear projection (token features -> segmentation logit)
        self.proj = nn.Conv3d(embed_dim, 1, kernel_size=1)

    def forward(self, tokens, target_patch_size=None):
        """
        tokens: (B, N, C) ViT token embeddings
        target_patch_size: (D, H, W) voxel resolution
        """

        B, N, C = tokens.shape

        # reshape tokens -> 3D patch grid
        x = tokens.transpose(1, 2).reshape(B, C, self.Dp, self.Hp, self.Wp)

        # linear projection
        logits = self.proj(x)

        # upsample to full voxel resolution
        if target_patch_size is not None and logits.shape[2:] != target_patch_size:
            logits = F.interpolate(
                logits, size=target_patch_size, mode="trilinear", align_corners=False
            )

        return logits
