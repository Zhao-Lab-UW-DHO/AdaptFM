# Dataset and Augmentation Pipeline for 3D ViT MAE Pretraining
# ------------------------------------------------------------
# Assumptions:
# - Volumes are stored as 3D TIFF files
# - Physical spacing is available in TIFF metadata
# - Single-channel intensity images (C=1)
# - All augmentations are applied BEFORE masking
#


from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates, zoom
from torch.utils.data import Dataset

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------


def read_tiff(path: Path):
    """
    Reads a 3D TIFF and extracts spacing from metadata.
    Returns:
        volume: np.ndarray [Z, Y, X]
        spacing: (sz, sy, sx)
    """

    volume = tifffile.imread(path)

    return volume.astype(np.float32)  # , (sz, sy, sx)


def normalize_patch(patch):
    """Per-patch z-score normalization (same as training)."""
    return (patch - patch.mean()) / (patch.std() + 1e-6)


def sliding_window_inference(
    model,
    volume,
    patch_size=(32, 64, 64),
    stride=(16, 32, 32),
    device="cuda",
    threshold=0.5,
):
    model.eval()

    Z, Y, X = volume.shape
    pz, py, px = patch_size

    logits_full = torch.zeros((1, 1, Z, Y, X), device=device)
    counts = torch.zeros_like(logits_full)

    with torch.no_grad():
        for z in range(0, Z, stride[0]):
            for y in range(0, Y, stride[1]):
                for x in range(0, X, stride[2]):
                    z1 = min(z + pz, Z)
                    y1 = min(y + py, Y)
                    x1 = min(x + px, X)

                    patch = volume[z:z1, y:y1, x:x1]

                    pad_z = pz - patch.shape[0]
                    pad_y = py - patch.shape[1]
                    pad_x = px - patch.shape[2]

                    patch = normalize_patch(patch)

                    if pad_z > 0 or pad_y > 0 or pad_x > 0:
                        patch = np.pad(
                            patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant"
                        )

                    patch_tensor = (
                        torch.from_numpy(patch)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )

                    logits_patch = model(patch_tensor)

                    dz = z1 - z
                    dy = y1 - y
                    dx = x1 - x

                    logits_patch = logits_patch[:, :, :dz, :dy, :dx]

                    logits_full[:, :, z:z1, y:y1, x:x1] += logits_patch
                    counts[:, :, z:z1, y:y1, x:x1] += 1

    logits_full /= counts.clamp_min(1)
    probs = torch.sigmoid(logits_full)
    seg = (probs > threshold).float()

    return (
        seg.squeeze().cpu().numpy().astype("uint8"),
        probs.squeeze().cpu().numpy(),
    )


# ------------------------------------------------------------
# Physical cropping
# ------------------------------------------------------------


def physical_random_crop(volume, crop_size_vox: tuple[int, int, int]):
    """Random crop by voxels (already computed to satisfy patch_size)."""
    vz, vy, vx = crop_size_vox
    Z, Y, X = volume.shape

    vz = min(vz, Z)
    vy = min(vy, Y)
    vx = min(vx, X)

    z0 = np.random.randint(0, Z - vz + 1) if Z > vz else 0
    y0 = np.random.randint(0, Y - vy + 1) if Y > vy else 0
    x0 = np.random.randint(0, X - vx + 1) if X > vx else 0

    return volume[z0 : z0 + vz, y0 : y0 + vy, x0 : x0 + vx]


# ------------------------------------------------------------
# Augmentations
# ------------------------------------------------------------


def anisotropic_scaling(volume, scale_range=(0.9, 1.1)):
    scales = np.random.uniform(*scale_range, size=3)
    return zoom(volume, scales, order=1)


def z_slice_dropout(volume, max_slices=3):
    vol = volume.copy()
    n = np.random.randint(0, max_slices + 1)
    if n > 0:
        idx = np.random.choice(vol.shape[0], n, replace=False)
        vol[idx] = 0
    return vol


def z_jitter(volume, max_shift=1):
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(volume, shift, axis=0)


def intensity_augment(volume):
    # intensity scaling
    scale = np.random.uniform(0.8, 1.2)
    volume = volume * scale

    # gamma
    gamma = np.random.uniform(0.7, 1.5)
    volume = np.power(np.clip(volume, 1e-6, None), gamma)

    # gaussian noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.01 * volume.std(), volume.shape)
        volume = volume + noise

    # poisson noise
    if np.random.rand() < 0.3:
        volume = np.random.poisson(np.clip(volume, 0, None)).astype(np.float32)

    return volume


def mild_elastic_deform(volume, alpha=2.0, sigma=8.0):
    """Low-frequency elastic deformation."""
    shape = volume.shape
    dz = np.random.randn(*shape) * alpha
    dy = np.random.randn(*shape) * alpha
    dx = np.random.randn(*shape) * alpha

    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )

    indices = (zz + dz, yy + dy, xx + dx)

    return map_coordinates(volume, indices, order=1, mode="reflect")


def physical_random_crop_pair(
    volume: np.ndarray,
    mask: np.ndarray,
    crop_size_vox: tuple[int, int, int],
):
    """
    Random voxel-aligned crop applied identically to volume and mask.
    """
    assert volume.shape == mask.shape

    vz, vy, vx = crop_size_vox
    Z, Y, X = volume.shape

    vz = min(vz, Z)
    vy = min(vy, Y)
    vx = min(vx, X)

    z0 = np.random.randint(0, Z - vz + 1) if Z > vz else 0
    y0 = np.random.randint(0, Y - vy + 1) if Y > vy else 0
    x0 = np.random.randint(0, X - vx + 1) if X > vx else 0

    vol_crop = volume[z0 : z0 + vz, y0 : y0 + vy, x0 : x0 + vx]
    mask_crop = mask[z0 : z0 + vz, y0 : y0 + vy, x0 : x0 + vx]

    return vol_crop, mask_crop


import numpy as np


def mild_elastic_deform_pair(
    volume: np.ndarray,
    mask: np.ndarray,
    alpha=2.0,
    sigma=8.0,
):
    """
    Low-frequency elastic deformation applied identically to volume and mask.
    Image uses linear interpolation; mask uses nearest neighbor.
    """
    assert volume.shape == mask.shape
    shape = volume.shape

    # shared random displacement fields
    dz = np.random.randn(*shape) * alpha
    dy = np.random.randn(*shape) * alpha
    dx = np.random.randn(*shape) * alpha

    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )

    indices = (zz + dz, yy + dy, xx + dx)

    vol_def = map_coordinates(
        volume,
        indices,
        order=1,  # linear
        mode="reflect",
    )

    mask_def = map_coordinates(
        mask,
        indices,
        order=0,  # nearest neighbor
        mode="reflect",
    )

    return vol_def, mask_def


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------


class OrganoidMAEDataset(Dataset):
    """
    Returns fixed-size 3D patches for ViT-MAE pretraining.
    """

    def __init__(
        self,
        paths: list[Path],
        patch_size=(32, 64, 64),  # voxel size
        augment=True,
        patches_per_image=5,
    ):

        self.paths = paths
        self.patch_size = patch_size
        self.augment = augment
        self.patches_per_image = patches_per_image

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx % len(self.paths)
        vol = read_tiff(self.paths[img_idx])

        # Random crop
        patch = physical_random_crop(vol, self.patch_size)

        # Augmentations
        if self.augment:
            patch = anisotropic_scaling(patch)
            patch = mild_elastic_deform(patch)
            patch = z_jitter(patch)
            # patch = z_slice_dropout(patch)
            patch = intensity_augment(patch)

        # Pad to make divisible by ViT patch size
        patch = center_crop_or_pad(patch, self.patch_size)

        # Normalize
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)

        # To tensor
        patch = torch.from_numpy(patch).unsqueeze(0).float()

        return {"image": patch}


# ------------------------------------------------------------
# NOTES
# ------------------------------------------------------------
# - This dataset returns ONLY what MAE needs
# - Easy to extend with:
#   - multiple augmented views (contrastive learning)
#   - additional reconstruction targets
#   - channel-wise volumes
# ------------------------------------------------------------


def center_crop_or_pad(patch: np.ndarray, target_shape):
    """
    Center-crop or zero-pad a 3D array to exactly target_shape.

    Parameters
    ----------
    patch : np.ndarray
        Input array of shape (Z, Y, X)
    target_shape : tuple
        Desired output shape (Zt, Yt, Xt)

    Returns
    -------
    np.ndarray
        Array of shape target_shape
    """
    assert patch.ndim == 3, f"Expected 3D array, got {patch.ndim}D"
    z, y, x = patch.shape
    tz, ty, tx = target_shape

    # ---------- Crop ----------
    z_start = max((z - tz) // 2, 0)
    y_start = max((y - ty) // 2, 0)
    x_start = max((x - tx) // 2, 0)

    patch = patch[
        z_start : z_start + min(z, tz),
        y_start : y_start + min(y, ty),
        x_start : x_start + min(x, tx),
    ]

    # ---------- Pad ----------
    z, y, x = patch.shape

    pad_z = max(tz - z, 0)
    pad_y = max(ty - y, 0)
    pad_x = max(tx - x, 0)

    pad_before = (
        pad_z // 2,
        pad_y // 2,
        pad_x // 2,
    )
    pad_after = (
        pad_z - pad_before[0],
        pad_y - pad_before[1],
        pad_x - pad_before[2],
    )

    patch = np.pad(
        patch,
        (
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2]),
        ),
        mode="constant",
        constant_values=0,
    )

    assert patch.shape == target_shape, (
        f"center_crop_or_pad failed: got {patch.shape}, expected {target_shape}"
    )

    return patch


from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class OrganoidSegmentationDataset(Dataset):
    """
    Patch-based 3D segmentation dataset for ViT fine-tuning.
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
        patch_size=(32, 64, 64),
        augment=True,
        patches_per_image=20,
    ):
        assert len(image_paths) == len(mask_paths)

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.augment = augment
        self.patches_per_image = patches_per_image

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)

        image = read_tiff(self.image_paths[img_idx])
        mask = read_tiff(self.mask_paths[img_idx])

        # ---- joint random crop ----
        image, mask = physical_random_crop_pair(image, mask, self.patch_size)

        # ---- augmentations ----
        if self.augment:
            image, mask = mild_elastic_deform_pair(image, mask)
            image, mask = anisotropic_scaling_pair(image, mask)
            image, mask = z_jitter_pair(image, mask)
            image = intensity_augment(image)

        # ---- center crop / pad ----
        image = center_crop_or_pad(image, self.patch_size)
        mask = center_crop_or_pad(mask, self.patch_size)

        # ---- normalize image only ----
        image = (image - image.mean()) / (image.std() + 1e-6)

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {"image": image, "mask": mask}


from pathlib import Path

from torch.utils.data import DataLoader


def build_dataloaders(
    data_root,
    batch_size=4,
    patch_size=(32, 64, 64),
    patches_per_image=20,
    num_workers=8,
):
    data_root = Path(data_root)

    image_paths = sorted((data_root / "images").glob("*.tiff"))
    mask_paths = sorted((data_root / "masks").glob("*.tiff"))

    assert len(image_paths) > 0, "No images found"
    assert len(image_paths) == len(mask_paths), "Image/mask count mismatch"

    dataset = OrganoidSegmentationDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        patch_size=patch_size,
        augment=True,
        patches_per_image=patches_per_image,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader


import numpy as np


def anisotropic_scaling_pair(image, mask, scale_range=(0.9, 1.1)):
    """
    Apply the same random anisotropic scaling to image and mask.
    image: np.ndarray [Z, Y, X]
    mask:  np.ndarray [Z, Y, X]
    """
    assert image.shape == mask.shape
    Z, Y, X = image.shape

    sz = np.random.uniform(*scale_range)
    sy = np.random.uniform(*scale_range)
    sx = np.random.uniform(*scale_range)

    # Scale
    image_scaled = zoom(
        image,
        zoom=(sz, sy, sx),
        order=1,  # linear
        mode="reflect",
    )

    mask_scaled = zoom(
        mask,
        zoom=(sz, sy, sx),
        order=0,  # nearest
        mode="reflect",
    )

    # Center crop or pad back to original shape
    image_scaled = center_crop_or_pad(image_scaled, (Z, Y, X))
    mask_scaled = center_crop_or_pad(mask_scaled, (Z, Y, X))

    return image_scaled, mask_scaled


def z_jitter_pair(image, mask, max_shift=2):
    """
    Randomly shift image and mask along Z axis.
    image: np.ndarray [Z, Y, X]
    mask:  np.ndarray [Z, Y, X]
    """
    assert image.shape == mask.shape
    Z = image.shape[0]

    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return image, mask

    def shift_z(vol, shift):
        out = np.zeros_like(vol)
        if shift > 0:
            out[shift:] = vol[:-shift]
        else:
            out[:shift] = vol[-shift:]
        return out

    image_shifted = shift_z(image, shift)
    mask_shifted = shift_z(mask, shift)

    return image_shifted, mask_shifted


from pathlib import Path

from torch.utils.data import Dataset


class OrganoidSegDataset(Dataset):
    """
    Returns full 3D images and voxel-level masks for segmentation.
    Reuses augmentations from pretraining.
    """

    def __init__(self, image_paths: list[Path], mask_paths: list[Path], augment=True):
        assert len(image_paths) == len(mask_paths), "Image and mask lists must match."
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_tiff(self.image_paths[idx])  # (D,H,W)
        mask = read_tiff(self.mask_paths[idx])  # (D,H,W)

        # Apply augmentations (same transform to image + mask)
        if self.augment:
            img, mask = self.apply_augmentations(img, mask)

        # Pad / crop to desired size if needed
        img = center_crop_or_pad(img, target_shape=img.shape)
        mask = center_crop_or_pad(mask, target_shape=mask.shape)

        # Normalize image
        img = (img - img.mean()) / (img.std() + 1e-6)

        # To tensor
        img = torch.from_numpy(img).unsqueeze(0).float()  # (1,D,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1,D,H,W)

        return {"image": img, "mask": mask}

    def apply_augmentations(self, img, mask):
        """
        Apply same augmentation to image and mask
        """
        # Anisotropic scaling
        img, mask = anisotropic_scaling_pair(img, mask)
        # Mild elastic deform
        img, mask = mild_elastic_deform_pair(img, mask)
        # Z-jitter
        img, mask = z_jitter_pair(img, mask)
        # Optional: z slice dropout
        # img, mask = z_slice_dropout(img, mask)
        # Intensity augment (only on image)
        img = intensity_augment(img)
        return img, mask


import numpy as np


def sliding_window_inference_3d(
    image,  # 3D torch tensor (1,D,H,W) or (B=1,1,D,H,W)
    model,  # your MAE3DSegmentation model
    patch_size=(4, 128, 128),  # patch size for the sliding window
    stride=None,  # stride of sliding window (None -> same as patch_size)
    device="cuda",
):
    """
    Perform sliding window inference on a 3D volume.
    Returns full-size voxel predictions.
    """
    model.eval()
    image = image.to(device)
    if len(image.shape) == 4:
        image = image.unsqueeze(0)  # add batch dim

    _, _, D, H, W = image.shape
    ps_d, ps_h, ps_w = patch_size
    stride = stride or patch_size

    st_d, st_h, st_w = stride

    # Output tensor
    output = torch.zeros((1, 1, D, H, W), device=device)
    count_map = torch.zeros((1, 1, D, H, W), device=device)  # for averaging overlaps

    # Compute sliding windows
    d_starts = list(range(0, max(D - ps_d + 1, 1), st_d))
    h_starts = list(range(0, max(H - ps_h + 1, 1), st_h))
    w_starts = list(range(0, max(W - ps_w + 1, 1), st_w))

    # Ensure last patch reaches the end
    if d_starts[-1] + ps_d < D:
        d_starts.append(D - ps_d)
    if h_starts[-1] + ps_h < H:
        h_starts.append(H - ps_h)
    if w_starts[-1] + ps_w < W:
        w_starts.append(W - ps_w)

    with torch.no_grad():
        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    d1 = d0 + ps_d
                    h1 = h0 + ps_h
                    w1 = w0 + ps_w

                    patch = image[:, :, d0:d1, h0:h1, w0:w1]

                    # Forward pass
                    logits_patch = model(patch)  # (B=1,1,ps_d,ps_h,ps_w)

                    # Resize to patch size if necessary
                    if logits_patch.shape[2:] != (ps_d, ps_h, ps_w):
                        logits_patch = F.interpolate(
                            logits_patch,
                            size=(ps_d, ps_h, ps_w),
                            mode="trilinear",
                            align_corners=False,
                        )

                    # Add patch prediction to output
                    output[:, :, d0:d1, h0:h1, w0:w1] += logits_patch
                    count_map[:, :, d0:d1, h0:h1, w0:w1] += 1

    # Average overlapping regions
    output = output / count_map
    return output


import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class OrganoidPatchDataset(Dataset):
    """
    Random patch sampling dataset for 3D voxel segmentation.
    """

    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        patch_size: tuple[int, int, int],
        samples_per_volume: int = 16,
        augment: bool = True,
        min_fg_fraction: float = 0.01,
    ):
        """
        patch_size: (D, H, W)
        samples_per_volume: number of patches drawn per volume per epoch
        min_fg_fraction: minimum foreground fraction to accept a patch
                         (helps avoid empty patches)
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.augment = augment
        self.min_fg_fraction = min_fg_fraction

        # dataset length = volumes × patches per volume
        self.total_samples = len(self.image_paths) * samples_per_volume

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which volume to sample from
        volume_idx = idx // self.samples_per_volume

        img = read_tiff(self.image_paths[volume_idx])  # (D,H,W)
        mask = read_tiff(self.mask_paths[volume_idx])  # (D,H,W)

        if self.augment:
            img, mask = self.apply_augmentations(img, mask)

        D, H, W = img.shape
        ps_d, ps_h, ps_w = self.patch_size

        # Random patch sampling with foreground constraint
        for _ in range(10):  # try 10 times to get FG patch
            d0 = random.randint(0, max(D - ps_d, 0))
            h0 = random.randint(0, max(H - ps_h, 0))
            w0 = random.randint(0, max(W - ps_w, 0))

            patch_img = img[d0 : d0 + ps_d, h0 : h0 + ps_h, w0 : w0 + ps_w]
            patch_mask = mask[d0 : d0 + ps_d, h0 : h0 + ps_h, w0 : w0 + ps_w]

            fg_fraction = patch_mask.mean()

            if fg_fraction >= self.min_fg_fraction:
                break  # accept patch

        # Normalize
        patch_img = (patch_img - patch_img.mean()) / (patch_img.std() + 1e-6)

        patch_img = torch.from_numpy(patch_img).unsqueeze(0).float()
        patch_mask = torch.from_numpy(patch_mask).unsqueeze(0).float()

        return {"image": patch_img, "mask": patch_mask}

    def apply_augmentations(self, img, mask):
        img, mask = anisotropic_scaling_pair(img, mask)
        img, mask = mild_elastic_deform_pair(img, mask)
        img, mask = z_jitter_pair(img, mask)
        img = intensity_augment(img)
        return img, mask


class OrganoidPatchDatasetGPU(Dataset):
    """
    Random patch sampling dataset for 3D voxel segmentation.
    """

    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        patch_size: tuple[int, int, int],
        samples_per_volume: int = 16,
        augment: bool = True,
        min_fg_fraction: float = 0.01,
    ):
        """
        patch_size: (D, H, W)
        samples_per_volume: number of patches drawn per volume per epoch
        min_fg_fraction: minimum foreground fraction to accept a patch
                         (helps avoid empty patches)
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.augment = augment
        self.min_fg_fraction = min_fg_fraction

        # dataset length = volumes × patches per volume
        self.total_samples = len(self.image_paths) * samples_per_volume

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which volume to sample from
        volume_idx = idx // self.samples_per_volume

        img = read_tiff(self.image_paths[volume_idx])  # (D,H,W)
        mask = read_tiff(self.mask_paths[volume_idx])  # (D,H,W)

        D, H, W = img.shape
        ps_d, ps_h, ps_w = self.patch_size

        # Random patch sampling with foreground constraint
        for _ in range(10):  # try 10 times to get FG patch
            d0 = random.randint(0, max(D - ps_d, 0))
            h0 = random.randint(0, max(H - ps_h, 0))
            w0 = random.randint(0, max(W - ps_w, 0))

            patch_img = img[d0 : d0 + ps_d, h0 : h0 + ps_h, w0 : w0 + ps_w]
            patch_mask = mask[d0 : d0 + ps_d, h0 : h0 + ps_h, w0 : w0 + ps_w]

            fg_fraction = patch_mask.mean()

            if fg_fraction >= self.min_fg_fraction:
                break  # accept patch

        # Normalize
        patch_img = (patch_img - patch_img.mean()) / (patch_img.std() + 1e-6)

        patch_img = torch.from_numpy(patch_img).unsqueeze(0).float()
        patch_mask = torch.from_numpy(patch_mask).unsqueeze(0).float()

        return {"image": patch_img, "mask": patch_mask}


from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tifffile as tiff


class OrganoidPatchDatasetGPUTCell(Dataset):
    """
    Random patch sampling dataset for 3D voxel segmentation.
    """

    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        patch_size: tuple[int, int, int],
        samples_per_volume: int = 16,
        augment: bool = True,
        min_fg_fraction: float = 0.01,
        num_preload_workers=8,
    ):
        """
        patch_size: (D, H, W)
        samples_per_volume: number of patches drawn per volume per epoch
        min_fg_fraction: minimum foreground fraction to accept a patch
                         (helps avoid empty patches)
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.augment = augment
        self.min_fg_fraction = min_fg_fraction
        self.fg_sampling_prob = 1.0

        # dataset length = volumes × patches per volume
        self.total_samples = len(self.image_paths) * samples_per_volume

        self.volumes = [None] * len(image_paths)
        self.mask_volumes = [None] * len(mask_paths)
        self.fg_voxels = [None] * len(mask_paths)

        def load_tiff(p: Path):
            vol = tiff.imread(p)
            return vol

        with ThreadPoolExecutor(max_workers=num_preload_workers) as ex:
            futures = {ex.submit(load_tiff, p): i for i, p in enumerate(image_paths)}

            for fut in as_completed(futures):
                i = futures[fut]
                self.volumes[i] = fut.result()

        with ThreadPoolExecutor(max_workers=num_preload_workers) as ex:
            futures = {ex.submit(load_tiff, p): i for i, p in enumerate(mask_paths)}

            for fut in as_completed(futures):
                i = futures[fut]
                mask = fut.result()

                self.mask_volumes[i] = mask
                self.fg_voxels[i] = np.argwhere(mask > 0)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):

        volume_idx = idx // self.samples_per_volume
        vol = self.volumes[volume_idx]
        mask = self.mask_volumes[volume_idx]

        D, H, W = vol.shape
        ps_d, ps_h, ps_w = self.patch_size

        if (
            random.random() < self.fg_sampling_prob
            and len(self.fg_voxels[volume_idx]) > 0
        ):
            # foreground-centered patch
            z, y, x = self.fg_voxels[volume_idx][
                np.random.randint(len(self.fg_voxels[volume_idx]))
            ]

            d0 = np.clip(z - ps_d // 2, 0, D - ps_d)
            h0 = np.clip(y - ps_h // 2, 0, H - ps_h)
            w0 = np.clip(x - ps_w // 2, 0, W - ps_w)

        else:
            # random patch
            d0 = random.randint(0, max(D - ps_d, 0))
            h0 = random.randint(0, max(H - ps_h, 0))
            w0 = random.randint(0, max(W - ps_w, 0))

        patch_img = vol[d0 : d0 + ps_d, h0 : h0 + ps_h, w0 : w0 + ps_w]
        patch_mask = mask[d0 : d0 + ps_d, h0 : h0 + ps_h, w0 : w0 + ps_w]

        patch_img = (patch_img - patch_img.mean()) / (patch_img.std() + 1e-6)

        patch_img = torch.from_numpy(patch_img).unsqueeze(0).float()
        patch_mask = torch.from_numpy(patch_mask).unsqueeze(0).float()

        return {"image": patch_img, "mask": patch_mask}
