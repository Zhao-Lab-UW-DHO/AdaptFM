"""
SAM-Med3D Inference Script — Their code is not a package. Thus need this custom wrapper
Going forward we will only support repos that can be installed as a package
===========================================================
"""

import copy
import os
import os.path as osp
from pathlib import Path
import re
import argparse
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from torchio.data.io import sitk_to_nib
import medim

# =============================================================================
# CLICK METHOD — identical to training
# =============================================================================

def get_next_click3D_torch_2(prev_seg, gt_semantic_seg):
    """
    prev_seg       : (B, 1, D, H, W)  raw logits
    gt_semantic_seg: (B, 1, D, H, W)  integer labels
    """
    mask_threshold = 0.5
    batch_points, batch_labels = [], []

    pred_masks = prev_seg > mask_threshold
    true_masks = gt_semantic_seg > 0
    fn_masks   = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks   = torch.logical_and(torch.logical_not(true_masks), pred_masks)
    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.argwhere(to_point_mask[i])
        if len(points) == 0:
            points = torch.argwhere(true_masks[i])
        if len(points) == 0:
            D, H, W = gt_semantic_seg.shape[-3:]
            batch_points.append(torch.tensor([[[D//2, H//2, W//2]]]))
            batch_labels.append(torch.tensor([[1]]))
            continue

        point = points[np.random.randint(len(points))]
        is_positive = bool(fn_masks[i, 0, point[1], point[2], point[3]])
        batch_points.append(point[1:].clone().detach().reshape(1, 1, 3))
        batch_labels.append(torch.tensor([[int(is_positive)]]))

    return batch_points, batch_labels


# =============================================================================
# CORE INFERENCE — single 128^3 patch
# =============================================================================

def sam_model_infer(model, roi_image, roi_gt=None,
                    prompt_generator=get_next_click3D_torch_2,
                    num_clicks=5):
    """
    roi_image : (1, 1, D, H, W) float tensor, ZNorm-ed
    roi_gt    : (1, 1, D, H, W) int tensor
    Returns   : np.ndarray uint8 (D, H, W)
    """
    model.eval()
    device = next(model.parameters()).device

    if roi_gt is not None and (roi_gt == 0).all():
        return np.zeros(roi_image.shape[-3:], dtype=np.uint8), None

    with torch.no_grad():
        input_tensor = roi_image.to(device).float()
        image_embeddings = model.image_encoder(input_tensor)

        img_shape  = input_tensor.shape[-3:]
        low_res_sz = tuple(s // 4 for s in img_shape)

        prev_masks    = torch.zeros(1, 1, *img_shape,   device=device)
        low_res_masks = torch.zeros(1, 1, *low_res_sz,  device=device)

        for click_idx in range(num_clicks):
            if roi_gt is not None:
                new_co, new_la = prompt_generator(prev_masks.cpu(), roi_gt.cpu())
                if isinstance(new_co, list): new_co = torch.cat(new_co, dim=0)
                if isinstance(new_la, list): new_la = torch.cat(new_la, dim=0)
                new_co = new_co.to(device)
                new_la = new_la.to(device)
            else:
                D, H, W = img_shape
                new_co = torch.tensor([[[D//2, H//2, W//2]]], device=device, dtype=torch.float)
                new_la = torch.tensor([[1]], device=device, dtype=torch.long)
                num_clicks = 1

            # Last click: pass points=None (matches training's random_insert logic)
            points_arg = None if click_idx == num_clicks - 1 else [new_co, new_la]

            sparse_emb, dense_emb = model.prompt_encoder(
                points=points_arg, boxes=None, masks=low_res_masks)

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

            prev_masks = F.interpolate(
                low_res_masks, size=img_shape, mode='trilinear', align_corners=False)

        final_hr = F.interpolate(
            low_res_masks, size=img_shape, mode='trilinear', align_corners=False)

    prob = torch.sigmoid(final_hr).cpu().numpy().squeeze()
    mask = (prob > 0.5).astype(np.uint8)
    return mask, low_res_masks.detach()


# =============================================================================
# I/O HELPERS
# =============================================================================

def read_arr_from_nifti(nii_path, get_meta_info=False):
    sitk_image = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(sitk_image)  # ZYX
    if not get_meta_info:
        return arr
    return arr, {
        "sitk_image_object":  sitk_image,
        "sitk_origin":        sitk_image.GetOrigin(),
        "sitk_direction":     sitk_image.GetDirection(),
        "sitk_spacing":       sitk_image.GetSpacing(),
        "original_numpy_shape": arr.shape,
    }


def get_subject_and_meta_info(img_path, gt_path):
    """Load image and label via sitk_to_nib — matches training data loader exactly."""
    sitk_image = sitk.ReadImage(img_path)
    sitk_label = sitk.ReadImage(gt_path)

    if sitk_image.GetOrigin() != sitk_label.GetOrigin():
        sitk_image.SetOrigin(sitk_label.GetOrigin())
    if sitk_image.GetDirection() != sitk_label.GetDirection():
        sitk_image.SetDirection(sitk_label.GetDirection())

    img_arr, _ = sitk_to_nib(sitk_image)
    lbl_arr, _ = sitk_to_nib(sitk_label)

    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.from_numpy(np.array(img_arr))),
        label=tio.LabelMap(tensor=torch.from_numpy(np.array(lbl_arr))),
    )
    _, meta_info = read_arr_from_nifti(img_path, get_meta_info=True)
    return subject, meta_info


def get_category_list_and_zero_mask(gt_path):
    arr, meta = read_arr_from_nifti(gt_path, get_meta_info=True)
    fg = [int(v) for v in np.unique(arr) if v != 0]
    return fg, np.zeros(meta["original_numpy_shape"], dtype=np.uint8)


def save_numpy_to_nifti(in_arr, out_path, meta_info_for_saving):
    """Save ZYX numpy array with original sitk metadata."""
    out_img = sitk.GetImageFromArray(in_arr)
    sitk_obj = meta_info_for_saving.get("sitk_image_object")
    if sitk_obj:
        out_img.SetOrigin(sitk_obj.GetOrigin())
        out_img.SetDirection(sitk_obj.GetDirection())
        out_img.SetSpacing(sitk_obj.GetSpacing())
    sitk.WriteImage(out_img, out_path)


# =============================================================================
# PREPROCESSING — matches training exactly
# =============================================================================

def data_preprocess(subject, meta_info, category_index, crop_size=128):
    """
    Replicates training pipeline:
      ToCanonical → CropOrPad(mask_name='label') → ZNormalization(x>0)
    Returns roi_image (1,1,D,H,W), roi_label (1,1,D,H,W), meta_info with affines.
    """
    # Binary label for this category
    lbl = subject.label.data.clone()
    new_lbl = torch.zeros_like(lbl)
    new_lbl[lbl == category_index] = 1
    subject.label.set_data(new_lbl)

    meta_info["original_subject_affine"]        = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape

    # 1. Canonicalize
    subject_canonical = tio.ToCanonical()(subject)

    # 2. CropOrPad centered on label centroid
    crop_transform = tio.CropOrPad(mask_name='label',
                                   target_shape=(crop_size, crop_size, crop_size))
    padding_params, cropping_params = crop_transform._compute_center_crop_or_pad(subject_canonical)
    subject_cropped = crop_transform(subject_canonical)

    meta_info["padding_params_functional"]  = padding_params
    meta_info["cropping_params_functional"] = cropping_params
    meta_info["canonical_subject_shape"]    = subject_canonical.spatial_shape
    meta_info["canonical_subject_affine"]   = subject_canonical.image.affine.copy()
    meta_info["roi_subject_affine"]         = subject_cropped.image.affine.copy()  # KEY affine

    # 3. Extract + normalize
    img3D = subject_cropped.image.data.clone().detach()  # (1, D, H, W) float64
    gt3D  = subject_cropped.label.data.clone().detach()

    norm = tio.ZNormalization(masking_method=lambda x: x > 0)
    img3D = norm(img3D).unsqueeze(0)  # (1, 1, D, H, W)
    gt3D  = gt3D.unsqueeze(0)

    return img3D, gt3D, meta_info


# =============================================================================
# POSTPROCESSING — maps canonical prediction back to original space
# =============================================================================

def data_postprocess(roi_pred_numpy, meta_info):
    """
    roi_pred_numpy: (X, Y, Z) uint8 in tio canonical space
    Returns       : (Z, Y, X) uint8 in original sitk space
    """
    pred_tensor = torch.from_numpy(roi_pred_numpy.astype(np.float32)).unsqueeze(0)

    pred_map = tio.LabelMap(tensor=pred_tensor, affine=meta_info["roi_subject_affine"])

    ref_shape = (1, *meta_info["original_subject_spatial_shape"])
    ref_image = tio.ScalarImage(
        tensor=torch.zeros(ref_shape),
        affine=meta_info["original_subject_affine"],
    )

    resampled = tio.Resample(target=ref_image, image_interpolation='nearest')(pred_map)
    out = resampled.data.squeeze(0).cpu().numpy().astype(np.uint8)
    return out.transpose(2, 1, 0)  # XYZ → ZYX for sitk


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================

def infer_full_volume(model, subject, meta_info, category_index,
                      num_clicks=5, crop_size=128):
    """
    Runs inference over the full canonical volume using a sliding window along Z.
    Normalization stats are computed from the centroid crop (matches training).
    Postprocessing uses affines from data_preprocess for correct spatial mapping.

    Returns: ZYX uint8 numpy array in original sitk space.
    """
    # --- Get correct affines via data_preprocess on centroid crop ---
    _subj = copy.deepcopy(subject)
    _meta = copy.deepcopy(meta_info)
    _, _, affine_meta = data_preprocess(_subj, _meta,
                                        category_index=category_index,
                                        crop_size=crop_size)
    # affine_meta["roi_subject_affine"]         = cropped subject affine (correct for mapping back)
    # affine_meta["original_subject_affine"]    = original tio subject affine (correct for ref space)
    # affine_meta["original_subject_spatial_shape"] = original spatial shape

    # --- Build canonical subject for sliding window ---
    subj_sw = copy.deepcopy(subject)
    lbl = subj_sw.label.data.clone()
    new_lbl = torch.zeros_like(lbl)
    new_lbl[lbl == category_index] = 1
    subj_sw.label.set_data(new_lbl)

    subj_canonical = tio.ToCanonical()(subj_sw)
    img_full = subj_canonical.image.data  # (1, X, Y, Z)
    lbl_full = subj_canonical.label.data  # (1, X, Y, Z)
    _, X, Y, Z = img_full.shape

    print(f"  Canonical shape: ({X},{Y},{Z}) | GT voxels: {(lbl_full>0).sum().item()}")

    # --- Compute norm stats from centroid crop (matches training normalization) ---
    all_coords = torch.argwhere(lbl_full[0] > 0)
    gcx = int(all_coords[:, 0].float().mean().item())
    gcy = int(all_coords[:, 1].float().mean().item())
    gcz = int(all_coords[:, 2].float().mean().item())

    def clip(v, lo, hi): return max(lo, min(hi, v))
    cgx0 = clip(gcx - crop_size//2, 0, X - crop_size)
    cgx1 = cgx0 + crop_size
    cgy0 = clip(gcy - crop_size//2, 0, Y - crop_size)
    cgy1 = cgy0 + crop_size
    cgz0 = clip(gcz - crop_size//2, 0, max(0, Z - crop_size))
    cgz1 = min(cgz0 + crop_size, Z)

    centroid_patch = img_full[:, cgx0:cgx1, cgy0:cgy1, cgz0:cgz1].clone()
    if centroid_patch.shape[-1] < crop_size:
        centroid_patch = F.pad(centroid_patch, (0, crop_size - centroid_patch.shape[-1]))

    cmask = centroid_patch > 0
    norm_mean = centroid_patch[cmask].mean()
    norm_std  = centroid_patch[cmask].std()
    print(f"  Norm stats — mean: {norm_mean:.4f}  std: {norm_std:.4f}")

    # --- Sliding window along Z ---
    stride = crop_size // 2  # 50% overlap
    if Z <= crop_size:
        z_starts = [0]
    else:
        z_starts = list(range(0, Z - crop_size, stride))
        if not z_starts or z_starts[-1] + crop_size < Z:
            z_starts.append(max(0, Z - crop_size))

    print(f"  Sliding window: {len(z_starts)} z-patches, stride={stride}")

    pred_acc  = np.zeros((X, Y, Z), dtype=np.float32)
    count_acc = np.zeros((X, Y, Z), dtype=np.float32)

    for z0 in z_starts:
        z1 = min(z0 + crop_size, Z)
        z0 = max(0, z1 - crop_size) if Z >= crop_size else 0
        actual_z = z1 - z0
        z_pad    = crop_size - actual_z

        lbl_z = lbl_full[0, :, :, z0:z1]
        if lbl_z.sum() == 0:
            continue

        # XY centroid for this z-slab
        lc = torch.argwhere(lbl_z > 0)
        cx = int(lc[:, 0].float().mean().item())
        cy = int(lc[:, 1].float().mean().item())

        x0 = clip(cx - crop_size//2, 0, X - crop_size); x1 = x0 + crop_size
        y0 = clip(cy - crop_size//2, 0, Y - crop_size); y1 = y0 + crop_size

        img_patch = img_full[:, x0:x1, y0:y1, z0:z1].clone()
        lbl_patch = lbl_full[:, x0:x1, y0:y1, z0:z1].clone()

        if lbl_patch.sum() == 0:
            continue

        if z_pad > 0:
            img_patch = F.pad(img_patch, (0, z_pad))
            lbl_patch = F.pad(lbl_patch, (0, z_pad))

        # Normalize with centroid stats — consistent across all patches
        img_normed = (img_patch - norm_mean) / (norm_std + 1e-8)
        img_normed = img_normed.unsqueeze(0).float()  # (1,1,128,128,128)
        lbl_5d     = lbl_patch.unsqueeze(0)           # (1,1,128,128,128)

        pred_np, _ = sam_model_infer(model, img_normed, roi_gt=lbl_5d, num_clicks=num_clicks)

        if z_pad > 0:
            pred_np = pred_np[:, :, :actual_z]

        pred_acc[x0:x1,  y0:y1,  z0:z1] += pred_np.astype(np.float32)
        count_acc[x0:x1, y0:y1, z0:z1]  += 1

    count_acc = np.maximum(count_acc, 1)
    roi_pred_numpy = (pred_acc / count_acc > 0.5).astype(np.uint8)
    print(f"  Predicted voxels: {roi_pred_numpy.sum()}")

    # --- Postprocess: canonical → original space ---
    # Use roi_subject_affine from data_preprocess (centroid crop affine) but
    # override with full canonical affine since prediction covers full volume
    affine_meta["roi_subject_affine"] = subj_canonical.image.affine.copy()

    return data_postprocess(roi_pred_numpy, affine_meta)


# =============================================================================
# TOP-LEVEL ENTRY POINT
# =============================================================================

def validate_paired_img_gt(model, img_path, gt_path, output_path,
                            num_clicks=5, crop_size=128, seed=233):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    exist_categories, final_pred = get_category_list_and_zero_mask(gt_path)
    _, gt_meta = read_arr_from_nifti(gt_path, get_meta_info=True)
    subject, meta_info = get_subject_and_meta_info(img_path, gt_path)

    for category_index in exist_categories:
        print(f"Category {category_index} | {Path(img_path).name}")

        cls_pred = infer_full_volume(
            model,
            copy.deepcopy(subject),
            copy.deepcopy(meta_info),
            category_index=category_index,
            num_clicks=num_clicks,
            crop_size=crop_size,
        )
        final_pred[cls_pred == 1] = category_index

    save_numpy_to_nifti(final_pred, output_path, gt_meta)
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument("--test_dir")
    parser.add_argument('--output_path')
    args = parser.parse_args()

    test_dir = args.test_dir

    raw_images_path = str(Path(test_dir) / 'imagesTr')
    labels_path = str(Path(test_dir) / 'labelsTr')

    output_path = args.output_path
    checkpoint = args.checkpoint

    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=checkpoint)
    images = [f.name for f in Path(raw_images_path).iterdir() 
              if f.is_file() and f.name.endswith('.nii.gz')]




    for image in images:
        img_path = str(Path(raw_images_path) / image)
        gt_path = str(Path(labels_path) / image)
        out_path = str(Path(output_path) / image)
        


        if not Path(gt_path).exists():
            print(f"GT not found for {image}, skipping.")
            continue

        validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=5)

if __name__ == "__main__":
    main()
