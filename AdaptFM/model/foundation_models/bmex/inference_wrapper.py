import argparse
import torch
from DUNet3D import DenseUNet3d
from bids import BIDSLayout
from utils.utils import reorient_to_std,fill_holes
import os
import json
import re
import shutil
import SimpleITK as sitk
import numpy as np
from monai.inferers import sliding_window_inference
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import sum as ndi_sum
from pathlib import Path


def _nifti_to_sidecar_path(nifti_path: str) -> str:
    # *.nii.gz  →  *.json
    base = os.path.splitext(os.path.splitext(nifti_path)[0])[0]
    return base + ".json"

def _to_bids_raw(path_on_disk: str, bids_root_dir: str) -> str:
    # make "bids:raw:/sub-xx/ses-xx/..." from absolute path under bids_root_dir
    import os
    rel = os.path.relpath(path_on_disk, start=bids_root_dir)
    return "bids:raw:" + rel.replace(os.sep, "/")

def write_sidecar_json(nifti_path: str, sources_paths: list, spatial_ref_path: str,
                       bids_root_dir: str, skull_stripped: bool, qi_value: float | None = None, type_field: str | None = None):
    """Write the requested fields into a sidecar JSON next to nifti_path; optionally attach QI."""
    sidecar = _nifti_to_sidecar_path(nifti_path)
    payload = {
        "Sources": [_to_bids_raw(p, bids_root_dir) for p in sources_paths],
        "SpatialReference": _to_bids_raw(spatial_ref_path, bids_root_dir),
        "SkullStripped": bool(skull_stripped),
    }
    if qi_value is not None:
        payload["QI"] = {
            "value": float(qi_value),
            "description": "Quality index [0, 1], where a higher value indicates better quality (0 being the worst)."
        }
    if type_field is not None:         
        payload["Type"] = type_field
        
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
def read_bids_filter(file_path):
    """Read and parse the BIDS filter file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def write_dataset_description_json(bids_out_root: str):
    """
    Write a BIDS-compliant dataset_description.json at the top-level of the output BIDS dir.
    Safe to call multiple times; it won't overwrite if exists.
    """
    dd_path = os.path.join(bids_out_root, "dataset_description.json")
    if os.path.exists(dd_path):
        return

    payload = {
        "Name": "BME-X Outputs",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "BME-X",
                "Version": "v1.0.5",
                "Container": {
                    "Type": "docker",
                    "Tag": "yuesun814/bme-x:v1.0.5"
                }
            }
        ]
    }

    os.makedirs(bids_out_root, exist_ok=True)
    with open(dd_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _load_qi_value(out_dir: str, stem: str):
    """
    Try reading {stem}-QI.txt in out_dir; return float or None if unavailable.
    """
    qi_txt = os.path.join(out_dir, f"{stem}-QI.txt")
    if os.path.isfile(qi_txt):
        try:
            with open(qi_txt, "r", encoding="utf-8") as f:
                import re
                text = f.read()
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
                return float(m.group(0)) if m else None
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument("--test_dir")
    parser.add_argument('--output_path')
    device = torch.device("cuda:0")

    args = parser.parse_args()
    #get model checkpoint
    checkpoint = args.checkpoint
    output_path = args.output_path
    
    model1 = DenseUNet3d()
    model1_dict = torch.load(checkpoint, map_location=('cpu'))
    model1.load_state_dict(model1_dict['state_dict'])

    model1.eval()
    model1.to(device)

    write_dataset_description_json(args.output_path) #see if we actually need this
    with torch.no_grad():

        bids_root_path = args.test_dir # users need to enter the root for the BID dataset

        layout = BIDSLayout(bids_root_path)
        t1w_files = layout.get(extension=['nii.gz', 'nii'], suffix=['T1w', 'T2w']) 

        #run inference loop using the same processing they specified

        for t1w_file in t1w_files:
            file_path = t1w_file.path
            
            file_name = os.path.basename(file_path)

            #subject_id 和 session_id
            subject_part = file_name.split("_")[0]
            session_part = file_name.split("_")[1]

            subject_id = subject_part.replace("sub-", "")
            session_id = session_part.replace("ses-", "")
            
            suffix = 'T1w' if ('T1w' in file_name or 'T1w' in file_path) else 'T2w'

            # Read and preprocess MRI image
            img_name = os.path.splitext(os.path.splitext(file_path)[0])[0]
            stem = os.path.basename(img_name)
            stem_no_suffix = re.sub(r'_(T1w|T2w)$', '', stem)


            # --- output dirs ---
            out_root = output_path
            out_dir = os.path.join(out_root, f"sub-{subject_id}", f"ses-{session_id}", "anat")
            os.makedirs(out_dir, exist_ok=True)
            dst_path = os.path.join(out_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dst_path)
            
            img_name = os.path.splitext(os.path.splitext(file_path)[0])[0]
            stem = os.path.basename(img_name)
            
            
            qi_val = _load_qi_value(out_dir, stem)

            write_sidecar_json(
                nifti_path=dst_path,
                sources_paths=[file_path],
                spatial_ref_path=file_path,
                bids_root_dir=bids_root_path,   
                skull_stripped=False,
                qi_value=qi_val
            )

            # reorient

            reoriented_base = os.path.join(out_dir, f"{stem}-reorient")

            T1w_img_reorient = reorient_to_std(file_path, reoriented_base)

            T1w_img_reorient = sitk.ReadImage(T1w_img_reorient)
            size, origin, spacing, direction = T1w_img_reorient.GetSize(), T1w_img_reorient.GetOrigin(), T1w_img_reorient.GetSpacing(), T1w_img_reorient.GetDirection()
            #print('Reorientation done')
            
            #rescale the image intensity to 0~1000
            arr = sitk.GetArrayFromImage(T1w_img_reorient).astype(np.float32)
            mask = np.isfinite(arr) & (arr != 0)
            vals = arr[mask] if np.any(mask) else arr[np.isfinite(arr)]
            
            if vals.size == 0:
                p1, p99 = 0.0, 1.0
            else:
                p1, p99 = np.nanpercentile(vals, [0.001, 99.999])
                if not np.isfinite(p1) or not np.isfinite(p99) or p1 >= p99:
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        p1, p99 = np.percentile(finite, [0.001, 99.999])
                    else:
                        p1, p99 = 0.0, 1.0
                    
            img_f = sitk.Cast(T1w_img_reorient, sitk.sitkFloat32)
            T1w_img_reorient = sitk.IntensityWindowing(img_f, p1, p99, 0.0, 1000.0)

            old_size = np.array(T1w_img_reorient.GetSize())  # [x, y, z]
            old_spacing = np.array(T1w_img_reorient.GetSpacing())  # [x, y, z]
            new_spacing = np.array([1.6, 1.6, 1.6])
            new_size = (old_size * old_spacing / new_spacing).astype(int)  

            old_origin = np.array(T1w_img_reorient.GetOrigin())
            old_center = old_origin + (old_size * old_spacing) / 2.0
            new_origin = old_center - (new_size * new_spacing) / 2.0

            resampler = sitk.ResampleImageFilter()
            resampler.SetSize([int(new_size[0]), int(new_size[1]), int(new_size[2])])
            resampler.SetOutputSpacing([1.6, 1.6, 1.6])
            resampler.SetOutputOrigin(new_origin.tolist())  # 这里要用 list
            resampler.SetOutputDirection(T1w_img_reorient.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)
            T1w_img_reorient_downsample = resampler.Execute(T1w_img_reorient)


            # Load and histogram-match the template
            print("Histogram matching")
            cwd = Path.cwd()
            bmex_root = (cwd.parent / "Brain_MRI_Enhancement")

            template = sitk.ReadImage(bmex_root / "Template" / "template.hdr")
            T1w_img_reorient_downsample = sitk.Cast(T1w_img_reorient_downsample, sitk.sitkFloat32)
            template = sitk.Cast(template, sitk.sitkFloat32)
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(1024)
            matcher.SetNumberOfMatchPoints(7)
            matcher.ThresholdAtMeanIntensityOn()
            T1w_img_reorient_downsample_hm = matcher.Execute(T1w_img_reorient_downsample, template)


            print("Skull stripping")
            T1w_img_reorient_downsample_hm = sitk.GetArrayFromImage(T1w_img_reorient_downsample_hm)
            T1w_img_reorient_downsample_hm = torch.tensor(T1w_img_reorient_downsample_hm).float()
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.unsqueeze(dim=0)
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.unsqueeze(dim=0)

            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm.to(device)

            # T1w_img = (T1w_img - torch.min(T1w_img)) / (torch.max(T1w_img) - torch.min(T1w_img))
            T1w_img_reorient_downsample_hm = T1w_img_reorient_downsample_hm / 10000.00


            logits = sliding_window_inference(T1w_img_reorient_downsample_hm, (64, 64, 64),
                                                1, model1,
                                                overlap=.85)
            promap = logits[:, :2, :, :, :]
            promap[:, 1, :, :, :] = logits[:, 2, :, :, :] + logits[:, 3, :, :, :]
            promap[:, 0, :, :, :] = 1 - promap[:, 1, :, :, :]

            print("Upsample to original space")
            promap_upsampled = F.interpolate(promap, size=[size[2], size[1], size[0]], mode='trilinear', align_corners=True)
            promap_upsampled = promap_upsampled.cpu().numpy()
            pre = np.argmax(promap_upsampled, axis=1).astype(np.uint8)

            # Apply binary opening
            pre_b = sitk.GetImageFromArray(pre[0, :, :, :])
            opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
            opening_filter.SetKernelRadius(3)
            pre_b_o = opening_filter.Execute(pre_b)

            # Extract largest connected component
            opened_image = sitk.GetArrayFromImage(pre_b_o)
            labeled_array, _ = ndimage_label(opened_image)
            sizes = ndi_sum(opened_image, labeled_array, range(labeled_array.max() + 1))
            mask = sizes < max(sizes)
            labeled_array[mask[labeled_array]] = 0
            labeled_array, _ = ndimage_label(labeled_array)

            # Fill hole
            #mask = sitk.GetArrayFromImage(labeled_array)
            labeled_array_mask = fill_holes(labeled_array, area_threshold=1)

            print("Save brainmask")
            s_path = os.path.join(out_dir, f"{stem}-reorient-brainmask.nii.gz")
            out = sitk.GetImageFromArray(labeled_array_mask)
            out.SetOrigin(origin)
            out.SetSpacing(spacing)
            out.SetDirection(direction)
            sitk.WriteImage(out, s_path)

    

if __name__=="__main__":
    main()