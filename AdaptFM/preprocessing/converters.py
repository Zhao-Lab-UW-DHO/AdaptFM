from pathlib import Path

import nd2
import numpy as np
import tifffile
import shutil
import SimpleITK as sitk
from pathlib import Path
import json
import re

def nd2_to_tiff_converter(input_dir: Path, output_dir: Path, progress_callback=None):
    """
    Converts all .nd2 files in the input directory to single-channel .tiff files
    in the output directory (one file per channel). Uses channel names for filenames if available.
    """
    nd2file_paths = sorted(list(input_dir.glob("*.nd2")))
    total_files = len(nd2file_paths)

    if total_files == 0:
        raise ValueError(f"No .nd2 files found in {input_dir}")

    for i, nd2file_path in enumerate(nd2file_paths):
        with nd2.ND2File(nd2file_path) as m:
            xarr = m.to_xarray()

            dims_present = [d for d in ["Z", "Y", "X", "C"] if d in xarr.dims]
            if dims_present:
                xarr = xarr.transpose(*dims_present)

            if (
                "C" in xarr.dims
            ):  # the FMs expect either RGB or single channel inputs, so split nd2 images per channel
                num_channels = xarr.sizes["C"]

                channel_names = (
                    xarr.coords["C"].values if "C" in xarr.coords else []
                )  # get cnames

                for c in range(num_channels):
                    channel_data = xarr.isel(C=c).values

                    try:
                        c_name = str(channel_names[c])
                        if c_name == str(c):
                            c_name = f"C{c}"
                    except IndexError:
                        c_name = f"C{c}"

                    # sanitize names
                    safe_c_name = "".join(
                        x if x.isalnum() or x in "-_" else "_"
                        for x in c_name.replace(" ", "_")
                    )

                    out_name = f"{nd2file_path.stem}_{safe_c_name}.tiff"
                    tifffile.imwrite(output_dir / out_name, channel_data)
            else:
                data = xarr.values
                out_name = f"{nd2file_path.stem}.tiff"
                tifffile.imwrite(output_dir / out_name, data)

        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)


def normalize_tiff_to_range(
    input_dir: Path,
    output_dir: Path,
    progress_callback=None,
    range_min=-1.0,
    range_max=1.0,
    axis_integer=None,
):
    """
    normalizes tiff files to float16 with a specified range. axis=None averages on the whole volume
    Typically axis (0,1,2) maps to (Z/Depth,Y/Height,X/Width)
    """
    exts = {".tiff", ".tif"}
    tiff_filepaths = sorted(
        [x for x in input_dir.glob("*") if x.suffix.lower() in exts]
    )
    total_files = len(tiff_filepaths)

    if axis_integer == "":
        axis_integer = None

    if total_files == 0:
        raise ValueError(f"No tiff files found in {input_dir}")

    for i, tiff_filepath in enumerate(tiff_filepaths):
        array = tifffile.imread(tiff_filepath).astype(np.float16)

        arr_min = np.min(array, axis=axis_integer, keepdims=True)
        arr_max = np.max(array, axis=axis_integer, keepdims=True)

        # prevent divby0
        diff = arr_max - arr_min
        diff = np.where(diff == 0, 1.0, diff)

        # scale to 0-1
        norm = (array) / diff

        # multiply sets the range and addition applies the offset
        if (range_min, range_max) != (0.0, 1.0):
            norm = norm * (range_max - range_min) + range_min

        tifffile.imwrite(output_dir / tiff_filepath.name, norm)

        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)


def conv_to_uint8(input_dir: Path, output_dir: Path, progress_callback=None):
    """
    Converts TIFF arrays in input_dir to uint8 by dynamically rescaling float values
    from their actual min/max range to [0, 255] to avoid clamping or underflow.
    """
    exts = {".tiff", ".tif"}
    tiff_filepaths = sorted(
        [x for x in input_dir.glob("*") if x.suffix.lower() in exts]
    )
    total_files = len(tiff_filepaths)

    if total_files == 0:
        raise ValueError(f"No tiff files found in {input_dir}")

    for i, tiff_filepath in enumerate(tiff_filepaths):
        array = tifffile.imread(tiff_filepath)

        if np.issubdtype(array.dtype, np.floating):
            arr_min = np.min(array)
            arr_max = np.max(array)

            # Dynamic min-max scaling to [0, 255]
            if arr_max > arr_min:
                norm = (array - arr_min) / (arr_max - arr_min)
                uint8_array = (norm * 255.0).astype(np.uint8)
            else:
                uint8_array = np.zeros_like(array, dtype=np.uint8)
        else:
            # Safely clip and convert non-float arrays
            uint8_array = np.clip(array, 0, 255).astype(np.uint8)

        tifffile.imwrite(output_dir / tiff_filepath.name, uint8_array)

        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)


def conv_to_bmask(input_dir: Path, output_dir: Path, progress_callback=None):
    """
    Converts object ID or multi-class uint8 segmentation masks into a unified
    binary mask (0 and 1) stored as uint8. Expects input arrays to be uint8.
    """
    exts = {".tiff", ".tif"}
    tiff_filepaths = sorted(
        [x for x in input_dir.glob("*") if x.suffix.lower() in exts]
    )
    total_files = len(tiff_filepaths)

    if total_files == 0:
        raise ValueError(f"No tiff files found in {input_dir}")

    for i, tiff_filepath in enumerate(tiff_filepaths):
        array = tifffile.imread(tiff_filepath)

        # Enforce uint8 input contract
        if array.dtype != np.uint8:
            raise TypeError(
                f"Expected uint8 array for binary mask conversion, but got {array.dtype} in {tiff_filepath.name}. "
                "Run 'Convert dtype to uint8' first."
            )

        # Map any foreground label/ID (> 0) to 1
        bmask = (array > 0).astype(np.uint8)

        tifffile.imwrite(output_dir / tiff_filepath.name, bmask)

        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)

def combine_labels_into_image_dir(
    input_dir: Path,
    output_dir: Path,
    progress_callback=None,
):
    """
    Moves extension-agnostic label files to the data folder. 
    Ensures a matching data file exists by stem, prevents overwriting,
    and renames the label to filename + _seg + ext.
    """
    def get_files(directory_path): # ignore hidden and dirs
        return [x for x in directory_path.glob("*") if x.is_file() and not x.name.startswith('.')]
        

    label_filepaths = sorted( get_files(input_dir) )
    total_files = len(label_filepaths)

    if total_files == 0:
        raise ValueError(f"No label files found in {input_dir}")

    data_stems = set( get_files(input_dir) )

    for i, label_filepath in enumerate(label_filepaths):

        if label_filepath.stem in data_stems:

            new_filename = f"{label_filepath.stem}_seg{label_filepath.suffix}"
            dest_filepath = output_dir / new_filename

            # do not overwrite
            if dest_filepath.exists():
                raise FileExistsError(
                    f"Destination file {dest_filepath} already exists. "
                    "Aborting move to prevent data loss."
                )

            shutil.move(str(label_filepath), str(dest_filepath))
        else:
            print(f"Skipping '{label_filepath.name}': No matching data file found in output directory.")

        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)

def batch_nii_to_tiff(input_dir: Path, output_dir: Path, progress_callback=None):
    """
    Converts all NIfTI files (.nii or .nii.gz) in input_dir to TIFF files in output_dir.
    """

    nii_filepaths = sorted(
        [x for x in input_dir.glob("*") if x.name.lower().endswith((".nii", ".nii.gz"))]
    )
    total_files = len(nii_filepaths)

    if total_files == 0:
        raise ValueError(f"No NIfTI files found in {input_dir}")

    for i, nii_filepath in enumerate(nii_filepaths):

        sitk_img = sitk.ReadImage(str(nii_filepath))
        img_array = sitk.GetArrayFromImage(sitk_img)

        if nii_filepath.name.lower().endswith(".nii.gz"):
            new_name = nii_filepath.name[:-7] + ".tiff"
        else:
            new_name = nii_filepath.stem + ".tiff"

        # Write to TIFF
        tifffile.imwrite(output_dir / new_name, img_array)

        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)

# as in bmex prepare dataset
# img = tiff.imread(input_path)
# img = sitk.GetImageFromArray(img)
# sitk.WriteImage(img, nii_path)
def batch_tiff_to_nii(input_dir: Path, output_dir: Path, progress_callback=None):
    """
    Converts all TIFF files in input_dir to compressed NIfTI files (.nii.gz) in output_dir.
    """
    exts = {".tiff", ".tif"}
    tiff_filepaths = sorted(
        [x for x in input_dir.glob("*") if x.suffix.lower() in exts]
    )
    total_files = len(tiff_filepaths)

    if total_files == 0:
        raise ValueError(f"No tiff files found in {input_dir}")

    for i, tiff_filepath in enumerate(tiff_filepaths):
        # as in bmex prepare dataset
        img_array = tifffile.imread(tiff_filepath)
        sitk_img = sitk.GetImageFromArray(img_array)
        new_name = tiff_filepath.stem + ".nii.gz"
        sitk.WriteImage(sitk_img, str(output_dir / new_name))

        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)

def copy_files_to_nnunet_inference(input_dir: Path, output_dir: Path, progress_callback=None):
    exts = {".tiff", ".tif", ".nii.gz"}
    filepaths = sorted(
        [x for x in input_dir.glob("*") if ((x.suffix.lower() in exts) and (not x.name.split('.')[0].endswith("_seg"))) ]
    )

    assert output_dir.parent.is_dir() and output_dir.parent.name == "nnUNet_raw", "FAILED: you need to select the dataset folder in the nnUNet_raw directory as the output_dir"

    set_name = output_dir.name.split("_", 1)[1]

    # find the highest existing ID in imagesTr to continue the sequence
    images_tr_dir = output_dir / "imagesTr"
    max_id = -1

    if images_tr_dir.exists():
        for f_path in images_tr_dir.glob("*_0000.*"):
            # Match the number right before "_0000." in the filename 
            # e.g., in "BrainTumor_004_0000.tiff", it captures "004"
            match = re.search(r'_(\d+)_0000\.', f_path.name)
            if match:
                file_id = int(match.group(1))
                if file_id > max_id:
                    max_id = file_id

    # if imagesTr had files, start after the highest ID. Otherwise, default to starting at 0.
    start_id = max_id + 1 if max_id >= 0 else 0

    images_ts_dir = output_dir / "imagesTs"
    images_ts_dir.mkdir(parents=True, exist_ok=True)

    # loop through filepaths, copy, and track mapping
    file_mapping = {}

    for i, filepath in enumerate(filepaths):
        current_id = start_id + i
        
        new_filename = f"{set_name}_{current_id:03d}_0000.tiff"
        dest_path = images_ts_dir / new_filename
        
        shutil.copy2(filepath, dest_path)
        file_mapping[filepath.name] = new_filename

    # write the mapping dictionary to filemapping.json
    mapping_filepath = output_dir / "filemapping.json"
    with open(mapping_filepath, "w") as f:
        json.dump(file_mapping, f, indent=4)