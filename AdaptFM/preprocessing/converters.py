import nd2
import tifffile
from pathlib import Path

from AdaptFM.segmentation.registry import SEGMENTATION_REGISTRY
import numpy as np

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
            
            if "C" in xarr.dims: # the FMs expect either RGB or single channel inputs, so split nd2 images per channel
                num_channels = xarr.sizes["C"]
                
                channel_names = xarr.coords["C"].values if "C" in xarr.coords else [] # get cnames
                
                for c in range(num_channels):
                    channel_data = xarr.isel(C=c).values
                    
                    try:
                        c_name = str(channel_names[c])
                        if c_name == str(c):
                            c_name = f"C{c}"
                    except IndexError:
                        c_name = f"C{c}"
                    
                    # sanitize names
                    safe_c_name = "".join(x if x.isalnum() or x in "-_" else "_" for x in c_name.replace(" ", "_"))
                    
                    out_name = f"{nd2file_path.stem}_{safe_c_name}.tiff"
                    tifffile.imwrite(output_dir / out_name, channel_data)
            else:
                data = xarr.values
                out_name = f"{nd2file_path.stem}.tiff"
                tifffile.imwrite(output_dir / out_name, data)
        
        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)


def normalize_tiff_to_range(input_dir: Path, output_dir: Path, progress_callback=None, range_min=-1., range_max=1., axis_integer=None):
    """
    normalizes tiff files to float16 with a specified range. axis=None averages on the whole volume
    Typically axis (0,1,2) maps to (Z/Depth,Y/Height,X/Width)
    """
    exts = {".tiff", ".tif"}
    tiff_filepaths = sorted([x for x in input_dir.glob("*") if x.suffix.lower() in exts])
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


def apply_organoidseg(input_dir: Path, output_dir: Path, progress_callback, minimum_size=0.0, sigma=0.0):
    """
    Applies OrganoidSeg segmentation to TIFF files in input_dir and saves uint8 masks to output_dir
    """
    exts = {".tiff", ".tif"}
    tiff_filepaths = sorted([x for x in input_dir.glob("*") if x.suffix.lower() in exts])
    total_files = len(tiff_filepaths)
    
    if total_files == 0:
        raise ValueError(f"No tiff files found in {input_dir}")

    for i, tiff_filepath in enumerate(tiff_filepaths):
        array = tifffile.imread(tiff_filepath)
            
        uint8_array = SEGMENTATION_REGISTRY.get('OrganoidSeg').run(array, {"minimum size":minimum_size, "sigma":sigma})

        tifffile.imwrite(output_dir / tiff_filepath.name, uint8_array)
        
        if progress_callback:
            percent_complete = int(((i + 1) / total_files) * 100)
            progress_callback(percent_complete)