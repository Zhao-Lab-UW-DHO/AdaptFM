import nd2
import tifffile
from pathlib import Path

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