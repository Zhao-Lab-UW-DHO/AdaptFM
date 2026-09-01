import argparse
import tifffile
from pathlib import Path
import numpy as np
from tqdm import tqdm

def run_postprocessing(input_dir: Path, output_dir: Path):
    
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

    for tiff_filepath in tqdm(tiff_filepaths,desc="Converting Masks"):
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
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    run_postprocessing(Path(args.input_dir), Path(args.output_dir))

