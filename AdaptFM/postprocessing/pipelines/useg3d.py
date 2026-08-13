import argparse
import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D
from pathlib import Path
import tifffile as tiff
import numpy as np
import logging

def get_all_planes(input_dir: Path):
    """
    input_dir/
        XY_planes/
        YZ_planes/
        XZ_planes/
    Each contains TIFF stacks with identical basenames.
    """

    xy_dir = input_dir / "XY_planes"
    yz_dir = input_dir / "YZ_planes"
    xz_dir = input_dir / "XZ_planes"

    for d in [xy_dir, yz_dir, xz_dir]:
        if not d.exists() or not d.is_dir():
            logging.warning(f"Directory not found: {d}")
            return {}

    def get_tiff_files(directory: Path):
        return {
            f.stem: f 
            for f in directory.iterdir() 
            if f.is_file() 
            and f.suffix.lower() in ('.tif', '.tiff') 
            and not f.name.startswith('.')  # hidden files
        }

    xy_files = get_tiff_files(xy_dir)
    yz_files = get_tiff_files(yz_dir)
    xz_files = get_tiff_files(xz_dir)

    # Only process images present in all three folders
    common = set(xy_files) & set(yz_files) & set(xz_files)

    if len(common) == 0:
        raise RuntimeError("No files found.")
    print("Found", len(common), "files.")

    results = {}

    for name in sorted(common):
        xy_stack = tiff.imread(xy_files[name])
        yz_stack = tiff.imread(yz_files[name])
        xz_stack = tiff.imread(xz_files[name])

        results[name] = {
            "xy": np.asarray(xy_stack),
            "yz": np.asarray(yz_stack),
            "xz": np.asarray(xz_stack)
        }

    return results



def run_postprocessing(input_dir: Path, output_dir: Path):

    all_images = get_all_planes(input_dir)

    for image_name, planes in all_images.items():
        try:
            indirect_aggregation_params = uSegment3D_params.get_2D_to_3D_aggregation_params()
            indirect_aggregation_params['indirect_method']['dtform_method'] = 'edt'

            assert planes["xy"].ndim == 3, f"Error, 2D predictions must be 3D (a stack of planes), found dimensions: {planes["xy"].ndim}"
            assert planes["xz"].ndim == 3, f"Error, 2D predictions must be 3D (a stack of planes), found dimensions: {planes["xz"].ndim}"
            assert planes["yz"].ndim == 3, f"Error, 2D predictions must be 3D (a stack of planes), found dimensions: {planes["yz"].ndim}"

            segmentation3D, (probability3D, gradients3D) = (
                uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(
                    segmentations=[planes["xy"], planes["xz"], planes["yz"]],
                    img_xy_shape=planes["xy"].shape,
                    precomputed_binary=None,
                    params=indirect_aggregation_params,
                    savefolder=None,
                    basename=None
                )
            )

            out_path = output_dir / f"{image_name}.tif"
            tiff.imwrite(out_path, segmentation3D)

        except Exception as e:
            print(f"Unable to merge planes on image {image_name}: {e}")

    return


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument('--output_dir')


    args = parser.parse_args()

    run_postprocessing(Path(args.input_dir), Path(args.output_dir))