import argparse
import logging
from pathlib import Path

import numpy as np
import segment3D.parameters as uSegment3D_params
import segment3D.usegment3d as uSegment3D
import tifffile as tiff


def get_all_planes(input_dir: Path):
    """
    input_dir/
        XY_planes/
        YZ_planes/
        XZ_planes/
    Each contains TIFF stacks with identical basenames. Missing directories are skipped,
    and their positions in the output will be populated with empty lists [].
    """
    plane_dirs = {
        "xy": input_dir / "XY_planes",
        "yz": input_dir / "YZ_planes",
        "xz": input_dir / "XZ_planes",
    }

    active_dirs = {}
    missing_planes = []

    for plane_name, directory in plane_dirs.items():
        if directory.exists() and directory.is_dir():
            logging.info(
                f"[PLANE FOUND] Located directory for '{plane_name.upper()}_planes': {directory}"
            )
            active_dirs[plane_name] = directory
        else:
            missing_planes.append(plane_name.upper())
            logging.warning(
                f"[PLANE MISSING WARNING] Directory '{directory}' DOES NOT EXIST or is not a directory. "
                f"Plane perspective '{plane_name.upper()}' will be treated as missing and replaced with an empty list []."
            )

    if not active_dirs:
        raise RuntimeError(
            f"Fatal Error: No valid plane directories (XY_planes, YZ_planes, XZ_planes) "
            f"were found inside {input_dir}."
        )

    if missing_planes:
        logging.warning(
            f"[REDUCED DIMENSION RUN] Operating with {len(active_dirs)}/3 planes. "
            f"Missing plane perspective(s): {', '.join(missing_planes)}. "
            f"If this was unintentional, please check directory naming inside '{input_dir}'."
        )
    else:
        logging.info(
            "[FULL 3D RUN] All 3 plane directories (XY, YZ, XZ) successfully detected."
        )

    def get_tiff_files(directory: Path):
        return {
            f.stem: f
            for f in directory.iterdir()
            if f.is_file()
            and f.suffix.lower() in (".tif", ".tiff")
            and not f.name.startswith(".")  # hidden files
        }

    files_by_plane = {
        plane: get_tiff_files(d) for plane, d in active_dirs.items()
    }

    # Intersect basenames across directories that actually exist
    common_sets = [set(files.keys()) for files in files_by_plane.values()]
    common = set.intersection(*common_sets) if common_sets else set()

    if len(common) == 0:
        dir_summary = "\n".join(
            f"  - {p.upper()}_planes ({active_dirs[p]}): {len(files)} TIFF file(s) found"
            for p, files in files_by_plane.items()
        )
        raise RuntimeError(
            "Plane Mismatch Error: Active plane directories were found, but they share NO matching TIFF file stems.\n"
            f"Directory breakdown:\n{dir_summary}\n"
            "Please ensure corresponding TIFF files across active folders share identical filenames."
         )
    print(
        f"Found {len(common)} matching files across active plane directories ({', '.join(active_dirs.keys())})."
    )

    results = {}

    for name in sorted(common):
        results[name] = {
            "xy": (
                np.asarray(tiff.imread(files_by_plane["xy"][name]))
                if "xy" in files_by_plane
                else []
            ),
            "yz": (
                np.asarray(tiff.imread(files_by_plane["yz"][name]))
                if "yz" in files_by_plane
                else []
            ),
            "xz": (
                np.asarray(tiff.imread(files_by_plane["xz"][name]))
                if "xz" in files_by_plane
                else []
            ),
        }

    return results


def run_postprocessing(input_dir: Path, output_dir: Path):

    all_images = get_all_planes(input_dir)

    for image_name, planes in all_images.items():
        try:
            indirect_aggregation_params = (
                uSegment3D_params.get_2D_to_3D_aggregation_params()
            )
            indirect_aggregation_params["indirect_method"]["dtform_method"] = (
                "edt"
            )

            # Validate dimensions only for present planes
            for plane_key in ["xy", "xz", "yz"]:
                plane_data = planes[plane_key]
                if isinstance(plane_data, np.ndarray) and plane_data.size > 0:
                    assert plane_data.ndim == 3, (
                        f"Error in '{image_name}' [{plane_key.upper()} plane]: "
                        f"2D predictions must be 3D (a stack of planes), found dimensions: {plane_data.ndim}"
                    )

            # Determine target 3D XY shape from any present plane stack
            if isinstance(planes["xy"], np.ndarray) and planes["xy"].size > 0:
                img_xy_shape = planes["xy"].shape
            elif isinstance(planes["xz"], np.ndarray) and planes["xz"].size > 0:
                s = planes["xz"].shape
                img_xy_shape = (s[1], s[0], s[2])  # Invert XZ transpose mapping
            elif isinstance(planes["yz"], np.ndarray) and planes["yz"].size > 0:
                s = planes["yz"].shape
                img_xy_shape = (s[2], s[0], s[1])  # Invert YZ transpose mapping
            else:
                raise ValueError(
                    f"No valid image arrays found for image '{image_name}' across any plane."
                )

            segmentation3D, (probability3D, gradients3D) = (
                uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(
                    segmentations=[planes["xy"], planes["xz"], planes["yz"]],
                    img_xy_shape=img_xy_shape,
                    precomputed_binary=None,
                    params=indirect_aggregation_params,
                    savefolder=None,
                    basename=None,
                )
            )

            out_path = output_dir / f"{image_name}.tiff"
            tiff.imwrite(out_path, segmentation3D)

        except Exception as e:
            print(f"Unable to merge planes on image {image_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    run_postprocessing(Path(args.input_dir), Path(args.output_dir))
