import argparse
import json
from pathlib import Path

import SimpleITK as sitk
from monai.data import Dataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

# run transforms on data as recommended by merlin

ImageTransforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        ToTensord(keys=["image"]),
    ]
)


def run_merlin_transforms(input_path):
    data = [{"image": input_path}]
    ds = Dataset(data=data, transform=ImageTransforms)
    item = ds[0]

    img = item["image"]  # torch tensor, shape (1, 224, 224, 160)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder2transform")
    parser.add_argument("--params")
    args = parser.parse_args()

    folder2transform = Path(args.folder2transform)
    params = json.loads(args.params)

    setID = params["Set ID"]
    setName = params["Set Name"]

    original_name = folder2transform.name
    output_name = f"{original_name}_transformed"

    output_folder = folder2transform.parent / output_name

    output_folder.mkdir(exist_ok=True, parents=True)

    nii_files = sorted(folder2transform.glob("*.nii.gz"))
    for f in nii_files:
        print(f"Processing {f.name}")

        img_tensor = run_merlin_transforms(f)  # shape (1, 224, 224, 160)
        img_np = img_tensor.numpy()[0]  # remove channel dimension

        # Save as NIfTI
        out_path = output_folder / f.name

        # img_np is shape (224, 224, 160)
        sitk_img = sitk.GetImageFromArray(img_np.astype("float32"))
        sitk.WriteImage(sitk_img, str(out_path))


if __name__ == "__main__":
    main()
