from AdaptFM.preprocessing.converters import (
    conv_to_uint8,
    nd2_to_tiff_converter,
    normalize_tiff_to_range,
    combine_labels_into_image_dir,
    batch_tiff_to_nii,
    batch_nii_to_tiff,
    copy_files_to_nnunet_inference
)

PREPROC_REGISTRY = {
    "Convert ND2 -> TIFF": nd2_to_tiff_converter,
    "Convert TIFF -> NIfTI (.nii.gz)": batch_tiff_to_nii,
    "Convert NIfTI (.nii.gz) -> TIFF": batch_nii_to_tiff,
    "Normalize TIFF to range [-1, 1]": normalize_tiff_to_range,
    "Scale and convert TIFF data dtype to uint8": conv_to_uint8,
    "Place image labels into image data folder": combine_labels_into_image_dir,
    "Send image data to nnUNet dataset folder": copy_files_to_nnunet_inference,
}
