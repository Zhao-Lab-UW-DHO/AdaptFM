from AdaptFM.preprocessing.converters import (
    conv_to_bmask,
    conv_to_uint8,
    nd2_to_tiff_converter,
    normalize_tiff_to_range,
    combine_labels_into_image_dir,
    batch_tiff_to_nii,
    batch_nii_to_tiff,
    copy_files_to_nnunet_inference
)

PREPROC_REGISTRY = {
    "ND2 -> TIFF": nd2_to_tiff_converter,
    "Normalize TIFF to range [-1, 1]": normalize_tiff_to_range,
    "Scale and convert tiff data dtype to uint8": conv_to_uint8,
    "Convert uint8 tiff data to binary mask (0|1)": conv_to_bmask,
    "Place image labels into image data folder": combine_labels_into_image_dir,
    "Convert tiff to .nii.gz": batch_tiff_to_nii,
    "Convert .nii.gz to tiff": batch_nii_to_tiff,
    "Send image data to nnUNet dataset folder": copy_files_to_nnunet_inference,
}
