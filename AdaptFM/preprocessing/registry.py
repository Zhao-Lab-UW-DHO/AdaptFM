from AdaptFM.preprocessing.converters import (
    conv_to_bmask,
    conv_to_uint8,
    nd2_to_tiff_converter,
    normalize_tiff_to_range,
)

PREPROC_REGISTRY = {
    "ND2 -> TIFF": nd2_to_tiff_converter,
    "Normalize TIFF to range [-1, 1]": normalize_tiff_to_range,
    "Scale and convert tiff data dtype to uint8": conv_to_uint8,
    "Convert uint8 tiff data to binary mask (0|1)": conv_to_bmask,
}