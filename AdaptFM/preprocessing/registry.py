from AdaptFM.preprocessing.converters import *

PREPROC_REGISTRY = {
    "ND2 -> TIFF": nd2_to_tiff_converter,
    "Normalize TIFF to range [-1, 1]": normalize_tiff_to_range,
    "Apply OrganoidSeg to tiff directory": apply_organoidseg
}