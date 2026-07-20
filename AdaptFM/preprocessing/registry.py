from AdaptFM.preprocessing.converters import nd2_to_tiff_converter

PREPROC_REGISTRY = {
    "ND2 -> TIFF": nd2_to_tiff_converter,
}