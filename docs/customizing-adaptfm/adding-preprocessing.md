# Adding New Preprocessing Algorithms to AdaptFM

Users can add new preprocessing pipelines to AdaptFM. This follows the same registry based system as other modules. The steps to adding a new pipeline are:

1. Navigate to AdaptFM > preprocessing > converters.py
2. Write a function defining your conversion. It should take ```input_dir``` and ```output_dir``` parameters. These should be ```Path``` objects. Any other parameters you include will be visible in the preprocessing widget. Example:

```python
def nd2_to_tiff_converter(input_dir: Path, output_dir: Path, progress_callback=None): # <-- write function takingg input_dir and output_dir. 
    """
    Converts all .nd2 files in the input directory to single-channel .tiff files
    in the output directory (one file per channel). Uses channel names for filenames if available.
    """
    nd2file_paths = sorted(list(input_dir.glob("*.nd2")))
    total_files = len(nd2file_paths)

    if total_files == 0:
        raise ValueError(f"No .nd2 files found in {input_dir}")

        ...
```

3. Once you have written you function, add it to the registry under AdaptFM > preprocessing > registry.py

```python
from AdaptFM.preprocessing.converters import (
    conv_to_bmask,
    conv_to_uint8,
    nd2_to_tiff_converter, # <-- IMPORT HERE
    normalize_tiff_to_range,
)

PREPROC_REGISTRY = {
    "ND2 -> TIFF": nd2_to_tiff_converter, # < -- ADD TO REGISTRY. THIS IS HOW THE PIPELINE WILL APPEAR IN THE WIDGET
    "Normalize TIFF to range [-1, 1]": normalize_tiff_to_range,
    "Scale and convert tiff data dtype to uint8": conv_to_uint8,
    "Convert uint8 tiff data to binary mask (0|1)": conv_to_bmask,
}
