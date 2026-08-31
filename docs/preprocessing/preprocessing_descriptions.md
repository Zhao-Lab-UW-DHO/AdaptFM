# Convert ND2 -> TIFF

This utility converts Nikon `.nd2` microscopy image files into standard `.tiff` format. Because many deep learning models and processing tools expect either RGB or single-channel inputs, this tool automatically splits multichannel `.nd2` images into individual, single-channel TIFF files.

## How It Works

The conversion processes your raw microscopy files using the following steps:

1. **Scan the input directory** - The tool identifies all `.nd2` files in your designated input folder.
2. **Extract and reorder dimensions** - It reads the complex multi-dimensional data and standardizes the axis order (Z, Y, X, C).
3. **Split by channel** - If the image contains multiple channels, the tool separates them. It attempts to read the original channel names from the metadata and sanitizes them for safe file naming.
4. **Save individual TIFFs** - Each channel is saved as its own `.tiff` file in the output directory, combining the original filename with the specific channel name (e.g., `image_C0.tiff`).

---

# Convert TIFF -> NIfTI (.nii.gz)

This utility performs batch conversion of standard TIFF image files into compressed NIfTI files (`.nii.gz`). NIfTI is a standard file format widely used in medical imaging and 3D volume analysis and is well supported in AdaptFM.

## How It Works

The conversion relies on SimpleITK to accurately translate the image arrays into the NIfTI format. The tool processes your data using these steps:

1. **Locate TIFF files** - The script scans the input directory for all files ending in `.tiff` or `.tif`. 
2. **Read the image array** - It uses the `tifffile` library to read the raw pixel data from the TIFF image into memory.
3. **Convert to SimpleITK format** - The raw array is converted into a SimpleITK image object, which natively supports the spatial metadata required for NIfTI formats.
4. **Compress and save** - The tool writes the new image to your output directory, replacing the old file extension with `.nii.gz` to apply standard NIfTI compression.

---

# Convert NIfTI (.nii.gz) -> TIFF

This utility performs batch conversion of NIfTI files (`.nii` or `.nii.gz`) into standard TIFF image files. TIFF files are well supported in AdaptFM.

## How It Works

The conversion relies on SimpleITK to extract the array data from the NIfTI format. The tool processes your data using these steps:

1. **Locate NIfTI files** - The script scans the input directory for all files ending in `.nii` or `.nii.gz`.
2. **Read the spatial image** - It uses the `SimpleITK` library to properly read the NIfTI file, preserving the structure of the spatial volume.
3. **Extract raw array** - The SimpleITK image object is converted back into a raw numerical array (NumPy).
4. **Save as TIFF** - The tool writes the raw array to your output directory using the `tifffile` library, stripping the `.nii.gz` extension and replacing it with `.tiff`.

---

# Normalize TIFF To Range [-1, 1]

This tool normalizes the pixel intensity values of standard TIFF images to a specific numerical range (defaulting to -1.0 to 1.0). This is a common preprocessing step required by many neural networks to ensure stable and efficient model training or inference as well as increasing the precision of the estimation as computers store floating point numbers in a greater density around 0.

## How It Works

The normalization process applies the following mathematical steps to your image arrays:

1. **Locate TIFF files** - The script scans the input directory for all files ending in `.tiff` or `.tif`.
2. **Convert to float16** - It loads the image array and converts the datatype to 16-bit floating-point numbers (`float16`) to support decimal values while saving memory.
3. **Calculate minimum and maximum values** - The tool calculates the minimum and maximum pixel values, either across the entire image volume or along a specific axis (like depth).
4. **Scale to range** - It first normalizes the pixels to a baseline `0.0` to `1.0` scale. Then, it multiplies and offsets these values to fit your target range (e.g., shifting the scale to range between `-1.0` and `1.0`).
5. **Save the output** - The normalized images are saved into the output directory with their original filenames.

---

# Scale and Convert TIFF Data dtype to uint8

This utility standardizes your TIFF images into 8-bit unsigned integers (`uint8`), which restricts pixel values to a range of 0 to 255. This format is lightweight and widely compatible with standard image viewers, annotation tools, and basic masking workflows. 

***Importantly, this tool dynamically rescales floating-point images to prevent data loss (underflow) or harsh cutoffs (clamping) during conversion.***

## How It Works

The tool standardizes your images into `uint8` using the following logic:

1. **Locate TIFF files** - The script scans the input directory for all `.tiff` or `.tif` files.
2. **Analyze data type** - It checks the datatype of the underlying image array. 
3. **Dynamically scale floats** - If the image is a floating-point type (e.g., values like `0.5` or `-1.2`), it calculates the actual minimum and maximum values present. It then proportionally scales these values to fit perfectly within a `0` to `255` range before converting to `uint8`.
4. **Safely clip integers** - If the image is already a non-float type, it simply clips any values below 0 or above 255 and converts the type to `uint8`.
5. **Save the output** - The standardized `uint8` images are saved into the output directory.

---

# Send Image Data to nnUNet Dataset Folder

This tool automatically formats and organizes your image dataset to meet the strict structural requirements of the nnUNet framework. Use this utility to take a dataset that you want to run a trained nnUNet model on and place it into the file structure that will allow for AdaptFM nnUNet inference.

***To use this tool, your output directory must be placed within an overarching `nnUNet_raw` folder and named according to the nnUNet dataset convention (e.g., `Dataset001_MyData`).***

## How It Works

The tool prepares your dataset for nnUNet inference through the following sequential steps:

1. **Filter for raw images** - It scans the input directory for image files (`.tiff`, `.tif`, `.nii.gz`) while explicitly ignoring any segmentation masks (files ending in `_seg`).
2. **Validate output path** - It checks that your destination folder is correctly situated inside an `nnUNet_raw` parent directory. 
3. **Determine next sequence ID** - The tool checks your existing training images folder (`imagesTr`) to find the highest existing file ID. It then starts numbering your new inference files sequentially right where the training data left off.
4. **Format and copy** - Files are copied into the testing/inference directory (`imagesTs`). They are automatically renamed to follow nnUNet's required format: `[DatasetName]_[ID]_0000.tiff` (the `_0000` indicates the imaging modality channel).
5. **Generate a mapping file** - Finally, it creates a `filemapping.json` document in the output directory. This serves as a dictionary linking your original image names to their new, nnUNet-compliant filenames.

---

# Place Image Labels into Image Data Folder

This utility helps organize your datasets by matching isolated label/segmentation files with their corresponding raw image data. It safely moves label files into your main image folder while automatically renaming them so they are easily identifiable as segmentation masks. 

***This process is extension-agnostic, meaning it will work regardless of the specific filetypes of your labels, as long as the base filenames match your raw data.***

## How It Works

The tool pairs and moves your files using the following logical steps:

1. **Identify label files** - The tool scans the input directory for your label files, ignoring any hidden files or subdirectories.
2. **Match files by name** - It looks at the base name (stem) of each label file and checks if a matching data file exists in the destination directory. 
3. **Rename with a suffix** - If a match is found, the tool prepares to move the label file and automatically appends `_seg` to the end of the filename (e.g., `cell_image.tiff` becomes `cell_image_seg.tiff`).
4. **Safely move files** - The tool transfers the file to the output directory. It has built-in protections and will abort the move if a file with the same name already exists in the destination, preventing accidental data loss.
5. **Skip unmatched files** - If a label file does not have a corresponding raw image in the output directory, it is simply skipped and left in the input folder.

---