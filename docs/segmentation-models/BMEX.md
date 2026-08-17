# BME-X

[BME-X](https://github.com/DBC-Lab/Brain_MRI_Enhancement) is a foundation model designed for magnetic resonance images (MRI). It's base model is used for motion correction, super resolution, and denoising. BME-X additionally has a segmentation model for segmenting brains from MRI images. 

Within AdaptFM you can use BME-X's **segmentation** model to do the following 
    - **Inference** - use the trained BME-X segmentation model to segment MRI brain images in bulk
    - **Fine-tuning** - adapt the existing segmentation model to better segment MRI brain images from your own dataset

***Before using BME-X you must first install the environment and package with AdaptFM's environment manager. Additionally, you must download the model checkpoint using this [dropbox link](https://www.dropbox.com/scl/fo/j55epethu8bhmdsjpv0i1/AEmWksuTfP94M0bDqN3ZHDo?rlkey=zsdcna67uwbajg4pri249jlbh&st=2he9arzw&e=2&dl=0)***

## Using AdaptFM to run Inference with BME-X

### Prepare the Data

***BME-X natively handles all normalization and preprocessing. This is done by a four step process in the pipeline. You do not need to reimplement this.***
1. Percentile clipping from the 0.0001 to 99.999 percentiles
2. Linear rescaling to map intensities to 0 - 1000
3. Histogram matching - match intensity values to a template4
4. Final scaling - divide values by 10,000 so that the model sees values between 0 and 0.1

***BME-X additionally requires a specific dataset format for running inference. The data must be in .nii.gz format, and be stored in [BIDS-MRI format](https://bids.neuroimaging.io/getting_started/folders_and_files/folders.html)***

```text
<root>
├──sub-01
    ├──ses-01
        ├──anat
            sub-01_ses-01_T1w.nii.gz
            sub-01_ses-01_T2w.nii.gz
```

You can find an example provided under test_BIDS in the [main repo](https://github.com/DBC-Lab/Brain_MRI_Enhancement)

### Run Inference

Once your data is in the proper format, you can use AdaptFM to run inference using:

1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference". Select "BME-X" as the model
2. Under "Dataset" select "Browse" and navigate to the folder containing the root of the BIDs dataset you would like to segment
3. Under "Output Directory" select the folder where you want to output your final predictions
4. Using "GPU Index" select the GPU you would like to use for running inference
5. Under "Model Checkpoint" navigate to the checkpoint you would like to use.
6. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.

## Using AdaptFM to Fine-tune BME-X

### Prepare the data

***MicroSAM expects all image inputs for training to be values between 0 and 255 and un uint8 format. AdaptFM employs MicroSAM's expected normalization strategy of min-max normalization followed by setting values between 0 and 255.***

As with other AdaptFM models, you must first prepare your data using the below steps

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.

