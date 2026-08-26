# Merlin-nnUNet

[Merlin-nnUNet](https://github.com/ashwinkumargb/Merlin-nnUNet) is a 3D CT segmentation model that is fine-tuned from the [Merlin](https://github.com/StanfordMIMI/Merlin) foundation model. It is based off of the [nnUNet](https://github.com/mic-dkfz/nnunet) framework.

***Merlin-nnUNet can only be used for inference and does not support fine-tuning.***

***Before using Merlin-nnUNet you must first install the environment and package with AdaptFM's environment manager. This will automatically download the model.***

## Using AdaptFM to run Inference with Merlin-nnUNet

***Importantly Merlin-nnUNet handles all data preprocessing and normalization. The preprocessing and normalizing involve:***
1. Orienting the image to RAS (Right-Anterior-Superior)
2. Resampling the image scale to 1.5 x 1.5 x 3mm voxel spacing using bi-linear interpolation
3. Pads the image with zeros so it is at least 224 x 224 x 160 voxels
4. It then leverages nnUNet's [CT Normalization scheme](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/normalization/default_normalization_schemes.py#L58)
    - Percentile clipping, lower bound of 00.5 percentile and upper bound of 99.5 percentile
    - Z Score normalization

### Running Infernce with Merlin-nnUNet

1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference"
2. Select the model Merlin
3. Under "Dataset" select "Browse" and navigate to the folder containing images you would like to segment. These images ***must*** be in the ``` nnUNet_raw/Dataset[Set ID]_[Set Name]``` folder and they must have the proper naming format. See the 'imagesTs' folder above as an example
4. Under "Output Directory" select the folder where you want to output your final predictions
5. Using "GPU Index" select the GPU you would like to use for running inference
6. Under "Model Checkpoint" navigate to the checkpoint you would like to use. This ***must*** be in the ``` nnUNet_results/Dataset[Set ID]_[Set Name]/nnUNetTrainerMerlin__nnUNetPlans__3d_fullres```. Importantly the 'Set ID' and 'Set Name' must match the 'Set ID' and 'Set Name' of the data you are running inference on
7. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.

