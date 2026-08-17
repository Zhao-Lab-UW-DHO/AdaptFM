# Merlin-nnUNet

[Merlin-nnUNet](https://github.com/ashwinkumargb/Merlin-nnUNet) is a 3D CT segmentation model that is fine-tuned from the [Merlin](https://github.com/StanfordMIMI/Merlin) foundation model. It is based off of the [nnU-Net](https://github.com/mic-dkfz/nnunet) framework.

***Merlin-nnUNet can only be used for inference and does not support fine-tuning.***

***Before using Merlin-nnUNet you must first install the environment and package with AdaptFM's environment manager. This will automatically download the model.***

## Using AdaptFM to run Inference with Merlin-nnUNet

***Importantly Merlin-nnUNet handles all data preprocessing and normalization. The preprocessing and normalizing involve:***
1. Orienting the image to RAS (Right-Anterior-Superior)
2. Resampling the image scale to 1.5 x 1.5 x 3mm voxel spacing using bilinear interpolation
3. Pads the image with zeros so it is at least 224 x 224 x 160 voxels
4. It then leverages nnU-Net's [CTNormalization scheme](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/preprocessing/normalization/default_normalization_schemes.py#L58)
    - percentile clipping, lower bound of 00.5 percentile and upper bound of 99.5 percentile
    - Z Score normalization

### Running Infernce with Merlin-nnUNet




