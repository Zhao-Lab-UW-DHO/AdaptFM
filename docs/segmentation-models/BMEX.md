# BME-X

[BME-X](https://github.com/DBC-Lab/Brain_MRI_Enhancement) is a foundation model designed for magnetic resonance images (MRI). It's base model is used for motion correction, super resolution, and denoising. BME-X additionally has a segmentation model for segmenting brains from MRI images. 

Within AdaptFM you can use BME-X's **segmentation** model to do the following 
    - **Inference** - use the trained BME-X segmentation model to segment MRI brain images in bulk
    - **Fine-tuning** - adapt the existing segmentation model to better segment MRI brain images from your own dataset

***Before using BME-X you must first install the environment and package with AdaptFM's environment manager. Additionally, you must download the model checkpoint using this [dropbox link](https://www.dropbox.com/scl/fo/j55epethu8bhmdsjpv0i1/AEmWksuTfP94M0bDqN3ZHDo?rlkey=zsdcna67uwbajg4pri249jlbh&st=2he9arzw&e=2&dl=0)***

## Using AdaptFM to run Inference with BME-X


