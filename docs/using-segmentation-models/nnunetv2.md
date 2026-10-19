# nnUNetV2 

[nnUNetv2](https://github.com/MIC-DKFZ/nnUNet) is a machine-learning framework designed to learn how to identify and segment structures in medical and biological images. nnUNetv2 is not a model on its own, it is a framework for users to train a model for identifying and segmenting objects in medical images. In AdaptFM, users can use their annotated images to train an nnUNet model and then use the trained model to automatically segment new images.

***As a general rule, nnUNetv2 serves as a strong baseline for comparison against other models. It trains a new model from scratch as opposed being previously trained and/or fine-tuned.***

***Before using nnUNetv2 you must first install the environment and package with AdaptFM's environment manager***

## Spatial Resolution & Image Sizing with nnU-Net

nnU-Net processes images without applying global spatial downsampling. During fine-tuning, the network extracts 3D sub-volume patches (typically 128x128x128 voxels) directly from the image. During inference, nnU-Net processes the complete image volume by sliding an overlapping 3D patch window across the entire spatial grid (using a default 50% step overlap). Predictions across all overlapping windows are stitched together, and the final segmentation mask is exported at the same array dimensions as your original input file.

## Training an nnUNet Model in AdaptFM

### Prepare the Data

To train an nnUNet model in AdaptFM, you must prepare your data with the below steps before using the model:

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.

### Train the Model

1. Within AdaptFM navigate to the "Models" menu at the top and select "Training"
2. Select the model nnUNetv2
3. In the "Dataset" field select "Browse" and navigate to the folder containing your raw and labeled images. This is the same folder in step 2 from "Prepare your data"
4. In the "Output Directory" field select "Browse" and navigate to the location where you want to output your results. nnUNet has a specific required folder structure (see structure under point 6). AdaptFM will automatically create the folders. If the folders were created in a previous run, AdaptFM will not overwrite the existing folders. 
5. Using the "GPU Index" field, select the GPU you plan to use for training
6. Using "Set Name", "Set ID", "Config", "fold" to define the name and ID for you training run. When you launch training, these fields will be used to create the following required folders in the "Output Directory"
    - Set Name - the name you would like to use to define your dataset 
    - Set ID - the ID associated with the dataset you are processing. This is used to distinguish between multiple training runs with the same "Set Name"
    - Config - how strong of resolution should the model use when finding objects. We generally recommend using 3d_fullres
        - 3d_fullres - processed the image at higher resolution to preserve fine anatomical or cellular detail
        - 3d_lowres - processed a downsampled version of the image, allowing the model to see a larger portion of the 3D volume at once, but with less fine detail
    - fold - which training split should be used for training and validation. We recommend leaving this as 'all' when doing training, especially if you plan to benchmark results against other models. Valid values are 'all', or an integer between 0–4.
    - trainer - the nnUNetTrainer you would like to use. You can use different trainers to change the network architecture, learning rate scheduler, number of epochs. A full list of standard trainers can be found [here](https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training/nnUNetTrainer/variants). To train only certain layers of your network, follow the instructions [below](#fine-tuning-specific-layers) 
    - trainer - the nnNUetTrainer class used during training. You can use different trainer classes to define different loss functions, lr schedulers, network architectures, and so on. A full list can be found [here](https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training/nnUNetTrainer/variants). AdaptFM will use the default nnUNetTrainer.

A couple of important points about the below folder structure
    - The below structure is **required** by nnUNet.
    - Because this structure is required, training data is copied and renamed into imagesTr and labelsTr. You can view the name_mapping.json file to see how the original names map to the nnUNet names.

```text
Output Directory
├──nnUNet_preprocessed
    ├──Dataset[Set ID]_[Set Name]
├──nnUNet_raw
    ├──Dataset[Set ID]_[Set Name]
        ├──name_mapping.json
        ├──imagesTr - Original images are moved here
            ├──[Set Name]_000_0000.tiff
            ├──[Set Name]_001_0000.tiff
            ├──[Set Name]_002_0000.tiff
            ├──[Set Name]_003_0000.tiff
            ├──...
        ├──labelsTr - Labeled / annotated images are moved here
            ├──[Set Name]_000.tiff
            ├──[Set Name]_001.tiff
            ├──[Set Name]_002.tiff
            ├──[Set Name]_003.tiff
            ├──...
        ├──imagesTs - MUST place images used for inference here. Additionally they MUST have the appropriate naming structure
            ├──[Set Name]_004_0000.tiff
            ├──[Set Name]_005_0000.tiff
            ├──[Set Name]_006_0000.tiff
            ├──[Set Name]_007_0000.tiff
            ├──...
├──nnUNet_results
    ├──Dataset[Set ID]_[Set Name]
        ├──nnUNetTrainer_nnUNet_Plans__[Config]
            ├──fold_[fold]
                ├──checkpoint_best.pth - Saved checkpoint from training. Model with the best performance on cross-validation
                ├──checkpoint_latest.pth - Saved checkpoint from training. Model saved from the latest training epoch
```

7. Once you are ready you can select "Run" to begin training the model. You can monitor output in the "Output Log" section. If at any point you would like to stop training simply click the "Terminate" button.

## Using nnUNet Inference Within AdaptFM

***You must train an nnUNet model before running inference. nnUNet does not have a base model for running inference***

Once you have completed training for nnUNet. You can use your train model to run inference using the following steps

1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference"
2. Select the model nnUNetv2
3. Under "Dataset" select "Browse" and navigate to the folder containing images you would like to segment. These images ***must*** be in the ``` nnUNet_raw/Dataset[Set ID]_[Set Name]``` folder and they must have the proper naming format. See the 'imagesTs' folder above as an example
4. Under "Output Directory" select the folder where you want to output your final predictions
5. Using "GPU Index" select the GPU you would like to use for running inference
6. Under "Model Checkpoint" navigate to the checkpoint you would like to use. This ***must*** be in the ``` nnUNet_results/Dataset[Set ID]_[Set Name]/nnUNetTrainer_nnUNet_Plans__[Config] ```. Importantly the 'Set ID' and 'Set Name' must match the 'Set ID' and 'Set Name' of the data you are running inference on
7. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.


#### Fine-Tuning Specific Layers

By default nnUNet will update weights in all layers when fine-tuning a model. If you would like to update only specific layers you can follow the below instructions. The below instructions follow the solutiosn proposed from the [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet/issues/2714)

1. Navigate to the nnUNet_adapt conda environment and open the nnUNetv2 trainer variants under nnUNet_adapt > lib > python3.10 > site-packages > nnunetv2 > training > nnUNetTrainer > variants
2. Create a new folder with an __init__.py file
3. In your newly created folder create a new file and add the following code

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerFreezeEncoder(nnUNetTrainer):

    def initialize(self):
        super().initialize()

        for name, param in self.network.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False
            else:
                param.requires_grad = True

```

4. Importantly, the above code will freeze the encoder while training the decoder layers. If you would like to name only specific layers, simply update the ```if name.startswith``` condition to specify the desired layers
5. Within the AdaptFM training widget, enter your newly created trainer in the 'trainer' field