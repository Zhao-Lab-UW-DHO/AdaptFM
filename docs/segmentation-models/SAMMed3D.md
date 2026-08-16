# SAM-Med3D

[SAM-Med3D](https://github.com/uni-medical/sam-med3d) is a deep learning model designed to segment objects in 3D medical images. It is primarily designed for segmenting 3D MRI, CT, or Ultrasound images. It is a 3D extension of the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). SAM-Med3D was trained used ~143,000 3D masks from across 245 categories. 

In AdaptFM, SAM-Med3D can be used in two ways:
    - **Inference** - You can use SAM-Med3D to predict images in bulk. **Because SAM-Med3D performs promptable segmentation, you must submit images along with 'mask prompt'** [details below](#creating-prompts-for-inference).
    - **Fine-tuning** - You can adapt the existing SAM-Med3D using your own labeled data so that it can perform better at segmenting a specific structure (e.g. organ) in a certain imaging modality (e.g. MRI, CT, ultrasound)

***Before using SAM-Med3D you must install the environment and package with AdaptFM's environment manager***

***Additionally, you must download the model's [checkpoint](https://github.com/uni-medical/sam-med3d#-checkpoint) using any of the associated links. You must provide this checkpoint's path when running inference or training***

## Using AdaptFM to run Inference with SAM-Med3D

***Importantly, SAM-Med3D natively handles all image normalization and preprocessing. This is done by default, and normalizes pixels using a z-score normalization strategy on the entire image volume. The exact implementation can be found [here](https://github.com/uni-medical/SAM-Med3D/blob/main/utils/infer_utils.py#L266). Consequently there is no need to independently normalize your data before running inference***

### Creating Prompts for Inference 

SAM-Med3D performs 'promptable' segmentation. This means you need ground truth labels to generate prompt points. These ground truth prompts tell the model where to 'look' for an object of interest. To prepare your images and prompts follow the below steps. This format and folder structure is **required** by the SAM-Med3D algorithm. 

1. Use AdaptFM's Annotation Algorithms to generate ground-truth 'mask prompts'. These should have segmentations of the object of interest. Note that they do not need to be perfect segmentations. They just need to give SAM-Med3D a place to 'look' for an object of interest.
2. Once you have created segmentations and saved them, move your original images into an 'imagesTr' folder and the segmentations into a 'labelsTr' folder. Your raw and labeled images must have the **exact** name. The structure should look like this. Again this is a SAM-Med3D design requirement.
3. ***Importantly, SAM-Med3D can only handle images that are in 'nii.gz' format.***

```text
Inference Folder
├──imagesTr
    ├──image_1.nii.gz
    ├──image_2.nii.gz
    ├──image_3.nii.gz
    ├──image_4.nii.gz
    ├──...
├──labelsTr
    ├──image_1.nii.gz
    ├──image_2.nii.gz
    ├──image_3.nii.gz
    ├──image_4.nii.gz
    ├──...
```

### Running Inference

Once you have generated ground truth mask prompts, you can use SAM-Med3D to segment objects of interest based on the provided prompts. 

1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference". Select "SAMMed3D" as the model
2. Under "Dataset" select "Browse" and navigate to the ***parent*** folder containingg your imagesTr and labelsTr subfolders. In the above example, you would select "Inference Folder" as you "Dataset"
3. Under "Output Directory" select the folder where you want to output your final predictions
4. Using "GPU Index" select the GPU you would like to use for running inference
5. Under "Model Checkpoint" navigate to the checkpoint you would like to use ***required***. You must either select the default SAM-Med3D checkpoint, or separate checkpoint you created from fine-tuning. 
6. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.

## Using AdaptFM to fine-tune SAM-Med3D

### Prepare the data

***Again, SAM-Med3D natively handles all image normalization and preprocessing. This is done by default, and normalizes pixels using a z-score normalization strategy on the entire image volume. The exact implementation can be found [here](https://github.com/uni-medical/SAM-Med3D/blob/main/train.py#L133). Consequently there is no need to independently normalize your data before running training***

As with other AdaptFM models, you must first prepare your data using the below steps

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.

### Fine-tuning the model

1. Within AdaptFM navigate to the "Models" menu at the top and select "Training"
2. Select the model MicroSAM
3. In the "Dataset" field select "Browse" and navigate to the folder containing your raw and labeled images. This is the same folder in step 2 from "Prepare your data"
4. In the "Output Directory" field select "Browse and navigate to the location where you want to output your results. This will create two sub-folders 
    -imagesTr - contains the original raw images
    -labelsTr - contains the labeled images
5. Using the "GPU Index" field, select the GPU you plan to use for training
6. You **must** select "Load Parameters" to further define the checkpoint path for SAM-Med3D. All other parameters can be left unchanged. If you don't have experience with machine-learning models we recommend leaving the default values for each (aside from the checkpoitn path which must be updated)
    - checkpoint - **REQUIRED** you must provide the file path for the SAM-Med3D checkpoint you want to fine-tune
    - task_name - name used to identify and organize the training task
    - click_type - Determines how prompts/clicks are generated during training to tell SAM-Med3D which object to segment. 
    - multi_click - True or False. Whether or not the model should try to identify objects by simulating multiple clicks
    - model_type - which SAM-Med3D architecture should be used for training
    - device - **Leave** as 'cuda'
    - work_dir - Directory where training checkpoints and other training outputs will be saved
    - num_workers - Number of CPU workers used to load and prepare training data
    - gpu_ids - **This is overridden by GPU Index**
    - multi_gpu - Not supported in AdaptFM
    - resume - True or False. Continues training from a previously saved checkpoint
    - lr_scheduler - determines how the learning rate chagnes during training
    - step_size - the training epochs at which the learning rate is reduced
    - gamma - How much the learning rate is reduced when the scheduler decreases it
    - num_epochs - number of complete passes through the training data
    - img_size - size of the 3D iamge region used as input to the model during training
    - batch_size - Number of 3D image samples processed together before teh model updates its weights
    - accumulation _steps - Number of training steps whose gradients are accumulated before updating the model, allowing an effectively larger batch size.
    - lr - Controls how strongly the model's weights are adjusted during each training update.
    - weight_decay - Regularization that discourages overly complex model weights and can help reduce overfitting.
