# Using AdaptFM To Create Annotations

You can use AdaptFM to create annotations to use as labels for fine-tuning models. Before fine-tuning, first make sure you have [installed](../../README.md#main-installation) and [tested](./Testing-Models.md) the model, since fine-tuning might not be needed.

Steps for creating annotations are:
1. Select the algorithm you would like to use.
1. Enter the parameters for segmentation. 
1. If using SAM2 or SAM3, follow the below steps (if not, go to step 4):
    - **For SAM2**:
        1. Press "Initialize (encode all slices)"
        1. Select the 'Segment click mode' checkbox. Click on objects to segment in **one layer**. Changing layers during click segmentation will cause SAM2 to improperly propagate segmentations.
        1. Unselect the 'Segment click mode' checkbox to move around the image.
        1. Click "Propagate through volume" once all desired objects on the current layer are selected.
        1. Scroll to other layers to segment objects as needed.
    - **For SAM3**:
        1. Press "Initialize (encode all slices)"
        1. Provide a text prompt for the segmentation model and click "Segment by text". SAM3 does not support click segmentation without text segmentation first. 
        1. Select the 'Segment click mode' checkbox. Segment objects in that layer (if needed) by clicking on them.
        1. Click "Propagate through volume" once all desired objects on the current layer are segmented.
        1. Move onto the succeeding steps for creating annotations.
1. Select the 'save dir' directory for saving the 3D image, and a name for the file. The "save dir" is the folder you should pick when [fine-tuning](./Fine-tuning-models.md) your models.

![Segmentation Demo](../../asset/demo_cropped.gif) ![SAM2 Demo](../../asset/SAM2_GIF.gif)
