# Using AdaptFM To Create Annotations

You can use AdaptFM to create annotations to use as labels for fine-tuning models. Before fine-tuning, first make sure you have [installed](../../README.md#main-installation) and [tested](./Testing-Models.md) the model, since fine-tuning might not be needed.

Steps for creating annotations are:
1. Select the algorithm you would like to use.
1. Enter the parameters for segmentation. 
1. Following parameter selection SAM2 and SAM3 have particular steps to follow for segmentation.
    - **For SAM2**:
        1. First, initialize the model with the given settings using the "Initialize (encode all slices)" button.
        1. Toggle on Segment click mode and click on the objects to segment on the current layer. Changing layers during click segmentation will cause SAM2 to improperly propagate segmentations.
        1. Toggle off Segment click mode to move around the image.
        1. Click "Propagate through volume" once all desired objects on the current layer are selected.
        1. Move onto the succeeding steps for creating annotations.
    - **For SAM3**:
        1. First, initialize the model with the given settings using the "Initialize (encode all slices)" button.
        1. Provide a text prompt for the segmentation model and click "Segment by text". You cannot click to segment without first Segmenting by text.
        1. Toggle on the Segment click mode and click on the objects to segment on the current layer.
        1. Click "Propagate through volume" once all desired objects on the current layer are segmented.
        1. Move onto the succeeding steps for creating annotations.
1. Select the 'save dir' directory for saving the 3D image, and a name for the file. The "save dir" is the folder you should pick when [fine-tuning](./Fine-tuning-models.md) your models.

![Segmentation Demo](../../asset/demo_cropped.gif) ![SAM2 Demo](../../asset/SAM2_GIF.gif)