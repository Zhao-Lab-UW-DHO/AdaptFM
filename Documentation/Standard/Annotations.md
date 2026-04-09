# Using AdaptFM To Create Annotations

You can use AdaptFM to create annotations to fine-tune models. Before fine-tuning, first make sure you have [installed](../../README.md/#Installation) and [tested](/Testing-models.md) the model, since fine-tuning might not be needed

Steps for creating annotations are:
1. Select the algorithm you would like to use
2. Enter the parameters for segmentation
3. For SAM2 and SAM3, you must initialize the encoder before trying click or text-based segmentation
4. Select the 'save dir' directory for saving the images, and a name for the file. The "save dir" is the folder you should pick when [fine-tuning](./Fine-tuning-Models.md) your models. 
