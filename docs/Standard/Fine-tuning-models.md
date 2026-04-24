# Using AdaptFM to Fine-Tune Models

AdaptFM supports GUI-based fine-tuning for various models. First, make sure you have [installed the tool](../../README.md/#Installation)

Steps:
1. Select the "Models" --> "Training" at the top of Napari.
1. Use the 'model' dropdown to select the model you would like to train.
1. Optionally, you can press 'Load parameters' to adjust the model's hyperparameters
- Note that SAM-Med3D and SSVT you need to specify the model checkpoint path.
1. Adjust parameters as needed including the choice of checkpoint. See [AdaptFM's README](../../README.md) to get the checkpoint for using SSVT.
1. "Select dataset folder" --> choose the folder you saved files to while making [annotations](Annotations.md)  
a. If you made annotations outside of AdaptFM ensure:
      - All images (original and annotations) are placed together in one folder
      - All images are in .tiff format
      - Annotations file names end with '_seg.tiff'
1. When you hit "Run" you will be prompted to select an output directory for your results, and a GPU to use.  
a. If using SSVT or SAM-Med3D, be sure to enter the full path to the model checkpoints

![Training Demo](../../asset/training-gif.gif)
