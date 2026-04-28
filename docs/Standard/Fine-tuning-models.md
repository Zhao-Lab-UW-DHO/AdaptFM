# Using AdaptFM to Fine-Tune Models

AdaptFM supports GUI-based fine-tuning for various models. First, make sure you have [installed the tool](../../README.md/#Installation)

Steps:
1. Select the "Models" --> "Training" at the top of Napari.
2. Use the 'model' dropdown to select the model you would like to train.
3. "Select dataset folder" --> choose the folder you saved files to while making [annotations](Annotations.md)
  - 
4. Press 'Load Parameters' to see the model's hyperparameters  
a. If using SSVT or SAM-Med3D, be sure to enter the full path to the model checkpoints

5.  Adjust parameters as needed including the choice of checkpoint. See [AdaptFM's README](../../README.md) to get the checkpoint for using SSVT.
a. If you made annotations outside of AdaptFM ensure:
      - All images (original and annotations) are placed together in one folder
      - All images are in .tiff format
      - Annotations file names end with '_seg.tiff'
6. When you hit "Run" you will be prompted to select an output directory for your results, and a GPU to use.  

![Training Demo](../../asset/training-gif.gif)
