# Using AdaptFM to Fine-Tune Models

AdaptFM supports GUI-based fine-tuning for various models. First, make sure you have [installed the tool](../../README.md/#Installation)

Steps:
1. Select the "Models" --> "Training" at the top of Napari.
1. Use the 'model' dropdown to select the model you would like to train.
1. Enter the function used for training to pull in hyper parameters. They are:
   - CellposeSAM - train_seg
   - MicroSAM - train_sam
   - SSVT - train_SSVT
   - SAM-Med-3D - launch_training
1. Adjust parameters as needed including the choice of checkpoint. See [AdaptFM's README](../../README.md) to get the checkpoint for using SSVT.
1. "Select dataset folder" --> choose the folder you saved files to while making [annotations](Annotations.md)
1. When you hit "Run" you will be prompted to select an output directory for your results, and a GPU to use.

![Training Demo](../../asset/training-gif.gif)
