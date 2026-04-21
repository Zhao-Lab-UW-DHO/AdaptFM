# Using AdaptFM to Fine-Tune Models

AdaptFM supports GUI-based fine-tuning for various models. First, make sure you have [installed the tool](../../README.md/#Installation)

Steps:
1. Prepare the data
   - For all models except SAM-Med3D:
      - Labels and images must exist in the folder and be in the .tiff format. 
      - Labels must have names identical to the image with the '_seg' suffix.
      - If the data has been generated with the [annotations](./Annotations.md) or [inference](Standard/Testing-Models.md) functionality within AdaptFM, preparing the data should be as simple as moving the segmentation outputs into the image folder.
   - For SAM-Med3D:
      - A checkpoint must be specified (see [AdaptFM's README](../../README.md) for the checkpoint to download).
      - The images must be in .nii.gz format.
      - The dataset folder must always have two subfolders one named 'imagesTr' (for images) and another 'labelsTr' (for labels), see [the imageTBAD huggingface demo dataset](https://huggingface.co/datasets/hbakhtiar/AdaptFM_Testing/tree/main/imageTBAD) for an example.
1. Select the "Models" --> "Training" at the top of Napari.
1. Use the 'model' dropdown to select the model you would like to train.
1. Enter the function used for training to pull in hyper parameters. They are:
   - CellposeSAM - train_seg
   - MicroSAM - train_sam
   - SSVT - train_SSVT
   - SAM-Med-3D - launch_training
1. Adjust parameters as needed including the choice of checkpoint. See [AdaptFM's README](../../README.md) to get the checkpoint for using SSVT.
1. "Select dataset folder" --> choose the folder made in the data preparation step.
1. When you hit "Run" you will be prompted to select an output directory for your results, and a GPU to use.

![Training Demo](../../asset/training-gif.gif)