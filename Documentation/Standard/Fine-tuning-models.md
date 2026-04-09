# Using AdaptFM to Fine-Tune Models

AdaptFM supports GUI-based fine-tuning for various models. First, make sure you have [installed the tool](../../README.md/#Installation)

Steps:
1. Select the "Models" --> "Training" at the top of Napari
2. Use the 'model' dropdown to select the model you would like to train
3. Enter the function used for training to pull in hyper parameters. They are:
   a. CellposeSAM - train_seg
   b. MicroSAM - train_sam
   c. SSVT - train_SSVT
   d. SAM-Med-3D - launc_training
5. "Select dataset folder" --> folder with raw images and associated masks used for training.
6. When you hit "Run" you will be prompted to select an output directory for your results, and a GPU to use.
