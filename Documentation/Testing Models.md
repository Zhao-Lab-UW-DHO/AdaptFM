# Using AdaptFM to Test Models

AdaptFM support GUI-based inference for various models. First, make sure you have installed the associated model in a separate conda environment. 

If you have already tested a model, you can benchmark it against a ground truth following the steps here. 

All inference pipelines follow the same steps:
- Select the "Models" --> "Inference" at the top of Napari
- Use the 'model' dropdown to select the model you would like to test
- "Select dataset folder" --> raw images you would liket to predict using the model
- "Select model checkpoint" --> you can optionally select the checkpoint for a trained model. This is not required for MicroSAM, CellposeSAM, or SAM-Med-3D. It is required for nnUNetv2.
- 



