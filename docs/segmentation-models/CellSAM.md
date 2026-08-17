# CellSAM

[CellSAM](https://github.com/vanvalenlab/cellSAM) is a pretrained foundation model designed to automatically identify and segment cells and other cellular structures in microscopy images. It was developed to work across varying biological targets and imaging modalities including brightfiled, fluorescence, phase-contrast, tissue, yeast, bacteria, and cell culture images.

***CellSAM can only be used for inference and does not support fine-tuning***

***Before using CellSAM you must first install the environment and package with AdaptFM's environment manager. To use it you will need to create a [deepcell account](https://deepcell.readthedocs.io/en/master/API-key.html#api-key-usage). In AdaptFM you will be asked to provide you deep cell access token. Without it, you will not be able to use CellSAM.***

## Using AdaptFM to run Inference with CellSAM

***Importantly, CellSAM natively handles all image normalization and preprocessing. The exact code can be found [here](https://github.com/vanvalenlab/cellSAM/blob/master/cellSAM/utils.py#L205) This is done in a two step process:***
1. Percentile thresholding - CellSAM finds the 99.9th percentile intensity. Any pixel brighter than that gets replaced with that percentile value
2. Contrast Limited Adaptive Histogram Equalization (CLAHE) per channel. Then rescales pixels to values between 0 and 1.

You can use CellSAM in AdaptFM to run inference on images in bulk using the following steps:
 
1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference". Select "CellSAM" as the model. 
2. Under "Dataset" select "Browse" and navigate to the folder containing images you would like to segment.
3. Under "Output Directory" select the folder where you want to output your final predictions
4. Using "GPU Index" select the GPU you would like to use for running inference
5. Under "Model Checkpoint" navigate to the checkpoint you would like to use ***(optional)***. If you would like to use CellSAM 'out-of-the-box' do not select a model checkpoint
6. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.
7. **CellSAM only performs 2D segmenation**. Per the [manuscript](https://www.nature.com/articles/s41592-025-02879-w) it can be extended to 3D using [u-segment3D](https://github.com/DanuserLab/u-segment3D).  This is directly implemented in AdaptFM. You do **not** need to run u-segement3D on the results again. 