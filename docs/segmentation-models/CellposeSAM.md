# CellposeSAM

[CellposeSAM](https://github.com/mouseland/cellpose) is a deep learning model designed to identify and segment cell and other biological structures in microscopy. It uses the segmentation capabilities of the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) with the segmentation capabilities of Cellpose. 

Within AdaptFM, you can use CellposeSAM in two ways:
    - **Inference** - Use the existing CellposeSAM model to segment new images
    - **Fine-tuning** - Adapt the existing CellposeSAM model using your own labeled images so that it performs better on a specific cell type, tissue, imaging modality, or experimental system

***Before using CellposeSAM you must first install the environment and package with AdaptFM's environment manager***

## Using AdaptFM to run Inference with CellposeSAM

***For 3D segmentation CellposeSAM requires images be stored in TIFF files. Depending on your file type, you might be able to use one of AdaptFM's preprocessing tools to convert your images to TIFF format. If we don't have a method for your imaging modality, please submit an issue and/or pull request.***

***Importantly, CellposeSAM natively handles all image normalization and preprocessing. This is done by default, and noramlizes the input so that 0=1st percentile of image values and 1=99th percentile.***

You can use CellposeSAM in AdaptFM to run inference on images in bulk using the following steps:
 
1. Within AdaptFM navigate to the "Models" menu at the top and select "Inference". Select "CellposeSAM" as the model. 
2. Under "Dataset" select "Browse" and navigate to the folder containing images you would like to segment.
3. Under "Output Directory" select the folder where you want to output your final predictions
4. Using "GPU Index" select the GPU you would like to use for running inference
5. Under "Model Checkpoint" navigate to the checkpoint you would like to use ***(optional)***. If you would like to use CellposeSAM 'out-of-the-box' do not select a model checkpoint
6. Once ready select 'Run' to start inference. You can monitor outputs in the Output Log and stop inference whenever using the "Terminate" button.

## Using AdaptFM to fine-tune CellposeSAM

### Prepare the data

***CellposeSAM requires that images used for 3D training be stored in TIFF format***

***Again, CellposeSAM natively handles all image normalization and preprocessing. This is done by default, and normalizes the input so that 0=1st percentile of image values and 1=99th percentile.***

As with other AdaptFM models, you must first prepare your data using the below steps

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.

### Fine-tuning the model

1. Within AdaptFM navigate to the "Models" menu at the top and select "Training"
2. Select the model CellposeSAM
3. In the "Dataset" field select "Browse" and navigate to the folder containing your raw and labeled images. This is the same folder in step 2 from "Prepare your data"
4. In the "Output Directory" field select "Browse and navigate to the location where you want to output your results.
5. Using the "GPU Index" field, select the GPU you plan to use for training
6. **Optionally** you can select "Load Parameters" to further define CellposeSAM fine-tuning. If you do not have programming and/or machine-learning experience, we do not recommend loading parameters or changing from any of the defaults. CellposeSAM has 31 adjustable parameters defined by the authors. Full descriptions can be found [here](https://github.com/MouseLand/cellpose/blob/main/cellpose/train.py#L322). We have provided descriptions of some of the most common parameters here:
    - batch_size: the number of patches to run on the GPU. If you are getting "Out of Memory" errors during training, try decreasing this number
    - n_epochs: how many times the entire training set is evaluated during training. If training is taking a particularly long time, try decreasing this number. Decreasing this too much can limit model performance
    - normalize: True or False. Whether or not you would like to use CellposeSAM's recommended normalization scheme (we recommend setting this to True). Normalizes the input so that 0=1st percentile of image values and 1=99th percentile.
    - save_path: Where you would like to save the trained model
    - save_every: Integer value. After how many epochs would you like to save the model
    - save_each: True or False. Save the network with a different filename every time it is saved
7. Once you are ready you can select "Run" to begin training the model. You can monitor output in the "Output Log" section. If at any point you would like to stop training simply click the "Terminate" button.