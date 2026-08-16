# CellposeSAM

[CellposeSAM](https://github.com/mouseland/cellpose) is a deep learning model designed to identify and segment cell and other biological structures in microscopy. It uses the segmentation capabilities of the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) with the segmentation capabilities of Cellpose. 

Within AdaptFM, you can use CellposeSAM in two ways:
    - **Inference** - Use the existing CellposeSAM model to segment new images
    - **Fine-tuning** - Adapt the existing CellposeSAM model using your own labeled images so that it performs better on a specific cell type, tissue, imaging modality, or experimental system

***Before using CellposeSAM you must first install the environment and package with AdaptFM's environment manager***

## Using AdaptFM to run Inference with CellposeSAM

***For 3D segmentation CellposeSAM requires images be stored in Tiff files. Depending on your file type, you might be able to use one of AdaptFM's preprocessing tools to convert your images to Tiff format. If we don't have a method for your imaging modality, please submit an issue and/or pull request.***

