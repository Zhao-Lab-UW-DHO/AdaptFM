# AdaptFM

An interactive framework for annotating, training, running inference, and benchmarking 3D segmentation models. 

In AdaptFM you can:
- Test and benchmark 3D segmentation models on your data. We currently support CellposeSAM, MicroSAM, nnUNetv2, SAM-Med-3D, and SSVT.
- Create 3D annotations using a number of segmentation algorithms, including SAM2 for click-based segmentation and SAM3 for text-based segmentation
- Use those annotations to fine-tune/train models on your data
- Add new models or annotation algorithms

[Installation](#Installation)
\n [Get Started](Documentation/Standard)

![Segmentation Demo](asset/demo_cropped.gif) ![SAM2 Demo](asset/SAM2_GIF.gif)

# Installation 

We recommend creating a separate conda environment for AdaptFM, and each associated model

Create a conda environment for AdaptFM using
```
conda create --name AdaptFM python=3.13
conda activate AdaptFM
```


