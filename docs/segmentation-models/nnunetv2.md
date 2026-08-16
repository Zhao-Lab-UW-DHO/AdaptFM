# nnUNetV2 

[nnUNetv2](https://github.com/MIC-DKFZ/nnUNet) is a machine-learning framework designed to learn how to identify and segment structures in medical and biological images. nnUNetv2 is not a model on its own, it is a framework for users can train a model for identifying and segmenting objects in medical images. In AdaptFM, users can use their annotated images to train an nnU-Net model and then use the trained model to automatically segment new images.

***As a general rule, nnUNetv2 serves as a strong baseline for comparison against other models. It trains a new model from scratch as opposed being previously trained and/or fine-tuned.***

## Training an nnU-Net model in AdaptFM

To train an nnU-Net model in AdaptFM, follow the below steps:

1. **Create annotations** - As with training or fine-tuning any model in AdaptFM, the first step is to generate labeled data using our annotation functionality.
2. **Place original and labeled images in one folder** - once all training images have been annotated, place the original images and labeled version in the same folder. ***The original images and labeled images must have identical names, differing only in their ending (labeled images ending with '_seg.ext')*** Note that if you save your image and its label with the save widget from within AdaptFM, they will the images will already be in this format.






#


