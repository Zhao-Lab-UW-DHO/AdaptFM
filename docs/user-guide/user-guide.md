# AdaptFM User Guide

AdaptFM is a Napari-based tool, imaging platform for viewing, annotating, training, benchmkaring, and deploying foundation-model segmentation workflows across 3D biological data. AdaptFM unifies image vieweing, semi-automated annotation using tools like SAM2 and SAM3, segmentation foundation-model training and inference (supporting 9 models for inference, 7 for fine-tuning), image preprocessing (e.g. format conversion, image normalization), postprocessing, and benchmarking. Users can install all models directly from AdaptFM. Importantly, every module in AdaptFM - preprocessing, annotation, post-processing, training, inference, and benchmarking is customizable, enabling seamless integration of new algorithms and foundation models as they emerge. 











AdaptFM has 9 distinct modules highlighted below. Click the links for each to learn more about how to use each module.

1. **AdaptFM Image Manager:** To open an image in AdaptFM you can use the "Open Image" button in the top right corner. AdaptFM also supports Napari's native File > Open File workflow for opening images. You can also open images in AdaptFM by using Napari's built-in "Drag and Drop" functionality.

2. [**AdaptFM Annotation Manager:**](annotations.md) This widget lets users run different segmentation algorithms on images. This is designed to help users quickly create training data and/or ground truth benchmarking data. To run segmentation foundation model training/inference use the Segmentation Models module. 

3. [**AdaptFM Save Widget:**](annotations.md/#adaptfm-save-widget). After creating annotations, save labeled images using the AdaptFM save widget. This saves images in the format needed to launch training

4. [**AdaptFM Preprocessing Module:**](preprocessing.md) This widgets lets users run various preprocessing algorithms or file conversions on image data. 

5. **AdaptFM Segmentation Models Module:** This module has two main parts:
    - [Fine-tuning segmentation models](fine-tuning-models.md): Use this to fine-tune an existing foundation model on your data
    - [Running Inference with segmentation models](testing-models.md): Use this to test various models on your data

6. **AdaptFM Post Processing Module:** 

7. **AdaptFM Benchmarking Module:**

8. **AdaptFM Model Installer:** 


