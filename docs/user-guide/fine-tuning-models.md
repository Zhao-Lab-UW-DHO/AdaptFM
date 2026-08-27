# AdaptFM Training Module

AdaptFM supports GUI-based fine-tuning/training for various models. If you are new to training/fine-tuning jump [here](#what-is-trainingfine-tuning). 

***All image normalization, rescaling, and tokenization are handled by each model according to its native preprocessing pipeline; AdaptFM does not impose additional image preprocessing***

The following models allow for fine-tuning and are supported within AdaptFM:

[CellposeSAM](../using-segmentation-models/cellposesam.md) - a deep learning model designed to identify and segment cell and other biological structures in microscopy.

[MicroSAM](../using-segmentation-models/microsam.md) - a deep learning model designed to segment objects in microscopy images

[SSVT](../using-segmentation-models/ssvt.md) - a custom vision transformer model pretrained on ~180,000 3D organoid confocal microscopy images.

[BME-X](../using-segmentation-models/bmex.md) - a foundation model designed for magnetic resonance images (MRI). Has a segmentation model for MRI Brains.

[nnUNetV2](../using-segmentation-models/nnunetv2.md) - a machine-learning framework designed to learn how to identify and segment structures in medical and biological images.

[SAM-Med3D](../using-segmentation-models/sammed3d.md) - a deep learning model designed to segment objects in 3D medical images (e.g. CT)

[CTFM](../using-segmentation-models/ctfm.md) - a 3D image-based pretrained foundation model for a number of radiological segmentation tasks

**Follow the instructions in the above links to launch training for each model.** All models follow the same general steps. Begin by selecting the "Segmentation Models" –> "Training" at the top of AdaptFM:

1. Use the 'Model' dropdown to select the model you would like to train or fine-tune.
2. Under 'Dataset' choose the folder you saved files to while making [annotations](annotations.md). If you made annotations outside of AdaptFM ensure
      - All images (original and annotations) are placed together in one folder
      - All images are in the proper format for the specified model (see the above links)
      - Annotations file names end with '_seg.tiff'
3. Under 'Output Directory' choose the folder where you would like to save your training progress/data. 
4. Use the 'GPU Index' field to choose which GPU you would like to use. If your system has only one GPU this will be chosen automatically
5. If available, select "Load Parameters" to see the model's hyperparameters. For detailed descriptions of the most common parameters, use the above link to see descriptions of the model's parameters.
6. Click "Run" to launch training. You can continue to monitor progress in the output log. 

![training-widget](../../asset/training-widget.png)


## What is training/fine-tuning?

Training or fine-tuning allows a segmentation model learn to segment objects in your specific type of images. This is helpful when a model does not perform well on your images out-of-the-box. **Before fine-tuning a model in AdaptFM, first assess the model's performance out-of-the-box.** AdaptFM uses the images and annotation you provide as examples for the model to learn from

When making annotations it is important to provide accurate examples of the objects you want the model to find, as well as representative examples of different appearances and challenging cases. For example, if segmenting cells, the training data should include cells of different sizes, shapes, intensities, and degress of overlap. 

### How much training data is needed

There is no universal number of images required for successful fine-tuning/training. The amount of data needed depends on the complexity of the segmentation task, the variability of the images, how different your images are from the model's original training data, quality of the annotations, among other factors. 

As a practical starting point, we recommend beginning with approximately **50-100 representative annotated images**. Some tasks may require fewer, while others might require more. If performance is still insufficient, analyze the types of objects.

More data is not necessarily better if the annotations are repetitive or unrepresentative. A smaller set of high-quality annotations covering the range of objects and image conditions encounterd in your application can be more useful than a much larger set of nearly identical images. Try to include the full spread of diversity in your dataset when making training examples. 
