# AdaptFM's Annotation Widget

You can use the annotation widget within AdaptFM to quickly create labeled data for training or evaluating models. The annotation algorithm supports a number of segmentation algorithms. These algorithms are a starting point. You can use these to get a starting segmentation of the image, and modify it using Napari's built in image editors. 

***These algorithms are intended for annotating images, not performing large-scale automated segmentation. For large-scale segmentation please see the supported foundation models.***

***In our experience, SAM2 offers the best and fastest results for image segmentation for most use cases. Be sure to install SAM2 using the AdaptFM Model Installer prior to using it.***

The annotation widget has 4 main parts:


1. **Algorithm Selector** - Use this drop down to select the algorithm you would like to use for segmentation. To learn more about each algorithm, how it works, and what their parameters do, click on the below links. 

***Because of the variety of medical images, we recommend quickly testing several of the below algorithms to see which works best for your data and segmentation needs***

[Canny Edge Detection 3D](../annotating/canny-edge-3d.md)
[Felzenszwalb 3D](../annotating/felzenszwalb-3d.md)
[High Frequency Segmentation](../annotating/high-freq-seg.md)
[Nuclear Log Gabor Segmentation](../annotating/nuc-log-gabor.md)
[Organoid Seg](../annotating/organoid-seg.md)
[Otsu Thresholding 3D](../annotating/otsu-threshold-3d.md)
[Segment Anything 2](../annotating/sam2.md)
[Segment Anything 3](../annotating/sam3.md)
[Sauvola Thresholding 3D](../annotating/sauvola-threshold-3d.md)

2. **Adjustable Parameters** - When you change models, a set of adjustable parameters will appear in this section. These paraemters will change how the algorithm is applied, and can improve (or worsen) segmentation quality. Use the above links to see how each parameter will impact segmentations. 

3. **Run auto-segmentation** - Once you have a selected an algorithm and adjusted parameters, click this button to apply the algorithm **to the image currently opened in Napari**. Importantly, this button is not available in SAM2 and SAM3. Those algorithms allow interactive segmentation (segmentation by clicking or entering a text concept)

4. **Run auto-segmentation on folder** - This button will let you run the selected algorithm (and parameters) on an entire folder of images. When selecting this option you will be prompted for the folder you would like to process. **You can adjust the save location and whether or not to save the original image alongside the segmentation by adjusting the save widget**. 

# AdaptFM's Save Widget

Once you are ready to save your image, you can use the AdaptFM Save Widget to save the labeled image. We recommend using this widget for saving, as it will save your data in the format required for training. 


The AdaptFM has 4 main settings

1. **Save Location** - the directory where you would like to save the image
2. **Filename** - the name given to the file when saving. Importantly, labeled images will have '_seg' attached to the name in order to automate training. 
3. **Segmentation Layer** - which labeled layer you would like to save
4. **Save Image Alongside** - when saving the labeled image, would you like to save the original image alongside. **We recommend checking this box if you plan to use your labeled data for training. This will save your data in a common folder that you can later select when launching training.**

