# Segment Anything 2 (SAM2)

Segment Anything Model 2 is a foundation model developed by [facebook research](https://github.com/facebookresearch/sam2) used for identifying objects in an image. Users can interactively identify objects by clicking on them in individual image slices. Because of its strong performance across numerous domains, we have incorporated SAM2 into AdaptFM so that users can rapidly generate labeled data for model training. SAM2 was designed to allow segmentation in both images and videos. 

***SAM2 allows users to segment objects by clicking on them. For 3D images, AdaptFM can either propagate that segmentation through the z-axis by treating the slices like frames in a video (using SAM2's video propagation capability), or independently segment each 2D slice.***

## How it works

SAM2 works by following the below steps

1. **Encode image** - this runs the image through the SAM2 model. This will not segment the image directly. It simply 'preps' the image so when a user clicks object they will be segmented. 
2. **Select objects in a layer** - The user clicks on objects of interest within a 2D image slice. This will automatically generate a segmentations for the objects the user selected 
3. **Extend the segmentation through the volume (optional)** - Because a 3D image consists of a series of 2D slices, users can choose to propagate teh segmentation across neighboring z-layers. SAM2 treats each z-layer as a separate 'frame' and tracks the selected object as it changes from slice to slice. 
4. **Review and refine** - The resulting segmentation can be reviewed and modified by the user as needed.

## Parameters

Wthin AdaptFM there are several adjustable parameters when using SAM2: model_size, propagation_direction, score_threshold, multimask_output, and GPU

1. **model_size** - SAM2 has four different base models that users can select, tiny, small, base_plus, and large. Generally speaking, larger models will run slower and use more memory, but should produce better segmentations. **In our experience, it is faster to use a smaller model (i.e. tiny) and manually refine segmentations if needed, rather than using a larger model.**
2. **propgataion_direction** - 