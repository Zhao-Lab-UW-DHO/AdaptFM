# Segment Anything 3 (SAM3)

Segment Anything Model 3 (SAM3) is a foundation model developed by [facebook research](https://github.com/facebookresearch/sam3) used for identifying objects in images and videos by using text concepts and click prompts. Users can segment objects by entering a text-concept (e.g. football) then further refine segmentations by clicking objects, similar to [Segment Anything 2](./SAM2.md)

***SAM3 allows users to segment objects in a 2D image slice by entering text concepts or clicking on objects of interest. For 3D images, AdaptFM can either propagate that segmentation through the z-axis by treating the slices like frames in a video (using SAM3's video propagation capability), or independently segment each 2D slice.***

## How it works

1. **Encode image** - this runs the image through the SAM2 model. This will not segment the image directly. It simply 'preps' the image so when a user enters a text concept objects will be segmented
2. **Enter a text concept** - users can enter a text-concept, (e.g. 'cells') and allow the algorithm to find those objects in the current 2D slice of the image. 
3. **Click objects to segment** - once the text concept has been entered and segmented objects, users can additionally segment objects by clicking on them
4. **Extend the segmentation through the volume (optional)** - Because a 3D image consists of a series of 2D slices, users can choose to propagate teh segmentation across neighboring z-layers. SAM3 treats each z-layer as a separate 'frame' and tracks the selected object as it changes from slice to slice.  
5. **Review and refine** - The resulting segmentation can be reviewed and modified by the user as needed.

## Parameters

Wthin AdaptFM there are several adjustable parameters when using SAM3: 

1. **Segment click mode** - when checked, clicking on the image will segment objects. Be sure to uncheck if clicking on the image for other reasons. 
2. **Initialise (encode all slices)** - this will run the image through the SAM3 model. Once the image is encoded you can start segmenting objects in the image. 
3. **Text concept** - the text concept a user wants to identify in the image (for example, 'cells', 'nucleus', 'cell membrane')
4. **Text Obj ID** - the label ID that will be assigned to the objects found by 'Text Concept'
5. **Segment by text (current slice)** - once a user has typed a text concept in the "Text concept" field, they can press this button to generate segmentations
6. **Click Object ID** - the ID of the current object to be segmented via clicking
7. **Propagate through volume** - once a user is done segmenting objects in a 2D slice, they can select this to extend the segmentation through the volume. This works in coordination with 'propagation_direction' to extend the segmentation 'forward' (higher z layers), 'backward' (lower z layers), or in both directions.
8. **Reset current object** - Allows users to remove the segmentation with the ID currently in "Click Object ID"
9. **Reset all** - Allows users to remove all segmentations from the image (across all layers)
10. **propgation_direction** - used with "Propagate through volume". 
    - Both - the segmentation will be extended across z layers in both directions
    - Forward - the segmentation will be extended across higher z layers
    - Backward - the segmentation will only be extended across lower z layers
11. **score_threshold** - how confident the model should be that a pixel is part of an object. Ranges from 0 to 1.0, with a default of 0.5 
    - Higher score_threshold - only pixels the model is very confident are part of the object will be included in the segmentation. Setting this too high might cause the model to miss pixels. 
    - Lower score_threshold - the model will include less confident pixels as part of the object. Setting this too low might cause the model to include too many pixels in the object. 
12. **text_conf_threshold** - how confident the model should be that a pixel is part of an object and matches the text concept. Ranges from 0 to 1 with a default of 0.25
    - Higher text_conf_threshold - only pixels the model is very confident match the text concept will be segmented. Setting this too high might cause the model to miss pixels
    - Lower text_conf_threshold - the model will include more pixels even if they match the text concept with less confidence. Setting this too low might cause the model to include too many pixels in the object.
11. **GPU** - The GPU you plan to run processing on

