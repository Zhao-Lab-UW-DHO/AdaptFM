# Segment Anything 3 (SAM3)

Segment Anything Model 3 (SAM3) is a foundation model developed by [facebook research](https://github.com/facebookresearch/sam3) used for identifying objects in images and videos by using text concepts and click prompts. Users can segment objects by entering a text-concept (e.g. football) then further refine segmentations by clicking objects, similar to [Segment Anything 2](./SAM2.md)

**For 3D images, AdaptFM can either propagate that segmentation through the z-axis by treating the slices like frames in a video (using SAM3's video propagation capability), or independently segment each 2D slice.***

## How it works

