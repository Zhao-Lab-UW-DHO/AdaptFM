# HighFreqSeg

HighFreqSeg is a custom segmentation technique that identifies small, fine-scale structures and intensity changes within an image. It is particularly useful for detecting small objects such as individual cells or nuclei, especially in confocal microscopy images with a predominantly dark background. 

***In AdaptFM, the algorithm is applied independently in each 2D z-slice. The resulting masks from all slices are then combined to produce a final 3D segmentation***

## How it works

The steps of HighFreqSeg are:

1. **Examine the image's spatial frequencies** - the algorithm analyzes the image to determine how much information is present at different scales, from broad intensity changes to fine details
2. **Identify the transition between broad and fine features** - the algorithm examines the distribution of frequencies in the image and identifies a cutoff point separating lower-frequency regions from higher-frequency regions
3. **Remove broad image features** - a 'high-pass' filter is applie to remove low-frequency information such as gradual changes in brightness, large-scale background variation, and other broad structures. 
4. **Keep fine image features** - The remaining image emphasizes sharp boundaries, small structures, and rapid changes in intensity that occur at higher spatial frequencies
5. **Select the strongest features** - A percentile threshold is applied to the high-frequency image. Only pixels with sufficiently strong high-frequency responses are classified as foreground.
6. **Connect neighboring pixels into objects** - The detected foreground pixels are grouped into connected regions to identify individual candidate objects.
7. **Remove small regions** - Regions below the specified minimum size are removed to reduce noise and unwanted detections.

## Parameters 

HighFreqSeg has two adjustable parameters: percentile and minimum size

1. **Percentile** - controls how strong a feature must be before it is included in the segmentation
    - Lower percentile - includes more pixels and detects weaker features
    - Higher percentile - includes only the strongest features and produces a more selective segmentation

2. **Minimum size** - controls which detected regions are retained based on their size
    - Lower minimum size - keeps smaller objects
    - Higher minimum size - removes more small regions adn suppresses small noise

