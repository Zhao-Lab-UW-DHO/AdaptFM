# Sauvola Threhsold (3D)

Sauvola thresholding is an image-processing technique that separates an image into foreground and background based on the local intensity of the image. Unlike a global threshold, which uses the same brightness cutoff for the entire image, Sauvola calculates a different threshold for different regions of the image. 

Sauvola thresholding is useful for images with uneven illumination, variable background intensity, or obejcts whose brightness changes across an image. 

***For images in AdaptFM, the algorithm is applied independently to each 2D slice. The resulting binary masks from all slices are combined into the final 3D segmentation***

## How it works

The general steps of Sauvola thresholding are:

1. **Examine the local neighborhood** - For each pixel, the algorirthm looks at a surrounding window of pixels rather than the entire image. 
2. **Measure local brightness** - determine the typical brightness and amount of variation within that neighborhood
3. **Calculate a local threshold** - a threshold is calculated specifically for that region based on its local brightness and contrast 
4. **Classify each pixel** - pixels brighter than their local threshold are classified as foreground, while darker pixels are classified as background
5. **Remove very small regions** - After thresholding, small isolated regions can be removed to reduce noise

## Parameters

Sauvola thresholding has three parameters that control how it detects objects: window size, k, and remove small objects

1. **Window size** - For each pixel, how large of a neighborhood does the algorithm 'look' to determine the local brightness and contrast of the image 
    - Smaller window size - the thresold adapts to smaller, more local changes in the image
    - Larger window size - the threshold is influenced by a larger surrounding area and changes more gradually across the image

2. **K** - determines how far the threshld is adjusted based on the characteristics of the surrounding region
    - Lower K - produces a less aggressive adjustment to the local threshold
    - Higher K - produces a stronger adjustment based on the local image characteristics

3. **Remove Small Objects** - specifies the minimum size of regions that should be retained after thresholding. 
    - Lower value - keeps smaller regions
    - Higher vale - removes more small regions and produces a cleaner mask



