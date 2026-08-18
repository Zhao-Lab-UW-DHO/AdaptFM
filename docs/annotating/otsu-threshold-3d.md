# Otsu Thresholding (3D)

Otsu thresholding is a technique that automatically separates an image into foreground and background based on pixel intensity. It determines a single intensity threshold that best separates the darker and brighter portions of the image. 

***In AdaptFM Otsu threhsolding uses one threshold value for the entire 3D volume.***

The method works particularly well when the image has a relative clear distinction between the intensity of the background and the structures of interest.

## How it works

Otsu thresholding identifies objects through the following steps:

1. **Examine the image intensities** - the algorithm looks at the range of pixel intensities throughout the image and determines how frequently each intensity occurs
2. **Consider possible thresholds** - the algorithm considers different intensity values as potential boundaries between background and foreground
3. **Separate the image into two groups** - For each possible threshold, pixels are divided into two groups:
    - pixels below the threshold are made background
    - pixels above the threhsold are made foreground
4. **Find the best separation** - the algorithm evaluates how well each possible threshold separates the two groups. 
5. **Select the optimal threshold** - the intensity value that provides the best separation is selected as the Otsu threshold
6. **Generate the segmentation** - Every pixel above the selected threshold is classified as foreground, while every pixel below the threshold is classified as background
7. **Remove small regions** - After thresholding, very small isolated regions can be removed to reduce noise and unwanted detections

## Parameters

Otsu thresholding has only one adjustable parameter in AdaptFM: remove small objects

1. **Remove Small Objects** - specifies the minimum size of regions that should be retained after thresholding
    - Lower value - keeps smaller regions
    - Higher value - removes more small regions and produces a segmentation with fewer objects

