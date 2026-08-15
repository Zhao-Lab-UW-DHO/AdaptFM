# OrganoidSeg

OrganoidSeg is a multi-step image processing method designed to identify organoids as foreground regions while excluding the surrounding background. This algorithm is particularly useful for identifying confocal microscopy images that have multiple organoids across multiple z layers in one image. 

The algorithm determines the approximate intensity of the background and foreground in each image slice. It then uses this information to identify potential organoid regions and applies additional processing to refine the segmentation. 

***In AdaptFM, this algorithm is run slice-by-slice in the z-direction. The individual 2D segmentation results are then combined to produce the final 3D segmentation***

## How it works

The steps of OrganoidSeg are:

1. **Examine each image slice** - The 3D image is divided into individual 2D z-slices so that each slice can be analyzed independently
2. **Determine the intensity threhsold** - Otsu thresholding is applied to each slice to estimate the intensity that separates the image background from brighter structures.
3. **Determine a common threshold** - The highest threshold identified across all slices is used as a reference for the subsequent processing. This helps maintain consistency across the volume.
4. **Identify potential organoid regions** - Each slice is processed to identify regions that are likely to correspond to organoids based on their intensity and spatial characteristics.
5. **Refine the segmentation** - Additional image-processing steps are used to refine the detected regions and remove or suppress structures that do not meet the desired object characteristics.
6. **Remove small regions** - Remove very small regions: Objects below the specified minimum size can be excluded from the segmentation.
7. **Combine the slices** - The processed 2D masks are combined back into a 3D volume to produce the final binary segmentation.

## Parameters

The algorithm has three parameters: minimum size, sigma, and number of CPU cores (n_jobs)

1. **Minimum size** - controls how small a detected object can be before it is excluded from the segmentation
    - Lower minimum size - smaller regions are retained in the segmentation
    - Higher minimum size - removes smaller regions and favors larger objects

2. **Sigma** - controls the amount of smoothing or blurring used during the object-processing steps
    - Lower sigma - preserves more fine detail in the image. 
    - Higher sigma - produces smoother regions and reduces sensitivity to small variations

3. **Number of CPU Cores** - determines how many CPU workers are used to process the image
    - Higher number - processes multiple slices simultaneously and can reduce processing time