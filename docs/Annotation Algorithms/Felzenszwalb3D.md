# Felzenszwalb Segmentation (3D)

Felzenszwalb segmentation is an image-processing technique that divides an image into region of pixels that have similar characteristics.  The algorhtm examines neighboring pixels and groups them together when they have similar image characteristics. Regions with larger differences between them are more likely to be separated. 

***For 3D images in AdaptFM, the algorithm is applied independently to each 2D slice. The resulting regions from all slices are then combined into the final 3D label image***

## Parameters 

Felzenszwalb segmentation has three parameters that control how regions are created: scale, sigma, and minimum size.

1. **Scale** - controls how aggressively the algorithm combines pixels into larger regions
    - Lower scale - produces more, smaller regions and preserves finer details
    - Higher scale - produces fewer, larger regions by combining regions that are more similar

2. **Sigma** - controls how much the image is smoothed/blurred before segmentation
    - Lower sigma - preserves fine detail, but might produce more small, noisy regions
    - Higher sigma - produces a smoother image, and can reduce fragmentation. Subtle structure might get lost if set too high

3. **Minimum size** - specifies the smallest region that will be kept in the final segmentation
    - Lower minimum size - keeps smaller regions and finer structures
    - Higher minimum size - removes smaller objects, which can eliminate noise or tiny objects

