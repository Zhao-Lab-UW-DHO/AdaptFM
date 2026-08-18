# Canny Edge Detection (3D)

Canny edge detection is an image-processing method used to identify boundaries between objects or regions of an image. The algorithm works by looking for locations where image intensity changes rapidly. For example, the boundary of an organoid or cell may appear brighter or darker than the surrounding background. Canny detects these changes and marks them as potential object boundaries. 

***Although typically a 2D algorithm, AdaptFM applies the algorithm to each individual z layer independently. The final segmentation are the stacked results of these individual 2D segmented planes***

## How it works

The general steps of Canny edge detection are:

1. **Blur the image** - the image is slightly blurred to reduce noise
2. **Find changes in intensity** - the algorithm finds places where there are sharp changes in intensity
3. **Identify strong edges** - Determines whether these boundaries represent true boundaries or noise
4. **Connect the edges** - Edge pixels are connected to make continuous boundaries
5. **Stack the results** - steps 1-4 are applied on each individual z layer. The final segmentation is produced by stacking these z layers together. 

## Parameters
Canny edge detection has three parameters that control how the algorithm identifies object boundaries: sigma, low threshold, and high threshold. 

1. **Sigma** – controls how much the image is 'smoothed' or 'blurred' before detecting edges.
    - Lower sigma – Less smoothing. More fine details and small edges are detected. However, if set too low, noise might also be detected as edges.
    - Higher sigma – More smoothing. Produces fewer small or noisy edges. However, subtle boundaries might be lost if set too high.

2. **Low Threshold** – controls how sensitive the algorithm is to relatively weak boundaries.
    - Lower low threshold – More weak boundaries are detected, but there might be more unwanted edges.
    - Higher low threshold – Only more noticeable boundaries are retained.

3. **High Threshold** – how strong an image boundary must be to be considered an edge.
    - Lower high threshold – more boundaries are considered edges.
    - Higher high threshold – only very clear, high‑contrast boundaries are detected.

