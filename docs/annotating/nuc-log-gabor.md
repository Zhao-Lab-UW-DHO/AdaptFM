# NucLogGabor (3D)

NucLogGabor is a custom method helpful for identifying nuclei in organoid images. This method works by identifying structure by looking for features that remain consistent across multiple spatial scales. It is particularly useful for detecting objects like nuclei that have distinct shape or boundary compared with their surroundings. 

The algorithm examines the pattern and alignment of image features at various spatial scales. This makes it depend less on the absolute intensity of the image and can identify structures even when their brightness varies across an image. 

***NucLogGaborGPU runs identically to NucLogGabor, but can offer faster, GPU-accelerated processing***

## How It Works

The algorithm uses the following conceptual steps to segment a 3D image

1. **Analyze the image at multiple scales** - The image is processed using a series of filters, each designed to detect structure of a different spatial scale
2. **Measure feature consistency** - The algorithm examines whether features detected at different scales occur in same locations 
3. **Create a congruency map** - Locations where features are consistently detected at multiple scales are given higher scores
4. **Select likely objects** - A percentile threshold is applied to the congruency map to identify regions most likely to correspond to structures of interest
5. **Optionally remove background** - An additional background-removal step can restrict the segmentation to regions with meaningful image content

## Parameters

NucLogGabor has five parameters that control which structures are detected: percentile, maximum frequency, frequency step, sigma, and remove background

1. **Percentile** - controls how strong a feature must be before it is included in the final segmentation 
    - Lower percentile - more pixels are included and detects more potential structures
    - High percentile - includes only the strongest features and produces a more selective segmentation

2. **Maximum Frequency** - controls the upper end of the range of spatial frequencies examined by the algorithm. Spatial frequency can be thought of as the size of image details being examined
    - Lower maximum frequency - focuses more on larger structures
    - Higher maximum frequency - allows smaller detailed to be detected

3. **Frequency Step** - determines how far apart the different spatial scales are from each other. In other words, how many spatial frequencies should be examined. 
    - Lower frequency step - examines more scales and provides a more detailed analysis across different feature sizes
    - Larger frequency step - examines fewer scales and can make the algorithm faster, but might miss some smaller features sizes

4. **Sigma** - how narrowly or broadly the algorithm looks for structures around each selected scale 
    - Lower Sigma - more narrowly focused on specific feature sizes
    - Higher sigma - responds to a broader range of feature sizes around each selected scale

5. **Remove Background** - determines whether the algorithm performs an additional step to exclude areas that appear to contain only background 
    - Enabled - removes detected features that fall outside the estimated foreground region
    - Disabled - keeps all features detected by the congruency map, including features that might occur in background areas. 

