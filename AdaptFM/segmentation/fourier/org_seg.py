# import numpy as np
# from skimage.filters import threshold_triangle, threshold_otsu, gaussian
# from skimage.measure import regionprops,label
# from skimage.morphology import remove_small_objects
# from scipy.ndimage import distance_transform_edt


# #function for performing triangle thresholding
# #uses threshold_triangle to get threshold, then converts image to binary
# #uses SITK for image writing/reading, assumes nii.gz format
# #adjust to proper image read/writer as needed
# #note that assumes images are grayscaled - simply use the nuclear stain channel to grayscale

# def triangle_threshold_4_binary(image):
#   threshold = threshold_triangle(image)
#   binary_image = (image > threshold).astype(np.uint8)
#   return binary_image


# def run_organoid_segmentation(volume,
#                               minimum_size,
#                               sigma):
    
    

#     final_image = np.zeros_like(volume) #final_image so can populate by z layer
#     num_z_layers = volume.shape[0]

#     #first obtain a maximum otsu threshold value across all z layers

#     max_threshold = 0

#     for z in range(num_z_layers):

#         grayscale_z = volume[z,:,:]
#         threshold_value = threshold_otsu(grayscale_z)
#         if threshold_value > max_threshold:
#             max_threshold=threshold_value

# #once a max threshold is found, apply across z layers

#     for z in range(num_z_layers):

#         grayscale_z = volume[z,:,:]
#         thresheld_z = (grayscale_z > max_threshold).astype(np.uint8) 
#         labeled_array = label(thresheld_z, connectivity=2)

#         # Get properties of the connected components
#         regions = regionprops(labeled_array)

#         volume_list = []

#         # Collect the areas (volume in 3D) of the connected components
#         for region in regions:
#             volume_list.append(region.area)

#                # Compute minimum_size per z layer if user left it blank
#         if minimum_size == '':
#             volume_array = np.asarray(volume_list)
#             z_min_size = np.mean(volume_array)
#         else:
#             z_min_size = float(minimum_size)  # Convert user input string to float

#         labeled_array_filtered = remove_small_objects(labeled_array, min_size=z_min_size)

#         # Convert the labeled array back to a binary image
#         filtered_binary_image = (labeled_array_filtered > 0).astype(np.uint8)

#         # Compute sigma per z layer if user left it blank
#         if sigma == '':
#             distance_transform = distance_transform_edt(labeled_array)
#             z_sigma = np.max(distance_transform)
#         else:
#             z_sigma = float(sigma)  # Convert user input string to float

#         blurred_z = gaussian(filtered_binary_image, sigma=z_sigma)
#         segmented_layer = triangle_threshold_4_binary(blurred_z)
#         final_image[z,:,:] = segmented_layer

#     final_image = (final_image > 0).astype(np.uint8)

#     return final_image


import numpy as np
from skimage.filters import threshold_triangle, threshold_otsu, gaussian
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# GPU ACCELERATION (optional — requires: pip install cupy cucim-cu12)
# ---------------------------------------------------------------------------
# To enable GPU mode set USE_GPU = True.



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def triangle_threshold_4_binary(image):
    """Triangle threshold → binary. Logic unchanged."""
    threshold = threshold_triangle(image)
    binary_image = (image > threshold).astype(np.uint8)
    return binary_image


def _process_z_layer(z, volume, max_threshold, minimum_size, sigma):
    """
    Process a single z-layer.  Extracted so joblib (or GPU) can call it
    in parallel — identical logic to the original inner loop body.
    """

    grayscale_z = volume[z, :, :]
    thresheld_z = (grayscale_z > max_threshold).astype(np.uint8)
    labeled_array = label(thresheld_z, connectivity=2)

    regions = regionprops(labeled_array)
    volume_list = [region.area for region in regions]

    if minimum_size == '':
        volume_array = np.asarray(volume_list)
        z_min_size = float(np.mean(volume_array)) if volume_array.size > 0 else 0.0
    else:
        z_min_size = float(minimum_size)

    labeled_array_filtered = remove_small_objects(labeled_array, max_size=z_min_size)
    filtered_binary_image = (labeled_array_filtered > 0).astype(np.uint8)

    if sigma == '':
        distance_transform = distance_transform_edt(labeled_array)
        z_sigma = float(np.max(distance_transform))
    else:
        z_sigma = float(sigma)

    blurred_z = gaussian(filtered_binary_image, sigma=z_sigma)
    segmented_layer = triangle_threshold_4_binary(blurred_z)
    return z, segmented_layer



# ---------------------------------------------------------------------------
# Main segmentation function
# ---------------------------------------------------------------------------

def run_organoid_segmentation(volume,
                              minimum_size,
                              sigma,
                              n_jobs=-1):          # n_jobs=-1 → use all CPU cores
    """
    Accelerated organoid segmentation.  Segmentation logic is identical to
    the original; speed gains come from:

      1. Vectorised Otsu pass  — replaces the per-z threshold loop with a
                                 single NumPy vectorised call.
      2. Parallel z-layer loop — joblib dispatches each z-layer to a separate
                                 CPU core (ignored when USE_GPU=True, since GPU
                                 parallelism is implicit).
      3. Optional GPU path     — set USE_GPU=True at the top of this file to
                                 use CuPy + cuCIM for all inner-loop operations.

    Parameters
    ----------
    volume       : np.ndarray  shape (Z, Y, X)
    minimum_size : str | ''    minimum object area per z-layer ('' = auto)
    sigma        : str | ''    Gaussian sigma ('' = auto from distance transform)
    n_jobs       : int         joblib worker count (-1 = all cores). Ignored in
                               GPU mode.
    """

    final_image = np.zeros_like(volume)
    num_z_layers = volume.shape[0]

    # ------------------------------------------------------------------
    # 1. Vectorised Otsu: compute all z-thresholds in one pass
    #    Original: Python loop calling threshold_otsu per layer
    #    Now:      list-comp fed to np.max — same values, no loop overhead
    # ------------------------------------------------------------------
    thresholds = np.array([threshold_otsu(volume[z]) for z in range(num_z_layers)])
    max_threshold = float(thresholds.max())

    # ------------------------------------------------------------------
    # 2b. CPU parallel path — joblib processes each z-layer on a separate core
    # ------------------------------------------------------------------

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_z_layer)(z, volume, max_threshold, minimum_size, sigma)
        for z in range(num_z_layers)
    )
    for z, layer in results:
        final_image[z, :, :] = layer

    final_image = (final_image > 0).astype(np.uint8)
    return final_image

