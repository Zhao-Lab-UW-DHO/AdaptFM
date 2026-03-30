import os
from scipy.fftpack import fftn, fftshift,ifftn,ifftshift
from kneed import KneeLocator
import numpy as np
from skimage.filters import threshold_otsu


#You should first segment images using organoid_segmentation.py and running through nnUNetv2, and crop_images.py
#place sample image into a folder and use this script to autosegment it

#function for computing the 3d power spectrum of an image

def get_3d_power_spectrum(image):

  image = image.astype(float) #ensure that we have a float
  fft_image = fftn(image) #get the Fourier transform of the image and shift it 
  fft_image = fftshift(fft_image) 
  power_spectrum = np.abs(fft_image)**2 #square the shifted fft to get the power spectrum

  return power_spectrum

#helper function for identifying the knee point of the power spectrum

def radial_average_3d(power_spectrum):
    """
    Computes the radial average of a 3D power spectrum while maintaining an anisotropic scaling factor for z.
    Do this because most images have a much smaller z dimnesion that x,y
    """
    dz,dy,dx = power_spectrum.shape
    center = (dz // 2, dy //2 , dx //2)
    z_scale = (dz/max(dy,dx)) # adjust for the aspect ratio
    z,y,x = np.indices([dz,dy,dx])
    aspect_ratio = dx/dy 

    # compute radial distances
    radius = np.sqrt((x - center[2]) ** 2 + ((y - center[1])* aspect_ratio) ** 2 + ((z - center[0]) * z_scale) ** 2)

    #flatten and sort power spectrum by radial distances
    radial_indices = np.argsort(radius.flat)
    radial_distances = radius.flat[radial_indices]
    sorted_power_spectrum = power_spectrum.flat[radial_indices]

        # Filter out zero power values
    nonzero_indices = sorted_power_spectrum > 0
    radial_distances = radial_distances[nonzero_indices]
    sorted_power_spectrum = sorted_power_spectrum[nonzero_indices]

    # Compute radial bins and average power per bin
    radial_bins = np.arange(0, np.max(radial_distances), 1)
    radial_power = np.zeros_like(radial_bins, dtype=np.float64)
    counts = np.zeros_like(radial_bins)

    for i, r in enumerate(np.floor(radial_distances).astype(int)):
        radial_power[r] += sorted_power_spectrum[i]
        counts[r] += 1

    counts[counts == 0] = 1  # Avoid division by zero
    radial_power /= counts  # Normalize

    return radial_bins, radial_power

  
#once you have the radial frequencies and power you can find at which frequency does power drop off

def detect_knee_3d(radial_frequencies,radial_power):

  nonzero_power_indices = radial_power > 0 #remove zero power regions b/c they are uninformative and can mess with knee detection
  filtered_frequencies = radial_frequencies[nonzero_power_indices]
  filtered_power = radial_power[nonzero_power_indices]

  knee_locator = KneeLocator(filtered_frequencies,filtered_power,curve='convex',direction='decreasing')
  return knee_locator.knee
  

def apply_knee_detection_3d(power_spectrum):

  radial_frequencies, radial_power = radial_average_3d(power_spectrum)
  knee_point = detect_knee_3d(radial_frequencies,radial_power)
  
  return knee_point


def apply_gaussian_low_pass_filter_3d(grayscale_image,cutoff, z_scaling=1.0):
    """
    Applies a Gaussian high-pass filter to a 3D image in the frequency domain.

    Parameters:
        image (np.ndarray): The 3D image to filter, shape (z, y, x).
        cutoff (float): The standard deviation (sigma) for the Gaussian filter.
        z_scaling (float): Factor to adjust the z-dimension distances (e.g., if z spacing is larger/smaller than xy spacing).

    Returns:
        np.ndarray: The filtered 3D image.
    """

    f_transform = fftn(grayscale_image)
    f_transform = fftshift(f_transform)

    depth, rows, columns = grayscale_image.shape
    cdepth, crow, ccol = depth // 2, rows // 2, columns // 2

    # Create a 3D mesh grid
    Z, Y, X = np.ogrid[:depth, :rows, :columns]
    
    # Adjust the distance metric to keep the z-ratio in check
    distance = np.sqrt(((X - ccol) ** 2) + ((Y - crow) ** 2) + ((Z - cdepth) ** 2) * (z_scaling ** 2))
    
    # Create the Gaussian low-pass filter
    gaussian_filter = np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))

    # Apply the Gaussian low-pass filter in the frequency domain
    f_transform *= gaussian_filter
    
    # Shift zero frequency component back
    f_transform_filtered = ifftshift(f_transform)
    
    # Perform inverse 3D Fourier transform
    filtered_image = np.abs(ifftn(f_transform_filtered))
    
    return filtered_image


def log_gabor_3d_filter(shape, f0, sigma_f):
    # Create a 3D meshgrid for the frequency domain
    z, y, x = np.meshgrid(np.arange(-shape[0]//2, shape[0]//2), 
                          np.arange(-shape[1]//2, shape[1]//2),
                          np.arange(-shape[2]//2, shape[2]//2),
                          indexing='ij')
    
    radius = np.sqrt(z**2 + y**2 + x**2)

    center_z, center_y, center_x = [dim // 2 for dim in radius.shape]
    radius[center_z, center_y, center_x] = 1
    
    # Create Log-Gabor filter
    log_gabor = np.exp(-(np.log(radius / f0) ** 2) / (2 * np.log(sigma_f) ** 2))
    log_gabor[radius < 1] = 0  # Remove low frequencies
    return log_gabor


def run_nuclear_segmentation(volume,
                             percentile,
                             max_freq,
                             frequency_step,
                             sigma,
                             remove_background: bool):
    
    #validate entries
    volume = volume.astype(np.float32) #important for FFT

    frequencies = np.arange(1,max_freq,frequency_step)

    num_z_layers = volume.shape[0]

    #if you want to try removing background noise use this code
    #obtain the power spectrum for the image
    #use the power spectrum and a knee point algorithm to identify the cutoff

    phase_response = []

    # # # Apply the filter in the frequency domain
    fft_img = fftshift(fftn(volume))

    #frequencies is a range that should be set by the user
    #sigma should be set by the user

    for f0 in frequencies:

        log_gabor = log_gabor_3d_filter(volume.shape, f0, sigma_f=sigma)
        filtered_fft = fft_img * log_gabor
        filtered_img = ifftn(filtered_fft)
        phase_response.append(np.angle(filtered_img))  # Extract phase 

    phase_sum = np.sum(np.exp(1j * np.array(phase_response)), axis=0)
    phase_congruency_map = np.abs(phase_sum) / len(phase_response)  # Normalize
    threshold = np.percentile(phase_congruency_map,percentile)
    segmented_nuclei = phase_congruency_map > threshold


    if remove_background is True:

        power_spectrum = get_3d_power_spectrum(volume)
        knee_point = apply_knee_detection_3d(power_spectrum)
        
        low_pass_image = apply_gaussian_low_pass_filter_3d(volume,cutoff=knee_point)
        foreground_binary_mask = np.zeros_like(low_pass_image)

        for z in range(num_z_layers):
            layer_threshold = threshold_otsu(low_pass_image[z,:,:])
            foreground_binary_mask[z,:,:] = low_pass_image[z,:,:] > layer_threshold

        segmented_nuclei = segmented_nuclei*foreground_binary_mask

    segmented_nuclei = segmented_nuclei.astype(np.uint8)


    return segmented_nuclei


