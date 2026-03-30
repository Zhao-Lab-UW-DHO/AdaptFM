import torch
import numpy as np
from skimage.filters import threshold_otsu
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from kneed import KneeLocator
import os
import tifffile as tiff

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPU-accelerated 3D power spectrum
def get_3d_power_spectrum(image):
    image = torch.tensor(image, dtype=torch.float32, device=device)
    fft_image = torch.fft.fftn(image)
    fft_image = torch.fft.fftshift(fft_image)
    power_spectrum = torch.abs(fft_image) ** 2
    return power_spectrum.cpu().numpy()  # convert back to numpy for compatibility

# GPU radial average
def radial_average_3d(power_spectrum):
    """
    Fully GPU-accelerated radial average of a 3D power spectrum.
    Maintains anisotropic z-scaling for typical volumetric images.
    """
    ps = torch.tensor(power_spectrum, dtype=torch.float32, device=device)
    dz, dy, dx = ps.shape
    center = (dz // 2, dy // 2, dx // 2)
    z_scale = dz / max(dy, dx)
    aspect_ratio = dx / dy

    # create 3D grid on GPU
    z, y, x = torch.meshgrid(
        torch.arange(dz, device=device),
        torch.arange(dy, device=device),
        torch.arange(dx, device=device),
        indexing='ij'
    )

    # compute radius with anisotropic scaling
    radius = torch.sqrt(
        (x - center[2]) ** 2 +
        ((y - center[1]) * aspect_ratio) ** 2 +
        ((z - center[0]) * z_scale) ** 2
    )

    # flatten
    radius_flat = radius.flatten()
    ps_flat = ps.flatten()

    # remove zero-power points
    mask = ps_flat > 0
    radius_flat = radius_flat[mask]
    ps_flat = ps_flat[mask]

    # bin indices (integer radii)
    bin_indices = torch.floor(radius_flat).long()

    # sum power per bin using bincount
    max_bin = bin_indices.max().item() + 1
    radial_power = torch.bincount(bin_indices, weights=ps_flat, minlength=max_bin)
    counts = torch.bincount(bin_indices, minlength=max_bin)

    counts[counts == 0] = 1  # avoid division by zero
    radial_power = radial_power / counts

    radial_bins = torch.arange(len(radial_power), device=device)

    return radial_bins.cpu().numpy(), radial_power.cpu().numpy()
# GPU-compatible knee detection (unchanged, CPU)
def detect_knee_3d(radial_frequencies, radial_power):
    nonzero_power_indices = radial_power > 0
    filtered_frequencies = radial_frequencies[nonzero_power_indices]
    filtered_power = radial_power[nonzero_power_indices]
    knee_locator = KneeLocator(filtered_frequencies, filtered_power, curve='convex', direction='decreasing')
    return knee_locator.knee

def apply_knee_detection_3d(power_spectrum):
    radial_frequencies, radial_power = radial_average_3d(power_spectrum)
    knee_point = detect_knee_3d(radial_frequencies, radial_power)
    return knee_point

# GPU Gaussian low-pass filter
def apply_gaussian_low_pass_filter_3d(grayscale_image, cutoff, z_scaling=1.0):
    img = torch.tensor(grayscale_image, dtype=torch.float32, device=device)
    fft_img = torch.fft.fftn(img)
    fft_img = torch.fft.fftshift(fft_img)

    dz, dy, dx = img.shape
    z, y, x = torch.meshgrid(torch.arange(dz, device=device),
                             torch.arange(dy, device=device),
                             torch.arange(dx, device=device),
                             indexing='ij')
    cz, cy, cx = dz // 2, dy // 2, dx // 2
    distance = torch.sqrt((x - cx)**2 + (y - cy)**2 + ((z - cz) * z_scaling)**2)
    gaussian_filter = torch.exp(-(distance**2) / (2 * (cutoff**2)))
    fft_img *= gaussian_filter

    fft_img = torch.fft.ifftshift(fft_img)
    filtered_img = torch.abs(torch.fft.ifftn(fft_img))
    return filtered_img.cpu().numpy()

# GPU Log-Gabor filter
def log_gabor_3d_filter(shape, f0, sigma_f):
    z, y, x = torch.meshgrid(
        torch.arange(-shape[0]//2, shape[0]//2, device=device),
        torch.arange(-shape[1]//2, shape[1]//2, device=device),
        torch.arange(-shape[2]//2, shape[2]//2, device=device),
        indexing='ij'
    )
    radius = torch.sqrt(z**2 + y**2 + x**2)
    center = [s // 2 for s in radius.shape]
    radius[center[0], center[1], center[2]] = 1
    log_gabor = torch.exp(-(torch.log(radius / f0)**2) / (2 * (torch.log(torch.tensor(sigma_f, device=device))**2)))
    log_gabor[radius < 1] = 0
    return log_gabor

# GPU-enabled nuclear segmentation
def run_nuclear_segmentation_gpu_chunked(volume, percentile, max_freq, frequency_step, sigma, remove_background: bool, chunk_size=5):
    """
    Fully GPU-accelerated nuclear segmentation with chunked frequency processing
    to avoid out-of-memory errors.
    """
    vol = torch.tensor(volume, dtype=torch.float32, device=device)
    dz, dy, dx = vol.shape
    frequencies = np.arange(1, max_freq, frequency_step)
    fft_vol = torch.fft.fftshift(torch.fft.fftn(vol))
    
    # Precompute coordinate grid for log-Gabor
    z, y, x = torch.meshgrid(
        torch.arange(-dz//2, dz//2, device=device),
        torch.arange(-dy//2, dy//2, device=device),
        torch.arange(-dx//2, dx//2, device=device),
        indexing='ij'
    )
    radius = torch.sqrt(z**2 + y**2 + x**2)
    center = (dz//2, dy//2, dx//2)
    radius[center[0], center[1], center[2]] = 1  # avoid log(0)

    phase_response_list = []

    # Process frequencies in chunks
    for i in range(0, len(frequencies), chunk_size):
        freq_chunk = frequencies[i:i+chunk_size]
        log_gabor_chunk = []
        for f0 in freq_chunk:
            lg = torch.exp(-(torch.log(radius / f0) ** 2) / (2 * (torch.log(torch.tensor(sigma, device=device))**2)))
            lg[radius < 1] = 0
            log_gabor_chunk.append(lg)
        log_gabor_chunk = torch.stack(log_gabor_chunk, dim=0)  # (F_chunk, Z, Y, X)
        fft_vol_chunk = fft_vol.unsqueeze(0).expand(len(freq_chunk), -1, -1, -1)
        
        # Apply filters and IFFT
        filtered_imgs_chunk = torch.fft.ifftn(fft_vol_chunk * log_gabor_chunk)
        phase_response_list.append(torch.angle(filtered_imgs_chunk).cpu())

        # free GPU memory immediately
        del log_gabor_chunk, fft_vol_chunk, filtered_imgs_chunk
        torch.cuda.empty_cache()

    # concatenate all phase responses
    phase_response = torch.cat(phase_response_list, dim=0)  # (F, Z, Y, X)
    phase_sum = torch.sum(torch.exp(1j * phase_response), axis=0)
    phase_congruency_map = np.abs(phase_sum) / len(frequencies)

    threshold = np.percentile(phase_congruency_map, percentile)
    segmented_nuclei = phase_congruency_map > threshold

    # Optional background removal
    if remove_background:
        power_spectrum = get_3d_power_spectrum(volume)
        knee_point = apply_knee_detection_3d(power_spectrum)
        low_pass_image = apply_gaussian_low_pass_filter_3d(volume, cutoff=knee_point)
        foreground_binary_mask = np.zeros_like(low_pass_image)
        for z in range(dz):
            layer_threshold = threshold_otsu(low_pass_image[z, :, :])
            foreground_binary_mask[z, :, :] = low_pass_image[z, :, :] > layer_threshold
        segmented_nuclei = segmented_nuclei * foreground_binary_mask

    if isinstance(segmented_nuclei, torch.Tensor):
        segmented_nuclei = segmented_nuclei.cpu().numpy()
    return segmented_nuclei.astype(np.uint8)