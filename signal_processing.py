import numpy as np
import scipy.ndimage as ndimage
import config

def add_awgn_noise(signal_map, std_dev=config.NOISE_STD_DEV):
    """
    Adds Additive White Gaussian Noise to the simulated reflection response
    to mimic real-world sensor thermal noise and environmental variability.
    """
    noise = np.random.normal(0, std_dev, signal_map.shape)
    noisy_signal = signal_map + noise
    
    # Clip to valid magnitude range [0, 1] for Gamma
    noisy_signal = np.clip(noisy_signal, 0, 1)
    return noisy_signal

def normalize_response(signal_map):
    """
    Normalizes the response map to the [0, 255] range for 
    subsequent image processing techniques.
    """
    min_val = np.min(signal_map)
    max_val = np.max(signal_map)
    
    if max_val == min_val:
        return np.zeros(signal_map.shape, dtype=np.uint8)
        
    normalized = 255.0 * (signal_map - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)

def apply_gaussian_filter(signal_map, sigma=1.5):
    """
    Applies a Gaussian filter to smooth the raw RF data and reduce high-frequency noise.
    This acts as a basic 2D signal processing low-pass filter.
    """
    filtered_signal = ndimage.gaussian_filter(signal_map, sigma=sigma)
    return filtered_signal
