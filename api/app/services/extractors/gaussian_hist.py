import numpy as np
from typing import Tuple

def _gaussian_hist(data: np.ndarray, bins: int, range_val: Tuple[float, float], sigma: float = None) -> np.ndarray:
    """Kernel-based histogram (Gaussian)"""
    hist = np.zeros(bins)
    min_val, max_val = range_val
    bin_centers = np.linspace(min_val, max_val, bins)
    if sigma is None:
        sigma = (max_val - min_val) / bins # Default sigma
        
    # Vectorized gaussian
    for i in range(bins):
        dist_sq = (data - bin_centers[i])**2
        weights = np.exp(-dist_sq / (2 * sigma**2))
        hist[i] = np.sum(weights)
        
    return hist / (np.sum(hist) + 1e-7)
