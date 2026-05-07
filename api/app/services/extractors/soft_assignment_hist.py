import numpy as np
from typing import Tuple

def _soft_assignment_hist(data: np.ndarray, bins: int, range_val: Tuple[float, float]) -> np.ndarray:
    """Linear interpolation histogram (Soft Assignment)"""
    hist = np.zeros(bins)
    min_val, max_val = range_val
    data_clipped = np.clip(data, min_val, max_val - 1e-7)
    
    # Calculate bin index and weights
    bin_width = (max_val - min_val) / bins
    idx_float = (data_clipped - min_val) / bin_width - 0.5
    
    idx_low = np.floor(idx_float).astype(int)
    idx_high = idx_low + 1
    
    weight_high = idx_float - idx_low
    weight_low = 1.0 - weight_high
    
    # Boundary handling
    idx_low = np.clip(idx_low, 0, bins - 1)
    idx_high = np.clip(idx_high, 0, bins - 1)
    
    # Accumulate
    np.add.at(hist, idx_low, weight_low)
    np.add.at(hist, idx_high, weight_high)
            
    return hist / (np.sum(hist) + 1e-7)
