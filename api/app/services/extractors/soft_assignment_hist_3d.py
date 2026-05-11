import numpy as np
from typing import Tuple

def _soft_assignment_hist_3d(img: np.ndarray, bins: int = 4, range_val: Tuple[float, float] = (0, 256)) -> np.ndarray:
    """
    3D Trilinear Soft Assignment Histogram.
    Distributes weight among 8 nearest neighbors in 3D color space.
    """
    hist = np.zeros((bins, bins, bins), dtype=np.float32)
    min_val, max_val = range_val
    
    # img shape (H, W, 3)
    data = img.reshape(-1, 3).astype(np.float32)
    data = np.clip(data, min_val, max_val - 1e-7)
    
    # Bin width
    w = (max_val - min_val) / bins
    
    # Calculate float coordinates relative to bin centers
    # Bin center k is at (k + 0.5) * w
    coords = (data / w) - 0.5
    
    # Floor to get the lower bin index
    idx_low = np.floor(coords).astype(np.int32)
    idx_high = idx_low + 1
    
    # Calculate weights for the high bin
    # If coords = 0.7, idx_low = 0, weight_high = 0.7
    w_high = coords - idx_low
    w_low = 1.0 - w_high
    
    # Clip indices to valid range
    idx_low = np.clip(idx_low, 0, bins - 1)
    idx_high = np.clip(idx_high, 0, bins - 1)
    
    # Iterate through 8 corners (000 to 111)
    # Using numpy indexing for speed
    for i in [0, 1]:
        xi = idx_low[:, 0] if i == 0 else idx_high[:, 0]
        wi = w_low[:, 0] if i == 0 else w_high[:, 0]
        
        for j in [0, 1]:
            yi = idx_low[:, 1] if j == 0 else idx_high[:, 1]
            wj = w_low[:, 1] if j == 0 else w_high[:, 1]
            
            for k in [0, 1]:
                zi = idx_low[:, 2] if k == 0 else idx_high[:, 2]
                wk = w_low[:, 2] if k == 0 else w_high[:, 2]
                
                # Combine weights
                weight = wi * wj * wk
                
                # Accumulate into histogram
                # np.add.at handles duplicate indices correctly
                np.add.at(hist, (xi, yi, zi), weight)
                
    # Normalize
    total = np.sum(hist)
    if total > 0:
        hist /= total
        
    return hist.flatten()
