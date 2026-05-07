import cv2
import numpy as np
from typing import List

def _extract_spm(img_bgr: np.ndarray) -> List[float]:
    """Spatial Pyramid Matching (160 dim: 1x1 + 2x2 grids with 32-bin histograms)"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:,:,0]
    
    def get_hist(region, bins=32):
        hist = cv2.calcHist([region], [0], None, [bins], [0, 180])
        cv2.normalize(hist, hist)
        return hist.flatten().tolist()
    
    # Level 0: 1x1
    features = get_hist(h_chan)
    
    # Level 1: 2x2
    h, w = h_chan.shape
    for i in range(2):
        for j in range(2):
            cell = h_chan[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
            features.extend(get_hist(cell))
            
    return features
