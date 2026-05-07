import cv2
import numpy as np
from typing import List

def _extract_edge_orientation(img_bgr: np.ndarray) -> List[float]:
    """Extracts 5-bin Edge Orientation Histogram (N, D, C45, C135, Non)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobelx, sobely)
    
    h, w = gray.shape
    total_pixels = h * w
    bins = np.zeros(5)
    
    # Threshold for edges
    edge_mask = mag > (np.max(mag) * 0.1)
    bins[4] = (total_pixels - np.sum(edge_mask)) / total_pixels # Non-directional
    
    if np.sum(edge_mask) > 0:
        valid_angs = ang[edge_mask] * 180 / np.pi
        for a in valid_angs:
            a = a % 180
            if a < 22.5 or a >= 157.5: bins[0] += 1
            elif a >= 67.5 and a < 112.5: bins[1] += 1
            elif a >= 22.5 and a < 67.5: bins[2] += 1
            else: bins[3] += 1
        
        sum_dirs = np.sum(bins[:4])
        if sum_dirs > 0:
            bins[:4] = bins[:4] / sum_dirs * (1.0 - bins[4])
            
    return bins.tolist()
