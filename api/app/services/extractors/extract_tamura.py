import cv2
import numpy as np
from typing import List

def _extract_tamura(img_bgr: np.ndarray) -> List[float]:
    """Extracts Tamura Texture Features: Coarseness, Contrast, Directionality"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 1. Contrast
    mean, std = cv2.meanStdDev(gray)
    mu4 = np.mean((gray - mean)**4)
    alpha4 = mu4 / (std[0][0]**4 + 1e-7)
    contrast = std[0][0] / (alpha4**0.25 + 1e-7)
    
    # 2. Directionality (Gradient based)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobelx, sobely)
    
    # Threshold magnitudes
    thresh = np.max(mag) * 0.1
    valid_angles = ang[mag > thresh]
    if len(valid_angles) > 0:
        hist, _ = np.histogram(valid_angles, bins=16, range=(0, 2*np.pi))
        directionality = 1.0 - float(np.sum(hist > (np.max(hist)*0.1)) / 16.0)
    else:
        directionality = 0.0
        
    # 3. Coarseness (Simplified)
    coarseness = 0.0
    for k in [1, 2, 3]:
        s = 2**k
        kernel = np.ones((s, s), np.float32) / (s*s)
        mean_scaled = cv2.filter2D(gray, -1, kernel)
        coarseness += np.mean(np.abs(gray - mean_scaled))
    coarseness /= 3.0
    
    return [float(coarseness), float(contrast), float(directionality)]
