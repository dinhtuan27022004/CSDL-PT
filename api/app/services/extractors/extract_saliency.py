import cv2
import numpy as np
from typing import List

def _extract_saliency(img_bgr: np.ndarray) -> List[float]:
    """Visual Saliency features (32 dim)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    # Simple Spectral Residual Saliency
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    log_mag = np.log(mag + 1e-7)
    
    # Mean filter for spectral residual
    avg_log_mag = cv2.blur(log_mag, (3, 3))
    spectral_residual = log_mag - avg_log_mag
    
    # Saliency map
    saliency = np.abs(np.fft.ifft2(np.fft.ifftshift(np.exp(spectral_residual + 1j * np.angle(fshift)))))
    saliency = cv2.GaussianBlur(saliency, (5, 5), 3)
    saliency = cv2.resize(saliency, (8, 4)) # Resize to small vector
    
    norm = np.linalg.norm(saliency) + 1e-7
    return (saliency.flatten() / norm).tolist()
