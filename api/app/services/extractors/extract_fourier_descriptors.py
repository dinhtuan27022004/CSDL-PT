import cv2
import numpy as np
from typing import List

def _extract_fourier_descriptors(img_bgr: np.ndarray) -> List[float]:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return [0.0] * 25
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 2: return [0.0] * 25
    
    cnt_complex = np.empty(len(cnt), dtype=complex)
    cnt_complex.real = cnt[:, 0, 0]
    cnt_complex.imag = cnt[:, 0, 1]
    fourier_result = np.fft.fft(cnt_complex)
    
    # Get magnitudes
    descriptors = np.abs(fourier_result)
    
    # Truncate to 25 if more
    if len(descriptors) > 25:
        descriptors = descriptors[:25]
    
    # DC component normalization
    if descriptors[0] != 0:
        descriptors = descriptors / descriptors[0]
    
    # Convert to list
    res = descriptors.tolist()
    
    # Pad with zeros if less than 25
    if len(res) < 25:
        res.extend([0.0] * (25 - len(res)))
        
    return res
