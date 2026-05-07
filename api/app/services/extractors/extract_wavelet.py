import cv2
import numpy as np
from typing import List

def _extract_wavelet(img_bgr: np.ndarray) -> List[float]:
    """Haar-like Wavelet Energy features (3 levels of decomposition)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    curr = cv2.resize(gray, (256, 256))
    features = []
    for _ in range(3):
        h, w = curr.shape
        ll = cv2.resize(curr, (w//2, h//2), interpolation=cv2.INTER_AREA)
        lh = np.abs(curr[0::2, 1::2] - curr[1::2, 1::2])
        hl = np.abs(curr[1::2, 0::2] - curr[1::2, 1::2])
        hh = np.abs(curr[0::2, 0::2] - curr[1::2, 1::2])
        features.extend([float(np.mean(ll)), float(np.mean(lh)), float(np.mean(hl)), float(np.mean(hh))])
        curr = ll
    return features[:12]
