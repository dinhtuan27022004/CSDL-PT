import cv2
import numpy as np
from typing import List

def _extract_correlogram(img_bgr: np.ndarray) -> List[float]:
    """Simplified Color Auto-Correlogram (8 bins, 4 distances)"""
    img_small = cv2.resize(img_bgr, (64, 64))
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    quant = (hsv[:,:,0] / 22.5).astype(int) 
    distances = [1, 3, 5, 7]
    features = []
    h, w = quant.shape
    for d in distances:
        for color in range(8):
            count, total = 0, 0
            for y in range(0, h, 2):
                for x in range(0, w, 2):
                    if quant[y, x] == color:
                        total += 1
                        for dy, dx in [(0, d), (d, 0), (d, d), (-d, d)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if quant[ny, nx] == color: count += 1
            features.append(float(count / (total * 4 + 1e-7)))
    return features
