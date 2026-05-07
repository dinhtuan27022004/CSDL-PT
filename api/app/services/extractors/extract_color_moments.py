import cv2
import numpy as np
from typing import List

def _extract_color_moments(img_bgr: np.ndarray) -> List[float]:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    moments = []
    for i in range(3):
        channel = img_hsv[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        diff = channel - mean
        skew = np.mean(diff**3) / (std**3 + 1e-7)
        moments.extend([float(mean), float(std), float(skew)])
    return moments
