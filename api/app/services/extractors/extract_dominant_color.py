import cv2
import numpy as np
from typing import List

def _extract_dominant_color(img_bgr: np.ndarray) -> List[float]:
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, flags)
    return centers[0].tolist()
