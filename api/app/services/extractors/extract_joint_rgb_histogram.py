import cv2
import numpy as np
from typing import List

def _extract_joint_rgb_histogram(img_bgr: np.ndarray) -> List[float]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([rgb], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().tolist()
    return hist
