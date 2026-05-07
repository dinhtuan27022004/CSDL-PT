import cv2
import numpy as np

def _extract_contrast(img_bgr: np.ndarray) -> float:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(img_gray)
    return float(min(1.0, std_dev / 127.5))
