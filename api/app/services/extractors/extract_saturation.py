import cv2
import numpy as np

def _extract_saturation(img_bgr: np.ndarray) -> float:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_s = np.mean(img_hsv[:, :, 1])
    return float(mean_s / 255.0)
