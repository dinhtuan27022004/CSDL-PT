import cv2
import numpy as np

def _extract_brightness(img_bgr: np.ndarray) -> float:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_v = np.mean(img_hsv[:, :, 2])
    return float(mean_v / 255.0)
