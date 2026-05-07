import cv2
import numpy as np

def _extract_sharpness(img_bgr: np.ndarray) -> float:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
