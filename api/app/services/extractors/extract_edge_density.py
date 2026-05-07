import cv2
import numpy as np

def _extract_edge_density(img_bgr: np.ndarray) -> float:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    edge_pixels = cv2.countNonZero(edges)
    total_pixels = edges.size
    return float(edge_pixels / total_pixels)
