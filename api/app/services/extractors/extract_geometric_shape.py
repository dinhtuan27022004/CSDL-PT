import cv2
import numpy as np
from typing import List

def _extract_geometric_shape(img_bgr: np.ndarray) -> List[float]:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return [0.0] * 6
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0: return [0.0] * 6
    circularity = (4 * np.pi * area) / (perimeter**2)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / hull_area) if hull_area > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = float(area / rect_area) if rect_area > 0 else 0
    aspect_ratio = float(w) / h if h > 0 else 0
    if len(cnt) >= 5:
        (x, y), (axes1, axes2), angle = cv2.fitEllipse(cnt)
        major_axis = max(axes1, axes2)
        minor_axis = min(axes1, axes2)
        eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2)) if major_axis > 0 else 0
    else: eccentricity = 0
    convexity = cv2.arcLength(hull, True) / perimeter
    
    # Ensure no NaNs are returned
    geo = [float(circularity), float(solidity), float(extent), float(aspect_ratio), float(eccentricity), float(convexity)]
    return [0.0 if np.isnan(x) else x for x in geo]
