import cv2
import numpy as np
from typing import List

def _extract_ehd(img_bgr: np.ndarray) -> List[float]:
    """MPEG-7 Edge Histogram Descriptor (80 dim: 16 cells * 5 edge types)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    
    # Kernel for 5 edge types
    kernels = [
        np.array([[1, 1], [-1, -1]]),    # Vertical
        np.array([[1, -1], [1, -1]]),    # Horizontal
        np.array([[np.sqrt(2), 0], [0, -np.sqrt(2)]]), # 45 degree
        np.array([[0, np.sqrt(2)], [-np.sqrt(2), 0]]), # 135 degree
        np.array([[2, -2], [-2, 2]])     # Non-directional
    ]
    
    features = []
    cells = 4
    ch, cw = 64, 64
    for i in range(cells):
        for j in range(cells):
            cell = gray[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            cell_hist = [0.0] * 5
            for k_idx, kernel in enumerate(kernels):
                edge_map = cv2.filter2D(cell, -1, kernel)
                cell_hist[k_idx] = float(np.mean(np.abs(edge_map)))
            features.extend(cell_hist)
            
    # Normalize
    norm = np.linalg.norm(features) + 1e-7
    return (np.array(features) / norm).tolist()
