import cv2
import numpy as np
from typing import List
from skimage.feature import graycomatrix, graycoprops

def _extract_glcm(img_bgr: np.ndarray) -> List[float]:
    """GLCM Texture features (Contrast, Correlation, Energy, Homogeneity) across 4x4 grid"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_std = cv2.resize(gray, (256, 256))
    cells = 4
    ch, cw = 64, 64
    features = []
    for i in range(cells):
        for j in range(cells):
            cell = img_std[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            # Quantize to 32 levels to save compute/memory
            cell_q = (cell // 8).astype(np.uint8)
            glcm = graycomatrix(cell_q, [1], [0], 32, symmetric=True, normed=True)
            for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
                features.append(float(graycoprops(glcm, prop)[0, 0]))
    return features
