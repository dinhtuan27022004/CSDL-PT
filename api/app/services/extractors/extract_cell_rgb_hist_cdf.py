import cv2
import numpy as np
from typing import List, Tuple

def _extract_cell_rgb_hist_cdf(img_bgr: np.ndarray) -> Tuple[List[float], List[float]]:
    """Extracts RGB Histogram and CDF for each of the 16 cells (4x4)"""
    img_std = cv2.resize(img_bgr, (256, 256))
    cells = 4
    ch, cw = 64, 64
    all_hists = []
    all_cdfs = []
    
    for i in range(cells):
        for j in range(cells):
            cell = img_std[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            cell_hists = []
            cell_cdfs = []
            for channel in range(3): # B, G, R
                hist = cv2.calcHist([cell], [channel], None, [8], [0, 256]).flatten()
                hist = hist / (ch * cw) # Normalize
                cdf = np.cumsum(hist)
                cell_hists.extend(hist.tolist())
                cell_cdfs.extend(cdf.tolist())
            all_hists.extend(cell_hists)
            all_cdfs.extend(cell_cdfs)
            
    return all_hists, all_cdfs
