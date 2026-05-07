import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from skimage.feature import local_binary_pattern
from ...core.logging import get_logger

logger = get_logger(__name__)

def _extract_lbp(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    if filename:
        filename = os.path.basename(filename)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_std = cv2.resize(img_gray, (256, 256))
    cells = 4
    cell_h, cell_w = 64, 64
    P, R = 8, 1
    lbp = local_binary_pattern(img_std, P, R, method="uniform")
    spatial_features = []
    for i in range(cells):
        for j in range(cells):
            cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            (hist, _) = np.histogram(cell.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            spatial_features.extend(hist.tolist())
    vis_relative_path = None
    if filename:
        try:
            lbp_uint8 = (lbp * (255 / (P + 1))).astype(np.uint8)
            vis_filename = f"lbp_{filename}.png"
            vis_path = visualizations_dir / vis_filename
            cv2.imwrite(str(vis_path), lbp_uint8)
            vis_relative_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"LBP vis failed: {e}")
    return spatial_features, vis_relative_path
