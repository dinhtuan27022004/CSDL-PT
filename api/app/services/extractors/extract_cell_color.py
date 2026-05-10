import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from ...core.logging import get_logger

logger = get_logger(__name__)

def _extract_cell_color(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    if filename:
        filename = os.path.basename(filename)
    img_std = cv2.resize(img_bgr, (256, 256))
    cells = 4
    ch, cw = 64, 64
    feature_vector = []
    vis_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(cells):
        for j in range(cells):
            cell = img_std[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            avg_color = np.mean(cell, axis=(0, 1))
            feature_vector.extend(avg_color.tolist())
            vis_img[i*ch:(i+1)*ch, j*cw:(j+1)*cw] = avg_color.astype(np.uint8)
    vis_path = None
    if filename:
        try:
            vis_filename = f"cell_{filename}.png"
            full_vis_path = Path(visualizations_dir) / vis_filename
            cv2.imwrite(str(full_vis_path), vis_img)
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Cell color vis failed: {e}")
    return feature_vector, vis_path
