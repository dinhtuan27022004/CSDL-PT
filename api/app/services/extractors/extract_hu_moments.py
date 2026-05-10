import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from ...core.logging import get_logger

logger = get_logger(__name__)

def _extract_hu_moments(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    if filename:
        filename = os.path.basename(filename)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    vis_path = None
    if filename:
        try:
            vis_filename = f"hu_{filename}.png"
            full_vis_path = Path(visualizations_dir) / vis_filename
            cv2.imwrite(str(full_vis_path), thresh)
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Hu vis failed: {e}")
    return hu_log.tolist(), vis_path
