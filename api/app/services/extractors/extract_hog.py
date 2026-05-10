import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from skimage.feature import hog
from skimage import exposure
from ...core.logging import get_logger

logger = get_logger(__name__)

def _extract_hog(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    if filename:
        filename = os.path.basename(filename)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128))
    fd, hog_image = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=True)
    vis_path = None
    if filename:
        try:
            hog_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            vis_filename = f"hog_{filename}.png"
            full_vis_path = Path(visualizations_dir) / vis_filename
            cv2.imwrite(str(full_vis_path), (hog_rescaled * 255).astype(np.uint8))
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"HOG vis failed: {e}")
    return fd.tolist(), vis_path
