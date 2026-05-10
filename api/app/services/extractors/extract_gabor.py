import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from ...core.logging import get_logger

logger = get_logger(__name__)

def _extract_gabor(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    if filename:
        filename = os.path.basename(filename)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_std = cv2.resize(img_gray, (256, 256))
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in (np.pi/4, np.pi/2):
                kernel = cv2.getGaborKernel((31, 31), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
    features = []
    vis_img = np.zeros_like(img_std, dtype=np.float32)
    cells = 4
    ch, cw = 64, 64
    for k in kernels:
        fimg = cv2.filter2D(img_std, cv2.CV_8UC3, k)
        vis_img += fimg.astype(np.float32)
        for i in range(cells):
            for j in range(cells):
                cell = fimg[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
                features.append(float(np.mean(cell)))
                features.append(float(np.var(cell)))
    vis_path = None
    if filename:
        try:
            vis_filename = f"gabor_{filename}.png"
            full_vis_path = Path(visualizations_dir) / vis_filename
            vis_norm = cv2.normalize(vis_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(str(full_vis_path), vis_norm)
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Gabor vis failed: {e}")
    return features, vis_path
