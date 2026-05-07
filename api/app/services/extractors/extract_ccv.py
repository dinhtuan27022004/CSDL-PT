import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from ...core.logging import get_logger

logger = get_logger(__name__)

def _extract_ccv(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    if filename:
        filename = os.path.basename(filename)
    img_small = cv2.resize(img_bgr, (256, 256))
    img_blur = cv2.GaussianBlur(img_small, (3, 3), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    h_bins, s_bins = 12, 4
    h_quant = (hsv[:,:,0] / (180/h_bins)).astype(int)
    s_quant = (hsv[:,:,1] / (256/s_bins)).astype(int)
    quantized = h_quant * s_bins + s_quant
    num_bins = h_bins * s_bins
    threshold = (img_small.shape[0] * img_small.shape[1]) * 0.01
    coherent = np.zeros(num_bins)
    incoherent = np.zeros(num_bins)
    for b in range(num_bins):
        mask = (quantized == b).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= threshold: coherent[b] += area
            else: incoherent[b] += area
    ccv = []
    for b in range(num_bins):
        total = coherent[b] + incoherent[b] + 1e-7
        ccv.append(float(coherent[b] / total))
        ccv.append(float(incoherent[b] / total))
    vis_path = None
    if filename:
        try:
            vis_filename = f"ccv_{filename}.png"
            full_vis_path = visualizations_dir / vis_filename
            vis_img = (quantized * (255/num_bins)).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(str(full_vis_path), vis_color)
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"CCV vis failed: {e}")
    return ccv, vis_path
