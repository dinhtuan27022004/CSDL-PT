import cv2
import numpy as np
from typing import List

def _extract_cld(img_bgr: np.ndarray) -> List[float]:
    """MPEG-7 Color Layout Descriptor (64 dim: Y channel 8x8 DCT coefficients)"""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y_chan = cv2.resize(ycrcb[:,:,0], (8, 8))
    # 8x8 DCT
    dct = cv2.dct(y_chan.astype(np.float32))
    return dct.flatten().tolist()
