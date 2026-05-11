import cv2
import numpy as np

def resize_logic_worker(image_bytes: bytes) -> bytes:
    """Standalone worker function for resizing (No Torch dependency)"""
    # Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return image_bytes
    
    h, w = img.shape[:2]
    target_w, target_h = 3840, 2160
    target_aspect = target_w / target_h
    curr_aspect = w / h
    
    # 1. Center Crop to 16:9
    if abs(curr_aspect - target_aspect) > 0.01:
        if curr_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            img = img[:, start_x:start_x+new_w]
        else:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            img = img[start_y:start_y+new_h, :]
    
    # 2. Resize to 4K
    if img.shape[1] != target_w or img.shape[0] != target_h:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 3. Re-encode to JPEG
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return buffer.tobytes()
