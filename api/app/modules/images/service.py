from pathlib import Path
import uuid
import json
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from .repository import ImageRepository
from .llm_service import LLMService
from .model import ImageMetadata
from .schema import ImageResponse
from ...core.config import get_settings
from ...core.logging import get_logger
from ...core.exceptions import ImageProcessingError
import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile
import torch
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

logger = get_logger(__name__)

def _extract_brightness(img_hsv: np.ndarray) -> float:
    mean_v = np.mean(img_hsv[:, :, 2])
    return float(mean_v / 255.0)

def _extract_contrast(img_gray: np.ndarray) -> float:
    std_dev = np.std(img_gray)
    return float(min(1.0, std_dev / 127.5))

def _extract_saturation(img_hsv: np.ndarray) -> float:
    mean_s = np.mean(img_hsv[:, :, 1])
    return float(mean_s / 255.0)

def _extract_edge_density(img_gray: np.ndarray) -> float:
    edges = cv2.Canny(img_gray, 100, 200)
    edge_pixels = cv2.countNonZero(edges)
    total_pixels = edges.size
    return float(edge_pixels / total_pixels)

def _extract_sharpness(img_gray: np.ndarray) -> float:
    # Laplacian variance is a common focus measure
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

def _extract_color_moments(img_hsv: np.ndarray) -> List[float]:
    moments = []
    # Loop H, S, V channels
    for i in range(3):
        channel = img_hsv[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        # Skewness
        diff = channel - mean
        skew = np.mean(diff**3) / (std**3 + 1e-7)
        moments.extend([float(mean), float(std), float(skew)])
    return moments

def _extract_lbp(img_gray: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    # Standardize size for 4x4 grid
    img_std = cv2.resize(img_gray, (256, 256))
    cells = 4
    cell_h, cell_w = 64, 64 # 256/4
    
    # P=8, R=1 for standard LBP
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

def _extract_histogram(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], List[float], Optional[str]]:
    # 8 bins per channel as requested
    bins = 8
    
    # --- HSV Extraction ---
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([img_hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([img_hsv], [2], None, [bins], [0, 256])
    hsv_vec = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    norm_hsv = np.linalg.norm(hsv_vec)
    if norm_hsv > 1e-10: hsv_vec /= norm_hsv
    
    # --- RGB Extraction ---
    hist_b = cv2.calcHist([img_bgr], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([img_bgr], [1], None, [bins], [0, 256])
    hist_r = cv2.calcHist([img_bgr], [2], None, [bins], [0, 256])
    rgb_vec = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    norm_rgb = np.linalg.norm(rgb_vec)
    if norm_rgb > 1e-10: rgb_vec /= norm_rgb
    
    # --- Visualization ---
    vis_relative_path = None
    if filename:
        try:
            h, w = 400, 600 # Larger canvas for better visibility
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis.fill(20) # Dark gray background
            
            def draw_channel_plots(hists, colors, offset_x, title):
                # Calculate CDFs
                cdfs = [np.cumsum(h) for h in hists]
                # Normalize CDFs to [0, 1] relative to sum
                cdfs = [c / (c[-1] + 1e-7) for c in cdfs]
                
                chart_w = w // 2 - 40
                chart_h = h - 100
                base_y = h - 40
                step = chart_w / (bins - 1)
                
                # Draw grid
                for i in range(5):
                    y_grid = int(base_y - (i/4) * chart_h)
                    cv2.line(vis, (int(offset_x + 20), y_grid), (int(offset_x + 20 + chart_w), y_grid), (40, 40, 40), 1)

                cv2.putText(vis, title, (int(offset_x + 20), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                for hist, cdf, color in zip(hists, cdfs, colors):
                    max_val = np.max(hist) if np.max(hist) > 0 else 1
                    
                    # Store points for PDF and CDF
                    pdf_pts = []
                    cdf_pts = []
                    
                    for i in range(bins):
                        lx = int(offset_x + 20 + i * step)
                        # PDF Y (normalized to local max)
                        ly_pdf = int(base_y - (hist[i][0] / max_val) * chart_h)
                        # CDF Y (normalized to 1.0)
                        ly_cdf = int(base_y - cdf[i] * chart_h)
                        
                        pdf_pts.append((lx, ly_pdf))
                        cdf_pts.append((lx, ly_cdf))
                    
                    # Draw PDF (Dashed/Thin)
                    for i in range(len(pdf_pts)-1):
                        cv2.line(vis, pdf_pts[i], pdf_pts[i+1], [int(c*0.5) for c in color], 1)
                        # Draw points
                        cv2.circle(vis, pdf_pts[i], 2, color, -1)
                    cv2.circle(vis, pdf_pts[-1], 2, color, -1)

                    # Draw CDF (Thick solid line)
                    for i in range(len(cdf_pts)-1):
                        cv2.line(vis, cdf_pts[i], cdf_pts[i+1], color, 3)

            # Draw HSV (H: Magenta, S: Orange, V: White)
            draw_channel_plots([hist_h, hist_s, hist_v], [(255, 0, 255), (0, 165, 255), (240, 240, 240)], 0, "HSV Density & CDF")
            # Draw RGB (Red, Green, Blue)
            draw_channel_plots([hist_r, hist_g, hist_b], [(0, 0, 255), (0, 255, 0), (255, 0, 0)], w // 2, "RGB Density & CDF")
            
            # Legend
            cv2.putText(vis, "Thin/Dots: Density (PDF)", (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(vis, "Thick Line: Cumulative (CDF)", (w//2 + 20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            vis_filename = f"hist_{filename}.png"
            vis_path = visualizations_dir / vis_filename
            cv2.imwrite(str(vis_path), vis)
            vis_relative_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Histogram visualization failed: {e}")
            
    # Calculate CDF vectors for return
    h_cdf = np.cumsum(hist_h).flatten()
    s_cdf = np.cumsum(hist_h).flatten() # Wait, i should use correct hist
    v_cdf = np.cumsum(hist_h).flatten() # Need to check previous logic
    # Fixing cumsum for all 6 channels
    h_cdf = np.cumsum(hist_h).flatten(); h_cdf = h_cdf / (h_cdf[-1] + 1e-7)
    s_cdf = np.cumsum(hist_s).flatten(); s_cdf = s_cdf / (s_cdf[-1] + 1e-7)
    v_cdf = np.cumsum(hist_v).flatten(); v_cdf = v_cdf / (v_cdf[-1] + 1e-7)
    hsv_cdf_vec = np.concatenate([h_cdf, s_cdf, v_cdf])

    r_cdf = np.cumsum(hist_r).flatten(); r_cdf = r_cdf / (r_cdf[-1] + 1e-7)
    g_cdf = np.cumsum(hist_g).flatten(); g_cdf = g_cdf / (g_cdf[-1] + 1e-7)
    b_cdf = np.cumsum(hist_b).flatten(); b_cdf = b_cdf / (b_cdf[-1] + 1e-7)
    rgb_cdf_vec = np.concatenate([r_cdf, g_cdf, b_cdf])

    return hsv_vec.tolist(), rgb_vec.tolist(), hsv_cdf_vec.tolist(), rgb_cdf_vec.tolist(), vis_relative_path
def _extract_gabor(img_gray: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    # Standardize size for 4x4 grid
    img_std = cv2.resize(img_gray, (256, 256))
    cells = 4
    cell_h, cell_w = 64, 64 # 256/4

    kernels = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [1, 3]:
            for lamda in [np.pi/4, np.pi/2]:
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
                
    spatial_features = []
    # Pre-calculate filtered images to avoid redundant filtering per cell
    filtered_images = [cv2.filter2D(img_std, cv2.CV_8UC3, k) for k in kernels]
    
    for i in range(cells):
        for j in range(cells):
            y1, y2 = i*cell_h, (i+1)*cell_h
            x1, x2 = j*cell_w, (j+1)*cell_w
            for fimg in filtered_images:
                cell = fimg[y1:y2, x1:x2]
                spatial_features.append(float(np.mean(cell)))
                spatial_features.append(float(np.var(cell)))
        
    vis_relative_path = None
    if filename:
        # Create vis by averaging responses
        vis_composite = np.mean(filtered_images, axis=0).astype(np.uint8)
        vis_composite = cv2.normalize(vis_composite, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_filename = f"gabor_{filename}.png"
        vis_path = visualizations_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_composite)
        vis_relative_path = f"/static/visualizations/{vis_filename}"
        
    return spatial_features, vis_relative_path

def _extract_ccv(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    # 48 bins total (16 per channel) quantized jointly to 64 bins for efficiency
    img_small = cv2.resize(img_bgr, (100, 100)) # Small for CCV labeling performance
    img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    h = (img_hsv[:,:,0] // 22.5).astype(np.int32) # 8 bins
    s = (img_hsv[:,:,1] // 64).astype(np.int32)   # 4 bins
    v = (img_hsv[:,:,2] // 64).astype(np.int32)   # 4 bins
    joint = (h * 16 + s * 4 + v).astype(np.int32) # 128 bins? Let's use 64 for 96-dim task (2*48)
    
    # Re-quantize to 48 bins to match the requested 96-dim vector (48 coherent + 48 incoherent)
    joint = (joint % 48).astype(np.int32)
    
    num_bins = 48
    coherent = np.zeros(num_bins)
    incoherent = np.zeros(num_bins)
    threshold = (img_small.shape[0] * img_small.shape[1]) * 0.01
    
    vis_img = np.zeros_like(img_small)
    
    for b in range(num_bins):
        mask = (joint == b).astype(np.uint8)
        if np.sum(mask) == 0: continue
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for l in range(1, num_labels):
            area = stats[l, cv2.CC_STAT_AREA]
            if area >= threshold:
                coherent[b] += area
                vis_img[labels == l] = [0, 255, 0] # Green for Coherent
            else:
                incoherent[b] += area
                vis_img[labels == l] = [0, 0, 255] # Red for Incoherent
                
    ccv = np.concatenate([coherent, incoherent])
    norm = np.linalg.norm(ccv)
    if norm > 1e-10: ccv = ccv / norm
        
    vis_relative_path = None
    if filename:
        vis_filename = f"ccv_{filename}.png"
        vis_path = visualizations_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_img)
        vis_relative_path = f"/static/visualizations/{vis_filename}"
        
    return ccv.tolist(), vis_relative_path

def _extract_fourier_descriptors(img_gray: np.ndarray) -> List[float]:
    # Extract shape using Fourier Descriptors (FD)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return [0.0] * 25
    
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 25: return [0.0] * 25
    
    # Complex representation
    complex_cnt = np.empty(len(cnt), dtype=complex)
    complex_cnt.real = cnt[:, 0, 0]
    complex_cnt.imag = cnt[:, 0, 1]
    
    # FFT
    fd = np.fft.fft(complex_cnt)
    # Scale invariant (normalize by first non-zero component)
    # Rotation invariant (use magnitude)
    res = np.abs(fd[1:26]) # Use first 25 components
    norm = np.linalg.norm(res)
    if norm > 1e-10: res = res / norm
    return res.tolist()

def _extract_dominant_color(img_bgr: np.ndarray) -> List[float]:
    # Resize image to small for speed while maintaining color distribution
    small_img = cv2.resize(img_bgr, (50, 50), interpolation=cv2.INTER_AREA)
    pixels = small_img.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # K-Means clustering (K=3) to find main color groups
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Identify the most frequent cluster center
    counts = np.bincount(labels.flatten())
    dominant_bgr = centers[np.argmax(counts)]
    
    # Convert the dominant color from BGR to CIE Lab for perceptual accuracy
    dominant_bgr_img = np.uint8([[dominant_bgr]])
    dominant_lab_img = cv2.cvtColor(dominant_bgr_img, cv2.COLOR_BGR2LAB)
    
    return dominant_lab_img[0, 0].astype(float).tolist()

def _extract_hog(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    # Standardize size for HOG (128x128) - Direct from original to preserve edges
    img_gray_hog = cv2.cvtColor(cv2.resize(img_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
    
    # Extract HOG with advanced L2-Hys block normalization
    fd, hog_image = hog(img_gray_hog, orientations=8, pixels_per_cell=(32, 32),
                        cells_per_block=(2, 2), block_norm='L2', visualize=True)
    
    # Global L2 Normalization for better Cosine Similarity
    norm = np.linalg.norm(fd)
    if norm > 1e-10:
        fd = fd / norm
    
    vis_relative_path = None
    # Save visualization to visualizations_dir
    if filename:
        try:
            # Improve contrast of hog_image for visualization
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            vis_filename = f"hog_{filename}.png"
            vis_path = visualizations_dir / vis_filename
            cv2.imwrite(str(vis_path), (hog_image_rescaled * 255).astype(np.uint8))
            vis_relative_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Could not save HOG visualization for {filename}: {e}")
        
    return fd.tolist(), vis_relative_path

def _extract_hu_moments(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    # Standardize size for HU (256x256)
    img_input = cv2.resize(img_bgr, (256, 256))
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Smooth slightly to reduce noise
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # Step 2: Use Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3: Find contours and use the largest one for shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        moments = cv2.moments(thresh)
        cnt = None
    else:
        cnt = max(contours, key=cv2.contourArea)
        moments = cv2.moments(cnt)
    
    vis_relative_path = None
    if filename:
        try:
            vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            if cnt is not None:
                cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
            
            if moments['m00'] > 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                cv2.drawMarker(vis, (cX, cY), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            
            vis_filename = f"hu_{filename}.png"
            vis_path = visualizations_dir / vis_filename
            cv2.imwrite(str(vis_path), vis)
            vis_relative_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"HU visualization failed: {e}")

    hu = cv2.HuMoments(moments).flatten()
    hu_transformed = []
    for h in hu:
        if abs(h) > 1e-20:
            hu_transformed.append(-1.0 * np.sign(h) * np.log10(abs(h)))
        else:
            hu_transformed.append(0.0)
    return hu_transformed, vis_relative_path

def _extract_geometric_shape(img_gray: np.ndarray) -> List[float]:
    # Threshold for geometric analysis
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [0.0] * 6
        
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if area < 10 or perimeter < 1.0:
        return [0.0] * 6
        
    # 1. Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    # 2. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    # 3. Extent
    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / (w * h) if (w * h) > 0 else 0
    # 4. Aspect Ratio
    aspect_ratio = w / h if h > 0 else 0
    # 5. Convexity
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = hull_perimeter / perimeter if perimeter > 0 else 0
    # 6. Eccentricity
    rect = cv2.minAreaRect(cnt)
    (x_r, y_r), (w_r, h_r), angle = rect
    if h_r > 0 and w_r > 0:
        a = max(w_r, h_r)
        b = min(w_r, h_r)
        eccentricity = np.sqrt(1 - (b/a)**2)
    else:
        eccentricity = 0
        
    return [
        float(circularity), float(solidity), float(extent), 
        float(aspect_ratio), float(convexity), float(eccentricity)
    ]

def _extract_cell_color(
    img_original, 
    visualizations_dir: Path, 
    filename: Optional[str] = None, 
    cells=4, 
    mode="mean",
    color_space="LAB"
):

    h, w, _ = img_original.shape
    
    # Resize về ảnh vuông (1:1) bằng cách nén (không crop)
    target_size = min(h, w)
    img = cv2.resize(img_original, (target_size, target_size))
    
    H, W, _ = img.shape
    cell_h = H // cells
    cell_w = W // cells
    
    # Chuyển đổi không gian màu sang mục tiêu (LAB, HSV, hoặc RGB)
    if color_space == "LAB":
        img_target = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif color_space == "HSV":
        img_target = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == "RGB":
        img_target = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_target = img  # Mặc định BGR
    
    features = []
    vis_img = np.zeros_like(img)

    for i in range(cells):
        for j in range(cells):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            
            cell = img_target[y1:y2, x1:x2]

            if mode == "mean":
                color = np.mean(cell.reshape(-1, 3), axis=0)
            elif mode == "dominant":
                pixels = cell.reshape(-1, 3)
                # Giảm số màu để tăng tốc (quantize nhẹ)
                pixels_q = (pixels // 16) * 16
                # Đếm màu
                colors, counts = np.unique(pixels_q, axis=0, return_counts=True)
                color = colors[np.argmax(counts)]
            else:
                raise ValueError("mode must be 'mean' or 'dominant'")
            
            color = color.astype(np.uint8)
            
            # Chuyển ngược về BGR để hiển thị lên ảnh minh họa đúng màu
            color_pixel = np.array([[color]], dtype=np.uint8)
            if color_space == "LAB":
                color_bgr = cv2.cvtColor(color_pixel, cv2.COLOR_LAB2BGR)[0, 0]
            elif color_space == "HSV":
                color_bgr = cv2.cvtColor(color_pixel, cv2.COLOR_HSV2BGR)[0, 0]
            elif color_space == "RGB":
                color_bgr = cv2.cvtColor(color_pixel, cv2.COLOR_RGB2BGR)[0, 0]
            else:
                color_bgr = color

            # append feature (giá trị trong không gian màu mục tiêu)
            features.extend(color.tolist())
            # vẽ lên ảnh minh họa (màu BGR)
            vis_img[y1:y2, x1:x2] = color_bgr

    feature_vector = np.array(features, dtype=np.float32)
    vis_filename = f"cell_color_{filename}.png"
    vis_path = visualizations_dir / vis_filename
    vis_img = cv2.resize(vis_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(vis_path), (vis_img).astype(np.uint8))
    return feature_vector.tolist(), f"/static/visualizations/{vis_filename}"

class ImageService:
    """Service for image CRUD, storage, and validation"""
    
    def __init__(self, 
        repository: ImageRepository
    ):
        self.repository = repository
        self.llm_service = LLMService()
        self.settings = get_settings()
        # Using ProcessPoolExecutor to test multiprocessing performance
        self.executor = ThreadPoolExecutor(max_workers=32)
    
    async def process(
        self, 
        db: Session, 
        files: List[UploadFile],
        force_llm: bool = False
    ) -> List[ImageMetadata]:
        logger.info(f"Processing {len(files)} images")
        results = []
        start_time = time.time()
        for file in files:
            img_content = await file.read()
            result = await self.extract_features(img_content, file.filename, force_llm=force_llm)
            result = self.repository.create(db, result, img_content)
            results.append(result)
        logger.info(f"Processed {len(results)} images successfully after {time.time() - start_time} seconds")
        return results

    def get_images(self, db: Session, limit: int = 500, offset: int = 0) -> List[ImageMetadata]:
        return self.repository.get_all(db, limit=limit, offset=offset)

    async def extract_features(self, image: bytes, filename: Optional[str] = None, force_llm: bool = False) -> ImageMetadata:
        """Extract all features from image (Visual + Deep Learning) with nested extraction logic"""
        try:
            file_bytes = np.frombuffer(image, dtype=np.uint8)
            img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_original is None:
                raise ValueError("Failed to decode image")
            
            # Use smaller version for traditional features to speed up
            img_small = cv2.resize(img_original, (int(img_original.shape[1]/6), int(img_original.shape[0]/6)))
            img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
            
            loop = asyncio.get_running_loop()
            
            # Prepare all extraction tasks to run in parallel threads
            tasks = [
                loop.run_in_executor(self.executor, _extract_brightness, img_hsv),
                loop.run_in_executor(self.executor, _extract_contrast, img_gray),
                loop.run_in_executor(self.executor, _extract_saturation, img_hsv),
                loop.run_in_executor(self.executor, _extract_edge_density, img_gray),
                loop.run_in_executor(self.executor, _extract_histogram, img_small, self.settings.visualizations_dir, filename),
                loop.run_in_executor(self.executor, _extract_hog, img_small, self.settings.visualizations_dir, filename),
                loop.run_in_executor(self.executor, _extract_hu_moments, img_small, self.settings.visualizations_dir, filename),
                loop.run_in_executor(self.executor, _extract_dominant_color, img_small),
                loop.run_in_executor(self.executor, _extract_cell_color, img_small, self.settings.visualizations_dir, filename),
                loop.run_in_executor(self.executor, _extract_lbp, img_gray, self.settings.visualizations_dir, filename),
                loop.run_in_executor(self.executor, _extract_color_moments, img_hsv),
                loop.run_in_executor(self.executor, _extract_sharpness, img_gray),
                loop.run_in_executor(self.executor, _extract_gabor, img_gray, self.settings.visualizations_dir, filename),
                loop.run_in_executor(self.executor, _extract_ccv, img_small, self.settings.visualizations_dir, filename),
                loop.run_in_executor(self.executor, _extract_fourier_descriptors, img_gray),
                loop.run_in_executor(self.executor, _extract_geometric_shape, img_gray),
                self.llm_service.analyze_and_embed(image, filename, force=force_llm)
            ]
            
            # Run all tasks concurrently and wait for results
            extraction_start = time.time()
            results = await asyncio.gather(*tasks)
            logger.info(f"Total feature extraction time: {time.time() - extraction_start:.4f} seconds")
            logger.info(f"results: {results}")
            
            # Unpack results in the same order as tasks
            (
                brightness, 
                contrast, 
                saturation, 
                edge_density, 
                (hsv_histogram_vector, rgb_histogram_vector, hsv_cdf_vector, rgb_cdf_vector, histogram_vis_path), 
                (hog_vector, hog_vis_path), 
                (hu_moments_vector, hu_vis_path), 
                dominant_color_vector, 
                (cell_color_vector, cell_color_vis_path),
                (lbp_vector, lbp_vis_path),
                color_moments_vector,
                sharpness,
                (gabor_vector, gabor_vis_path),
                (ccv_vector, ccv_vis_path),
                zernike_vector,
                geo_vector,
                semantic_data
            ) = results

            return ImageMetadata(
                file_name = filename,
                brightness = brightness,
                contrast = contrast,
                saturation = saturation,
                edge_density = edge_density,
                hsv_histogram_vector = hsv_histogram_vector,
                rgb_histogram_vector = rgb_histogram_vector,
                hsv_cdf_vector = hsv_cdf_vector,
                rgb_cdf_vector = rgb_cdf_vector,
                hog_vector = hog_vector,
                hu_moments_vector = hu_moments_vector,
                dominant_color_vector = dominant_color_vector,
                cell_color_vector = cell_color_vector,
                lbp_vector = lbp_vector,
                color_moments_vector = color_moments_vector,
                sharpness = sharpness,
                gabor_vector = gabor_vector,
                ccv_vector = ccv_vector,
                zernike_vector = zernike_vector,
                geo_vector = geo_vector,
                
                # Semantic features from LLM
                category = semantic_data.get("category"),
                description = semantic_data.get("description"),
                entities = semantic_data.get("entities"),
                llm_embedding = semantic_data.get("llm_embedding"),
                
                hog_vis_path = hog_vis_path,
                hu_vis_path = hu_vis_path,
                cell_color_vis_path = cell_color_vis_path,
                lbp_vis_path = lbp_vis_path,
                gabor_vis_path = gabor_vis_path,
                ccv_vis_path = ccv_vis_path,
                histogram_vis_path = histogram_vis_path
            )
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

    async def search_similar(
        self, 
        db: Session, 
        query_image_content: bytes, 
        filename: str,
        limit: int = 5,
        force_llm: bool = False
    ) -> Dict[str, Any]:
        """Hybrid Search with 5-way weighted similarity mapping"""
        try:
            query_metadata = await self.extract_features(query_image_content, filename, force_llm=force_llm)
            
            # Use Repository for high-performance 5-way search
            search_results = self.repository.search(
                db=db, 
                query_metadata=query_metadata,
                limit=limit
            )
            
            # Map database results (tuples) to ImageResponse schema objects
            response_data = []
            for row in search_results:
                (
                    record, total_sim, b_sim, c_sim, s_sim, e_sim, 
                    hsv_h_sim, rgb_h_sim, hog_sim, hu_sim, dom_sim, cell_color_sim,
                    lbp_sim, moments_sim, sharp_sim,
                    gabor_sim, ccv_sim, zernike_sim, geo_sim, 
                    embedding_sim, entity_sim, category_sim,
                    hsv_cdf_sim, rgb_cdf_sim
                ) = row
                
                # Convert 0-1 similarity to 0-100% for the UI
                res = ImageResponse.model_validate(record)
                res.similarity = round(float(total_sim) * 100.0, 2)
                res.semantic_similarity = round(float(embedding_sim) * 100.0, 2)
                res.entity_similarity = round(float(entity_sim) * 100.0, 2)
                res.category_similarity = round(float(category_sim) * 100.0, 2)
                res.dominant_color_similarity = round(float(dom_sim) * 100.0, 2)
                res.brightness_similarity = round(float(b_sim) * 100.0, 2)
                res.contrast_similarity = round(float(c_sim) * 100.0, 2)
                res.saturation_similarity = round(float(s_sim) * 100.0, 2)
                res.edge_density_similarity = round(float(e_sim) * 100.0, 2)
                res.hsv_histogram_similarity = round(float(hsv_h_sim) * 100.0, 2)
                res.rgb_histogram_similarity = round(float(rgb_h_sim) * 100.0, 2)
                res.hsv_cdf_similarity = round(float(hsv_cdf_sim) * 100.0, 2)
                res.rgb_cdf_similarity = round(float(rgb_cdf_sim) * 100.0, 2)
                res.hog_similarity = round(float(hog_sim) * 100.0, 2)
                res.hu_moments_similarity = round(float(hu_sim) * 100.0, 2)
                res.cell_color_similarity = round(float(cell_color_sim) * 100.0, 2)
                res.lbp_similarity = round(float(lbp_sim) * 100.0, 2)
                res.color_moments_similarity = round(float(moments_sim) * 100.0, 2)
                res.sharpness_similarity = round(float(sharp_sim) * 100.0, 2)
                res.gabor_similarity = round(float(gabor_sim) * 100.0, 2)
                res.ccv_similarity = round(float(ccv_sim) * 100.0, 2)
                res.zernike_similarity = round(float(zernike_sim) * 100.0, 2)
                res.geometric_similarity = round(float(geo_sim) * 100.0, 2)
                response_data.append(res)
                
            return {
                "query_features": query_metadata,
                "results": response_data
            }
        except Exception as e:
            logger.error(f"Error in search_similar: {e}")
            raise
