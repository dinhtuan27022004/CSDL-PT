from pathlib import Path
import uuid
import json
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from ..repositories.image_repository import ImageRepository
from .llm_service import LLMService
from .cache_service import CacheService
from ..models.image import ImageMetadata
from ..schemas.image import ImageResponse
from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import ImageProcessingError
import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile
import torch
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel, SamModel, SamProcessor, EfficientNetModel, ConvNextV2Model
import torch.nn.functional as F
import time
import asyncio
import os
import gc
import math
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage import exposure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = get_logger(__name__)

# --- Helper Functions (Stateless) ---

# --- Helper Functions (Stateless) ---

def _soft_assignment_hist(data: np.ndarray, bins: int, range_val: Tuple[float, float]) -> np.ndarray:
    """Linear interpolation histogram (Soft Assignment)"""
    hist = np.zeros(bins)
    min_val, max_val = range_val
    data_clipped = np.clip(data, min_val, max_val - 1e-7)
    
    # Calculate bin index and weights
    bin_width = (max_val - min_val) / bins
    idx_float = (data_clipped - min_val) / bin_width - 0.5
    
    idx_low = np.floor(idx_float).astype(int)
    idx_high = idx_low + 1
    
    weight_high = idx_float - idx_low
    weight_low = 1.0 - weight_high
    
    # Boundary handling
    idx_low = np.clip(idx_low, 0, bins - 1)
    idx_high = np.clip(idx_high, 0, bins - 1)
    
    # Accumulate
    np.add.at(hist, idx_low, weight_low)
    np.add.at(hist, idx_high, weight_high)
            
    return hist / (np.sum(hist) + 1e-7)

def _gaussian_hist(data: np.ndarray, bins: int, range_val: Tuple[float, float], sigma: float = None) -> np.ndarray:
    """Kernel-based histogram (Gaussian)"""
    hist = np.zeros(bins)
    min_val, max_val = range_val
    bin_centers = np.linspace(min_val, max_val, bins)
    if sigma is None:
        sigma = (max_val - min_val) / bins # Default sigma
        
    # Vectorized gaussian
    for i in range(bins):
        dist_sq = (data - bin_centers[i])**2
        weights = np.exp(-dist_sq / (2 * sigma**2))
        hist[i] = np.sum(weights)
        
    return hist / (np.sum(hist) + 1e-7)

def _extract_brightness(img_bgr: np.ndarray) -> float:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_v = np.mean(img_hsv[:, :, 2])
    return float(mean_v / 255.0)

def _extract_contrast(img_bgr: np.ndarray) -> float:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(img_gray)
    return float(min(1.0, std_dev / 127.5))

def _extract_saturation(img_bgr: np.ndarray) -> float:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_s = np.mean(img_hsv[:, :, 1])
    return float(mean_s / 255.0)

def _extract_edge_density(img_bgr: np.ndarray) -> float:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    edge_pixels = cv2.countNonZero(edges)
    total_pixels = edges.size
    return float(edge_pixels / total_pixels)

def _extract_sharpness(img_bgr: np.ndarray) -> float:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

def _extract_color_moments(img_bgr: np.ndarray) -> List[float]:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    moments = []
    for i in range(3):
        channel = img_hsv[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        diff = channel - mean
        skew = np.mean(diff**3) / (std**3 + 1e-7)
        moments.extend([float(mean), float(std), float(skew)])
    return moments

def _extract_lbp(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
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

def _extract_all_color_features(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Dict[str, Any]:
    """Unified extractor for all 7 spaces and 3 assignment methods"""
    bins = 8
    spaces = {
        "rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "hsv": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV),
        "lab": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab),
        "ycrcb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb),
        "hls": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS),
        "xyz": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2XYZ),
        "gray": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    }
    
    results = {}
    
    for name, img in spaces.items():
        is_gray = (name == "gray")
        channels = 1 if is_gray else 3
        
        # --- 1D Histograms & CDFs ---
        h_std, h_interp, h_gauss = [], [], []
        c_std, c_interp, c_gauss = [], [], []
        
        data_list = [img] if is_gray else [img[:,:,i] for i in range(3)]
        
        for channel_data in data_list:
            flat = channel_data.flatten().astype(float)
            
            # Std
            h = cv2.calcHist([channel_data.astype(np.uint8)], [0], None, [bins], [0, 256])
            h = cv2.normalize(h, h).flatten().tolist()
            h_std.extend(h)
            c_std.extend(np.cumsum(h).tolist())
            
            # Interp
            hi = _soft_assignment_hist(flat, bins, (0, 256)).tolist()
            h_interp.extend(hi)
            c_interp.extend(np.cumsum(hi).tolist())
            
            # Gauss
            hg = _gaussian_hist(flat, bins, (0, 256)).tolist()
            h_gauss.extend(hg)
            c_gauss.extend(np.cumsum(hg).tolist())
            
        results[f"{name}_hist_std"] = h_std
        results[f"{name}_hist_interp"] = h_interp
        results[f"{name}_hist_gauss"] = h_gauss
        results[f"{name}_cdf_std"] = c_std
        results[f"{name}_cdf_interp"] = c_interp
        results[f"{name}_cdf_gauss"] = c_gauss
        
        # --- Joint Histograms (3D) ---
        if not is_gray:
            # Standard
            hj = cv2.calcHist([img], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
            results[f"joint_{name}_std"] = cv2.normalize(hj, hj).flatten().tolist()
            
            # For Interp/Gauss 3D, we use a simplified version (1D of quantized joint)
            # as 3D interpolation is compute-heavy.
            # But to be thorough, we calculate it by flattening if needed.
            # Here we'll reuse the 1D Interp logic on the quantized space for now.
            q = (img // 64)
            q_flat = (q[:,:,0] * 16 + q[:,:,1] * 4 + q[:,:,2]).flatten().astype(float)
            results[f"joint_{name}_interp"] = _soft_assignment_hist(q_flat, 64, (0, 64)).tolist()
            results[f"joint_{name}_gauss"] = _gaussian_hist(q_flat, 64, (0, 64)).tolist()
            
        # --- Cell Color (4x4 Grid Mean) ---
        img_std = cv2.resize(img, (256, 256))
        cells = 4
        ch, cw = 64, 64
        cell_vec = []
        for i in range(cells):
            for j in range(cells):
                cell = img_std[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
                avg = np.mean(cell, axis=(0, 1)) if not is_gray else [np.mean(cell)]
                cell_vec.extend(avg.tolist() if not is_gray else avg)
        results[f"cell_{name}_vector"] = cell_vec

    # --- Visualization ---
    vis_path = None
    if filename:
        try:
            vis_filename = f"multi_hist_{filename}.png"
            full_vis_path = visualizations_dir / vis_filename
            
            fig, axes = plt.subplots(len(spaces), 3, figsize=(15, 2 * len(spaces)))
            for idx, (name, img) in enumerate(spaces.items()):
                # Plot Std, Interp, Gauss for first channel (or gray)
                h_std = results[f"{name}_hist_std"][:bins]
                h_interp = results[f"{name}_hist_interp"][:bins]
                h_gauss = results[f"{name}_hist_gauss"][:bins]
                
                axes[idx, 0].bar(range(bins), h_std, color='blue', alpha=0.6)
                axes[idx, 0].set_title(f"{name.upper()} Std")
                axes[idx, 1].bar(range(bins), h_interp, color='green', alpha=0.6)
                axes[idx, 1].set_title(f"{name.upper()} Interp")
                axes[idx, 2].bar(range(bins), h_gauss, color='red', alpha=0.6)
                axes[idx, 2].set_title(f"{name.upper()} Gauss")
            
            plt.tight_layout()
            plt.savefig(str(full_vis_path))
            plt.close(fig)
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Multi-Hist vis failed: {e}")
            
    results["vis_path"] = vis_path
    return results

def _extract_joint_rgb_histogram(img_bgr: np.ndarray) -> List[float]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([rgb], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().tolist()
    return hist

def _extract_hog(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128))
    fd, hog_image = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=True)
    vis_path = None
    if filename:
        try:
            hog_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            vis_filename = f"hog_{filename}.png"
            full_vis_path = visualizations_dir / vis_filename
            cv2.imwrite(str(full_vis_path), (hog_rescaled * 255).astype(np.uint8))
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"HOG vis failed: {e}")
    return fd.tolist(), vis_path

def _extract_hu_moments(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    vis_path = None
    if filename:
        try:
            vis_filename = f"hu_{filename}.png"
            full_vis_path = visualizations_dir / vis_filename
            cv2.imwrite(str(full_vis_path), thresh)
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Hu vis failed: {e}")
    return hu_log.tolist(), vis_path

def _extract_gabor(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
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
            full_vis_path = visualizations_dir / vis_filename
            vis_norm = cv2.normalize(vis_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(str(full_vis_path), vis_norm)
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Gabor vis failed: {e}")
    return features, vis_path

def _extract_ccv(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
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

def _extract_fourier_descriptors(img_bgr: np.ndarray) -> List[float]:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return [0.0] * 25
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 2: return [0.0] * 25
    
    cnt_complex = np.empty(len(cnt), dtype=complex)
    cnt_complex.real = cnt[:, 0, 0]
    cnt_complex.imag = cnt[:, 0, 1]
    fourier_result = np.fft.fft(cnt_complex)
    
    # Get magnitudes
    descriptors = np.abs(fourier_result)
    
    # Truncate to 25 if more
    if len(descriptors) > 25:
        descriptors = descriptors[:25]
    
    # DC component normalization
    if descriptors[0] != 0:
        descriptors = descriptors / descriptors[0]
    
    # Convert to list
    res = descriptors.tolist()
    
    # Pad with zeros if less than 25
    if len(res) < 25:
        res.extend([0.0] * (25 - len(res)))
        
    return res

def _extract_geometric_shape(img_bgr: np.ndarray) -> List[float]:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return [0.0] * 6
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0: return [0.0] * 6
    circularity = (4 * np.pi * area) / (perimeter**2)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / hull_area) if hull_area > 0 else 0
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = float(area / rect_area) if rect_area > 0 else 0
    aspect_ratio = float(w) / h if h > 0 else 0
    if len(cnt) >= 5:
        (x, y), (axes1, axes2), angle = cv2.fitEllipse(cnt)
        major_axis = max(axes1, axes2)
        minor_axis = min(axes1, axes2)
        eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2)) if major_axis > 0 else 0
    else: eccentricity = 0
    convexity = cv2.arcLength(hull, True) / perimeter
    
    # Ensure no NaNs are returned
    geo = [float(circularity), float(solidity), float(extent), float(aspect_ratio), float(eccentricity), float(convexity)]
    return [0.0 if np.isnan(x) else x for x in geo]

def _extract_dominant_color(img_bgr: np.ndarray) -> List[float]:
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, flags)
    return centers[0].tolist()
def _extract_cell_color(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
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
            full_vis_path = visualizations_dir / vis_filename
            cv2.imwrite(str(full_vis_path), vis_img)
            vis_path = f"/static/visualizations/{vis_filename}"
        except Exception as e:
            logger.warning(f"Cell color vis failed: {e}")
    return feature_vector, vis_path
def _extract_tamura(img_bgr: np.ndarray) -> List[float]:
    """Extracts Tamura Texture Features: Coarseness, Contrast, Directionality"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # 1. Contrast
    mean, std = cv2.meanStdDev(gray)
    mu4 = np.mean((gray - mean)**4)
    alpha4 = mu4 / (std[0][0]**4 + 1e-7)
    contrast = std[0][0] / (alpha4**0.25 + 1e-7)
    
    # 2. Directionality (Gradient based)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobelx, sobely)
    
    # Threshold magnitudes
    thresh = np.max(mag) * 0.1
    valid_angles = ang[mag > thresh]
    if len(valid_angles) > 0:
        hist, _ = np.histogram(valid_angles, bins=16, range=(0, 2*np.pi))
        directionality = 1.0 - float(np.sum(hist > (np.max(hist)*0.1)) / 16.0)
    else:
        directionality = 0.0
        
    # 3. Coarseness (Simplified)
    # Average of local variations at different scales
    coarseness = 0.0
    for k in [1, 2, 3]:
        s = 2**k
        kernel = np.ones((s, s), np.float32) / (s*s)
        mean_scaled = cv2.filter2D(gray, -1, kernel)
        coarseness += np.mean(np.abs(gray - mean_scaled))
    coarseness /= 3.0
    
    return [float(coarseness), float(contrast), float(directionality)]

def _extract_edge_orientation(img_bgr: np.ndarray) -> List[float]:
    """Extracts 5-bin Edge Orientation Histogram (N, D, C45, C135, Non)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobelx, sobely)
    
    # Quantize angles into 4 directions + 1 non-directional
    # 0: Horizontal, 1: Vertical, 2: 45 deg, 3: 135 deg
    h, w = gray.shape
    total_pixels = h * w
    bins = np.zeros(5)
    
    # Threshold for edges
    edge_mask = mag > (np.max(mag) * 0.1)
    bins[4] = (total_pixels - np.sum(edge_mask)) / total_pixels # Non-directional
    
    if np.sum(edge_mask) > 0:
        valid_angs = ang[edge_mask] * 180 / np.pi
        # Map angles to 4 bins
        # 0-22.5, 157.5-180 -> 0
        # 67.5-112.5 -> 1
        # 22.5-67.5 -> 2
        # 112.5-157.5 -> 3
        for a in valid_angs:
            a = a % 180
            if a < 22.5 or a >= 157.5: bins[0] += 1
            elif a >= 67.5 and a < 112.5: bins[1] += 1
            elif a >= 22.5 and a < 67.5: bins[2] += 1
            else: bins[3] += 1
        
        # Normalize the 4 direction bins to sum to (1 - non_directional)
        sum_dirs = np.sum(bins[:4])
        if sum_dirs > 0:
            bins[:4] = bins[:4] / sum_dirs * (1.0 - bins[4])
            
    return bins.tolist()

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

def _extract_glcm(img_bgr: np.ndarray) -> List[float]:
    """GLCM Texture features (Contrast, Correlation, Energy, Homogeneity) across 4x4 grid"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_std = cv2.resize(gray, (256, 256))
    cells = 4
    ch, cw = 64, 64
    features = []
    for i in range(cells):
        for j in range(cells):
            cell = img_std[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            # Quantize to 32 levels to save compute/memory
            cell_q = (cell // 8).astype(np.uint8)
            glcm = graycomatrix(cell_q, [1], [0], 32, symmetric=True, normed=True)
            for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
                features.append(float(graycoprops(glcm, prop)[0, 0]))
    return features

def _extract_wavelet(img_bgr: np.ndarray) -> List[float]:
    """Haar-like Wavelet Energy features (3 levels of decomposition)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    curr = cv2.resize(gray, (256, 256))
    features = []
    for _ in range(3):
        h, w = curr.shape
        ll = cv2.resize(curr, (w//2, h//2), interpolation=cv2.INTER_AREA)
        lh = np.abs(curr[0::2, 1::2] - curr[1::2, 1::2])
        hl = np.abs(curr[1::2, 0::2] - curr[1::2, 1::2])
        hh = np.abs(curr[0::2, 0::2] - curr[1::2, 1::2])
        features.extend([float(np.mean(ll)), float(np.mean(lh)), float(np.mean(hl)), float(np.mean(hh))])
        curr = ll
    return features[:12]

def _extract_ehd(img_bgr: np.ndarray) -> List[float]:
    """MPEG-7 Edge Histogram Descriptor (80 dim: 16 cells * 5 edge types)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    
    # Kernel for 5 edge types
    kernels = [
        np.array([[1, 1], [-1, -1]]),    # Vertical
        np.array([[1, -1], [1, -1]]),    # Horizontal
        np.array([[np.sqrt(2), 0], [0, -np.sqrt(2)]]), # 45 degree
        np.array([[0, np.sqrt(2)], [-np.sqrt(2), 0]]), # 135 degree
        np.array([[2, -2], [-2, 2]])     # Non-directional
    ]
    
    features = []
    cells = 4
    ch, cw = 64, 64
    for i in range(cells):
        for j in range(cells):
            cell = gray[i*ch:(i+1)*ch, j*cw:(j+1)*cw]
            cell_hist = [0.0] * 5
            for k_idx, kernel in enumerate(kernels):
                edge_map = cv2.filter2D(cell, -1, kernel)
                cell_hist[k_idx] = float(np.mean(np.abs(edge_map)))
            features.extend(cell_hist)
            
    # Normalize
    norm = np.linalg.norm(features) + 1e-7
    return (np.array(features) / norm).tolist()

def _extract_cld(img_bgr: np.ndarray) -> List[float]:
    """MPEG-7 Color Layout Descriptor (64 dim: Y channel 8x8 DCT coefficients)"""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y_chan = cv2.resize(ycrcb[:,:,0], (8, 8))
    # 8x8 DCT
    dct = cv2.dct(y_chan.astype(np.float32))
    return dct.flatten().tolist()

def _extract_spm(img_bgr: np.ndarray) -> List[float]:
    """Spatial Pyramid Matching (160 dim: 1x1 + 2x2 grids with 32-bin histograms)"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:,:,0]
    
    def get_hist(region, bins=32):
        hist = cv2.calcHist([region], [0], None, [bins], [0, 180])
        cv2.normalize(hist, hist)
        return hist.flatten().tolist()
    
    # Level 0: 1x1
    features = get_hist(h_chan)
    
    # Level 1: 2x2
    h, w = h_chan.shape
    for i in range(2):
        for j in range(2):
            cell = h_chan[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
            features.extend(get_hist(cell))
            
    return features

def _extract_saliency(img_bgr: np.ndarray) -> List[float]:
    """Visual Saliency features (32 dim)"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    # Simple Spectral Residual Saliency
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    log_mag = np.log(mag + 1e-7)
    
    # Mean filter for spectral residual
    avg_log_mag = cv2.blur(log_mag, (3, 3))
    spectral_residual = log_mag - avg_log_mag
    
    # Saliency map
    saliency = np.abs(np.fft.ifft2(np.fft.ifftshift(np.exp(spectral_residual + 1j * np.angle(fshift)))))
    saliency = cv2.GaussianBlur(saliency, (5, 5), 3)
    saliency = cv2.resize(saliency, (8, 4)) # Resize to small vector
    
    norm = np.linalg.norm(saliency) + 1e-7
    return (saliency.flatten() / norm).tolist()

def _extract_bovw(img_bgr: np.ndarray) -> List[float]:
    """Bag of Visual Words (512 dim) using SIFT features"""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=500)
        kp, des = sift.detectAndCompute(gray, None)
        
        if des is None or len(des) == 0:
            return [0.0] * 512
            
        # Try to load vocabulary if it exists
        vocab_path = os.path.join(os.getcwd(), "vocab_512.npy")
        if os.path.exists(vocab_path):
            vocab = np.load(vocab_path)
            # Find nearest visual words
            from scipy.cluster.vq import vq
            words, _ = vq(des, vocab)
            hist, _ = np.histogram(words, bins=range(513), density=True)
            return hist.tolist()
        else:
            # Fallback: Just return a normalized summary of descriptors if no vocab
            # (This is not true BoVW but prevents errors)
            summary = np.mean(des, axis=0)
            # Pad to 512 if SIFT is 128
            if len(summary) == 128:
                summary = np.tile(summary, 4)
            norm = np.linalg.norm(summary) + 1e-7
            return (summary / norm).tolist()
    except Exception as e:
        logger.error(f"BoVW extraction failed: {e}")
        return [0.0] * 512

def _extract_correlogram(img_bgr: np.ndarray) -> List[float]:
    """Simplified Color Auto-Correlogram (8 bins, 4 distances)"""
    img_small = cv2.resize(img_bgr, (64, 64))
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    quant = (hsv[:,:,0] / 22.5).astype(int) 
    distances = [1, 3, 5, 7]
    features = []
    h, w = quant.shape
    for d in distances:
        for color in range(8):
            count, total = 0, 0
            for y in range(0, h, 2):
                for x in range(0, w, 2):
                    if quant[y, x] == color:
                        total += 1
                        for dy, dx in [(0, d), (d, 0), (d, d), (-d, d)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if quant[ny, nx] == color: count += 1
            features.append(float(count / (total * 4 + 1e-7)))
    return features

# --- ImageService Class ---

class ImageService:
    """Service for image CRUD, storage, and validation with batch processing and OOM protection"""
    
    def __init__(self, 
        repository: ImageRepository,
        llm_service: LLMService,
        cache_service: CacheService
    ):
        self.repository = repository
        self.llm_service = llm_service
        self.cache = cache_service
        self.settings = get_settings()
        
        # Lane 1 & 2 Executors
        self.executor = ThreadPoolExecutor(max_workers=32)
        # Use ThreadPool for traditional features on Windows to save RAM
        self.cpu_executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Login to HuggingFace Hub for higher rate limits
        if hasattr(self.settings, 'hf_token') and self.settings.hf_token:
            try:
                login(token=self.settings.hf_token)
                logger.info("Successfully logged in to HuggingFace Hub")
            except Exception as e:
                logger.warning(f"Failed to login to HuggingFace Hub: {e}")
        
        # Models are loaded dynamically during batch processing
        self.clip_model = None
        self.clip_processor = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.siglip_model = None
        self.siglip_processor = None
        self.convnext_model = None
        self.convnext_processor = None
        self.efficientnet_model = None
        self.efficientnet_processor = None
        self.dreamsim_model = None
        self.dreamsim_preprocess = None
        self.sam_model = None
        self.sam_processor = None

    def _load_clip(self):
        """Loads CLIP model into GPU memory"""
        if self.clip_model is not None:
            return
        logger.info(f"Loading CLIP model: {self.settings.clip_model_name} on {self.device}...")
        try:
            self.clip_model = CLIPModel.from_pretrained(self.settings.clip_model_name, local_files_only=True, low_cpu_mem_usage=True).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.settings.clip_model_name, local_files_only=True, low_cpu_mem_usage=True)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")

    def _unload_clip(self):
        """Unloads CLIP model and frees GPU memory"""
        if self.clip_model is not None:
            logger.info("Unloading CLIP model...")
            del self.clip_model
            del self.clip_processor
            self.clip_model = None
            self.clip_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("CLIP model unloaded")

    def _load_dinov2(self):
        """Loads DINOv2 model into GPU memory"""
        if self.dinov2_model is not None:
            return
        logger.info(f"Loading DINOv2 model: {self.settings.dinov2_model_name} on {self.device}...")
        try:
            self.dinov2_processor = AutoImageProcessor.from_pretrained(self.settings.dinov2_model_name, local_files_only=True)
            self.dinov2_model = AutoModel.from_pretrained(self.settings.dinov2_model_name, local_files_only=True, low_cpu_mem_usage=True).to(self.device)
            logger.info("DINOv2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")

    def _unload_dinov2(self):
        """Unloads DINOv2 model and frees GPU memory"""
        if self.dinov2_model is not None:
            logger.info("Unloading DINOv2 model...")
            del self.dinov2_model
            del self.dinov2_processor
            self.dinov2_model = None
            self.dinov2_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("DINOv2 model unloaded")

    def _extract_clip_batch(self, images_pil: List[Image.Image]) -> List[List[float]]:
        """Extract CLIP features from a batch of images (Lane 3)"""
        if not self.clip_model or not self.clip_processor or not images_pil:
            return [[0.0] * 768] * len(images_pil)
        try:
            all_features = []
            batch_size = 32
            for i in range(0, len(images_pil), batch_size):
                logger.info(f"CLIP Processing batch {i // batch_size + 1} of {math.ceil(len(images_pil) / batch_size)}")
                chunk = images_pil[i:i + batch_size]
                inputs = self.clip_processor(images=chunk, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    outputs = self.clip_model.get_image_features(**inputs)
                    
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        image_embeds = outputs.pooler_output
                    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                        image_embeds = outputs.image_embeds
                    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                        image_embeds = outputs.last_hidden_state[:, 0]
                    else:
                        image_embeds = outputs
                        
                    image_embeds = image_embeds / (image_embeds.norm(p=2, dim=-1, keepdim=True) + 1e-7)
                    all_features.extend(image_embeds.cpu().numpy().tolist())
            return all_features
        except Exception as e:
            logger.error(f"Batch CLIP extraction failed: {e}")
            return [[0.0] * 768] * len(images_pil)

    def _extract_dinov2_batch(self, images_pil: List[Image.Image]) -> List[List[float]]:
        """Extract DINOv2 features from a batch of images (Lane 3)"""
        if not self.dinov2_model or not self.dinov2_processor or not images_pil:
            return [[0.0] * 1536] * len(images_pil)
        try:
            all_features = []
            batch_size = 16
            for i in range(0, len(images_pil), batch_size):
                logger.info(f"DINOv2 Processing batch {i // batch_size + 1} of {math.ceil(len(images_pil) / batch_size)}")
                chunk = images_pil[i:i + batch_size]
                inputs = self.dinov2_processor(images=chunk, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.dinov2_model(**inputs)
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        embeddings = outputs.pooler_output
                    else:
                        embeddings = outputs.last_hidden_state[:, 0]
                    embeddings = embeddings / (embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-7)
                    all_features.extend(embeddings.cpu().numpy().tolist())
            return all_features
        except Exception as e:
            logger.error(f"Batch DINOv2 extraction failed: {e}")
            return [[0.0] * 1536] * len(images_pil)

    # --- SigLIP ---
    def _load_siglip(self):
        """Loads SigLIP model into GPU memory"""
        if self.siglip_model is not None:
            return
        logger.info(f"Loading SigLIP model: {self.settings.siglip_model_name} on {self.device}...")
        try:
            self.siglip_model = AutoModel.from_pretrained(self.settings.siglip_model_name, low_cpu_mem_usage=True).to(self.device)
            self.siglip_processor = AutoImageProcessor.from_pretrained(self.settings.siglip_model_name)
            logger.info("SigLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SigLIP model: {e}")

    def _unload_siglip(self):
        if self.siglip_model is not None:
            logger.info("Unloading SigLIP model...")
            del self.siglip_model
            del self.siglip_processor
            self.siglip_model = None
            self.siglip_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("SigLIP model unloaded")

    def _extract_siglip_batch(self, images_pil: List[Image.Image]) -> List[List[float]]:
        """Extract SigLIP features from a batch of images"""
        if not self.siglip_model or not self.siglip_processor or not images_pil:
            return [[0.0] * 768] * len(images_pil)
        try:
            all_features = []
            batch_size = 32
            for i in range(0, len(images_pil), batch_size):
                logger.info(f"SigLIP Processing batch {i // batch_size + 1} of {math.ceil(len(images_pil) / batch_size)}")
                chunk = images_pil[i:i + batch_size]
                inputs = self.siglip_processor(images=chunk, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.siglip_model.get_image_features(**inputs)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeds = outputs.pooler_output
                    elif isinstance(outputs, torch.Tensor):
                        embeds = outputs
                    else:
                        embeds = outputs.last_hidden_state[:, 0]
                    embeds = embeds / (embeds.norm(p=2, dim=-1, keepdim=True) + 1e-7)
                    all_features.extend(embeds.cpu().numpy().tolist())
            return all_features
        except Exception as e:
            logger.error(f"Batch SigLIP extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [[0.0] * 768] * len(images_pil)

    # --- ConvNeXt V2 ---
    def _load_convnext(self):
        """Loads ConvNeXt V2 model into GPU memory"""
        if self.convnext_model is not None:
            return
        logger.info(f"Loading ConvNeXt V2 model: {self.settings.convnext_model_name} on {self.device}...")
        try:
            self.convnext_model = ConvNextV2Model.from_pretrained(self.settings.convnext_model_name, low_cpu_mem_usage=True).to(self.device)
            self.convnext_processor = AutoImageProcessor.from_pretrained(self.settings.convnext_model_name)
            logger.info("ConvNeXt V2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ConvNeXt V2 model: {e}")

    def _unload_convnext(self):
        if self.convnext_model is not None:
            logger.info("Unloading ConvNeXt V2 model...")
            del self.convnext_model
            del self.convnext_processor
            self.convnext_model = None
            self.convnext_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("ConvNeXt V2 model unloaded")

    def _extract_convnext_batch(self, images_pil: List[Image.Image]) -> List[List[float]]:
        """Extract ConvNeXt V2 features from a batch of images"""
        if not self.convnext_model or not self.convnext_processor or not images_pil:
            return [[0.0] * 1024] * len(images_pil)
        try:
            all_features = []
            batch_size = 16
            for i in range(0, len(images_pil), batch_size):
                logger.info(f"ConvNeXt Processing batch {i // batch_size + 1} of {math.ceil(len(images_pil) / batch_size)}")
                chunk = images_pil[i:i + batch_size]
                inputs = self.convnext_processor(images=chunk, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.convnext_model(**inputs)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeds = outputs.pooler_output
                    else:
                        embeds = outputs.last_hidden_state.mean(dim=[2, 3])
                    embeds = embeds / (embeds.norm(p=2, dim=-1, keepdim=True) + 1e-7)
                    all_features.extend(embeds.cpu().numpy().tolist())
            return all_features
        except Exception as e:
            logger.error(f"Batch ConvNeXt extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [[0.0] * 1024] * len(images_pil)

    # --- EfficientNet ---
    def _load_efficientnet(self):
        """Loads EfficientNet-B7 model into GPU memory"""
        if self.efficientnet_model is not None:
            return
        logger.info(f"Loading EfficientNet model: {self.settings.efficientnet_model_name} on {self.device}...")
        try:
            self.efficientnet_model = EfficientNetModel.from_pretrained(self.settings.efficientnet_model_name, low_cpu_mem_usage=True).to(self.device)
            self.efficientnet_processor = AutoImageProcessor.from_pretrained(self.settings.efficientnet_model_name)
            logger.info("EfficientNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model: {e}")

    def _unload_efficientnet(self):
        if self.efficientnet_model is not None:
            logger.info("Unloading EfficientNet model...")
            del self.efficientnet_model
            del self.efficientnet_processor
            self.efficientnet_model = None
            self.efficientnet_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("EfficientNet model unloaded")

    def _extract_efficientnet_batch(self, images_pil: List[Image.Image]) -> List[List[float]]:
        """Extract EfficientNet-B7 features from a batch of images"""
        if not self.efficientnet_model or not self.efficientnet_processor or not images_pil:
            return [[0.0] * 2560] * len(images_pil)
        try:
            all_features = []
            batch_size = 8
            for i in range(0, len(images_pil), batch_size):
                logger.info(f"EfficientNet Processing batch {i // batch_size + 1} of {math.ceil(len(images_pil) / batch_size)}")
                chunk = images_pil[i:i + batch_size]
                inputs = self.efficientnet_processor(images=chunk, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.efficientnet_model(**inputs)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeds = outputs.pooler_output
                    else:
                        embeds = outputs.last_hidden_state.mean(dim=[2, 3])
                    embeds = embeds / (embeds.norm(p=2, dim=-1, keepdim=True) + 1e-7)
                    all_features.extend(embeds.cpu().numpy().tolist())
            return all_features
        except Exception as e:
            logger.error(f"Batch EfficientNet extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [[0.0] * 2560] * len(images_pil)

    # --- DreamSim ---
    def _load_dreamsim(self):
        """Loads DreamSim Ensemble model into GPU memory"""
        if self.dreamsim_model is not None:
            return
        logger.info(f"Loading DreamSim Ensemble model on {self.device}...")
        try:
            from dreamsim import dreamsim
            self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True, device=self.device)
            logger.info("DreamSim model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DreamSim model: {e}")

    def _unload_dreamsim(self):
        if self.dreamsim_model is not None:
            logger.info("Unloading DreamSim model...")
            del self.dreamsim_model
            del self.dreamsim_preprocess
            self.dreamsim_model = None
            self.dreamsim_preprocess = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("DreamSim model unloaded")

    def _extract_dreamsim_batch(self, images_pil: List[Image.Image]) -> List[List[float]]:
        """Extract DreamSim Ensemble features from a batch of images"""
        if not self.dreamsim_model or not self.dreamsim_preprocess or not images_pil:
            return [[0.0] * 1792] * len(images_pil)
        try:
            all_features = []
            for i, img in enumerate(images_pil):
                logger.info(f"DreamSim Processing image {i + 1} of {len(images_pil)}")
                preprocessed = self.dreamsim_preprocess(img).to(self.device)
                with torch.no_grad():
                    embedding = self.dreamsim_model.embed(preprocessed)
                    embedding = embedding / (embedding.norm(p=2, dim=-1, keepdim=True) + 1e-7)
                    all_features.append(embedding.cpu().numpy().flatten().tolist())
            return all_features
        except Exception as e:
            logger.error(f"Batch DreamSim extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [[0.0] * 1792] * len(images_pil)

    # --- SAM (Segment Anything) ---
    def _load_sam(self):
        """Loads SAM ViT-B model into GPU memory"""
        if self.sam_model is not None:
            return
        logger.info(f"Loading SAM model: {self.settings.sam_model_name} on {self.device}...")
        try:
            self.sam_model = SamModel.from_pretrained(self.settings.sam_model_name, low_cpu_mem_usage=True).to(self.device)
            self.sam_processor = SamProcessor.from_pretrained(self.settings.sam_model_name)
            logger.info("SAM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")

    def _unload_sam(self):
        if self.sam_model is not None:
            logger.info("Unloading SAM model...")
            del self.sam_model
            del self.sam_processor
            self.sam_model = None
            self.sam_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("SAM model unloaded")

    def _extract_sam_batch(self, images_pil: List[Image.Image]) -> List[List[float]]:
        """Extract SAM high-fidelity features from a batch of images (256x7x7 = 12544-dim)"""
        if not self.sam_model or not self.sam_processor or not images_pil:
            return [[0.0] * 12544] * len(images_pil)
        try:
            all_features = []
            for i, img in enumerate(images_pil):
                logger.info(f"SAM Processing image {i + 1} of {len(images_pil)}")
                inputs = self.sam_processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.sam_model.get_image_embeddings(inputs["pixel_values"])
                    # outputs shape: (1, 256, 64, 64) → pool to (1, 256, 7, 7) → flatten to 12544
                    pooled = F.adaptive_avg_pool2d(outputs, (7, 7))
                    flat = pooled.flatten(start_dim=1)
                    flat = flat / (flat.norm(p=2, dim=-1, keepdim=True) + 1e-7)
                    all_features.append(flat.cpu().numpy().flatten().tolist())
            return all_features
        except Exception as e:
            logger.error(f"Batch SAM extraction failed: {e}")
            return [[0.0] * 12544] * len(images_pil)

    def get_images(self, db: Session, limit: int = 500, offset: int = 0) -> List[ImageMetadata]:
        return self.repository.get_all(db, limit=limit, offset=offset)

    async def extract_features_batch(
        self, 
        images_bytes: List[bytes], 
        filenames: List[Optional[str]], 
        force_llm: bool = False,
        required_features: Optional[Set[str]] = None
    ) -> List[ImageMetadata]:
        """Memory-safe concurrent feature extraction using 4-lane pipeline with selective extraction support"""
        # Determine if we need to run specific lanes or models
        run_lane2 = True # API Lane (Metadata) usually always needed for display
        run_lane3 = True # GPU Lane
        run_lane4 = True # LLM Lane
        
        if required_features is not None:
            # Check if any GPU models are needed
            gpu_models = {"clip", "dinov2", "siglip", "convnext", "efficientnet", "dreamsim", "sam"}
            run_lane3 = any(f in required_features for f in gpu_models)
            run_lane4 = "semantic" in required_features or force_llm
            # Lane 1 (CPU) is light, we always run it to get basic stats

        n = len(images_bytes)
        logger.info(f"Starting 4-Lane pipeline for N={n} images")
        
        img_smalls = []
        cached_vectors = [{} for _ in range(n)]

        for i in range(n):
            file_bytes = np.frombuffer(images_bytes[i], dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None: raise ValueError(f"Failed to decode image at index {i}")
            
            small = cv2.resize(img, (int(img.shape[1]/6), int(img.shape[0]/6)))
            img_smalls.append(small)
            
        for i, img in enumerate(img_smalls):
            cached_vectors[i]["width"] = img.shape[1]
            cached_vectors[i]["height"] = img.shape[0]

        all_indices = list(range(n))
        
        logger.info(f"Processing Features: All 7 GPU models will be extracted for {n} images")

        loop = asyncio.get_running_loop()

        # ============================================================
        # LANE 1: Traditional CPU Feature Extraction (runs in parallel)
        # ============================================================
        async def run_cpu_lane():
            logger.info("Lane 1: Starting Traditional CPU Feature Extraction...")
            cpu_start = time.time()
            all_results = []
            for i in range(n):
                logger.info(f"Lane 1: Extracting features for image {i+1}/{n}...")
                t = [
                    loop.run_in_executor(self.cpu_executor, _extract_brightness, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_contrast, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_saturation, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_edge_density, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_all_color_features, img_smalls[i], self.settings.visualizations_dir, filenames[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_hog, img_smalls[i], self.settings.visualizations_dir, filenames[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_hu_moments, img_smalls[i], self.settings.visualizations_dir, filenames[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_dominant_color, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_lbp, img_smalls[i], self.settings.visualizations_dir, filenames[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_color_moments, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_sharpness, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_gabor, img_smalls[i], self.settings.visualizations_dir, filenames[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_ccv, img_smalls[i], self.settings.visualizations_dir, filenames[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_fourier_descriptors, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_geometric_shape, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_tamura, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_edge_orientation, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_glcm, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_wavelet, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_correlogram, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_ehd, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_cld, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_spm, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_saliency, img_smalls[i]),
                    loop.run_in_executor(self.cpu_executor, _extract_bovw, img_smalls[i])
                ]
                try:
                    image_features = await asyncio.gather(*t)
                    all_results.append(image_features)
                except Exception as e:
                    logger.error(f"Error extracting traditional features for image {i+1} ({filenames[i]}): {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
            logger.info(f"Lane 1: CPU Traditional Features Completed in {time.time() - cpu_start:.2f}s")
            return all_results

        # ============================================================
        # LANE 2: LLM Vision API (runs in parallel)
        # ============================================================
        async def run_api_lane():
            if not run_lane2: return [{} for _ in range(n)]
            logger.info("Lane 2: Starting LLM Vision (VLM) Analysis...")
            vlm_start = time.time()
            names = [filenames[i] or f"upload_{i}_{int(time.time())}" for i in range(n)]
            res = await self.llm_service.analyze_vision_batch(images_bytes, names)
            logger.info(f"Lane 2: VLM Analysis Completed in {time.time() - vlm_start:.2f}s")
            return res

        # ============================================================
        # LANE 3: Sequential GPU Model Extraction (runs in parallel with Lane 1 & 2)
        # Models: CLIP → DINOv2 → SigLIP → ConvNeXt → EfficientNet → DreamSim → SAM
        # ============================================================
        async def run_gpu_lane():
            if not run_lane3:
                logger.info("Lane 3: Skipping all GPU models (not required by weights)")
                return {m: [None] * n for m in ["clip", "dinov2", "siglip", "convnext", "efficientnet", "dreamsim", "sam"]}

            gpu_start = time.time()
            results = {m: [None] * n for m in ["clip", "dinov2", "siglip", "convnext", "efficientnet", "dreamsim", "sam"]}
            
            # Prepare PIL images once for all GPU models
            batch_images = [Image.fromarray(cv2.cvtColor(img_smalls[i], cv2.COLOR_BGR2RGB)) for i in all_indices]
            
            # --- Model 1: CLIP ---
            if required_features is None or "clip" in required_features:
                model_start = time.time()
                self._load_clip()
                clip_vectors = self._extract_clip_batch(batch_images)
                for idx, vec in zip(all_indices, clip_vectors):
                    results["clip"][idx] = vec
                self._unload_clip()
                logger.info(f"Lane 3: CLIP completed in {time.time() - model_start:.2f}s")
            else:
                logger.info("Lane 3: Skipping CLIP (weight is 0)")
            
            # --- Model 2: DINOv2 ---
            if required_features is None or "dinov2" in required_features:
                model_start = time.time()
                self._load_dinov2()
                dinov2_vectors = self._extract_dinov2_batch(batch_images)
                for idx, vec in zip(all_indices, dinov2_vectors):
                    results["dinov2"][idx] = vec
                self._unload_dinov2()
                logger.info(f"Lane 3: DINOv2 completed in {time.time() - model_start:.2f}s")
            else:
                logger.info("Lane 3: Skipping DINOv2 (weight is 0)")
            
            # --- Model 3: SigLIP ---
            if required_features is None or "siglip" in required_features:
                model_start = time.time()
                self._load_siglip()
                siglip_vectors = self._extract_siglip_batch(batch_images)
                for idx, vec in zip(all_indices, siglip_vectors):
                    results["siglip"][idx] = vec
                self._unload_siglip()
                logger.info(f"Lane 3: SigLIP completed in {time.time() - model_start:.2f}s")
            else:
                logger.info("Lane 3: Skipping SigLIP")
            
            # --- Model 4: ConvNeXt V2 ---
            if required_features is None or "convnext" in required_features:
                model_start = time.time()
                self._load_convnext()
                convnext_vectors = self._extract_convnext_batch(batch_images)
                for idx, vec in zip(all_indices, convnext_vectors):
                    results["convnext"][idx] = vec
                self._unload_convnext()
                logger.info(f"Lane 3: ConvNeXt V2 completed in {time.time() - model_start:.2f}s")
            else:
                logger.info("Lane 3: Skipping ConvNeXt V2")
            
            # --- Model 5: EfficientNet ---
            if required_features is None or "efficientnet" in required_features:
                model_start = time.time()
                self._load_efficientnet()
                efficientnet_vectors = self._extract_efficientnet_batch(batch_images)
                for idx, vec in zip(all_indices, efficientnet_vectors):
                    results["efficientnet"][idx] = vec
                self._unload_efficientnet()
                logger.info(f"Lane 3: EfficientNet completed in {time.time() - model_start:.2f}s")
            else:
                logger.info("Lane 3: Skipping EfficientNet")
            
            # --- Model 6: DreamSim ---
            if required_features is None or "dreamsim" in required_features:
                model_start = time.time()
                self._load_dreamsim()
                dreamsim_vectors = self._extract_dreamsim_batch(batch_images)
                for idx, vec in zip(all_indices, dreamsim_vectors):
                    results["dreamsim"][idx] = vec
                self._unload_dreamsim()
                logger.info(f"Lane 3: DreamSim completed in {time.time() - model_start:.2f}s")
            else:
                logger.info("Lane 3: Skipping DreamSim")
            
            # --- Model 7: SAM ---
            if required_features is None or "sam" in required_features:
                model_start = time.time()
                self._load_sam()
                sam_vectors = self._extract_sam_batch(batch_images)
                for idx, vec in zip(all_indices, sam_vectors):
                    results["sam"][idx] = vec
                self._unload_sam()
                logger.info(f"Lane 3: SAM completed in {time.time() - model_start:.2f}s")
            else:
                logger.info("Lane 3: Skipping SAM")
            
            # Clean up PIL images
            del batch_images
            gc.collect()
            
            logger.info(f"Lane 3: All 7 GPU Models Completed in {time.time() - gpu_start:.2f}s")
            return results

        # ============================================================
        # EXECUTE: Lane 1, 2, 3 in parallel
        # ============================================================
        lane1_task = asyncio.create_task(run_cpu_lane())
        lane2_task = asyncio.create_task(run_api_lane())
        lane3_task = asyncio.create_task(run_gpu_lane())
        
        # Wait for all 3 lanes to complete
        lane1_res, lane2_res, lane3_res = await asyncio.gather(lane1_task, lane2_task, lane3_task)
        
        # Free img_smalls after all lanes are done
        img_smalls.clear()
        gc.collect()

        # ============================================================
        # LANE 4: LLM Text Embedding (runs after Lane 2 completes with descriptions)
        # ============================================================
        llm_embeddings = [None] * n
        if run_lane4:
            logger.info("Lane 4: Starting LLM Text Embedding...")
            lane4_start = time.time()
            texts_to_embed = []
            for vd in lane2_res:
                combined = f"Category: {vd.get('category')}. Entities: {vd.get('entities')}. Description: {vd.get('description')}"
                texts_to_embed.append(combined)
            llm_embeddings = self.llm_service.extract_embeddings_batch(texts_to_embed)
            logger.info(f"Lane 4: LLM Text Embedding Completed in {time.time() - lane4_start:.2f}s")
        else:
            logger.info("Lane 4: Skipping LLM Embedding (not required by weights)")

        # ============================================================
        # ASSEMBLE: Combine all results into ImageMetadata
        # ============================================================
        final_metadatas = []
        try:
            for i in range(n):
                c_res = lane1_res[i]
                v_res = lane2_res[i]
                
                # Debugging dimensions
                if i == 0:
                    logger.info(f"DEBUG: Vector dimensions for image 0:")
                    logger.info(f"  - CLIP: {len(lane3_res['clip'][i]) if lane3_res['clip'][i] is not None else 'Skipped'}")
                    logger.info(f"  - DINOv2: {len(lane3_res['dinov2'][i]) if lane3_res['dinov2'][i] is not None else 'Skipped'}")
                    logger.info(f"  - SigLIP: {len(lane3_res['siglip'][i]) if lane3_res['siglip'][i] is not None else 'Skipped'}")
                    logger.info(f"  - ConvNeXt: {len(lane3_res['convnext'][i]) if lane3_res['convnext'][i] is not None else 'Skipped'}")
                    logger.info(f"  - EfficientNet: {len(lane3_res['efficientnet'][i]) if lane3_res['efficientnet'][i] is not None else 'Skipped'}")
                    logger.info(f"  - DreamSim: {len(lane3_res['dreamsim'][i]) if lane3_res['dreamsim'][i] is not None else 'Skipped'}")
                    logger.info(f"  - SAM: {len(lane3_res['sam'][i]) if lane3_res['sam'][i] is not None else 'Skipped'}")
                    logger.info(f"  - LLM: {len(llm_embeddings[i]) if llm_embeddings[i] is not None else 'Skipped'}")

                color_feat = c_res[4]
                final_metadatas.append(ImageMetadata(
                    file_name = filenames[i],
                    brightness = c_res[0],
                    contrast = c_res[1],
                    saturation = c_res[2],
                    edge_density = c_res[3],
                    
                    # RGB
                    rgb_hist_std = color_feat["rgb_hist_std"],
                    rgb_hist_interp = color_feat["rgb_hist_interp"],
                    rgb_hist_gauss = color_feat["rgb_hist_gauss"],
                    rgb_cdf_std = color_feat["rgb_cdf_std"],
                    rgb_cdf_interp = color_feat["rgb_cdf_interp"],
                    rgb_cdf_gauss = color_feat["rgb_cdf_gauss"],
                    joint_rgb_std = color_feat["joint_rgb_std"],
                    joint_rgb_interp = color_feat["joint_rgb_interp"],
                    joint_rgb_gauss = color_feat["joint_rgb_gauss"],
                    cell_rgb_vector = color_feat["cell_rgb_vector"],

                    # HSV
                    hsv_hist_std = color_feat["hsv_hist_std"],
                    hsv_hist_interp = color_feat["hsv_hist_interp"],
                    hsv_hist_gauss = color_feat["hsv_hist_gauss"],
                    hsv_cdf_std = color_feat["hsv_cdf_std"],
                    hsv_cdf_interp = color_feat["hsv_cdf_interp"],
                    hsv_cdf_gauss = color_feat["hsv_cdf_gauss"],
                    joint_hsv_std = color_feat["joint_hsv_std"],
                    joint_hsv_interp = color_feat["joint_hsv_interp"],
                    joint_hsv_gauss = color_feat["joint_hsv_gauss"],
                    cell_hsv_vector = color_feat["cell_hsv_vector"],

                    # Lab
                    lab_hist_std = color_feat["lab_hist_std"],
                    lab_hist_interp = color_feat["lab_hist_interp"],
                    lab_hist_gauss = color_feat["lab_hist_gauss"],
                    lab_cdf_std = color_feat["lab_cdf_std"],
                    lab_cdf_interp = color_feat["lab_cdf_interp"],
                    lab_cdf_gauss = color_feat["lab_cdf_gauss"],
                    joint_lab_std = color_feat["joint_lab_std"],
                    joint_lab_interp = color_feat["joint_lab_interp"],
                    joint_lab_gauss = color_feat["joint_lab_gauss"],
                    cell_lab_vector = color_feat["cell_lab_vector"],

                    # YCrCb
                    ycrcb_hist_std = color_feat["ycrcb_hist_std"],
                    ycrcb_hist_interp = color_feat["ycrcb_hist_interp"],
                    ycrcb_hist_gauss = color_feat["ycrcb_hist_gauss"],
                    ycrcb_cdf_std = color_feat["ycrcb_cdf_std"],
                    ycrcb_cdf_interp = color_feat["ycrcb_cdf_interp"],
                    ycrcb_cdf_gauss = color_feat["ycrcb_cdf_gauss"],
                    joint_ycrcb_std = color_feat["joint_ycrcb_std"],
                    joint_ycrcb_interp = color_feat["joint_ycrcb_interp"],
                    joint_ycrcb_gauss = color_feat["joint_ycrcb_gauss"],
                    cell_ycrcb_vector = color_feat["cell_ycrcb_vector"],

                    # HLS
                    hls_hist_std = color_feat["hls_hist_std"],
                    hls_hist_interp = color_feat["hls_hist_interp"],
                    hls_hist_gauss = color_feat["hls_hist_gauss"],
                    hls_cdf_std = color_feat["hls_cdf_std"],
                    hls_cdf_interp = color_feat["hls_cdf_interp"],
                    hls_cdf_gauss = color_feat["hls_cdf_gauss"],
                    joint_hls_std = color_feat["joint_hls_std"],
                    joint_hls_interp = color_feat["joint_hls_interp"],
                    joint_hls_gauss = color_feat["joint_hls_gauss"],
                    cell_hls_vector = color_feat["cell_hls_vector"],

                    # XYZ
                    xyz_hist_std = color_feat["xyz_hist_std"],
                    xyz_hist_interp = color_feat["xyz_hist_interp"],
                    xyz_hist_gauss = color_feat["xyz_hist_gauss"],
                    xyz_cdf_std = color_feat["xyz_cdf_std"],
                    xyz_cdf_interp = color_feat["xyz_cdf_interp"],
                    xyz_cdf_gauss = color_feat["xyz_cdf_gauss"],
                    joint_xyz_std = color_feat["joint_xyz_std"],
                    joint_xyz_interp = color_feat["joint_xyz_interp"],
                    joint_xyz_gauss = color_feat["joint_xyz_gauss"],
                    cell_xyz_vector = color_feat["cell_xyz_vector"],

                    # Gray
                    gray_hist_std = color_feat["gray_hist_std"],
                    gray_hist_interp = color_feat["gray_hist_interp"],
                    gray_hist_gauss = color_feat["gray_hist_gauss"],
                    gray_cdf_std = color_feat["gray_cdf_std"],
                    gray_cdf_interp = color_feat["gray_cdf_interp"],
                    gray_cdf_gauss = color_feat["gray_cdf_gauss"],
                    cell_gray_vector = color_feat["cell_gray_vector"],

                    histogram_vis_path = color_feat["vis_path"],
                    hog_vector = c_res[5][0],
                    hog_vis_path = c_res[5][1],
                    hu_moments_vector = c_res[6][0],
                    hu_vis_path = c_res[6][1],
                    dominant_color_vector = c_res[7],
                    lbp_vector = c_res[8][0],
                    lbp_vis_path = c_res[8][1],
                    color_moments_vector = c_res[9],
                    sharpness = c_res[10],
                    gabor_vector = c_res[11][0],
                    gabor_vis_path = c_res[11][1],
                    ccv_vector = c_res[12][0],
                    ccv_vis_path = c_res[12][1],
                    zernike_vector = c_res[13],
                    geo_vector = c_res[14],
                    tamura_vector = c_res[15],
                    edge_orientation_vector = c_res[16],
                    glcm_vector = c_res[17],
                    wavelet_vector = c_res[18],
                    correlogram_vector = c_res[19],
                    ehd_vector = c_res[20],
                    cld_vector = c_res[21],
                    spm_vector = c_res[22],
                    saliency_vector = c_res[23],
                    bovw_vector = c_res[24],

                    width = cached_vectors[i].get("width"),
                    height = cached_vectors[i].get("height"),
                    category = v_res.get("category"),
                    description = v_res.get("description"),
                    entities = v_res.get("entities"),
                    llm_embedding = llm_embeddings[i],
                    clip_vector = lane3_res["clip"][i],
                    dinov2_vector = lane3_res["dinov2"][i],
                    siglip_vector = lane3_res["siglip"][i],
                    convnext_vector = lane3_res["convnext"][i],
                    efficientnet_vector = lane3_res["efficientnet"][i],
                    dreamsim_vector = lane3_res["dreamsim"][i],
                    sam_vector = lane3_res["sam"][i],
                ))
        except Exception as e:
            logger.error(f"Error during metadata assembly at index {i}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        return final_metadatas

    async def extract_features(
        self, 
        image: bytes, 
        filename: Optional[str] = None, 
        force_llm: bool = False,
        required_features: Optional[Set[str]] = None
    ) -> ImageMetadata:
        """Backward compatibility for single image feature extraction with selective support"""
        results = await self.extract_features_batch([image], [filename], force_llm=force_llm, required_features=required_features)
        return results[0]

    async def process(
        self, 
        db: Session, 
        files: List[UploadFile],
        force_llm: bool = False
    ) -> List[ImageMetadata]:
        """Process multiple uploaded images using the batch pipeline"""
        logger.info(f"Processing batch of {len(files)} images")
        start_time = time.time()
        
        batch_size = 128
        final_results = []
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            images_data = []
            filenames = []
            
            for file in batch_files:
                content = await file.read()
                images_data.append(content)
                filenames.append(file.filename)
                
            logger.info(f"--- Processing Sub-batch {i//batch_size + 1} / {math.ceil(len(files)/batch_size)} ---")
            results_metadata = await self.extract_features_batch(images_data, filenames, force_llm=force_llm)
            
            for j, metadata in enumerate(results_metadata):
                result = self.repository.create(db, metadata, images_data[j])
                final_results.append(result)
            
            # Giải phóng RAM cho sub-batch
            images_data.clear()
            import gc
            gc.collect()
            
        logger.info(f"Batch processing completed {len(final_results)} images in {time.time() - start_time:.2f}s")
        return final_results

    async def search_similar(
        self, 
        db: Session, 
        query_image_content: bytes, 
        filename: str,
        search_settings: Any,
        limit: int = 50,
        force_llm: bool = False
    ) -> Dict[str, Any]:
        """Hybrid Search with selective feature extraction based on active weights"""
        try:
            # 1. Determine required features from weights
            required_features = None
            mode = getattr(search_settings, 'mode', 'optimized')
            
            if mode == "optimized":
                target = getattr(search_settings, 'optimization_target', 'clip')
                weights = self.repository._load_weights(target)
                required_features = {k for k, v in weights.items() if v != 0}
                
                # IMPORTANT: If comparison mode is ON, we MUST extract the target model's features
                # even if it has 0 weight in the optimized set
                if getattr(search_settings, 'compare_with_gt', False):
                    required_features.add(target)
                
                logger.info(f"Optimized mode ({target}): Required features = {required_features}")
            elif mode == "manual":
                weights = getattr(search_settings, 'weights', {})
                required_features = {k for k, v in weights.items() if v > 0}
                logger.info(f"Manual mode: Required features = {required_features}")
            else:
                logger.info("Equal mode: All features required")

            # 2. Extract only necessary features for the query image
            query_metadata = await self.extract_features(
                query_image_content, 
                filename, 
                force_llm=force_llm, 
                required_features=required_features
            )
            search_results, gt_search_results = self.repository.search(
                db=db, 
                query_metadata=query_metadata,
                search_settings=search_settings,
                limit=limit
            )
            
            response_data = []
            response_data = []
            for row in search_results:
                record = row[0]
                total_sim = row[1]
                row_dict = row._asdict()
                
                res = ImageResponse.model_validate(record)
                res.similarity = round(float(total_sim or 0.0) * 100.0, 2)
                
                # Map all similarity scores dynamically
                for key, val in row_dict.items():
                    if key.endswith('_sim'):
                        field_name = key.replace('_sim', '_similarity')
                        if key == "semantic_sim":
                            res.semantic_similarity = round(float(val or 0.0) * 100.0, 2)
                        elif hasattr(res, field_name):
                            setattr(res, field_name, round(float(val or 0.0) * 100.0, 2))
                
                # Ensure existing specific mappings are correct
                res.dominant_color_similarity = round(float(row_dict.get('dominant_color_sim', 0)) * 100.0, 2)
                
                response_data.append(res)
            
            gt_response_data = None
            if gt_search_results:
                gt_response_data = []
                for row in gt_search_results:
                    record, sim = row
                    res = ImageResponse.model_validate(record)
                    res.similarity = round(float(sim or 0.0) * 100.0, 2)
                    gt_response_data.append(res)
                    
            return {
                "query_image": query_metadata,
                "results": response_data,
                "gt_results": gt_response_data
            }
        except Exception as e:
            logger.error(f"Error in search_similar: {e}")
            raise
