import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from .soft_assignment_hist import _soft_assignment_hist
from .gaussian_hist import _gaussian_hist
from ...core.logging import get_logger

logger = get_logger(__name__)

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
        channels = [img] if is_gray else [img[:,:,i] for i in range(3)]
        
        # --- 1D Histograms & CDFs (Std, Interp, Gauss) ---
        for method, func in [("std", None), ("interp", _soft_assignment_hist), ("gauss", _gaussian_hist)]:
            h_vec, c_vec = [], []
            for ch in channels:
                if method == "std":
                    h = cv2.normalize(cv2.calcHist([ch.astype(np.uint8)], [0], None, [bins], [0, 256]), None).flatten().tolist()
                else:
                    h = func(ch.flatten().astype(float), bins, (0, 256)).tolist()
                h_vec.extend(h)
                c_vec.extend(np.cumsum(h).tolist())
            results[f"{name}_hist_{method}"] = h_vec
            results[f"{name}_cdf_{method}"] = c_vec

        # --- Joint Histograms (3D) ---
        if not is_gray:
            hj = cv2.normalize(cv2.calcHist([img], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]), None).flatten().tolist()
            q = (img // 64)
            q_flat = (q[:,:,0] * 16 + q[:,:,1] * 4 + q[:,:,2]).flatten().astype(float)
            results.update({
                f"joint_{name}_std": hj,
                f"joint_{name}_interp": _soft_assignment_hist(q_flat, 64, (0, 64)).tolist(),
                f"joint_{name}_gauss": _gaussian_hist(q_flat, 64, (0, 64)).tolist()
            })
            
        # --- Cell Color (4x4 Grid Mean) ---
        img_std = cv2.resize(img, (256, 256))
        cell_vec = []
        for i in range(4):
            for j in range(4):
                cell = img_std[i*64:(i+1)*64, j*64:(j+1)*64]
                avg = np.mean(cell, axis=(0, 1)) if not is_gray else [np.mean(cell)]
                cell_vec.extend(avg.tolist() if not is_gray else avg)
        results[f"cell_{name}_vector"] = cell_vec

        # --- Color Moments (Mean, Std, Skew) ---
        means, stds, skews = [], [], []
        for ch in channels:
            mean = np.mean(ch)
            std = np.std(ch)
            diff = ch - mean
            skew = np.mean(diff**3) / (std**3 + 1e-7)
            means.append(float(mean))
            stds.append(float(std))
            skews.append(float(skew))
        results[f"{name}_mean"] = means
        results[f"{name}_std"] = stds
        results[f"{name}_skew"] = skews

    # --- Visualization (21 separate images) ---
    vis_paths = {}
    cell_color_vis_path = None
    if filename:
        try:
            fname = os.path.basename(filename)
            # 1. Histogram Grid
            for name, method in [(n, m) for n in spaces.keys() for m in ["std", "interp", "gauss"]]:
                fig, ax = plt.subplots(figsize=(4, 3))
                data = results[f"{name}_hist_{method}"]
                n_ch = 1 if name == "gray" else 3
                colors = ['#94a3b8'] if name == "gray" else (['#ef4444', '#10b981', '#3b82f6'] if name == "rgb" else ['#3b82f6', '#10b981', '#ef4444'])
                
                for c in range(n_ch):
                    ch_d = data[c*bins : (c+1)*bins]
                    ax.plot(range(bins), ch_d, color=colors[c], lw=2, alpha=0.8)
                    ax.fill_between(range(bins), ch_d, color=colors[c], alpha=0.1)
                
                ax.set_title(f"{name.upper()} {method.upper()}", fontsize=9, color='#94a3b8')
                ax.set_facecolor('#0f172a')
                vis_fn = f"hist_{name}_{method}_{fname}.png"
                plt.savefig(str(Path(visualizations_dir) / vis_fn), facecolor='#020617', bbox_inches='tight', dpi=100)
                plt.close(fig)
                vis_paths[f"{name}_{method}"] = f"/static/visualizations/{vis_fn}"

            # 2. Cell Color Grid
            cell_rgb = results["cell_rgb_vector"]
            grid = np.zeros((256, 256, 3), dtype=np.uint8)
            for i in range(4):
                for j in range(4):
                    r, g, b = cell_rgb[(i*4+j)*3:(i*4+j)*3+3]
                    grid[i*64:(i+1)*64, j*64:(j+1)*64] = [int(r), int(g), int(b)]
            cc_fn = f"cell_color_{fname}.png"
            cv2.imwrite(str(Path(visualizations_dir) / cc_fn), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            cell_color_vis_path = f"/static/visualizations/{cc_fn}"
        except Exception as e: logger.warning(f"Feature vis failed: {e}")
            
    results.update({"vis_path": json.dumps(vis_paths), "cell_color_vis_path": cell_color_vis_path})
    return results
