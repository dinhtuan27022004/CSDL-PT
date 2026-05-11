import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from .soft_assignment_hist import _soft_assignment_hist
from .soft_assignment_hist_3d import _soft_assignment_hist_3d
from ...core.logging import get_logger

logger = get_logger(__name__)

def _extract_all_color_features(img_bgr: np.ndarray, visualizations_dir: Path, filename: Optional[str] = None) -> Dict[str, Any]:
    """Consolidated extractor for color meta-vectors using Interpolated method only"""
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
    
    # Meta-vector accumulators
    meta_hist, meta_cdf, meta_joint, meta_cell = [], [], [], []
    meta_mean, meta_std, meta_skew = [], [], []
    
    # For visualization mapping
    vis_data = {} # space -> hist_data

    for name, img in spaces.items():
        is_gray = (name == "gray")
        channels = [img] if is_gray else [img[:,:,i] for i in range(3)]
        
        # 1. 1D Histograms & CDFs (Interpolated)
        space_hists = []
        for ch in channels:
            h = _soft_assignment_hist(ch.flatten().astype(float), bins, (0, 256))
            h_list = h.tolist()
            space_hists.extend(h_list)
            meta_hist.extend(h_list)
            # Segmented CDF: resets per channel
            meta_cdf.extend(np.cumsum(h).tolist())
        vis_data[name] = space_hists
            
        # 2. Joint Histograms (3D Trilinear Soft Assignment)
        if not is_gray:
            hj = _soft_assignment_hist_3d(img, bins=4, range_val=(0, 256))
            meta_joint.extend(hj.tolist())
            
        # 3. Cell Color (4x4 Grid Mean)
        img_std = cv2.resize(img, (256, 256))
        for i in range(4):
            for j in range(4):
                cell = img_std[i*64:(i+1)*64, j*64:(j+1)*64]
                avg = np.mean(cell, axis=(0, 1)) if not is_gray else [np.mean(cell)]
                meta_cell.extend(avg.tolist() if not is_gray else avg)

        # 4. Color Moments (Mean, Std, Skew)
        for ch in channels:
            m = np.mean(ch)
            s = np.std(ch)
            diff = ch - m
            k = np.mean(diff**3) / (s**3 + 1e-7)
            meta_mean.append(float(m))
            meta_std.append(float(s))
            meta_skew.append(float(k))

    results = {
        "meta_hist_interp": meta_hist,
        "meta_cdf_interp": meta_cdf,
        "meta_joint_interp": meta_joint,
        "meta_cell_vector": meta_cell,
        "meta_moments_mean": meta_mean,
        "meta_moments_std": meta_std,
        "meta_moments_skew": meta_skew
    }

    # --- Visualization ---
    vis_paths = {}
    cell_color_vis_path = None
    if filename:
        try:
            fname = os.path.basename(filename)
            # 1. Histogram Visualization (Interpolated only)
            for name, data in vis_data.items():
                fig, ax = plt.subplots(figsize=(4, 3))
                n_ch = 1 if name == "gray" else 3
                colors = ['#94a3b8'] if name == "gray" else (['#ef4444', '#10b981', '#3b82f6'] if name == "rgb" else ['#3b82f6', '#10b981', '#ef4444'])
                
                for c in range(n_ch):
                    ch_d = data[c*bins : (c+1)*bins]
                    ax.plot(range(bins), ch_d, color=colors[c], lw=2, alpha=0.8)
                    ax.fill_between(range(bins), ch_d, color=colors[c], alpha=0.1)
                
                ax.set_title(f"{name.upper()} INTERP", fontsize=9, color='#94a3b8')
                ax.set_facecolor('#0f172a')
                vis_fn = f"hist_{name}_interp_{fname}.png"
                plt.savefig(str(Path(visualizations_dir) / vis_fn), facecolor='#020617', bbox_inches='tight', dpi=100)
                plt.close(fig)
                vis_paths[name] = f"/static/visualizations/{vis_fn}"

            # 2. Cell Color Grid Visualization
            cell_rgb = meta_cell[:48] # First 48 are RGB
            grid = np.zeros((256, 256, 3), dtype=np.uint8)
            for i in range(4):
                for j in range(4):
                    r, g, b = cell_rgb[(i*4+j)*3:(i*4+j)*3+3]
                    grid[i*64:(i+1)*64, j*64:(j+1)*64] = [int(r), int(g), int(b)]
            cc_fn = f"cell_color_{fname}.png"
            cv2.imwrite(str(Path(visualizations_dir) / cc_fn), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            cell_color_vis_path = f"/static/visualizations/{cc_fn}"
        except Exception as e: 
            logger.warning(f"Feature vis failed: {e}")
            
    results.update({"vis_path": json.dumps(vis_paths), "cell_color_vis_path": cell_color_vis_path})
    return results
