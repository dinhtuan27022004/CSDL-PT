import os
import time
import asyncio
import io
from typing import List, Optional, Tuple, Set
import numpy as np
from PIL import Image as PILImage
import cv2

from ...core.logging import get_logger
from ..extractors import (
    _extract_brightness, _extract_contrast, _extract_saturation, 
    _extract_edge_density, _extract_sharpness, _extract_lbp, 
    _extract_all_color_features, _extract_hog, _extract_hu_moments, 
    _extract_gabor, _extract_ccv, _extract_fourier_descriptors,
    _extract_geometric_shape, _extract_dominant_color, _extract_tamura,
    _extract_edge_orientation, _extract_glcm, _extract_wavelet,
    _extract_ehd, _extract_cld, _extract_spm, _extract_saliency, _extract_correlogram
)

logger = get_logger(__name__)

def _process_single_image_worker(args):
    """Worker function for Lane 1 - Runs all extractors for a single image in a separate process"""
    img, fname, vis_dir = args
    
    # Local imports inside worker to ensure they are available in the spawned process
    from ..extractors import (
        _extract_brightness, _extract_contrast, _extract_saturation, 
        _extract_edge_density, _extract_sharpness, _extract_lbp, 
        _extract_all_color_features, _extract_hog, _extract_hu_moments, 
        _extract_gabor, _extract_ccv, _extract_fourier_descriptors,
        _extract_geometric_shape, _extract_dominant_color, _extract_tamura,
        _extract_edge_orientation, _extract_glcm, _extract_wavelet,
        _extract_ehd, _extract_cld, _extract_spm, _extract_saliency, _extract_correlogram
    )
    
    extractors = [
        (_extract_brightness, []),
        (_extract_contrast, []),
        (_extract_saturation, []),
        (_extract_edge_density, []),
        (_extract_all_color_features, [vis_dir]),
        (_extract_hog, [vis_dir]),
        (_extract_hu_moments, [vis_dir]),
        (_extract_dominant_color, []),
        (_extract_lbp, [vis_dir]),
        (_extract_sharpness, []),
        (_extract_gabor, [vis_dir]),
        (_extract_ccv, [vis_dir]),
        (_extract_fourier_descriptors, []),
        (_extract_geometric_shape, []),
        (_extract_tamura, []),
        (_extract_edge_orientation, []),
        (_extract_glcm, []),
        (_extract_wavelet, []),
        (_extract_correlogram, []),
        (_extract_ehd, []),
        (_extract_cld, []),
        (_extract_spm, []),
        (_extract_saliency, [])
    ]
    
    results = []
    for func, extra_args in extractors:
        f_args = [img] + extra_args
        if func.__name__ in ["_extract_all_color_features", "_extract_hog", "_extract_hu_moments", "_extract_lbp", "_extract_gabor", "_extract_ccv"]:
            f_args.append(fname)
        results.append(func(*f_args))
    return results

class TraditionalLane:
    """Handler for Lane 1: CPU-based Traditional Feature Extraction (Multiprocessed)"""
    def __init__(self, settings, cpu_executor):
        self.settings = settings
        self.cpu_executor = cpu_executor
        self.vis_dir = str(self.settings.visualizations_dir) # Stringify for safer pickling

    async def run(self, img_smalls: List[np.ndarray], filenames: List[str], loop):
        n = len(img_smalls)
        if n == 0: return []
        
        logger.info(f"Lane 1: Starting Parallel Traditional Feature Extraction for {n} images...")
        cpu_start = time.time()
        
        # Prepare arguments for the worker pool
        # We process ALL images in parallel across the process pool
        tasks = []
        for i in range(n):
            fname = filenames[i] or f"image_{i+1}"
            args = (img_smalls[i], fname, self.vis_dir)
            tasks.append(loop.run_in_executor(self.cpu_executor, _process_single_image_worker, args))
        
        # Await all images simultaneously
        all_results = await asyncio.gather(*tasks)
            
        logger.info(f"Lane 1: Completed {n} images in {time.time() - cpu_start:.2f}s")
        return all_results


class SemanticLane:
    """Handler for Lane 2 & 4: LLM Vision API and Text Embeddings"""
    def __init__(self, llm_service):
        self.llm_service = llm_service

    async def run_vlm(self, images_bytes: List[bytes], filenames: List[str]):
        logger.info("Lane 2: Starting LLM Vision (VLM) Analysis...")
        vlm_start = time.time()
        res = await self.llm_service.analyze_vision_batch(images_bytes, filenames)
        logger.info(f"Lane 2: Completed in {time.time() - vlm_start:.2f}s")
        return res

    async def run_embeddings(self, vlm_results, filenames):
        logger.info("Lane 4: Starting LLM Text Embedding...")
        lane4_start = time.time()
        texts = [f"Category: {r.get('category')}. Entities: {r.get('entities')}. Description: {r.get('description')}" for r in vlm_results]
        embeddings = self.llm_service.extract_embeddings_batch(texts, filenames=filenames)
        logger.info(f"Lane 4: Completed in {time.time() - lane4_start:.2f}s")
        return embeddings


class PerceptualLane:
    """Handler for Lane 3: DreamSim GPU Feature Extraction"""
    def __init__(self, executor):
        self.executor = executor
        self._model = None
        self._preprocess = None

    def _get_model(self):
        if self._model is None:
            from dreamsim import dreamsim
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Lane 3: Loading DreamSim model on {device}...")
            self._model, self._preprocess = dreamsim(pretrained=True, device=device)
        return self._model, self._preprocess

    def _extract_sync(self, images_bytes_list):
        if not images_bytes_list: return []
        try:
            import torch
            model, preprocess = self._get_model()
            device = next(model.parameters()).device
            
            # Batch preprocessing
            tensors = []
            for img_bytes in images_bytes_list:
                img_pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                t_raw = preprocess(img_pil)
                # Ensure it's 3D [C, H, W] before stacking into 4D [B, C, H, W]
                if hasattr(t_raw, 'squeeze'):
                    tensors.append(t_raw.squeeze())
                else:
                    tensors.append(t_raw)
            
            # Process in a single batch for GPU efficiency
            batch_tensor = torch.stack(tensors).to(device)
            
            with torch.no_grad():
                # dreamsim's embed method typically supports batches
                feats = model.embed(batch_tensor)
            
            # Return as list of lists
            return feats.cpu().numpy().reshape(len(images_bytes_list), -1).tolist()
            
        except Exception as e:
            logger.error(f"Lane 3 (DreamSim) batch failed: {e}. Falling back to sequential.")
            # Fallback to sequential if batching fails (e.g. OOM or shape mismatch)
            try:
                results = []
                for img_bytes in images_bytes_list:
                    img_pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_tensor = preprocess(img_pil).to(device)
                    with torch.no_grad():
                        feat = model.embed(img_tensor)
                    results.append(feat.cpu().numpy().flatten().tolist())
                return results
            except Exception as e2:
                logger.error(f"Lane 3 fallback failed: {e2}")
                return [None] * len(images_bytes_list)

    async def run(self, images_bytes: List[bytes], loop):
        logger.info("Lane 3: Scheduling DreamSim Feature Extraction...")
        ds_start = time.time()
        results = await loop.run_in_executor(self.executor, self._extract_sync, images_bytes)
        logger.info(f"Lane 3: Completed in {time.time() - ds_start:.2f}s")
        return results
