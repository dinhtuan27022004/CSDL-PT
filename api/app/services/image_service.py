import os
import gc
import traceback
import time
import math
import asyncio
from pathlib import Path
import uuid
import json
import hashlib
from typing import List, Optional, Dict, Any, Tuple, Set
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage import exposure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .extractors import (
    _soft_assignment_hist, _gaussian_hist, _extract_brightness, _extract_contrast,
    _extract_saturation, _extract_edge_density, _extract_sharpness, _extract_color_moments,
    _extract_lbp, _extract_all_color_features, _extract_joint_rgb_histogram, _extract_hog,
    _extract_hu_moments, _extract_gabor, _extract_ccv, _extract_fourier_descriptors,
    _extract_geometric_shape, _extract_dominant_color, _extract_cell_color, _extract_tamura,
    _extract_edge_orientation, _extract_cell_rgb_hist_cdf, _extract_glcm, _extract_wavelet,
    _extract_ehd, _extract_cld, _extract_spm, _extract_saliency, _extract_correlogram
)

logger = get_logger(__name__)


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

        # Models are loaded dynamically during batch processing

    def get_images(self, db: Session, limit: int = 10000, offset: int = 0) -> List[ImageMetadata]:
        return self.repository.get_all(db, limit=limit, offset=offset)

    async def recompute_vlm_missing(self, db: Session, force: bool = False):
        """Sync VLM analysis and embeddings for all images in the uploads folder"""
        # 1. Get records for images in the upload folder
        records = db.query(ImageMetadata).filter(ImageMetadata.file_path.ilike("%/static/uploads/%")).all()
            
        if not records:
            return {"message": "No images found in uploads folder", "count": 0}
            
        n_total = len(records)
        logger.info(f"Syncing VLM/Embeddings for {n_total} images in uploads (force={force})")
        
        batch_size = 1000
        processed_count = 0
        
        for i in range(0, n_total, batch_size):
            sub_batch = records[i : i + batch_size]
            images_bytes, filenames, valid_records = [], [], []
            
            for record in sub_batch:
                rel_path = record.file_path.replace("/static/uploads/", "")
                phys_path = self.settings.uploads_dir / rel_path
                if phys_path.exists():
                    try:
                        with open(phys_path, "rb") as f:
                            images_bytes.append(f.read())
                        filenames.append(record.file_name)
                        valid_records.append(record)
                    except Exception as e: logger.error(f"Error reading {phys_path}: {e}")
            
            if not images_bytes: continue
                
            try:
                # 2. VLM (Uses cache internally)
                vlm_results = await self.llm_service.analyze_vision_batch(images_bytes, filenames)
                
                # 3. LLM Text Embedding
                texts = [f"Category: {r['category']}. Entities: {r['entities']}. Description: {r['description']}" for r in vlm_results]
                llm_embeddings = self.llm_service.extract_embeddings_batch(texts)
                
                # 4. Update Database
                for j, record in enumerate(valid_records):
                    if j < len(vlm_results):
                        record.category, record.description, record.entities = vlm_results[j]["category"], vlm_results[j]["description"], vlm_results[j]["entities"]
                        if j < len(llm_embeddings): record.llm_embedding = llm_embeddings[j]
                
                db.commit()
                processed_count += len(valid_records)
                logger.info(f"Synced batch {i//batch_size + 1}/{math.ceil(n_total/batch_size)}")
            except Exception as e:
                db.rollback()
                logger.error(f"Batch sync failed: {e}")
                
        return {"message": "VLM Sync Completed", "total": n_total, "processed": processed_count}

    async def extract_features_batch(
        self, 
        images_bytes: List[bytes], 
        filenames: List[Optional[str]], 
        force_llm: bool = False,
        required_features: Optional[Set[str]] = None
    ) -> List[ImageMetadata]:
        """Memory-safe concurrent feature extraction using 4-lane pipeline with selective extraction support"""
        run_lane2 = True # API Lane (Metadata) usually always needed for display
        run_lane3 = True # DreamSim Lane
        run_lane4 = True # LLM Lane
        
        if required_features is not None:
            run_lane3 = "dreamsim" in required_features
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

        loop = asyncio.get_running_loop()

        # ============================================================
        # LANE 1: Traditional CPU Feature Extraction
        # ============================================================
        async def run_cpu_lane():
            logger.info(f"Lane 1: Starting Traditional Feature Extraction for {n} images...")
            cpu_start = time.time()
            
            # Extractors registry: (function, list of extra arguments)
            extractors = [
                (_extract_brightness, []),
                (_extract_contrast, []),
                (_extract_saturation, []),
                (_extract_edge_density, []),
                (_extract_all_color_features, [self.settings.visualizations_dir]),
                (_extract_hog, [self.settings.visualizations_dir]),
                (_extract_hu_moments, [self.settings.visualizations_dir]),
                (_extract_dominant_color, []),
                (_extract_lbp, [self.settings.visualizations_dir]),
                (_extract_color_moments, []),
                (_extract_sharpness, []),
                (_extract_gabor, [self.settings.visualizations_dir]),
                (_extract_ccv, [self.settings.visualizations_dir]),
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

            all_results = []
            for i in range(n):
                img = img_smalls[i]
                fname = filenames[i] or f"image_{i+1}"
                logger.info(f"Lane 1 [{i+1}/{n}]: Extracting features for '{fname}'...")
                
                tasks = []
                for func, extra_args in extractors:
                    args = [img] + extra_args
                    # Add filename if the extractor expects it for visualizations
                    if func in [_extract_all_color_features, _extract_hog, _extract_hu_moments, _extract_lbp, _extract_gabor, _extract_ccv]:
                        args.append(fname)
                        
                    tasks.append(loop.run_in_executor(self.cpu_executor, func, *args))
                
                try:
                    res = await asyncio.gather(*tasks)
                    all_results.append(res)
                    logger.info(f"Lane 1 [{i+1}/{n}]: Completed all CPU features for '{fname}'")
                except Exception as e:
                    logger.error(f"Lane 1 Error on image {i+1} ({fname}): {e}")
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
        # LANE 3: DreamSim GPU Feature Extraction (runs in parallel)
        # ============================================================
        def _run_dreamsim_sync(images_bytes_list):
            try:
                from dreamsim import dreamsim
                import torch
                from PIL import Image as PILImage
                import io
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Cache model in the class to avoid reloading every time
                if not hasattr(self, '_dreamsim_model'):
                    logger.info(f"Lane 3: Loading DreamSim model on {device}...")
                    self._dreamsim_model, self._dreamsim_preprocess = dreamsim(pretrained=True, device=device)
                
                results = []
                for img_bytes in images_bytes_list:
                    img_pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_tensor = self._dreamsim_preprocess(img_pil).to(device)
                    with torch.no_grad():
                        feat = self._dreamsim_model.embed(img_tensor)
                    results.append(feat.cpu().numpy().flatten().tolist())
                return results
            except Exception as e:
                logger.error(f"Lane 3 (DreamSim) failed in executor: {e}")
                return [None] * len(images_bytes_list)

        async def run_dreamsim_lane():
            if not run_lane3: return [None] * n
            logger.info("Lane 3: Scheduling DreamSim Feature Extraction...")
            ds_start = time.time()
            
            # Use run_in_executor to avoid blocking the main event loop
            results = await loop.run_in_executor(self.executor, _run_dreamsim_sync, images_bytes)
            
            logger.info(f"Lane 3: DreamSim Completed in {time.time() - ds_start:.2f}s")
            return results

        # Start tasks
        lane1_task = run_cpu_lane()
        lane2_task = run_api_lane()
        lane3_task = run_dreamsim_lane()

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
            llm_embeddings = self.llm_service.extract_embeddings_batch(texts_to_embed, filenames=filenames)
            logger.info(f"Lane 4: LLM Text Embedding Completed in {time.time() - lane4_start:.2f}s")
        else:
            logger.info("Lane 4: Skipping LLM Embedding (not required by weights)")

        # ============================================================
        # ASSEMBLE: Combine results into ImageMetadata
        # ============================================================
        final_metadatas = []
        try:
            for i in range(n):
                c, v = lane1_res[i], lane2_res[i]
                col = c[4] # all_color_features dict
                
                # Create metadata object with core features
                meta = ImageMetadata(
                    file_name = filenames[i],
                    width = cached_vectors[i].get("width"),
                    height = cached_vectors[i].get("height"),
                    brightness = c[0], contrast = c[1], saturation = c[2], edge_density = c[3],
                    histogram_vis_path = col["vis_path"],
                    cell_color_vis_path = col["cell_color_vis_path"],
                    hog_vector = c[5][0], hog_vis_path = c[5][1],
                    hu_moments_vector = c[6][0], hu_vis_path = c[6][1],
                    dominant_color_vector = c[7],
                    lbp_vector = c[8][0], lbp_vis_path = c[8][1],
                    color_moments_vector = c[9],
                    sharpness = c[10],
                    gabor_vector = c[11][0], gabor_vis_path = c[11][1],
                    ccv_vector = c[12][0], ccv_vis_path = c[12][1],
                    zernike_vector = c[13], geo_vector = c[14], tamura_vector = c[15],
                    edge_orientation_vector = c[16], glcm_vector = c[17],
                    wavelet_vector = c[18], correlogram_vector = c[19],
                    ehd_vector = c[20], cld_vector = c[21], spm_vector = c[22],
                    saliency_vector = c[23],
                    category = v.get("category"), description = v.get("description"), entities = v.get("entities"),
                    llm_embedding = llm_embeddings[i],
                    dreamsim_vector = lane3_res[i]
                )
                
                # Dynamically fill color space features (hist, cdf, joint, vectors)
                spaces = ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"]
                feats = ["hist_std", "hist_interp", "hist_gauss", "cdf_std", "cdf_interp", "cdf_gauss"]
                
                for s in spaces:
                    for f in feats:
                        field = f"{s}_{f}"
                        if hasattr(meta, field) and field in col:
                            setattr(meta, field, col[field])
                    
                    # Joint features
                    if s != "gray":
                        for m in ["std", "interp", "gauss"]:
                            field = f"joint_{s}_{m}"
                            if hasattr(meta, field) and field in col:
                                setattr(meta, field, col[field])
                    
                    # Cell vectors
                    field = f"cell_{s}_vector"
                    if hasattr(meta, field) and field in col:
                        setattr(meta, field, col[field])

                final_metadatas.append(meta)
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
        files: List[Any],
        force_llm: bool = False
    ) -> List[ImageMetadata]:
        """Process multiple images (UploadFile or Tuple[bytes, str]) using the batch pipeline"""
        n_total = len(files)
        logger.info(f"Processing batch of {n_total} images")
        start_time = time.time()
        
        batch_size = 128
        final_results = []
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            images_data = []
            filenames = []
            file_hashes = []
            
            for file in batch_files:
                if hasattr(file, "read"): # FastAPI UploadFile
                    content = await file.read()
                    filename = file.filename
                else: # Tuple (bytes, filename)
                    content, filename = file
                
                # Check for duplicates using MD5 hash
                f_hash = hashlib.md5(content).hexdigest()
                
                # Fast database check
                existing = db.query(ImageMetadata).filter(ImageMetadata.file_hash == f_hash).first()
                if existing:
                    logger.info(f"Skipping duplicate image: {filename} (Already exists as ID {existing.id})")
                    continue
                
                images_data.append(content)
                filenames.append(filename)
                file_hashes.append(f_hash)
                
            if not images_data:
                continue
                
            logger.info(f"--- Processing Sub-batch {i//batch_size + 1} / {math.ceil(n_total/batch_size)} ---")
            results_metadata = await self.extract_features_batch(images_data, filenames, force_llm=force_llm)
            
            for j, metadata in enumerate(results_metadata):
                metadata.file_hash = file_hashes[j] # Save the hash
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
                target = "optimized"
                weights = self.repository._load_weights(target)
                required_features = {k for k, v in weights.items() if v != 0}
                
                logger.info(f"Optimized mode: Required features = {required_features}")
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
            search_results = self.repository.search(
                db=db, 
                query_metadata=query_metadata,
                search_settings=search_settings,
                limit=limit
            )
            
            response_data = []
            for idx, row in enumerate(search_results):
                record = row[0]
                total_sim = row[1]
                row_dict = row._asdict()
                
                res = ImageResponse.model_validate(record)
                res.similarity = round(float(total_sim or 0.0) * 100.0, 2)
                
                # Map all similarity scores dynamically
                for key, val in row_dict.items():
                    if key.endswith('_similarity') and key != 'similarity':
                        if hasattr(res, key):
                            setattr(res, key, float(val or 0.0))
                
                # Map Visualization URLs
                mapping = {
                    "hog_vis_path": "hogPreviewUrl",
                    "hu_vis_path": "huPreviewUrl",
                    "cell_color_vis_path": "cellColorPreviewUrl",
                    "lbp_vis_path": "lbpPreviewUrl",
                    "gabor_vis_path": "gaborPreviewUrl",
                    "ccv_vis_path": "ccvPreviewUrl",
                    "histogram_vis_path": "histogramPreviewUrl"
                }
                
                for p_key, ui_key in mapping.items():
                    val = getattr(record, p_key, None)
                    if val:
                        setattr(res, ui_key, val)
                
                res.previewUrl = record.file_path
                response_data.append(res)
                    
            return {
                "query_image": query_metadata,
                "results": response_data
            }
        except Exception as e:
            logger.error(f"Error in search_similar: {e}")
            raise
