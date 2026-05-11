import os
import gc
import time
import math
import asyncio
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cv2
import numpy as np

from ..repositories.image_repository import ImageRepository
from .llm_service import LLMService
from .cache_service import CacheService
from ..models.image import ImageMetadata
from ..schemas.image import ImageResponse
from ..core.config import get_settings
from ..core.logging import get_logger

# Import modular components
from .image.lanes import TraditionalLane, SemanticLane, PerceptualLane
from .image.metadata import assemble_metadatas
from ..utils.image_processing import resize_logic_worker

logger = get_logger(__name__)

class ImageService:
    """Service for image CRUD, storage, and validation using modular lane handlers"""
    
    def __init__(self, 
        repository: ImageRepository,
        llm_service: LLMService,
        cache_service: CacheService
    ):
        self.repository = repository
        self.llm_service = llm_service
        self.cache = cache_service
        self.settings = get_settings()
        
        # Shared Executors
        self.io_executor = ThreadPoolExecutor(max_workers=8)
        self.gpu_executor = ThreadPoolExecutor(max_workers=8)
        # Using ProcessPool for CPU bound tasks to bypass GIL
        # Set to 1 worker for sequential processing to strictly control memory
        self.cpu_executor = ProcessPoolExecutor(max_workers=1)
        
        # Initialize Lane Handlers
        self.traditional_lane = TraditionalLane(self.settings, self.cpu_executor)
        self.semantic_lane = SemanticLane(self.llm_service)
        self.perceptual_lane = PerceptualLane(self.gpu_executor)

    # --- Core Extraction Pipeline ---

    async def extract_features_batch(
        self, 
        images_bytes: List[bytes], 
        filenames: List[Optional[str]], 
        force_llm: bool = False,
        required_features: Optional[Set[str]] = None
    ) -> List[ImageMetadata]:
        """Orchestrate the 4-lane pipeline using modular handlers"""
        n = len(images_bytes)
        loop = asyncio.get_running_loop()
        
        # Feature Selection Logic
        traditional_features = {
            "hog", "hu_moments", "lbp", "sharpness", "gabor", "ccv", 
            "brightness", "contrast", "saturation", "edge_density",
            "meta_hist", "meta_cdf", "meta_joint", "meta_cell", "meta_moments"
        }
        
        if required_features:
            run_traditional = any(f in traditional_features for f in required_features)
            run_vlm = any(f in {"category", "description", "entity"} for f in required_features)
            run_dreamsim = "dreamsim" in required_features
            run_llm = ("semantic" in required_features or force_llm)
        else:
            run_traditional = run_vlm = run_dreamsim = run_llm = True

        # Phase 1: Pre-processing (Decode and Resize)
        img_smalls = []
        dims = []
        for i in range(n):
            file_bytes = np.frombuffer(images_bytes[i], dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image {filenames[i]}")
            
            # Resizing to 1/6 for traditional feature extraction efficiency
            img_smalls.append(cv2.resize(img, (img.shape[1]//6, img.shape[0]//6)))
            dims.append((img.shape[1], img.shape[0]))

        # Phase 2: Parallel Lane Execution (1, 2, 3)
        lane1_task = self.traditional_lane.run(img_smalls, filenames, loop) if run_traditional else asyncio.sleep(0, [([None]*25) for _ in range(n)])
        lane2_task = self.semantic_lane.run_vlm(images_bytes, filenames) if run_vlm else asyncio.sleep(0, [{} for _ in range(n)])
        lane3_task = self.perceptual_lane.run(images_bytes, loop) if run_dreamsim else asyncio.sleep(0, [None] * n)

        lane1_res, lane2_res, lane3_res = await asyncio.gather(lane1_task, lane2_task, lane3_task)
        
        # Phase 3: Dependent Lane Execution (Lane 4 depends on Lane 2)
        llm_embeddings = await self.semantic_lane.run_embeddings(lane2_res, filenames) if run_llm else [None] * n

        # Phase 4: Final Metadata Assembly
        results = assemble_metadatas(n, filenames, dims, lane1_res, lane2_res, lane3_res, llm_embeddings)
        
        # Aggressive Memory Cleanup (Only after everything is finished)
        img_smalls.clear()
        del lane1_res, lane2_res, lane3_res, llm_embeddings, images_bytes
        gc.collect()
        
        return results

    async def extract_features(
        self, 
        image: bytes, 
        filename: Optional[str] = None, 
        force_llm: bool = False,
        required_features: Optional[Set[str]] = None
    ) -> ImageMetadata:
        """Compatibility wrapper for single image processing"""
        results = await self.extract_features_batch([image], [filename], force_llm=force_llm, required_features=required_features)
        return results[0]

    # --- Persistence & Internal Logic ---

    async def _process_and_persist(
        self, 
        db: Session, 
        images: List[bytes], 
        names: List[str], 
        hashes: List[str], 
        save_to_disk: bool = False, 
        force_llm: bool = False
    ) -> List[Any]:
        """Unified persistence logic (Reusable)"""
        if not images: return []
            
        metas = await self.extract_features_batch(images, names, force_llm=force_llm)
        results = []
        
        for j, meta in enumerate(metas):
            try:
                meta.file_hash = hashes[j]
                if save_to_disk:
                    res = self.repository.create(db, meta, images[j])
                    results.append(res)
                else:
                    meta.file_path = f"/static/uploads/{names[j]}"
                    db.add(meta)
                    db.commit() # Immediate save per requirement
                    results.append(ImageResponse.model_validate(meta))
            except Exception as e:
                db.rollback()
                logger.error(f"Persistence failed for {names[j]}: {e}")
                
        return results

    # --- Public API Endpoints ---

    def get_images(self, db: Session, limit: int = 10000, offset: int = 0) -> List[ImageMetadata]:
        return self.repository.get_all(db, limit=limit, offset=offset)

    async def _standardize_image(self, image_bytes: bytes) -> bytes:
        """Standardize image to 16:9 4K resolution before saving"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.cpu_executor, resize_logic_worker, image_bytes)

    async def process(self, db: Session, files: List[Any], force_llm: bool = False) -> List[ImageMetadata]:
        """Process images from API upload request"""
        final_results = []
        batch_size = 32
        n_total = len(files)
        n_batches = math.ceil(n_total / batch_size)
        
        for i in range(0, n_total, batch_size):
            logger.info(f"--- Processing Upload Batch {i//batch_size + 1} / {n_batches} ---")
            batch = files[i:i+batch_size]
            imgs, names, hashes = [], [], []
            
            for f in batch:
                content = await f.read() if hasattr(f, "read") else f[0]
                fname = f.filename if hasattr(f, "read") else f[1]
                
                # --- Auto Resize & Standardize before any other logic ---
                logger.info(f"Standardizing {fname} to 16:9 4K...")
                content = await self._standardize_image(content)
                
                f_hash = hashlib.md5(content).hexdigest()
                
                existing = db.query(ImageMetadata).filter(
                    (ImageMetadata.file_hash == f_hash) | 
                    ((ImageMetadata.file_name == fname) & (ImageMetadata.file_hash == None))
                ).first()
                
                if existing:
                    logger.info(f"Skipping existing image: {fname}")
                    if existing.file_hash is None: 
                        existing.file_hash = f_hash
                        db.commit()
                    continue
                imgs.append(content); names.append(fname); hashes.append(f_hash)
            
            results = await self._process_and_persist(db, imgs, names, hashes, save_to_disk=True, force_llm=force_llm)
            final_results.extend(results)
            
            # Aggressive cleanup for the next batch
            imgs.clear(); names.clear(); hashes.clear()
            del results
            gc.collect()
            
        return final_results

    async def recompute_all(self, db: Session) -> List[ImageResponse]:
        """Sync missing images from uploads directory to database and fix invalid DreamSim vectors with detailed logging"""
        start_time = time.time()
        logger.info("Starting Global Recompute/Sync process...")
        
        existing = db.query(ImageMetadata.id, ImageMetadata.file_hash, ImageMetadata.file_name, ImageMetadata.dreamsim_vector).all()
        
        # Helper to check vector validity
        def is_vec_valid(vec):
            if vec is None: return False
            try:
                v_arr = np.array(vec)
                return v_arr.shape[0] > 0 and np.any(v_arr != 0)
            except:
                return False

        ex_map = {r.file_hash: {"id": r.id, "name": r.file_name, "dreamsim": r.dreamsim_vector} for r in existing if r.file_hash}
        ex_names = {r.file_name for r in existing}
        
        logger.info(f"Database contains {len(existing)} total records. {len(ex_map)} have valid hashes.")

        def scan_files():
            scanned_new = []
            scanned_fix = []
            skipped_count = 0
            valid_dreamsim_count = 0
            
            for root, _, filenames in os.walk(self.settings.uploads_dir):
                for f in filenames:
                    if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): continue
                    p = Path(root) / f
                    rel = str(p.relative_to(self.settings.uploads_dir)).replace("\\", "/")
                    try:
                        with open(p, "rb") as file_obj:
                            content = file_obj.read()
                            h = hashlib.md5(content).hexdigest()
                        
                        if h in ex_map:
                            record = ex_map[h]
                            if not is_vec_valid(record["dreamsim"]):
                                scanned_fix.append((record["id"], content, rel))
                            else:
                                valid_dreamsim_count += 1
                                skipped_count += 1
                            continue
                            
                        # If hash missing, check if we need to update hash for known filename
                        if rel in ex_names:
                            scanned_new.append(("UPDATE_HASH", rel, h))
                        else:
                            scanned_new.append(("PROCESS", content, rel, h))
                    except Exception as e:
                        logger.error(f"Failed to scan {p}: {e}")
            return scanned_new, scanned_fix, skipped_count, valid_dreamsim_count

        loop = asyncio.get_event_loop()
        scanned_items, fix_items, total_skipped, valid_ds_count = await loop.run_in_executor(self.io_executor, scan_files)
        
        logger.info(f"Scan Result: Total Skipped={total_skipped} (Healthy DreamSim={valid_ds_count})")
        logger.info(f"Actions Needed: {len(scanned_items)} New/Update items, {len(fix_items)} DreamSim fixes needed.")
        
        to_process = []
        for item in scanned_items:
            if item[0] == "UPDATE_HASH":
                rel, h = item[1], item[2]
                rec = db.query(ImageMetadata).filter(ImageMetadata.file_name == rel).first()
                if rec and rec.file_hash is None:
                    rec.file_hash = h; db.commit()
            else:
                to_process.append(item[1:]) # (content, rel, h)
            
        final = []
        batch_size = 32
        
        # 1. Process New Images
        if to_process:
            n_total = len(to_process)
            n_batches = math.ceil(n_total / batch_size)
            logger.info(f"Phase 1: Extracting features for {n_total} NEW images in {n_batches} batches.")
            
            for i in range(0, n_total, batch_size):
                b_start = time.time()
                batch = to_process[i:i+batch_size]
                results = await self._process_and_persist(db, [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch])
                for r in batch:
                    logger.info(f"  [NEW IMAGE] {r[1]}")
                final.extend(results)
                logger.info(f"Batch {i//batch_size + 1}/{n_batches} completed in {time.time() - b_start:.2f}s")
                
                # Aggressive cleanup
                del batch, results
                gc.collect()

        # 2. Fix Invalid DreamSim Vectors
        if fix_items:
            n_fix = len(fix_items)
            batch_size_fix = 32
            n_fix_batches = math.ceil(n_fix / batch_size_fix)
            logger.info(f"Phase 2: Fixing {n_fix} invalid DreamSim vectors in {n_fix_batches} batches.")
            
            for i in range(0, n_fix, batch_size_fix):
                b_start = time.time()
                batch = fix_items[i:i+batch_size_fix]
                contents = [x[1] for x in batch]; ids = [x[0] for x in batch]
                
                try:
                    ds_results = await self.perceptual_lane.run(contents, loop)
                    for j, ds_vec in enumerate(ds_results):
                        if ds_vec:
                            db.query(ImageMetadata).filter(ImageMetadata.id == ids[j]).update({"dreamsim_vector": ds_vec})
                            logger.info(f"  [FIXED DREAMSIM] {batch[j][2]}")
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.error(f"Failed to fix DreamSim batch: {e}")
                
                logger.info(f"Fix Batch {i//batch_size + 1}/{n_fix_batches} completed in {time.time() - b_start:.2f}s")
                gc.collect()

        duration = time.time() - start_time
        logger.info(f"Global Recompute Finished. Processed {len(final)} new images and {len(fix_items)} fixes in {duration:.2f}s.")
        return final

    async def search_similar(self, db: Session, content: bytes, filename: str, search_settings: Any, limit: int = 50, force_llm: bool = False) -> Dict[str, Any]:
        """Perform hybrid search based on dynamic search settings"""
        mode = getattr(search_settings, 'mode', 'optimized')
        req_features = None
        
        if mode == "optimized":
            weights = self.repository._load_weights("optimized")
            req_features = {k for k, v in weights.items() if v != 0}
        elif mode == "manual":
            req_features = {k for k, v in getattr(search_settings, 'weights', {}).items() if v > 0}
            
        query_meta = (await self.extract_features_batch([content], [filename], force_llm=force_llm, required_features=req_features))[0]
        results = self.repository.search(db, query_meta, limit=limit, search_settings=search_settings)
        
        resp = []
        mapping = {
            "hog_vis_path": "hogPreviewUrl", "hu_vis_path": "huPreviewUrl", "cell_color_vis_path": "cellColorPreviewUrl", 
            "lbp_vis_path": "lbpPreviewUrl", "gabor_vis_path": "gaborPreviewUrl", "ccv_vis_path": "ccvPreviewUrl", 
            "histogram_vis_path": "histogramPreviewUrl"
        }
        
        for row in results:
            record, sim = row[0], row[1]
            res = ImageResponse.model_validate(record)
            res.similarity = round(float(sim or 0.0) * 100.0, 2)
            res.previewUrl = record.file_path
            
            for k, v in row._asdict().items():
                if k.endswith('_similarity') and hasattr(res, k): setattr(res, k, float(v or 0.0))
            
            for pk, uk in mapping.items():
                val = getattr(record, pk, None)
                if val: setattr(res, uk, val)
            resp.append(res)
            
        return {"query_image": query_meta, "results": resp}

    async def recompute_vlm_missing(self, db: Session, force: bool = False):
        """Minimal sync for images with missing or failed VLM/LLM analysis"""
        query = db.query(ImageMetadata)
        
        if not force:
            # Filter for missing data: category is null, description is null/empty, or embedding is null
            from sqlalchemy import or_
            query = query.filter(or_(
                ImageMetadata.category == None,
                ImageMetadata.category == "",
                ImageMetadata.llm_embedding == None,
                ImageMetadata.description == None
            ))
            
        records = query.all()
        if not records:
            return {"message": "No images with missing VLM data found.", "processed": 0}
            
        logger.info(f"Syncing VLM for {len(records)} images...")
        processed = 0
        batch_size = 32
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            imgs, names, valid_records = [], [], []
            
            for r in batch:
                # Resolve physical path
                p = self.settings.uploads_dir / r.file_name
                if not p.exists() and r.file_path:
                    # Try resolving from web path
                    rel = r.file_path.replace("/static/uploads/", "")
                    p = self.settings.uploads_dir / rel
                
                if p.exists():
                    try:
                        with open(p, "rb") as f:
                            imgs.append(f.read())
                            names.append(r.file_name)
                            valid_records.append(r)
                    except Exception as e:
                        logger.error(f"Failed to read image {p}: {e}")
            
            if not imgs: continue
            
            try:
                # 1. Run VLM (Category, Description, Entities)
                vlm_results = await self.semantic_lane.run_vlm(imgs, names)
                
                # 2. Run Embeddings
                txts = [f"Category: {r.get('category','')}. Entities: {r.get('entities',[])}. Description: {r.get('description','')}" for r in vlm_results]
                embeddings = await self.semantic_lane.run_embeddings(vlm_results, names)
                
                # 3. Update Database
                for j, r in enumerate(valid_records):
                    res = vlm_results[j]
                    r.category = res.get("category")
                    r.description = res.get("description")
                    r.entities = res.get("entities")
                    if j < len(embeddings):
                        r.llm_embedding = embeddings[j]
                    logger.info(f"  [VLM FIXED] {r.file_name}")
                
                db.commit()
                processed += len(valid_records)
                logger.info(f"Processed VLM batch {i//batch_size + 1}: {len(valid_records)} images.")
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to process VLM batch: {e}")
            
            gc.collect()
            
        return {"message": f"VLM Sync Completed. Processed {processed} images.", "processed": processed}
