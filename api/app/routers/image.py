from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import threading
from ..schemas.image import ImageResponse, SearchResponse, SearchSettings
from ..services.image_service import ImageService
from ..services.llm_service import LLMService
from ..services.cache_service import CacheService
from ..repositories.image_repository import ImageRepository
from ..deps import get_db
from ..core.logging import get_logger
from ..core.config import get_settings
import os
import time
import math
import asyncio

router = APIRouter(prefix="/api/images", tags=["images"])
# Reuse processing service to avoid re-loading weights
_image_service = None
_service_lock = threading.Lock()
logger = get_logger(__name__)

def get_image_service():
    global _image_service
    with _service_lock:
        if _image_service is None:
            logger.info("Initializing ImageService singleton with injected dependencies...")
            # Explicitly inject dependencies for reusability
            repo = ImageRepository()
            cache = CacheService()
            llm = LLMService(cache_service=cache)
            _image_service = ImageService(
                repository=repo,
                llm_service=llm,
                cache_service=cache
            )
    return _image_service

def _parse_features_json(features_json):
    if features_json is None:
        return None
    if isinstance(features_json, dict):
        return features_json
    if isinstance(features_json, str):
        try:
            return json.loads(features_json)
        except json.JSONDecodeError:
            return None
    return None

@router.post("/upload", response_model=List[ImageResponse])
async def upload_images(
    files: List[UploadFile] = File(...),
    paths: str = Form(""),  # JSON string: ["cat/img1.jpg", "img2.jpg", ...]
    force_llm: bool = Form(False),
    db: Session = Depends(get_db),
    service: ImageService = Depends(get_image_service)
):
    try:
        logger.info(f"Received {len(files)} images (force_llm={force_llm})")
        
        # Parse paths and update ground_truth.json
        paths_list = json.loads(paths) if paths else []
        if paths_list:
            gt_file = "ground_truth.json"
            gt_data = {}
            if os.path.exists(gt_file):
                try:
                    with open(gt_file, 'r') as f:
                        gt_data = json.load(f)
                except:
                    gt_data = {}
            
            updated = False
            for p in paths_list:
                parts = p.replace("\\", "/").split("/")
                if len(parts) > 1:
                    folder_name = parts[-2]
                    file_name = parts[-1]
                    if folder_name not in gt_data:
                        gt_data[folder_name] = []
                    if file_name not in gt_data[folder_name]:
                        gt_data[folder_name].append(file_name)
                        updated = True
            
            if updated:
                with open(gt_file, 'w') as f:
                    json.dump(gt_data, f, indent=2)
                logger.info(f"Updated {gt_file} with new folder-based labels")

        image_records = await service.process(db, files, force_llm=force_llm)
        db.commit()
        return image_records
    except Exception as e:
        db.rollback()
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from ..schemas.image import ImageResponse, SearchResponse, SearchSettings, PaginatedImageResponse

@router.get("/", response_model=PaginatedImageResponse)
async def get_images(
    page: int = 1,
    size: int = 50,
    db: Session = Depends(get_db),
    service: ImageService = Depends(get_image_service)
):
    try:
        offset = (page - 1) * size
        images = service.get_images(db, limit=size, offset=offset)
        
        from ..models.image import ImageMetadata
        total = db.query(ImageMetadata).count()
        
        return PaginatedImageResponse(
            total=total,
            items=images,
            page=page,
            size=size,
            pages=math.ceil(total / size)
        )
    except Exception as e:
        logger.error(f"Error fetching paginated images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=SearchResponse)
async def search_similar_images(
    file: UploadFile = File(...),
    limit: int = Form(5),
    force_llm: bool = Form(False),
    search_settings: Optional[str] = Form(None), # JSON string of booleans
    db: Session = Depends(get_db),
    service: ImageService = Depends(get_image_service)
):
    try:
        # Parse search settings
        settings = SearchSettings()
        if search_settings:
            try:
                settings = SearchSettings.model_validate_json(search_settings)
            except Exception as e:
                logger.warning(f"Failed to parse search settings JSON: {e}. Using defaults.")

        content = await file.read()
        search_results = await service.search_similar(
            db, content, file.filename, 
            search_settings=settings,
            limit=settings.limit or limit, 
            force_llm=force_llm
        )
        
        return SearchResponse(
            query_image=search_results["query_image"],
            results=search_results["results"]
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recompute-vlm")
async def recompute_vlm_data(
    force: bool = False,
    db: Session = Depends(get_db),
    service: ImageService = Depends(get_image_service)
):
    """Trigger recomputation of missing or failed VLM data for uploaded images"""
    try:
        result = await service.recompute_vlm_missing(db, force=force)
        return result
    except Exception as e:
        logger.error(f"Recomputation route failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-db")
async def reset_database(db: Session = Depends(get_db)):
    """Wipe the image_metadata table and recreate it with the current schema"""
    try:
        from ..db.base import Base
        from ..models.image import ImageMetadata
        from sqlalchemy import text
        
        engine = db.get_bind()
        
        # Drop table with CASCADE
        db.execute(text("DROP TABLE IF EXISTS image_metadata CASCADE"))
        db.commit()
        
        # Recreate all tables (including image_metadata)
        Base.metadata.create_all(bind=engine)
        
        # Clear visualizations directory
        settings = get_settings()
        if settings.visualizations_dir.exists():
            import shutil
            for item in settings.visualizations_dir.iterdir():
                if item.is_file(): item.unlink()
                elif item.is_dir(): shutil.rmtree(item)
                
        # Reset ground_truth.json
        gt_file = "ground_truth.json"
        if os.path.exists(gt_file):
            with open(gt_file, 'w') as f:
                json.dump({}, f)

        return {"message": "Database and visualizations reset successfully"}
    except Exception as e:
        logger.error(f"Reset database failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
