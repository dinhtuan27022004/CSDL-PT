from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from .schema import ImageResponse, SearchResponse
from .service import ImageService
from .repository import ImageRepository
from ...deps import get_db
from ...core.logging import get_logger

router = APIRouter(prefix="/api/images", tags=["images"])
# Reuse processing service to avoid re-loading weights
_image_service = None
logger = get_logger(__name__)

def get_image_service():
    global _image_service
    if _image_service is None:
        _image_service = ImageService(ImageRepository())
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
    db: Session = Depends(get_db),
    service: ImageService = Depends(get_image_service)
):
    try:
        logger.info(f"Received {len(files)} images")
        image_records = await service.process(db, files)
        # Flush to DB (the service handles it, but commit ensures persistence)
        db.commit()
        return image_records
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[ImageResponse])
async def get_images(
    limit: int = 500,
    offset: int = 0,
    db: Session = Depends(get_db),
    service: ImageService = Depends(get_image_service)
):
    try:
        images = service.get_images(db, limit=limit, offset=offset)
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=SearchResponse)
async def search_similar_images(
    file: UploadFile = File(...),
    limit: int = 5,
    db: Session = Depends(get_db),
    service: ImageService = Depends(get_image_service)
):
    try:

        content = await file.read()
        search_results = await service.search_similar(db, content, file.filename, limit=limit)
        
        return SearchResponse(
            query_image=search_results["query_features"],
            results=search_results["results"]
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))