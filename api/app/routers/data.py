from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import threading
from ..services.data_service import DataService
from ..repositories.image_repository import ImageRepository
from ..deps import get_db
from ..core.logging import get_logger

router = APIRouter(prefix="/api/data", tags=["data"])
_data_service = None
_service_lock = threading.Lock()
logger = get_logger(__name__)

def get_data_service():
    global _data_service
    with _service_lock:
        if _data_service is None:
            repo = ImageRepository()
            _data_service = DataService(repository=repo)
    return _data_service

@router.post("/generate-ground-truth")
def generate_ground_truth(
    db: Session = Depends(get_db),
    service: DataService = Depends(get_data_service)
):
    """Generate ground_truth.json (Runs in threadpool to avoid blocking)"""
    try:
        result = service.generate_ground_truth(db)
        return result
    except Exception as e:
        logger.error(f"Ground truth generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/select-diverse-gt")
def select_diverse_gt(
    db: Session = Depends(get_db),
    service: DataService = Depends(get_data_service)
):
    """Select 50 non-overlapping clusters (Runs in threadpool)"""
    try:
        result = service.select_diverse_ground_truth(db)
        return result
    except Exception as e:
        logger.error(f"Diverse GT selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-gt3")
def generate_gt3(
    payload: dict,
    db: Session = Depends(get_db),
    service: DataService = Depends(get_data_service)
):
    """Generate ground_truth_3.json from folder structure"""
    try:
        folder_path = payload.get("folder_path")
        if not folder_path:
            raise HTTPException(status_code=400, detail="folder_path is required")
            
        result = service.generate_ground_truth_3(db, folder_path)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GT3 generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
def get_stats(
    mode: str = "full",
    db: Session = Depends(get_db),
    service: DataService = Depends(get_data_service)
):
    """Get statistics for a specific ground truth file (Runs in threadpool)"""
    if mode == "full":
        filename = "ground_truth.json"
    elif mode == "diverse":
        filename = "ground_truth_2.json"
    else:
        filename = "ground_truth_3.json"

    logger.info(f"--- FETCHING STATS FOR MODE: {mode}, FILENAME: {filename} ---")
    try:
        result = service.get_stats_for_file(db, filename)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Stats for {mode} not found.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
