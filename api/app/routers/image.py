from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
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
import subprocess

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
    force_llm: bool = Form(False),
    db: Session = Depends(get_db),
    service: ImageService = Depends(get_image_service)
):
    try:
        logger.info(f"Received {len(files)} images (force_llm={force_llm})")
        image_records = await service.process(db, files, force_llm=force_llm)
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
            results=search_results["results"],
            gt_results=search_results.get("gt_results")
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluation")
async def get_evaluation(gt: str = "clip"):
    settings = get_settings()
    filename = f"evaluation_results_{gt}.json"
    if not os.path.exists(filename):
        # Fallback to default if it exists and gt is clip
        if gt == "clip" and os.path.exists(settings.evaluation_results_file):
            filename = settings.evaluation_results_file
        else:
            raise HTTPException(status_code=404, detail=f"Evaluation results for {gt} not found")
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {str(e)}")

@router.get("/evaluation-all")
async def get_all_evaluation():
    settings = get_settings()
    results = {}
    for gt in ["clip", "dinov2", "siglip", "dreamsim"]:
        filename = f"evaluation_results_{gt}.json"
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    results[gt] = json.load(f)
            except:
                continue
    return results

@router.post("/optimize")
async def trigger_optimization(
    gt: str = "all", 
    trials: int = 50, 
    allow_negative: bool = False,
    exclude_embeddings: bool = False
):
    settings = get_settings()
    # Path to python executable
    python_exe = str(settings.base_dir / "env" / "Scripts" / "python.exe")
    script_path = str(settings.base_dir / "optimize_weights.py")
    
    # Define models to optimize
    available_models = ["clip", "dinov2", "siglip", "dreamsim"]
    targets = available_models if gt == "all" else [gt]
    
    started_models = []
    
    for target in targets:
        if target not in available_models and gt != "all":
            continue
            
        cmd = [
            python_exe,
            script_path,
            "--gt", target,
            "--trials", str(trials)
        ]
        if allow_negative:
            cmd.append("--allow-negative")
        if exclude_embeddings:
            cmd.append("--exclude-embeddings")
        
        try:
            # Run each model in its own background process
            subprocess.Popen(cmd, cwd=str(settings.base_dir))
            started_models.append(target)
        except Exception as e:
            logger.error(f"Failed to start optimization for {target}: {e}")
            
    if not started_models:
        raise HTTPException(status_code=500, detail="Failed to start any optimization processes")
        
    return {
        "message": f"Optimization started for {len(started_models)} models in parallel", 
        "models": started_models, 
        "trials": trials, 
        "allow_negative": allow_negative,
        "exclude_embeddings": exclude_embeddings
    }

@router.get("/weights")
async def get_current_weights():
    settings = get_settings()
    if not os.path.exists(settings.weights_file):
        return {}
    try:
        with open(settings.weights_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read weights: {str(e)}")
