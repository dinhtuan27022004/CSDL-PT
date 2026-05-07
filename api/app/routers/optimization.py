from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any
import json
import os
from ..deps import get_db
from ..core.config import get_settings
from ..core.logging import get_logger
from ..services.optimization_service import OptimizationService

router = APIRouter(prefix="/api/optimization", tags=["optimization"])
logger = get_logger(__name__)

@router.get("/evaluation")
async def get_evaluation():
    filename = "evaluation_results.json"
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Evaluation results not found. Please run optimization first.")
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {str(e)}")

@router.post("/optimize")
async def trigger_optimization(
    background_tasks: BackgroundTasks,
    trials: int = 50, 
    allow_negative: bool = False,
    db: Session = Depends(get_db)
):
    try:
        service = OptimizationService(db)
        background_tasks.add_task(service.run_optimization, trials, allow_negative)
        
        return {
            "message": "Optimization started in background using folder-based Ground Truth",
            "trials": trials, 
            "allow_negative": allow_negative
        }
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

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
