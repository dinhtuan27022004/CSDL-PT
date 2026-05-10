from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import json
import os
from ..deps import get_db
from ..core.config import get_settings
from ..core.logging import get_logger
from ..services.optimization_service import OptimizationService

router = APIRouter(prefix="/api/optimization", tags=["optimization"])
logger = get_logger(__name__)

@router.get("/evaluation")
def get_evaluation():
    filename = "evaluation_results.json"
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Evaluation results not found. Please run optimization first.")
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation results: {str(e)}")

@router.post("/optimize")
def trigger_optimization(
    background_tasks: BackgroundTasks,
    trials: int = 50,
    db: Session = Depends(get_db)
):
    try:
        service = OptimizationService(db)
        background_tasks.add_task(service.run_optimization, trials)
        return {
            "status": "Optimization started",
            "trials": trials
        }
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

@router.get("/weights")
def get_current_weights():
    settings = get_settings()
    if not os.path.exists(settings.weights_file):
        return {}
    try:
        with open(settings.weights_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read weights: {str(e)}")

@router.get("/worst-queries")
def get_worst_queries(
    top_n: int = 5,
    db: Session = Depends(get_db)
):
    """Return the top_n query images with the lowest per-image mAP@5 from cache."""
    try:
        filename = "evaluation_results.json"
        if not os.path.exists(filename):
            raise HTTPException(status_code=404, detail="Evaluation results not found. Please run optimization first.")
        with open(filename, 'r') as f:
            data = json.load(f)
        
        worst_queries = data.get("worst_queries", [])
        return worst_queries[:top_n]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get worst queries from cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
