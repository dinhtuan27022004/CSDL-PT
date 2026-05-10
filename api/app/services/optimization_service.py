import os
from sqlalchemy.orm import Session
from ..core.config import get_settings
from ..core.logging import get_logger
from .optimization.feature_store import SharedFeatureStore
from .optimization.weight_optimizer import WeightOptimizer

logger = get_logger(__name__)
settings = get_settings()

class OptimizationService:
    """High-level service for managing image retrieval weight optimization tasks"""
    
    def __init__(self, db: Session):
        self.db = db

    def run_optimization(self, trials: int = 50):
        """
        Background task to run weight optimization using Optuna.
        Always uses the default ground_truth.json for validation labels.
        """
        try:
            logger.info(f"Starting background optimization: trials={trials}")
            
            # 1. Resolve Ground Truth path
            gt_path = os.path.join(settings.base_dir, "ground_truth.json")
            if not os.path.exists(gt_path):
                logger.error(f"No ground truth file found at {gt_path}. Aborting.")
                return

            # 2. Load Feature Store (In-memory similarity matrices)
            store = SharedFeatureStore(self.db)
            if not store.load(gt_path=gt_path):
                logger.error("Failed to load feature matrices into memory store.")
                return
                
            # 3. Initialize Optimizer
            optimizer = WeightOptimizer(gt_name="folder", shared_store=store)
            
            # 4. Prepare Evaluation Clusters
            if not optimizer.prepare(gt_path=gt_path):
                logger.error("Failed to prepare optimizer (no Ground Truth matches found).")
                return
            
            # 5. Optimization Loop
            # Warm-up: Use equal weights as baseline so UI doesn't show 0%
            feature_names = store.feature_names
            if feature_names:
                baseline_weights = {name: 1.0 / len(feature_names) for name in feature_names}
                optimizer.save_results(baseline_weights)
            
            # Start Optuna study
            logger.info(f"Running {trials} optimization trials...")
            best_weights = optimizer.optimize(n_trials=trials)
            
            # 6. Finalize: Save the absolute best weights found
            optimizer.save_results(best_weights)
            logger.info("Weight optimization completed successfully.")
            logger.info(f"Best mAP@5 reached: {optimizer.best_test_map5:.4f}")
            
        except Exception as e:
            logger.error(f"Critical error during optimization task: {e}", exc_info=True)
