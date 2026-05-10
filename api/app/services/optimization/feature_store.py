import numpy as np
import threading
import time
from sqlalchemy.orm import Session
from ...models.image import ImageMetadata
from ...core.logging import get_logger
from ...core.similarity_specs import get_all_feature_specs
from .similarity import SimilarityCalculator
from .constants import METRIC_MAP, COLUMN_MAP

logger = get_logger(__name__)

class SharedFeatureStore:
    """Memory-resident store for pre-computed similarity matrices to speed up optimization trials"""
    
    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_session: Session):
        self.db = db_session
        self.feature_matrices = {}
        self.image_ids = []
        self.images = []
        self.feature_names = []
        self._is_loaded = False

    def load(self, gt_path: Optional[str] = None) -> bool:
        """Load feature data from DB, optionally filtered by ground truth filenames"""
        if self._is_loaded:
            return True
            
        try:
            import json
            import os
            
            # 1. Determine which images to load
            allowed_filenames = None
            if gt_path and os.path.exists(gt_path):
                try:
                    with open(gt_path, "r") as f:
                        gt_data = json.load(f)
                    # Extract all unique filenames from clusters
                    allowed_filenames = set()
                    for cluster in gt_data.values():
                        for fname in cluster:
                            allowed_filenames.add(os.path.basename(fname))
                    logger.info(f"Filtering Feature Store to {len(allowed_filenames)} images from GT.")
                except Exception as e:
                    logger.error(f"Failed to read GT for filtering: {e}")

            logger.info("Loading Shared Feature Store from DB...")
            query = self.db.query(ImageMetadata)
            
            if allowed_filenames is not None:
                # Filter by filenames found in GT
                # Note: We use the basename to match
                all_images = query.all()
                self.images = [img for img in all_images if os.path.basename(img.file_name) in allowed_filenames]
            else:
                self.images = query.all()
                
            if not self.images:
                logger.error("No images found for the given criteria.")
                return False
            
            self.image_ids = [img.id for img in self.images]
            specs = get_all_feature_specs()
            
            for name, metric_type in specs.items():
                try:
                    col_name = COLUMN_MAP.get(name, name)
                    
                    if metric_type in ["scalar", "sharpness", "cosine", "l2_color", "l2_cell"]:
                        raw_data = [getattr(img, col_name) for img in self.images]
                        
                        if metric_type in ["scalar", "sharpness"]:
                            vec = np.array([float(v) if v is not None else 0.0 for v in raw_data])
                            self.feature_matrices[name] = SimilarityCalculator.get_matrix(vec, metric=METRIC_MAP[metric_type])
                        else:
                            sample_v = next((v for v in raw_data if v is not None), None)
                            dim = len(sample_v) if sample_v is not None else 512
                            vecs = np.array([np.array(v) if v is not None else np.zeros(dim) for v in raw_data])
                            self.feature_matrices[name] = SimilarityCalculator.get_matrix(vecs, metric=METRIC_MAP[metric_type])
                            
                    elif metric_type == "category":
                        vals = [getattr(img, col_name) for img in self.images]
                        self.feature_matrices[name] = SimilarityCalculator.get_discrete_matrix(vals, type="category")
                        
                    elif metric_type == "entity":
                        vals = [getattr(img, col_name) for img in self.images]
                        self.feature_matrices[name] = SimilarityCalculator.get_discrete_matrix(vals, type="entities")
                        
                    time.sleep(0.01) # Yield control
                except Exception as e:
                    logger.error(f"Failed to load feature matrix for {name}: {e}")

            self.feature_names = list(self.feature_matrices.keys())
            self._is_loaded = True
            logger.info(f"Store loaded with {len(self.feature_names)} matrices.")
            return True
        except Exception as e:
            logger.error(f"Error loading feature store: {e}")
            return False
