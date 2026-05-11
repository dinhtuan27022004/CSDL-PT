from sqlalchemy.orm import Session
from sqlalchemy import text, func, case, cast, Float, literal, Text
from sqlalchemy.dialects.postgresql import JSONB
from typing import List, Optional, Tuple, Any
from ..models.image import ImageMetadata
from ..core.logging import get_logger
from ..core.config import get_settings
from ..core.similarity_specs import get_all_feature_specs
import math
import os
import json

logger = get_logger(__name__)

# Mapping from feature names to actual ImageMetadata column names if different
COLUMN_MAP = {
    "hog": "hog_vector", "hu_moments": "hu_moments_vector", "lbp": "lbp_vector",
    "gabor": "gabor_vector", "ccv": "ccv_vector", "fourier": "fourier_vector",
    "geo": "geo_vector", "tamura": "tamura_vector",
    "edge_orientation": "edge_orientation_vector", "glcm": "glcm_vector", "wavelet": "wavelet_vector",
    "correlogram": "correlogram_vector", "ehd": "ehd_vector", "cld": "cld_vector",
    "spm": "spm_vector", "saliency": "saliency_vector", "semantic": "llm_embedding",
    "entity": "entities",
    
    # --- Consolidated Meta Features ---
    "meta_hist_interp": "meta_hist_interp",
    "meta_cdf_interp": "meta_cdf_interp",
    "meta_joint_interp": "meta_joint_interp",
    "meta_cell": "meta_cell_vector",
    "meta_moments_mean": "meta_moments_mean",
    "meta_moments_std": "meta_moments_std",
    "meta_moments_skew": "meta_moments_skew"
}

class ImageRepository:
    """Repository for image metadata database operations"""
    def __init__(self):
        self.settings = get_settings()

    def _load_weights(self, suffix: Optional[str] = None) -> dict:
        """Load optimized weights from file if it exists"""
        filename = self.settings.weights_file
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load weights file {filename}: {e}")
        return {}

    def create(self, db: Session, image_record: ImageMetadata, image: bytes) -> ImageMetadata:
        try:
            img_path = self.settings.uploads_dir / image_record.file_name
            # Ensure parent directory exists (e.g. if file_name is "images/wallpaper_0.jpg")
            img_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(img_path, "wb") as f:
                f.write(image)
            image_record.file_path = f"/static/uploads/{image_record.file_name.replace('\\', '/')}"
            db.add(image_record)
            db.flush()
            db.refresh(image_record)
            logger.info(f"Created image metadata: {image_record}")
            return image_record
        except Exception as e:
            logger.error(f"Error creating image metadata: {e}")
            db.rollback()
            raise

    def update(self, db: Session, image_id: int, **kwargs) -> Optional[ImageMetadata]:
        try:
            image = self.get_by_id(db, image_id)
            if not image:
                return None
                
            for key, value in kwargs.items():
                setattr(image, key, value)
            
            db.commit()
            db.refresh(image)
            return image
        except Exception as e:
            logger.error(f"Error updating image metadata: {e}")
            db.rollback()
            raise

    def get_by_id(self, db: Session, image_id: int) -> Optional[ImageMetadata]:
        return db.query(ImageMetadata).filter(ImageMetadata.id == image_id).first()

    def get_all(self, db: Session, limit: int = 10000, offset: int = 0) -> List[ImageMetadata]:
        return db.query(ImageMetadata)\
            .order_by(ImageMetadata.created_at.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()


    def _get_similarity_map(self, query_metadata: ImageMetadata) -> Dict[str, Any]:
        """Builds a map of SQLAlchemy expressions for all similarity components using central specs"""
        specs = get_all_feature_specs()
        sim_map = {}

        def raised_cosine_sim(dist_expr):
            """Applies y = (1 + cos(pi * x)) / 2 to distance x"""
            # Ensure x is clamped between 0 and 1 before applying cosine
            clamped_x = func.least(1.0, func.greatest(0.0, dist_expr))
            return (1.0 + func.cos(literal(math.pi) * clamped_x)) / 2.0

        for name, metric in specs.items():
            col_name = COLUMN_MAP.get(name, name)
            col = getattr(ImageMetadata, col_name)
            query_val = getattr(query_metadata, col_name)

            if metric == "scalar":
                dist = func.abs(col - query_val)
                sim_map[name] = raised_cosine_sim(dist)
            
            elif metric == "sharpness":
                diff = func.abs(col - query_val)
                denom = func.abs(col + query_val) + 1e-7
                dist = diff / denom
                sim_map[name] = raised_cosine_sim(dist)
            
            elif metric == "cosine":
                # For PGVector, cosine_distance is already in [0, 2], but usually [0, 1] for unit vectors
                # We normalize it to [0, 1] for the formula
                dist = func.coalesce(col.cosine_distance(query_val), 1.0)
                sim_map[name] = raised_cosine_sim(dist)
            
            elif metric == "l2_color":
                max_d = math.sqrt(255*255*3)
                dist = col.l2_distance(query_val) / max_d
                sim_map[name] = raised_cosine_sim(dist)
            
            elif metric == "l2_cell":
                space = name.split("_")[1] if "_" in name else "rgb"
                max_d = math.sqrt(255*255*3*16) if space != "gray" else math.sqrt(255*255*16)
                dist = func.coalesce(col.l2_distance(query_val), 0.0) / max_d
                sim_map[name] = raised_cosine_sim(dist)
            
            elif metric == "category":
                q_cat = (query_val or "General").lower()
                sim_map[name] = case((func.lower(col) == q_cat, 1.0), else_=0.0)
            
            elif metric == "entity":
                query_ents = list(set([e.lower() for e in (query_val or [])]))
                if query_ents:
                    intersection = sum([case((col.cast(Text).ilike(f'%"{e}"%'), 1.0), else_=0.0) for e in query_ents])
                    union = len(query_ents) + func.jsonb_array_length(col) - intersection
                    # For Jaccard, similarity is already [0, 1]. 
                    # We can treat (1 - similarity) as distance and apply the kernel
                    jaccard_sim = intersection / func.greatest(1.0, union)
                    sim_map[name] = raised_cosine_sim(1.0 - jaccard_sim)
                else:
                    sim_map[name] = literal(0.0)

        return sim_map

    def _apply_weights(self, sim_map: Dict[str, Any], weights: Dict[str, float]) -> Any:
        """Applies weights to similarity expressions and returns the total similarity expression"""
        if not weights:
            # Fallback: Equal weights
            default_w = 1.0 / len(sim_map)
            return sum([expr * default_w for expr in sim_map.values()])
            
        weighted_components = []
        for name, expr in sim_map.items():
            w = weights.get(name, 0.0)
            if w != 0:
                weighted_components.append(expr * w)
        
        return sum(weighted_components) if weighted_components else literal(0.0)

    def search(self, db: Session, query_metadata: ImageMetadata, limit: int = 10, search_settings: Any = None) -> List[Any]:
        """Perform dynamic weighted similarity search based on enabled components"""
        try:
            # 1. Get base similarity expressions
            sim_map = self._get_similarity_map(query_metadata)

            # 2. Determine Weights based on Mode
            mode = getattr(search_settings, 'mode', 'optimized')
            logger.info(f"--- Search Mode: {mode} ---")

            if mode == "manual":
                weights = getattr(search_settings, 'weights', {})
            elif mode == "equal":
                weights = None # Triggers equal weights in _apply_weights
            else: # optimized
                weights = self._load_weights()

            # 3. Build Total Similarity Expression
            total_sim = self._apply_weights(sim_map, weights)

            # 4. Execute Query
            results_query = db.query(
                ImageMetadata,
                cast(total_sim, Float).label('similarity')
            )
            
            # Add all individual similarity components with _similarity suffix
            for name, expr in sim_map.items():
                results_query = results_query.add_columns(cast(expr, Float).label(f"{name}_similarity"))
            
            final_results = results_query.order_by(text('similarity DESC')).limit(limit).all()
            
            return final_results
        except Exception as e:
            logger.error(f"Error performing dynamic hybrid search: {e}")
            raise
