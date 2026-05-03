from sqlalchemy.orm import Session
from sqlalchemy import text, func, case, cast, Float, literal
from sqlalchemy.dialects.postgresql import JSONB
from typing import List, Optional, Tuple, Any
from ..models.image import ImageMetadata
from ..core.logging import get_logger
from ..core.config import get_settings
import math
import os
import json

logger = get_logger(__name__)

class ImageRepository:
    """Repository for image metadata database operations"""
    def __init__(self):
        self.settings = get_settings()

    def _load_weights(self, suffix: Optional[str] = None) -> dict:
        """Load optimized weights from file if it exists, optionally with a model suffix"""
        filename = f"weights_{suffix}.json" if suffix else self.settings.weights_file
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
            with open(img_path, "wb") as f:
                f.write(image)
            image_record.file_path = f"/static/uploads/{image_record.file_name}"
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

    def get_all(self, db: Session, limit: int = 500, offset: int = 0) -> List[ImageMetadata]:
        return db.query(ImageMetadata)\
            .order_by(ImageMetadata.created_at.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()

    def search(self, db: Session, query_metadata: ImageMetadata, limit: int = 10, search_settings: Any = None) -> Tuple[List[Any], Optional[List[Any]]]:
        """Perform dynamic weighted similarity search based on enabled components"""
        try:
            # Helper to make expressions NaN-proof and range-bound [0, 1]
            def safe_sim(expr):
                # Using coalesce and least/greatest is enough since we prevent 
                # negative sqrt inputs at the source.
                return func.coalesce(
                    func.least(1.0, func.greatest(0.0, expr)), 
                    0.0
                )

            # 1. Scalar Similarities
            brightness_sim = safe_sim(1.0 - func.greatest(0.0, func.abs(ImageMetadata.brightness - query_metadata.brightness)))
            contrast_sim = safe_sim(1.0 - func.greatest(0.0, func.abs(ImageMetadata.contrast - query_metadata.contrast)))
            saturation_sim = safe_sim(1.0 - func.greatest(0.0, func.abs(ImageMetadata.saturation - query_metadata.saturation)))
            edge_density_sim = safe_sim(1.0 - func.greatest(0.0, func.abs(ImageMetadata.edge_density - query_metadata.edge_density)))
            
            sharp_diff = func.abs(ImageMetadata.sharpness - query_metadata.sharpness)
            sharp_denom = func.abs(ImageMetadata.sharpness + query_metadata.sharpness) + 1e-7
            sharpness_sim = safe_sim(1.0 - func.greatest(0.0, sharp_diff / sharp_denom))

            # Build complete sim_map with all available features
            sim_map = {
                "brightness": brightness_sim,
                "contrast": contrast_sim,
                "saturation": saturation_sim,
                "edge_density": edge_density_sim,
                "sharpness": sharpness_sim,
            }

            # 2. Color Space Similarities (Dynamic Generation)
            for space in ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"]:
                for method in ["std", "interp", "gauss"]:
                    # Hist
                    h_col = f"{space}_hist_{method}"
                    sim_map[h_col] = safe_sim(1.0 - func.greatest(0.0, getattr(ImageMetadata, h_col).cosine_distance(getattr(query_metadata, h_col))))
                    # CDF
                    c_col = f"{space}_cdf_{method}"
                    sim_map[c_col] = safe_sim(1.0 - func.greatest(0.0, getattr(ImageMetadata, c_col).cosine_distance(getattr(query_metadata, c_col))))
                    # Joint (except gray)
                    if space != "gray":
                        j_col = f"joint_{space}_{method}"
                        sim_map[j_col] = safe_sim(1.0 - func.greatest(0.0, getattr(ImageMetadata, j_col).cosine_distance(getattr(query_metadata, j_col))))
                
                # Cell Color (L2)
                cell_col = f"cell_{space}_vector"
                max_dist = math.sqrt(255*255*3*16) if space != "gray" else math.sqrt(255*255*16)
                sim_map[f"cell_{space}"] = safe_sim(1.0 - func.greatest(0.0, func.coalesce(getattr(ImageMetadata, cell_col).l2_distance(getattr(query_metadata, cell_col)), 0.0)) / max_dist)

            # 3. Traditional Features
            sim_map.update({
                "hog": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.hog_vector.cosine_distance(query_metadata.hog_vector))),
                "hu_moments": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.hu_moments_vector.cosine_distance(query_metadata.hu_moments_vector))),
                "lbp": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.lbp_vector.cosine_distance(query_metadata.lbp_vector))),
                "color_moments": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.color_moments_vector.cosine_distance(query_metadata.color_moments_vector))),
                "gabor": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.gabor_vector.cosine_distance(query_metadata.gabor_vector))),
                "ccv": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.ccv_vector.cosine_distance(query_metadata.ccv_vector))),
                "zernike": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.zernike_vector.cosine_distance(query_metadata.zernike_vector))),
                "geo": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.geo_vector.cosine_distance(query_metadata.geo_vector))),
                "tamura": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.tamura_vector.cosine_distance(query_metadata.tamura_vector))),
                "edge_orientation": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.edge_orientation_vector.cosine_distance(query_metadata.edge_orientation_vector))),
                "glcm": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.glcm_vector.cosine_distance(query_metadata.glcm_vector))),
                "wavelet": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.wavelet_vector.cosine_distance(query_metadata.wavelet_vector))),
                "correlogram": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.correlogram_vector.cosine_distance(query_metadata.correlogram_vector))),
                "ehd": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.ehd_vector.cosine_distance(query_metadata.ehd_vector))),
                "cld": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.cld_vector.cosine_distance(query_metadata.cld_vector))),
                "spm": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.spm_vector.cosine_distance(query_metadata.spm_vector))),
                "saliency": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.saliency_vector.cosine_distance(query_metadata.saliency_vector))),
                "bovw": safe_sim(1.0 - func.greatest(0.0, ImageMetadata.bovw_vector.cosine_distance(query_metadata.bovw_vector))),
            })

            # 4. Deep Learning / Semantic
            sim_map.update({
                "semantic": safe_sim(1.0 - func.greatest(0.0, func.coalesce(ImageMetadata.llm_embedding.cosine_distance(query_metadata.llm_embedding), 1.0))),
                "clip": safe_sim(1.0 - func.greatest(0.0, func.coalesce(ImageMetadata.clip_vector.cosine_distance(query_metadata.clip_vector), 1.0))),
                "dinov2": safe_sim(1.0 - func.greatest(0.0, func.coalesce(ImageMetadata.dinov2_vector.cosine_distance(query_metadata.dinov2_vector), 1.0))),
                "siglip": safe_sim(1.0 - func.greatest(0.0, func.coalesce(ImageMetadata.siglip_vector.cosine_distance(query_metadata.siglip_vector), 1.0))),
                "convnext": safe_sim(1.0 - func.greatest(0.0, func.coalesce(ImageMetadata.convnext_vector.cosine_distance(query_metadata.convnext_vector), 1.0))),
                "efficientnet": safe_sim(1.0 - func.greatest(0.0, func.coalesce(ImageMetadata.efficientnet_vector.cosine_distance(query_metadata.efficientnet_vector), 1.0))),
                "dreamsim": safe_sim(1.0 - func.greatest(0.0, func.coalesce(ImageMetadata.dreamsim_vector.cosine_distance(query_metadata.dreamsim_vector), 1.0))),
                "sam": safe_sim(1.0 - func.greatest(0.0, func.coalesce(ImageMetadata.sam_vector.cosine_distance(query_metadata.sam_vector), 1.0))),
            })
            
            # 5. Discrete / Dominant Color
            sim_map["dominant_color"] = safe_sim(1.0 - ImageMetadata.dominant_color_vector.l2_distance(query_metadata.dominant_color_vector) / (math.sqrt(255*255*3)))
            
            query_category = (query_metadata.category or "General").lower()
            sim_map["category"] = case((func.lower(ImageMetadata.category) == query_category, 1.0), else_=0.0)
            
            query_entities = [e.lower() for e in (query_metadata.entities or [])]
            if query_entities:
                entity_match_sum = sum([
                    case((ImageMetadata.entities.cast(JSONB).contains([tag]), 1.0), else_=0.0) 
                    for tag in query_entities
                ])
                sim_map["entity"] = entity_match_sum / len(query_entities)
            else:
                sim_map["entity"] = literal(0.0)

            if not sim_map:
                total_sim = literal(0.0)
            else:
                # Priority and Mode handling:
                # 1. manual: Use weights provided in settings. If empty, use equal.
                # 2. equal: Force equal weights.
                # 3. optimized: Load from weights.json. Fallback to equal if missing.
                
                weights = None
                mode = getattr(search_settings, 'mode', 'optimized')
                logger.info(f"--- Search Mode: {mode} ---")
                
                # Debug: Check if query metadata has features
                logger.info(f"Query Image Features Sample - Brightness: {query_metadata.brightness}, Contrast: {query_metadata.contrast}")
                if query_metadata.clip_vector:
                    logger.info(f"Query Image has CLIP vector (dim: {len(query_metadata.clip_vector)})")
                else:
                    logger.warning("Query Image is MISSING CLIP vector!")

                if mode == "manual":
                    weights = getattr(search_settings, 'weights', None)
                    logger.info(f"Manual weights keys: {list(weights.keys()) if weights else 'None'}")
                elif mode == "equal":
                    weights = {} # Triggers equal weights fallback
                    logger.info("Using Equal Weights mode")
                else: # optimized
                    opt_target = getattr(search_settings, 'optimization_target', 'clip')
                    weights = self._load_weights(opt_target)
                    logger.info(f"Optimized weights ({opt_target}) keys: {list(weights.keys()) if weights else 'None'}")
                
                weighted_components = []
                weight_sum = 0.0
                
                if not weights:
                    # Case: equal weights
                    default_w = 1.0 / len(sim_map)
                    logger.info(f"No weights found, using equal weight: {default_w:.4f} for {len(sim_map)} features")
                    for name, sim_expr in sim_map.items():
                        weighted_components.append(sim_expr * default_w)
                        weight_sum += default_w
                else:
                    logger.info(f"Applying weights to {len(sim_map)} available features")
                    for name, sim_expr in sim_map.items():
                        w = weights.get(name, 0.0)
                        if w != 0:
                            weighted_components.append(sim_expr * w)
                            weight_sum += w
                    logger.info(f"Total active components: {len(weighted_components)}, Total Weight Sum: {weight_sum:.4f}")
                
                if weighted_components:
                    total_sim = sum(weighted_components)
                else:
                    logger.error("CRITICAL: No weighted components! Total similarity will be 0.")
                    total_sim = literal(0.0)

            results_query = db.query(
                ImageMetadata,
                cast(total_sim, Float).label('similarity')
            )
            
            # Add all individual similarity components with _sim suffix
            for name, expr in sim_map.items():
                results_query = results_query.add_columns(cast(expr, Float).label(f"{name}_sim"))
            
            final_results = results_query.order_by(text('similarity DESC')).limit(limit).all()
            
            gt_results = None
            if getattr(search_settings, 'compare_with_gt', False):
                opt_target = getattr(search_settings, 'optimization_target', 'clip')
                gt_sim_expr = sim_map.get(opt_target, literal(0.0))
                gt_results = db.query(
                    ImageMetadata,
                    cast(gt_sim_expr, Float).label('similarity')
                ).order_by(text('similarity DESC')).limit(limit).all()
            
            return final_results, gt_results
        except Exception as e:
            logger.error(f"Error performing dynamic hybrid search: {e}")
            raise
