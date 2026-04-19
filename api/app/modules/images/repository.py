from sqlalchemy.orm import Session
from sqlalchemy import text, func, case, cast, Float, literal
from sqlalchemy.dialects.postgresql import JSONB
from typing import List, Optional, Tuple
from .model import ImageMetadata
from ...core.logging import get_logger
from ...core.config import get_settings
logger = get_logger(__name__)
import math
class ImageRepository:
    """Repository for image metadata database operations"""
    def __init__(self):
        self.settings = get_settings()

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

    def get_all(self, db: Session, limit: int = 500, offset: int = 0) -> List[ImageMetadata]:
        return db.query(ImageMetadata)\
            .order_by(ImageMetadata.created_at.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()

    def search(self, 
                db: Session, 
                query_metadata: ImageMetadata,
                limit: int = 10) -> List[Tuple]:
        """Perform 5-way weighted similarity search (20% each) directly in SQL"""
        try:
            brightness_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.brightness - query_metadata.brightness))
            contrast_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.contrast - query_metadata.contrast))
            saturation_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.saturation - query_metadata.saturation))
            edge_density_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.edge_density - query_metadata.edge_density))
            hsv_histogram_sim = 1.0 - func.sqrt(ImageMetadata.hsv_histogram_vector.cosine_distance(query_metadata.hsv_histogram_vector))
            rgb_histogram_sim = 1.0 - func.sqrt(ImageMetadata.rgb_histogram_vector.cosine_distance(query_metadata.rgb_histogram_vector))
            hsv_cdf_sim = 1.0 - func.sqrt(ImageMetadata.hsv_cdf_vector.cosine_distance(query_metadata.hsv_cdf_vector))
            rgb_cdf_sim = 1.0 - func.sqrt(ImageMetadata.rgb_cdf_vector.cosine_distance(query_metadata.rgb_cdf_vector))
            hog_sim = 1 - func.sqrt(ImageMetadata.hog_vector.cosine_distance(query_metadata.hog_vector))
            hu_sim = 1.0 - func.sqrt(ImageMetadata.hu_moments_vector.cosine_distance(query_metadata.hu_moments_vector))
            dom_sim = 1.0 - ImageMetadata.dominant_color_vector.l2_distance(query_metadata.dominant_color_vector) / (math.sqrt(255*255*3))
            cell_color_sim = 1.0 - ImageMetadata.cell_color_vector.l2_distance(query_metadata.cell_color_vector) / (math.sqrt(255*255*3*16))
            lbp_sim = 1.0 - func.sqrt(ImageMetadata.lbp_vector.cosine_distance(query_metadata.lbp_vector))
            color_moments_sim = 1.0 - func.sqrt(ImageMetadata.color_moments_vector.cosine_distance(query_metadata.color_moments_vector))
            sharpness_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.sharpness - query_metadata.sharpness) / (func.abs(ImageMetadata.sharpness + query_metadata.sharpness) + 1e-7))
            gabor_sim = 1.0 - func.sqrt(ImageMetadata.gabor_vector.cosine_distance(query_metadata.gabor_vector))
            ccv_sim = 1.0 - func.sqrt(ImageMetadata.ccv_vector.cosine_distance(query_metadata.ccv_vector))
            zernike_sim = 1.0 - func.sqrt(ImageMetadata.zernike_vector.cosine_distance(query_metadata.zernike_vector))
            geo_sim = 1.0 - func.sqrt(ImageMetadata.geo_vector.cosine_distance(query_metadata.geo_vector))
            embedding_sim = 1.0 - func.sqrt(func.coalesce(ImageMetadata.llm_embedding.cosine_distance(query_metadata.llm_embedding), 1.0))
            
            # 19. Category Similarity (Exact Match) - Normalized to lowercase
            query_category = (query_metadata.category or "General").lower()
            category_sim = case((func.lower(ImageMetadata.category) == query_category, 1.0), else_=0.0)
            
            # 20. Entity Similarity (Overlap Ratio)
            query_entities = [e.lower() for e in (query_metadata.entities or [])]
            if query_entities:
                # Summing existence of each query tag in the image's entities JSON array
                # Using @> operator with a small JSONB array for maximum robustness
                entity_match_sum = sum([
                    case((ImageMetadata.entities.cast(JSONB).contains([tag]), 1.0), else_=0.0) 
                    for tag in query_entities
                ])
                entity_sim = entity_match_sum / len(query_entities)
            else:
                entity_sim = literal(0.0)
            
            # Total Similarity: Custom combined sum from USER + CDF components
            total_sim = entity_sim + brightness_sim + contrast_sim + category_sim + saturation_sim + \
                        edge_density_sim + hsv_histogram_sim + rgb_histogram_sim + \
                        hsv_cdf_sim + rgb_cdf_sim
            total_sim = cell_color_sim
            # Query returns the record plus all individual similarity components
            results = db.query(
                ImageMetadata,
                total_sim.label('similarity'),
                brightness_sim.label('brightness_sim'),
                contrast_sim.label('contrast_sim'),
                saturation_sim.label('saturation_sim'),
                edge_density_sim.label('edge_density_sim'),
                hsv_histogram_sim.label('hsv_histogram_sim'),
                rgb_histogram_sim.label('rgb_histogram_sim'),
                hog_sim.label('hog_sim'),
                hu_sim.label('hu_sim'),
                dom_sim.label('dom_sim'),
                cell_color_sim.label('cell_color_sim'),
                lbp_sim.label('lbp_similarity'),
                color_moments_sim.label('color_moments_similarity'),
                sharpness_sim.label('sharpness_similarity'),
                gabor_sim.label('gabor_similarity'),
                ccv_sim.label('ccv_similarity'),
                zernike_sim.label('zernike_similarity'),
                geo_sim.label('geo_sim'),
                embedding_sim.label('embedding_sim'),
                entity_sim.label('entity_sim'),
                category_sim.label('category_sim'),
                hsv_cdf_sim.label('hsv_cdf_sim'),
                rgb_cdf_sim.label('rgb_cdf_sim')
            ).order_by(text('similarity DESC')).limit(limit).all()
            
            return results
        except Exception as e:
            logger.error(f"Error performing advanced hybrid search: {e}")
            raise 
