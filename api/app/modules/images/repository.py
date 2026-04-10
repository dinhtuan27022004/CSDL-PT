from sqlalchemy.orm import Session
from sqlalchemy import text, func
from typing import List, Optional, Tuple
from .model import ImageMetadata
from ...core.logging import get_logger
from ...core.config import get_settings
logger = get_logger(__name__)

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

    def get_by_id(self, db: Session, image_id: int) -> Optional[ImageMetadata]:
        return db.query(ImageMetadata).filter(ImageMetadata.id == image_id).first()

    def search(self, 
                db: Session, 
                query_metadata: ImageMetadata,
                limit: int = 10) -> List[Tuple]:
        """Perform 5-way weighted similarity search (20% each) directly in SQL"""
        try:
            # Individual Similarity Formulas (1.0 - Distance)
            # DINOv2 uses SQRT punishment model (defined below)
            
            # Traditional attributes use Square Root Difference Similarity (1.0 - sqrt(|X-Q|))
            # This is extremely sensitive to even minor differences (High sensitivity punishment)
            b_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.brightness - query_metadata.brightness))
            c_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.contrast - query_metadata.contrast))
            s_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.saturation - query_metadata.saturation))
            e_sim = 1.0 - func.sqrt(func.abs(ImageMetadata.edge_density - query_metadata.edge_density))
            
            # Vector attributes use 1.0 - sqrt(cosine_distance)
            h_sim = 1.0 - ImageMetadata.histogram_vector.cosine_distance(query_metadata.histogram_vector)
            hog_sim = 1.0 - ImageMetadata.hog_vector.cosine_distance(query_metadata.hog_vector)
            hu_sim = 1.0 - ImageMetadata.hu_moments_vector.cosine_distance(query_metadata.hu_moments_vector)
            
            # DINOv2 uses SQL punishment model
            dino_sim = 1.0 - func.sqrt(ImageMetadata.dinov2_vector.cosine_distance(query_metadata.dinov2_vector) / 2.0)
            
            # CLIP uses the same normalized punishment model
            clip_sim = 1.0 - func.sqrt(ImageMetadata.clip_vector.cosine_distance(query_metadata.clip_vector) / 2.0)
            
            # Dominant Color Similarity (Perceptual Distance in Lab space)
            # We use a sensitive normalization for color match
            dom_dist = ImageMetadata.dominant_color_vector.l2_distance(query_metadata.dominant_color_vector)
            dom_sim = 1.0 - func.sqrt(dom_dist / 200.0)
            
            # Hybrid scoring: Mix of CLIP (Semantic) + Histogram (Distribution) + Dominant Color (Theme)
            # You can adjust these weights. Right now: 60% CLIP, 20% Histogram, 20% Dominant Color
            total_sim = dom_sim
            
            # Query returns the record plus all individual similarity components
            results = db.query(
                ImageMetadata,
                total_sim.label('similarity'),
                dino_sim.label('dino_sim'),
                clip_sim.label('clip_sim'),
                dom_sim.label('dom_sim'),
                b_sim.label('b_sim'),
                c_sim.label('c_sim'),
                s_sim.label('s_sim'),
                e_sim.label('e_sim'),
                h_sim.label('h_sim'),
                hog_sim.label('hog_sim'),
                hu_sim.label('hu_sim')
            ).order_by(text('similarity DESC')).limit(limit).all()
            
            return results
        except Exception as e:
            logger.error(f"Error performing 5-way hybrid search: {e}")
            raise
