"""
Database Service
Manages database connections and operations
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional
import logging

from ..models.database import Base, ImageMetadata
from ..config import get_settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database connection and operations manager"""
    
    def __init__(self):
        """Initialize database service"""
        self.settings = get_settings()
        self.engine = create_engine(
            self.settings.database_url, 
            echo=False
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables ensured")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise


    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def create_image_metadata(
        self,
        db: Session,
        file_name: str,
        unique_filename: str,
        features: dict
    ) -> ImageMetadata:

        try:
            image_record = ImageMetadata(
                file_name=file_name,
                unique_filename=unique_filename,
                width=features.get('width'),
                height=features.get('height'),
                brightness=features.get('brightness'),
                contrast=features.get('contrast'),
                saturation=features.get('saturation'),
                edge_density=features.get('edge_density'),
                dominant_color_hex=features.get('dominant_color_hex'),
                features_json=features.get('features_json'),
                texture_json=features.get('texture_json'),
                shape_json=features.get('shape_json'),
                embedding_json=features.get('embedding_json') if isinstance(features.get('embedding_json'), str) else str(features.get('embedding_json')) # Ensure string if list
            )
            
            db.add(image_record)
            db.commit()
            db.refresh(image_record)
            
            logger.info(f"Created image metadata: ID {image_record.id}")
            return image_record
            
        except Exception as e:
            logger.error(f"Error creating image metadata: {e}")
            db.rollback()
            raise
    
    def update_image_metadata(
        self,
        db: Session,
        image_id: int,
        features: dict
    ) -> Optional[ImageMetadata]:
        """
        Update image metadata with new features
        """
        try:
            image = self.get_image_by_id(db, image_id)
            if not image:
                return None
                
            # Update fields
            image.width = features.get('width')
            image.height = features.get('height')
            image.brightness = features.get('brightness')
            image.contrast = features.get('contrast')
            image.saturation = features.get('saturation')
            image.edge_density = features.get('edge_density')
            image.dominant_color_hex = features.get('dominant_color_hex')
            image.features_json = features.get('features_json')
            image.texture_json = features.get('texture_json')
            image.shape_json = features.get('shape_json')
            
            # Handle embedding_json list->str conversion if needed
            emb = features.get('embedding_json')
            if emb is not None:
                image.embedding_json = emb if isinstance(emb, str) else str(emb)
            
            db.commit()
            db.refresh(image)
            
            logger.info(f"Updated image metadata: ID {image.id}")
            return image
            
        except Exception as e:
            logger.error(f"Error updating image metadata: {e}")
            db.rollback()
            raise
    
    def get_images(
        self,
        db: Session,
        limit: int = 100,
        offset: int = 0
    ) -> List[ImageMetadata]:

        try:
            images = db.query(ImageMetadata)\
                .order_by(ImageMetadata.created_at.desc())\
                .limit(limit)\
                .offset(offset)\
                .all()
            
            return images
            
        except Exception as e:
            logger.error(f"Error fetching images: {e}")
            raise
    
    def get_image_by_id(self, db: Session, image_id: int) -> Optional[ImageMetadata]:
        """
        Get image by ID
        
        Args:
            db: Database session
            image_id: Image ID
            
        Returns:
            ImageMetadata or None
        """
        try:
            return db.query(ImageMetadata)\
                .filter(ImageMetadata.id == image_id)\
                .first()
        except Exception as e:
            logger.error(f"Error fetching image {image_id}: {e}")
            raise
