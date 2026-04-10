from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from pgvector.sqlalchemy import Vector
from datetime import datetime
from ...db.base import Base

class ImageMetadata(Base):
    """Image metadata table"""
    __tablename__ = "image_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    brightness = Column(Float)
    contrast = Column(Float)
    saturation = Column(Float)
    edge_density = Column(Float)
    histogram_vector = Column(Vector(48))  # 16 bins * 3 channels
    hog_vector = Column(Vector(1764))  # HOG features (128x128 with 16x16 pixels per cell)
    hu_moments_vector = Column(Vector(7))  # 7 invariant Hu Moments for shape matching
    
    file_path = Column(Text)
    hog_vis_path = Column(Text)
    hu_vis_path = Column(Text)
    dinov2_vector = Column(Vector(768))  # DINOv2 ViT-B/14 generates 768-dimensional embeddings
    clip_vector = Column(Vector(512))    # CLIP ViT-B/32 generates 512-dimensional embeddings
    dominant_color_vector = Column(Vector(3)) # Top Dominant Color in LAB space (3-dim)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ImageMetadata(id={self.id}, file_name='{self.file_name}')>"

    def to_dict(self):
        return {
            "id": self.id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "edge_density": self.edge_density,
            "histogram_vector": self.histogram_vector,
            "hog_vector": self.hog_vector,
            "hu_moments_vector": self.hu_moments_vector,
            "file_path": self.file_path,
            "hog_vis_path": self.hog_vis_path,
            "hu_vis_path": self.hu_vis_path,
            "dinov2_vector": self.dinov2_vector,
            "created_at": self.created_at
        }