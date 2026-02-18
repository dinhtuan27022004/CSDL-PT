"""
Database Models
SQLAlchemy ORM models for image metadata
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class ImageMetadata(Base):
    """Image metadata table"""
    __tablename__ = "image_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    unique_filename = Column(String(255), nullable=False, unique=True)  # Store UUID filename
    width = Column(Integer)
    height = Column(Integer)
    brightness = Column(Float)
    contrast = Column(Float)
    saturation = Column(Float)
    edge_density = Column(Float)
    dominant_color_hex = Column(String(7))
    features_json = Column(Text)
    texture_json = Column(Text)
    shape_json = Column(Text)
    embedding_json = Column(Text)  # Storing as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ImageMetadata(id={self.id}, file_name='{self.file_name}')>"
