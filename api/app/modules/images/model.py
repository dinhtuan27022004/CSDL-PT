from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from datetime import datetime
from ...db.base import Base

class ImageMetadata(Base):
    """Image metadata table"""
    __tablename__ = "image_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    
    brightness = Column(Float)
    contrast = Column(Float)
    saturation = Column(Float)
    edge_density = Column(Float)
    hsv_histogram_vector = Column(Vector(24))  # 8 bins * 3 channels
    rgb_histogram_vector = Column(Vector(24))  # 8 bins * 3 channels
    hsv_cdf_vector = Column(Vector(24)) # Cumulative HSV Hist
    rgb_cdf_vector = Column(Vector(24)) # Cumulative RGB Hist
    hog_vector = Column(Vector(1568))  # HOG features
    hu_moments_vector = Column(Vector(7))  # 7 invariant Hu Moments
    dominant_color_vector = Column(Vector(3)) 
    cell_color_vector = Column(Vector(48)) 
    lbp_vector = Column(Vector(160)) # Local Binary Patterns (10 bins * 16 cells)
    color_moments_vector = Column(Vector(9)) # Mean, Std, Skew for 3 channels (HSV)
    sharpness = Column(Float)
    gabor_vector = Column(Vector(512)) # Gabor Filter Bank (32 stats * 16 cells)
    ccv_vector = Column(Vector(96)) # Color Coherence Vector (2 * 48 histogram bins)
    zernike_vector = Column(Vector(25)) # Zernike Moments (Order 8)
    geo_vector = Column(Vector(6)) # Geometric Shape Profile (Roundness, Solidity, etc.)
    
    # Semantic features from LLM
    category = Column(String(100))
    description = Column(Text)
    entities = Column(JSONB) # List of detected entities (Postgres JSONB for search)
    llm_embedding = Column(Vector(1024)) # BGE-M3 embedding size
    
    file_path = Column(Text)
    hog_vis_path = Column(Text)
    hu_vis_path = Column(Text)
    cell_color_vis_path = Column(Text)
    lbp_vis_path = Column(Text)
    gabor_vis_path = Column(Text)
    ccv_vis_path = Column(Text)
    histogram_vis_path = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ImageMetadata(id={self.id}, file_name='{self.file_name}')>"

