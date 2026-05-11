from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from datetime import datetime
from ..db.base import Base

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
    # --- Consolidated Color Meta-Features (Interpolated Only) ---
    meta_hist_interp = Column(Vector(152))
    meta_cdf_interp = Column(Vector(152))
    meta_joint_interp = Column(Vector(384))
    meta_cell_vector = Column(Vector(304))

    meta_moments_mean = Column(Vector(19))
    meta_moments_std = Column(Vector(19))
    meta_moments_skew = Column(Vector(19))


    hog_vector = Column(Vector(1568))  # HOG features
    hu_moments_vector = Column(Vector(7))  # 7 invariant Hu Moments
    lbp_vector = Column(Vector(160)) # Local Binary Patterns (10 bins * 16 cells)

    sharpness = Column(Float)
    gabor_vector = Column(Vector(512)) # Gabor Filter Bank (32 stats * 16 cells)
    ccv_vector = Column(Vector(96)) # Color Coherence Vector (2 * 48 histogram bins)
    fourier_vector = Column(Vector(25)) # Fourier Descriptors
    geo_vector = Column(Vector(6)) # Geometric Shape Profile (Roundness, Solidity, etc.)
    
    # Advanced Traditional Features
    tamura_vector = Column(Vector(3)) # Coarseness, Contrast, Directionality
    edge_orientation_vector = Column(Vector(5)) # Edge Orientation Histogram
    glcm_vector = Column(Vector(64)) # GLCM Texture (4 stats * 16 cells)
    wavelet_vector = Column(Vector(12)) # DWT Wavelet Energy (3 levels * 4 subbands?)
    correlogram_vector = Column(Vector(32)) # Color Auto-correlogram (8 bins * 4 distances)
    ehd_vector = Column(Vector(80)) # MPEG-7 Edge Histogram (5 orientations * 16 cells)
    cld_vector = Column(Vector(64)) # MPEG-7 Color Layout (YCbCr DCT)
    spm_vector = Column(Vector(160)) # Spatial Pyramid Matching (1x1(32) + 2x2(32*4))
    saliency_vector = Column(Vector(32)) # Features from Salient regions

    
    # Semantic features from LLM
    category = Column(String(100))
    description = Column(Text)
    entities = Column(JSONB) # List of detected entities (Postgres JSONB for search)
    llm_embedding = Column(Vector(1024)) # BGE-M3 embedding size
    dreamsim_vector = Column(Vector(1792), nullable=True) # DreamSim features
    
    file_path = Column(Text)
    hog_vis_path = Column(Text)
    hu_vis_path = Column(Text)
    cell_color_vis_path = Column(Text)
    lbp_vis_path = Column(Text)
    gabor_vis_path = Column(Text)
    ccv_vis_path = Column(Text)
    histogram_vis_path = Column(Text)
    
    file_hash = Column(String(32), index=True, nullable=True) # MD5 hash for duplicate detection
    
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ImageMetadata(id={self.id}, file_name='{self.file_name}')>"
