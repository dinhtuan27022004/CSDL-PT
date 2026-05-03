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
    # --- RGB Features ---
    rgb_hist_std = Column(Vector(24))
    rgb_hist_interp = Column(Vector(24))
    rgb_hist_gauss = Column(Vector(24))
    rgb_cdf_std = Column(Vector(24))
    rgb_cdf_interp = Column(Vector(24))
    rgb_cdf_gauss = Column(Vector(24))
    joint_rgb_std = Column(Vector(64))
    joint_rgb_interp = Column(Vector(64))
    joint_rgb_gauss = Column(Vector(64))
    cell_rgb_vector = Column(Vector(48))

    # --- HSV Features ---
    hsv_hist_std = Column(Vector(24))
    hsv_hist_interp = Column(Vector(24))
    hsv_hist_gauss = Column(Vector(24))
    hsv_cdf_std = Column(Vector(24))
    hsv_cdf_interp = Column(Vector(24))
    hsv_cdf_gauss = Column(Vector(24))
    joint_hsv_std = Column(Vector(64))
    joint_hsv_interp = Column(Vector(64))
    joint_hsv_gauss = Column(Vector(64))
    cell_hsv_vector = Column(Vector(48))

    # --- Lab Features ---
    lab_hist_std = Column(Vector(24))
    lab_hist_interp = Column(Vector(24))
    lab_hist_gauss = Column(Vector(24))
    lab_cdf_std = Column(Vector(24))
    lab_cdf_interp = Column(Vector(24))
    lab_cdf_gauss = Column(Vector(24))
    joint_lab_std = Column(Vector(64))
    joint_lab_interp = Column(Vector(64))
    joint_lab_gauss = Column(Vector(64))
    cell_lab_vector = Column(Vector(48))

    # --- YCrCb Features ---
    ycrcb_hist_std = Column(Vector(24))
    ycrcb_hist_interp = Column(Vector(24))
    ycrcb_hist_gauss = Column(Vector(24))
    ycrcb_cdf_std = Column(Vector(24))
    ycrcb_cdf_interp = Column(Vector(24))
    ycrcb_cdf_gauss = Column(Vector(24))
    joint_ycrcb_std = Column(Vector(64))
    joint_ycrcb_interp = Column(Vector(64))
    joint_ycrcb_gauss = Column(Vector(64))
    cell_ycrcb_vector = Column(Vector(48))

    # --- HLS Features ---
    hls_hist_std = Column(Vector(24))
    hls_hist_interp = Column(Vector(24))
    hls_hist_gauss = Column(Vector(24))
    hls_cdf_std = Column(Vector(24))
    hls_cdf_interp = Column(Vector(24))
    hls_cdf_gauss = Column(Vector(24))
    joint_hls_std = Column(Vector(64))
    joint_hls_interp = Column(Vector(64))
    joint_hls_gauss = Column(Vector(64))
    cell_hls_vector = Column(Vector(48))

    # --- XYZ Features ---
    xyz_hist_std = Column(Vector(24))
    xyz_hist_interp = Column(Vector(24))
    xyz_hist_gauss = Column(Vector(24))
    xyz_cdf_std = Column(Vector(24))
    xyz_cdf_interp = Column(Vector(24))
    xyz_cdf_gauss = Column(Vector(24))
    joint_xyz_std = Column(Vector(64))
    joint_xyz_interp = Column(Vector(64))
    joint_xyz_gauss = Column(Vector(64))
    cell_xyz_vector = Column(Vector(48))

    # --- Gray Features ---
    gray_hist_std = Column(Vector(8))
    gray_hist_interp = Column(Vector(8))
    gray_hist_gauss = Column(Vector(8))
    gray_cdf_std = Column(Vector(8))
    gray_cdf_interp = Column(Vector(8))
    gray_cdf_gauss = Column(Vector(8))
    cell_gray_vector = Column(Vector(16))

    hog_vector = Column(Vector(1568))  # HOG features
    hu_moments_vector = Column(Vector(7))  # 7 invariant Hu Moments
    dominant_color_vector = Column(Vector(3)) 
    lbp_vector = Column(Vector(160)) # Local Binary Patterns (10 bins * 16 cells)
    color_moments_vector = Column(Vector(9)) # Mean, Std, Skew for 3 channels (HSV)
    sharpness = Column(Float)
    gabor_vector = Column(Vector(512)) # Gabor Filter Bank (32 stats * 16 cells)
    ccv_vector = Column(Vector(96)) # Color Coherence Vector (2 * 48 histogram bins)
    zernike_vector = Column(Vector(25)) # Zernike Moments (Order 8)
    geo_vector = Column(Vector(6)) # Geometric Shape Profile (Roundness, Solidity, etc.)
    
    # Advanced Traditional Features
    tamura_vector = Column(Vector(3)) # Coarseness, Contrast, Directionality
    edge_orientation_vector = Column(Vector(5)) # Edge Orientation Histogram
    glcm_vector = Column(Vector(64)) # GLCM Texture (4 stats * 16 cells)
    wavelet_vector = Column(Vector(12)) # DWT Wavelet Energy (3 levels * 4 subbands?)
    correlogram_vector = Column(Vector(32)) # Color Auto-correlogram (8 bins * 4 distances)
    bovw_vector = Column(Vector(512)) # Bag of Visual Words (512 clusters)
    ehd_vector = Column(Vector(80)) # MPEG-7 Edge Histogram (5 orientations * 16 cells)
    cld_vector = Column(Vector(64)) # MPEG-7 Color Layout (YCbCr DCT)
    spm_vector = Column(Vector(160)) # Spatial Pyramid Matching (1x1(32) + 2x2(32*4))
    saliency_vector = Column(Vector(32)) # Features from Salient regions
    
    clip_vector = Column(Vector(768)) # CLIP Large Vector
    dinov2_vector = Column(Vector(1536)) # DINOv2 Giant Vector
    siglip_vector = Column(Vector(768)) # SigLIP Base Vector
    convnext_vector = Column(Vector(1024)) # ConvNeXt V2 Base Vector
    efficientnet_vector = Column(Vector(2560)) # EfficientNet-B7 Vector
    dreamsim_vector = Column(Vector(1792)) # DreamSim Ensemble Vector (Actual: 1792)
    sam_vector = Column(Vector(12544)) # SAM ViT-B High-Fidelity Vector (256x7x7)
    
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
