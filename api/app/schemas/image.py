from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime

class SearchSettings(BaseModel):
    """Configuration for weighted similarity search"""
    mode: str = "optimized"  # "optimized" | "manual" | "equal"
    weights: Optional[Dict[str, float]] = None
    limit: int = 20
    
    # Boolean flags for toggling similarity components
    use_meta_hist_interp: bool = True
    use_meta_cdf_interp: bool = True
    use_meta_joint_interp: bool = True
    use_meta_cell: bool = True
    use_meta_moments: bool = True

    use_tamura: bool = True
    use_edge_orientation: bool = True
    use_dreamsim: bool = True

class ImageBase(BaseModel):
    file_name: str
    width: Optional[int] = None
    height: Optional[int] = None
    
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    edge_density: Optional[float] = None
    sharpness: Optional[float] = None

    # --- Consolidated Meta-Features ---
    meta_hist_interp: Optional[List[float]] = None
    meta_cdf_interp: Optional[List[float]] = None
    meta_joint_interp: Optional[List[float]] = None
    meta_cell_vector: Optional[List[float]] = None

    meta_moments_mean: Optional[List[float]] = None
    meta_moments_std: Optional[List[float]] = None
    meta_moments_skew: Optional[List[float]] = None

    file_path: Optional[str] = None
    hog_vis_path: Optional[str] = None
    hu_vis_path: Optional[str] = None
    cell_color_vis_path: Optional[str] = None
    lbp_vis_path: Optional[str] = None
    gabor_vis_path: Optional[str] = None
    ccv_vis_path: Optional[str] = None
    histogram_vis_path: Optional[str] = None
    
    # Semantic features
    category: Optional[str] = None
    description: Optional[str] = None
    entities: Optional[List[str]] = None
    llm_embedding: Optional[List[float]] = None

    model_config = ConfigDict(from_attributes=True)

class ImageResponse(ImageBase):
    """Image response schema with overall and component similarity scores"""
    id: int
    similarity: Optional[float] = None
    semantic_similarity: Optional[float] = None
    entity_similarity: Optional[float] = None
    category_similarity: Optional[float] = None
    brightness_similarity: Optional[float] = None
    contrast_similarity: Optional[float] = None
    saturation_similarity: Optional[float] = None
    edge_density_similarity: Optional[float] = None
    sharpness_similarity: Optional[float] = None
    dreamsim_similarity: Optional[float] = None

    # Meta Similarities
    meta_hist_interp_similarity: Optional[float] = None
    meta_cdf_interp_similarity: Optional[float] = None
    meta_joint_interp_similarity: Optional[float] = None
    meta_cell_similarity: Optional[float] = None
    meta_moments_mean_similarity: Optional[float] = None
    meta_moments_std_similarity: Optional[float] = None
    meta_moments_skew_similarity: Optional[float] = None


    hog_similarity: Optional[float] = None
    hu_moments_similarity: Optional[float] = None
    lbp_similarity: Optional[float] = None
    gabor_similarity: Optional[float] = None
    ccv_similarity: Optional[float] = None
    fourier_similarity: Optional[float] = None
    geo_similarity: Optional[float] = None
    tamura_similarity: Optional[float] = None
    edge_orientation_similarity: Optional[float] = None
    glcm_similarity: Optional[float] = None
    wavelet_similarity: Optional[float] = None
    correlogram_similarity: Optional[float] = None
    ehd_similarity: Optional[float] = None
    cld_similarity: Optional[float] = None
    spm_similarity: Optional[float] = None
    saliency_similarity: Optional[float] = None

    
    # UI/Visualization URLs
    previewUrl: Optional[str] = None
    histogramPreviewUrl: Optional[str] = None
    hogPreviewUrl: Optional[str] = None
    huPreviewUrl: Optional[str] = None
    cellColorPreviewUrl: Optional[str] = None
    lbpPreviewUrl: Optional[str] = None
    gaborPreviewUrl: Optional[str] = None
    ccvPreviewUrl: Optional[str] = None
    
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class SearchResponse(BaseModel):
    """Search response schema including query image features and optional comparison results"""
    query_image: ImageBase
    results: List[ImageResponse]

class PaginatedImageResponse(BaseModel):
    total: int
    items: List[ImageResponse]
    page: int
    size: int
    pages: int
