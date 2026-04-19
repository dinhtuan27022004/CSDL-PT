from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

class ImageBase(BaseModel):
    file_name: str
    
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    edge_density: Optional[float] = None
    hsv_histogram_vector: Optional[List[float]] = None
    rgb_histogram_vector: Optional[List[float]] = None
    hsv_cdf_vector: Optional[List[float]] = None
    rgb_cdf_vector: Optional[List[float]] = None
    hog_vector: Optional[List[float]] = None
    hu_moments_vector: Optional[List[float]] = None
    dominant_color_vector: Optional[List[float]] = None
    lbp_vector: Optional[List[float]] = None
    color_moments_vector: Optional[List[float]] = None
    sharpness: Optional[float] = None
    
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
    hsv_histogram_similarity: Optional[float] = None
    rgb_histogram_similarity: Optional[float] = None
    hsv_cdf_similarity: Optional[float] = None
    rgb_cdf_similarity: Optional[float] = None
    hog_similarity: Optional[float] = None
    hu_moments_similarity: Optional[float] = None
    dominant_color_similarity: Optional[float] = None
    cell_color_similarity: Optional[float] = None
    lbp_similarity: Optional[float] = None
    color_moments_similarity: Optional[float] = None
    sharpness_similarity: Optional[float] = None
    gabor_similarity: Optional[float] = None
    ccv_similarity: Optional[float] = None
    zernike_similarity: Optional[float] = None
    geometric_similarity: Optional[float] = None
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)
class SearchResponse(BaseModel):
    """Search response schema including query image features"""
    query_image: ImageBase
    results: List[ImageResponse]
