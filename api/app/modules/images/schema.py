from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime

class ImageBase(BaseModel):
    file_name: str
    file_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    edge_density: Optional[float] = None
    histogram_json: Optional[str] = None
    hog_vis_path: Optional[str] = None
    hu_vis_path: Optional[str] = None

class ImageResponse(ImageBase):
    """Image response schema with overall and component similarity scores"""
    id: int
    similarity: Optional[float] = None
    # Component similarities (0-100%)
    dino_similarity: Optional[float] = None
    clip_similarity: Optional[float] = None
    dominant_color_similarity: Optional[float] = None
    brightness_similarity: Optional[float] = None
    contrast_similarity: Optional[float] = None
    saturation_similarity: Optional[float] = None
    edge_density_similarity: Optional[float] = None
    histogram_similarity: Optional[float] = None
    hog_similarity: Optional[float] = None
    hu_moments_similarity: Optional[float] = None
    
    created_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)

class SearchResponse(BaseModel):
    """Search response schema including query image features"""
    query_image: dict
    results: List[ImageResponse]
