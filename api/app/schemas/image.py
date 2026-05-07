from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime

class SearchSettings(BaseModel):
    """Configuration for weighted similarity search"""
    mode: str = "optimized"  # "optimized" | "manual" | "equal"
    weights: Optional[Dict[str, float]] = None
    limit: int = 20
    
    # Boolean flags for toggling similarity components
    # --- Color Space Toggles ---
    # RGB
    use_rgb_hist_std: bool = True
    use_rgb_hist_interp: bool = True
    use_rgb_hist_gauss: bool = True
    use_rgb_cdf_std: bool = True
    use_rgb_cdf_interp: bool = True
    use_rgb_cdf_gauss: bool = True
    use_joint_rgb_std: bool = True
    use_joint_rgb_interp: bool = True
    use_joint_rgb_gauss: bool = True
    use_cell_rgb: bool = True

    # HSV
    use_hsv_hist_std: bool = True
    use_hsv_hist_interp: bool = True
    use_hsv_hist_gauss: bool = True
    use_hsv_cdf_std: bool = True
    use_hsv_cdf_interp: bool = True
    use_hsv_cdf_gauss: bool = True
    use_joint_hsv_std: bool = True
    use_joint_hsv_interp: bool = True
    use_joint_hsv_gauss: bool = True
    use_cell_hsv: bool = True

    # Lab
    use_lab_hist_std: bool = True
    use_lab_hist_interp: bool = True
    use_lab_hist_gauss: bool = True
    use_lab_cdf_std: bool = True
    use_lab_cdf_interp: bool = True
    use_lab_cdf_gauss: bool = True
    use_joint_lab_std: bool = True
    use_joint_lab_interp: bool = True
    use_joint_lab_gauss: bool = True
    use_cell_lab: bool = True

    # YCrCb
    use_ycrcb_hist_std: bool = True
    use_ycrcb_hist_interp: bool = True
    use_ycrcb_hist_gauss: bool = True
    use_ycrcb_cdf_std: bool = True
    use_ycrcb_cdf_interp: bool = True
    use_ycrcb_cdf_gauss: bool = True
    use_joint_ycrcb_std: bool = True
    use_joint_ycrcb_interp: bool = True
    use_joint_ycrcb_gauss: bool = True
    use_cell_ycrcb: bool = True

    # HLS
    use_hls_hist_std: bool = True
    use_hls_hist_interp: bool = True
    use_hls_hist_gauss: bool = True
    use_hls_cdf_std: bool = True
    use_hls_cdf_interp: bool = True
    use_hls_cdf_gauss: bool = True
    use_joint_hls_std: bool = True
    use_joint_hls_interp: bool = True
    use_joint_hls_gauss: bool = True
    use_cell_hls: bool = True

    # XYZ
    use_xyz_hist_std: bool = True
    use_xyz_hist_interp: bool = True
    use_xyz_hist_gauss: bool = True
    use_xyz_cdf_std: bool = True
    use_xyz_cdf_interp: bool = True
    use_xyz_cdf_gauss: bool = True
    use_joint_xyz_std: bool = True
    use_joint_xyz_interp: bool = True
    use_joint_xyz_gauss: bool = True
    use_cell_xyz: bool = True

    # Gray
    use_gray_hist_std: bool = True
    use_gray_hist_interp: bool = True
    use_gray_hist_gauss: bool = True
    use_gray_cdf_std: bool = True
    use_gray_cdf_interp: bool = True
    use_gray_cdf_gauss: bool = True
    use_cell_gray: bool = True

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

    # RGB
    rgb_hist_std: Optional[List[float]] = None
    rgb_hist_interp: Optional[List[float]] = None
    rgb_hist_gauss: Optional[List[float]] = None
    rgb_cdf_std: Optional[List[float]] = None
    rgb_cdf_interp: Optional[List[float]] = None
    rgb_cdf_gauss: Optional[List[float]] = None
    joint_rgb_std: Optional[List[float]] = None
    joint_rgb_interp: Optional[List[float]] = None
    joint_rgb_gauss: Optional[List[float]] = None
    cell_rgb_vector: Optional[List[float]] = None

    # HSV
    hsv_hist_std: Optional[List[float]] = None
    hsv_hist_interp: Optional[List[float]] = None
    hsv_hist_gauss: Optional[List[float]] = None
    hsv_cdf_std: Optional[List[float]] = None
    hsv_cdf_interp: Optional[List[float]] = None
    hsv_cdf_gauss: Optional[List[float]] = None
    joint_hsv_std: Optional[List[float]] = None
    joint_hsv_interp: Optional[List[float]] = None
    joint_hsv_gauss: Optional[List[float]] = None
    cell_hsv_vector: Optional[List[float]] = None

    # Lab
    lab_hist_std: Optional[List[float]] = None
    lab_hist_interp: Optional[List[float]] = None
    lab_hist_gauss: Optional[List[float]] = None
    lab_cdf_std: Optional[List[float]] = None
    lab_cdf_interp: Optional[List[float]] = None
    lab_cdf_gauss: Optional[List[float]] = None
    joint_lab_std: Optional[List[float]] = None
    joint_lab_interp: Optional[List[float]] = None
    joint_lab_gauss: Optional[List[float]] = None
    cell_lab_vector: Optional[List[float]] = None

    # YCrCb
    ycrcb_hist_std: Optional[List[float]] = None
    ycrcb_hist_interp: Optional[List[float]] = None
    ycrcb_hist_gauss: Optional[List[float]] = None
    ycrcb_cdf_std: Optional[List[float]] = None
    ycrcb_cdf_interp: Optional[List[float]] = None
    ycrcb_cdf_gauss: Optional[List[float]] = None
    joint_ycrcb_std: Optional[List[float]] = None
    joint_ycrcb_interp: Optional[List[float]] = None
    joint_ycrcb_gauss: Optional[List[float]] = None
    cell_ycrcb_vector: Optional[List[float]] = None

    # HLS
    hls_hist_std: Optional[List[float]] = None
    hls_hist_interp: Optional[List[float]] = None
    hls_hist_gauss: Optional[List[float]] = None
    hls_cdf_std: Optional[List[float]] = None
    hls_cdf_interp: Optional[List[float]] = None
    hls_cdf_gauss: Optional[List[float]] = None
    joint_hls_std: Optional[List[float]] = None
    joint_hls_interp: Optional[List[float]] = None
    joint_hls_gauss: Optional[List[float]] = None
    cell_hls_vector: Optional[List[float]] = None

    # XYZ
    xyz_hist_std: Optional[List[float]] = None
    xyz_hist_interp: Optional[List[float]] = None
    xyz_hist_gauss: Optional[List[float]] = None
    xyz_cdf_std: Optional[List[float]] = None
    xyz_cdf_interp: Optional[List[float]] = None
    xyz_cdf_gauss: Optional[List[float]] = None
    joint_xyz_std: Optional[List[float]] = None
    joint_xyz_interp: Optional[List[float]] = None
    joint_xyz_gauss: Optional[List[float]] = None
    cell_xyz_vector: Optional[List[float]] = None

    # Gray
    gray_hist_std: Optional[List[float]] = None
    gray_hist_interp: Optional[List[float]] = None
    gray_hist_gauss: Optional[List[float]] = None
    gray_cdf_std: Optional[List[float]] = None
    gray_cdf_interp: Optional[List[float]] = None
    gray_cdf_gauss: Optional[List[float]] = None
    cell_gray_vector: Optional[List[float]] = None
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

    # RGB Similarities
    rgb_hist_std_similarity: Optional[float] = None
    rgb_hist_interp_similarity: Optional[float] = None
    rgb_hist_gauss_similarity: Optional[float] = None
    rgb_cdf_std_similarity: Optional[float] = None
    rgb_cdf_interp_similarity: Optional[float] = None
    rgb_cdf_gauss_similarity: Optional[float] = None
    joint_rgb_std_similarity: Optional[float] = None
    joint_rgb_interp_similarity: Optional[float] = None
    joint_rgb_gauss_similarity: Optional[float] = None
    cell_rgb_similarity: Optional[float] = None

    # HSV Similarities
    hsv_hist_std_similarity: Optional[float] = None
    hsv_hist_interp_similarity: Optional[float] = None
    hsv_hist_gauss_similarity: Optional[float] = None
    hsv_cdf_std_similarity: Optional[float] = None
    hsv_cdf_interp_similarity: Optional[float] = None
    hsv_cdf_gauss_similarity: Optional[float] = None
    joint_hsv_std_similarity: Optional[float] = None
    joint_hsv_interp_similarity: Optional[float] = None
    joint_hsv_gauss_similarity: Optional[float] = None
    cell_hsv_similarity: Optional[float] = None

    # Lab Similarities
    lab_hist_std_similarity: Optional[float] = None
    lab_hist_interp_similarity: Optional[float] = None
    lab_hist_gauss_similarity: Optional[float] = None
    lab_cdf_std_similarity: Optional[float] = None
    lab_cdf_interp_similarity: Optional[float] = None
    lab_cdf_gauss_similarity: Optional[float] = None
    joint_lab_std_similarity: Optional[float] = None
    joint_lab_interp_similarity: Optional[float] = None
    joint_lab_gauss_similarity: Optional[float] = None
    cell_lab_similarity: Optional[float] = None

    # YCrCb Similarities
    ycrcb_hist_std_similarity: Optional[float] = None
    ycrcb_hist_interp_similarity: Optional[float] = None
    ycrcb_hist_gauss_similarity: Optional[float] = None
    ycrcb_cdf_std_similarity: Optional[float] = None
    ycrcb_cdf_interp_similarity: Optional[float] = None
    ycrcb_cdf_gauss_similarity: Optional[float] = None
    joint_ycrcb_std_similarity: Optional[float] = None
    joint_ycrcb_interp_similarity: Optional[float] = None
    joint_ycrcb_gauss_similarity: Optional[float] = None
    cell_ycrcb_similarity: Optional[float] = None

    # HLS Similarities
    hls_hist_std_similarity: Optional[float] = None
    hls_hist_interp_similarity: Optional[float] = None
    hls_hist_gauss_similarity: Optional[float] = None
    hls_cdf_std_similarity: Optional[float] = None
    hls_cdf_interp_similarity: Optional[float] = None
    hls_cdf_gauss_similarity: Optional[float] = None
    joint_hls_std_similarity: Optional[float] = None
    joint_hls_interp_similarity: Optional[float] = None
    joint_hls_gauss_similarity: Optional[float] = None
    cell_hls_similarity: Optional[float] = None

    # XYZ Similarities
    xyz_hist_std_similarity: Optional[float] = None
    xyz_hist_interp_similarity: Optional[float] = None
    xyz_hist_gauss_similarity: Optional[float] = None
    xyz_cdf_std_similarity: Optional[float] = None
    xyz_cdf_interp_similarity: Optional[float] = None
    xyz_cdf_gauss_similarity: Optional[float] = None
    joint_xyz_std_similarity: Optional[float] = None
    joint_xyz_interp_similarity: Optional[float] = None
    joint_xyz_gauss_similarity: Optional[float] = None
    cell_xyz_similarity: Optional[float] = None

    # Gray Similarities
    gray_hist_std_similarity: Optional[float] = None
    gray_hist_interp_similarity: Optional[float] = None
    gray_hist_gauss_similarity: Optional[float] = None
    gray_cdf_std_similarity: Optional[float] = None
    gray_cdf_interp_similarity: Optional[float] = None
    gray_cdf_gauss_similarity: Optional[float] = None
    cell_gray_similarity: Optional[float] = None

    hog_similarity: Optional[float] = None
    hu_moments_similarity: Optional[float] = None
    dominant_color_similarity: Optional[float] = None
    lbp_similarity: Optional[float] = None
    color_moments_similarity: Optional[float] = None
    gabor_similarity: Optional[float] = None
    ccv_similarity: Optional[float] = None
    zernike_similarity: Optional[float] = None
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
