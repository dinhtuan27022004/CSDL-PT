# Mapping from spec metrics to internal numpy metrics
METRIC_MAP = {
    "scalar": "scalar",
    "sharpness": "sharpness",
    "cosine": "cosine",
    "l2_color": "l2_color",
    "l2_cell": "l2_cell_color"
}

# Mapping from feature names to actual ImageMetadata column names
COLUMN_MAP = {
    "hog": "hog_vector", "hu_moments": "hu_moments_vector", "lbp": "lbp_vector",
    "gabor": "gabor_vector", "ccv": "ccv_vector",
    "zernike": "zernike_vector", "geo": "geo_vector", "tamura": "tamura_vector",
    "edge_orientation": "edge_orientation_vector", "glcm": "glcm_vector", "wavelet": "wavelet_vector",
    "correlogram": "correlogram_vector", "ehd": "ehd_vector", "cld": "cld_vector",
    "spm": "spm_vector", "saliency": "saliency_vector", "semantic": "llm_embedding",
    "dreamsim": "dreamsim_vector", "dominant_color": "dominant_color_vector",
    "category": "category", "entity": "entities"
}

# Dynamic mapping for cell vectors and color moments
for space in ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"]:
    COLUMN_MAP[f"cell_{space}"] = f"cell_{space}_vector"
    for m_type in ["mean", "std", "skew"]:
        COLUMN_MAP[f"{space}_{m_type}"] = f"{space}_{m_type}_vector"
