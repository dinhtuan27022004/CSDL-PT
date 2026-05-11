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
    "gabor": "gabor_vector", "ccv": "ccv_vector", "fourier": "fourier_vector",
    "geo": "geo_vector", "tamura": "tamura_vector",
    "edge_orientation": "edge_orientation_vector", "glcm": "glcm_vector", "wavelet": "wavelet_vector",
    "correlogram": "correlogram_vector", "ehd": "ehd_vector", "cld": "cld_vector",
    "spm": "spm_vector", "saliency": "saliency_vector", "semantic": "llm_embedding",
    "category": "category", "entity": "entities",
    
    # --- Consolidated Meta Features ---
    "meta_hist_interp": "meta_hist_interp",
    "meta_cdf_interp": "meta_cdf_interp",
    "meta_joint_interp": "meta_joint_interp",
    "meta_cell": "meta_cell_vector",
    "meta_moments_mean": "meta_moments_mean",
    "meta_moments_std": "meta_moments_std",
    "meta_moments_skew": "meta_moments_skew"
}
