from typing import Dict

# Metric Types:
# - 'scalar': 1.0 - abs(a - b)
# - 'sharpness': 1.0 - abs(a - b) / (abs(a + b) + 1e-7)
# - 'cosine': 1.0 - cosine_distance(a, b)
# - 'l2_color': 1.0 - l2_dist / max_l2_rgb
# - 'l2_cell': 1.0 - l2_dist / max_l2_cell
# - 'category': discrete match
# - 'entity': jaccard intersection

# Base Scalars
SCALAR_FEATURES = {
    "brightness": "scalar",
    "contrast": "scalar",
    "saturation": "scalar",
    "edge_density": "scalar",
    "sharpness": "sharpness"
}

# Consolidated Color Meta-Features
META_COLOR_FEATURES = {
    "meta_hist_interp": "cosine",
    "meta_cdf_interp": "cosine",
    "meta_joint_interp": "cosine",
    "meta_cell": "cosine",
    "meta_moments_mean": "l2_cell",
    "meta_moments_std": "l2_cell",
    "meta_moments_skew": "l2_cell"
}

# Traditional & Deep Learning Vectors
VECTOR_FEATURES = {
    "hog": "cosine",
    "hu_moments": "cosine",
    "lbp": "cosine",
    "gabor": "cosine",
    "ccv": "cosine",
    "fourier": "cosine",
    "geo": "cosine",
    "tamura": "cosine",
    "edge_orientation": "cosine",
    "glcm": "cosine",
    "wavelet": "cosine",
    "correlogram": "cosine",
    "ehd": "cosine",
    "cld": "cosine",
    "spm": "cosine",
    "saliency": "cosine",
    "semantic": "cosine"
}

# Special Discrete/Custom Features
SPECIAL_FEATURES = {
    "category": "category",
    "entity": "entity"
}

def get_all_feature_specs() -> Dict[str, str]:
    """Combines all feature definitions into one master dictionary"""
    all_specs = {}
    all_specs.update(SCALAR_FEATURES)
    all_specs.update(META_COLOR_FEATURES)
    all_specs.update(VECTOR_FEATURES)
    all_specs.update(SPECIAL_FEATURES)
    return all_specs
