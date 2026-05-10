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

# Color Space Features (Histograms, CDFs, Joints, Cell Vectors)
COLOR_SPACES = ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"]
COLOR_METHODS = ["std", "interp", "gauss"]

def get_color_feature_specs() -> Dict[str, str]:
    specs = {}
    for space in COLOR_SPACES:
        # Hist & CDF
        for method in COLOR_METHODS:
            specs[f"{space}_hist_{method}"] = "cosine"
            specs[f"{space}_cdf_{method}"] = "cosine"
            if space != "gray":
                specs[f"joint_{space}_{method}"] = "cosine"
        
        # Cell Vector
        specs[f"cell_{space}"] = "l2_cell"
        
        # Color Moments
        specs[f"{space}_mean"] = "cosine"
        specs[f"{space}_std"] = "cosine"
        specs[f"{space}_skew"] = "cosine"
    return specs

# Traditional & Deep Learning Vectors
VECTOR_FEATURES = {
    "hog": "cosine",
    "hu_moments": "cosine",
    "lbp": "cosine",
    "gabor": "cosine",
    "ccv": "cosine",
    "zernike": "cosine",
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
    "dominant_color": "l2_color",
    "category": "category",
    "entity": "entity"
}

def get_all_feature_specs() -> Dict[str, str]:
    """Combines all feature definitions into one master dictionary"""
    all_specs = {}
    all_specs.update(SCALAR_FEATURES)
    all_specs.update(get_color_feature_specs())
    all_specs.update(VECTOR_FEATURES)
    all_specs.update(SPECIAL_FEATURES)
    return all_specs
