from typing import List, Dict, Any
from ...models.image import ImageMetadata

def assemble_metadatas(n, filenames, dims, l1, l2, l3, l4):
    """Combines all lane results into ImageMetadata objects"""
    metadatas = []
    for i in range(n):
        c, v = l1[i], l2[i]
        col = c[4]
        meta = ImageMetadata(
            file_name=filenames[i], width=dims[i][0], height=dims[i][1],
            brightness=c[0], contrast=c[1], saturation=c[2], edge_density=c[3],
            histogram_vis_path=col["vis_path"], cell_color_vis_path=col["cell_color_vis_path"],
            hog_vector=c[5][0], hog_vis_path=c[5][1],
            hu_moments_vector=c[6][0], hu_vis_path=c[6][1],
            dominant_color_vector=c[7], lbp_vector=c[8][0], lbp_vis_path=c[8][1],
            sharpness=c[9], gabor_vector=c[10][0], gabor_vis_path=c[10][1],
            ccv_vector=c[11][0], ccv_vis_path=c[11][1],
            zernike_vector=c[12], geo_vector=c[13], tamura_vector=c[14],
            edge_orientation_vector=c[15], glcm_vector=c[16],
            wavelet_vector=c[17], correlogram_vector=c[18],
            ehd_vector=c[19], cld_vector=c[20], spm_vector=c[21], saliency_vector=c[22],
            category=v.get("category"), description=v.get("description"), entities=v.get("entities"),
            llm_embedding=l4[i], dreamsim_vector=l3[i]
        )
        fill_color_spaces(meta, col)
        metadatas.append(meta)
    return metadatas

def fill_color_spaces(meta, col):
    """Dynamically fills color space features into metadata object"""
    spaces = ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"]
    feats = ["hist_std", "hist_interp", "hist_gauss", "cdf_std", "cdf_interp", "cdf_gauss"]
    for s in spaces:
        for attr in ["mean_vector", "std_vector", "skew_vector"]:
            val = col.get(f"{s}_{attr.split('_')[0]}")
            if val is not None:
                setattr(meta, f"{s}_{attr}", val)
        
        for f in feats:
            field = f"{s}_{f}"
            if hasattr(meta, field):
                val = col.get(field)
                if val is not None:
                    setattr(meta, field, val)
        
        if s != "gray":
            for m in ["std", "interp", "gauss"]:
                field = f"joint_{s}_{m}"
                if hasattr(meta, field):
                    val = col.get(field)
                    if val is not None:
                        setattr(meta, field, val)
        
        setattr(meta, f"cell_{s}_vector", col.get(f"cell_{s}_vector"))
