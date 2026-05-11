from typing import List, Dict, Any
from ...models.image import ImageMetadata

def assemble_metadatas(n, filenames, dims, l1, l2, l3, l4):
    """Combines all lane results into ImageMetadata objects"""
    metadatas = []
    for i in range(n):
        # Safety checks for lane results
        c = l1[i] if (l1 and len(l1) > i) else [None] * 25
        v = l2[i] if (l2 and len(l2) > i) else {}
        
        # Lane 1 - Traditional Features
        col = c[4] if (len(c) > 4 and c[4] is not None) else {}
        
        meta = ImageMetadata(
            file_name=filenames[i], width=dims[i][0], height=dims[i][1],
            brightness=c[0], contrast=c[1], saturation=c[2], edge_density=c[3],
            histogram_vis_path=col.get("vis_path"), 
            cell_color_vis_path=col.get("cell_color_vis_path"),
            hog_vector=c[5][0] if (len(c) > 5 and c[5]) else None, 
            hog_vis_path=c[5][1] if (len(c) > 5 and c[5]) else None,
            hu_moments_vector=c[6][0] if (len(c) > 6 and c[6]) else None, 
            hu_vis_path=c[6][1] if (len(c) > 6 and c[6]) else None,
            lbp_vector=c[7][0] if (len(c) > 7 and c[7]) else None, 
            lbp_vis_path=c[7][1] if (len(c) > 7 and c[7]) else None,
            sharpness=c[8] if len(c) > 8 else None, 
            gabor_vector=c[9][0] if (len(c) > 9 and c[9]) else None, 
            gabor_vis_path=c[9][1] if (len(c) > 9 and c[9]) else None,
            ccv_vector=c[10][0] if (len(c) > 10 and c[10]) else None, 
            ccv_vis_path=c[10][1] if (len(c) > 10 and c[10]) else None,
            fourier_vector=c[11] if len(c) > 11 else None, 
            geo_vector=c[12] if len(c) > 12 else None, 
            tamura_vector=c[13] if len(c) > 13 else None,
            edge_orientation_vector=c[14] if len(c) > 14 else None, 
            glcm_vector=c[15] if len(c) > 15 else None,
            wavelet_vector=c[16] if len(c) > 16 else None, 
            correlogram_vector=c[17] if len(c) > 17 else None,
            ehd_vector=c[18] if len(c) > 18 else None, 
            cld_vector=c[19] if len(c) > 19 else None, 
            spm_vector=c[20] if len(c) > 20 else None, 
            saliency_vector=c[21] if len(c) > 21 else None,
            category=v.get("category"), description=v.get("description"), entities=v.get("entities"),
            llm_embedding=l4[i] if (l4 and len(l4) > i) else None, 
            dreamsim_vector=l3[i] if (l3 and len(l3) > i) else None
        )
        fill_color_spaces(meta, col)
        metadatas.append(meta)
    return metadatas

def fill_color_spaces(meta, col):
    """Fills consolidated meta-features into metadata object"""
    meta_fields = [
        "meta_hist_interp", "meta_cdf_interp", "meta_joint_interp", 
        "meta_cell_vector", "meta_moments_mean", "meta_moments_std", "meta_moments_skew"
    ]
    for field in meta_fields:
        val = col.get(field)
        if val is not None:
            setattr(meta, field, val)
