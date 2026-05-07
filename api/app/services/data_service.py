import os
import json
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..repositories.image_repository import ImageRepository
from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.similarity_specs import get_all_feature_specs
from ..core.exceptions import ImageProcessingError
from collections import Counter

logger = get_logger(__name__)

class DataService:
    """Service for dataset management, ground truth generation, and analytics"""
    
    def __init__(self, repository: ImageRepository):
        self.repository = repository
        self.settings = get_settings()

    def get_stats_for_file(self, db: Session, filename: str, force_recompute: bool = False):
        """Get stats from cache or recompute if forced/missing"""
        import numpy as np
        import time
        
        # 1. Check cache first
        cache_filename = filename.replace(".json", "_stats.json")
        cache_path = self.settings.base_dir / cache_filename
        
        if not force_recompute and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    logger.info(f"Loading stats from cache: {cache_filename}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache {cache_filename}: {e}")

        # 2. Recompute
        path = self.settings.base_dir / filename
        if not path.exists():
            return None
            
        with open(path, "r") as f:
            gt_data = json.load(f)
            
        if not gt_data:
            return None

        # Fetch all images once
        all_db_images = self.repository.get_all(db, limit=50000)
        n_total = len(all_db_images)
        
        img_map = {os.path.basename(img.file_name): i for i, img in enumerate(all_db_images)}
        vectors = []
        valid_indices = []
        for i, img in enumerate(all_db_images):
            if img.dreamsim_vector is not None:
                vectors.append(img.dreamsim_vector)
                valid_indices.append(i)
        
        if not vectors: return None
        
        X = np.array(vectors, dtype=np.float32)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        orig_to_norm = {orig_idx: norm_idx for norm_idx, orig_idx in enumerate(valid_indices)}

        sim_distribution = [0] * 10
        avg_sims = []
        all_appearances = []
        
        logger.info(f"Recomputing stats for {filename}...")
        for i, (query_fname, matches) in enumerate(gt_data.items()):
            if not matches: continue
            q_name = matches[0]
            q_idx = img_map.get(q_name)
            if q_idx is None or q_idx not in orig_to_norm: continue
            
            q_vec = X_norm[orig_to_norm[q_idx]]
            cluster_sims = []
            for m in matches:
                all_appearances.append(m)
                m_idx = img_map.get(m)
                if m_idx is not None and m_idx in orig_to_norm:
                    m_vec = X_norm[orig_to_norm[m_idx]]
                    sim = float(np.dot(q_vec, m_vec))
                else:
                    sim = 0.5
                cluster_sims.append(sim)
                
            avg_cluster_sim = sum(cluster_sims) / len(cluster_sims)
            avg_sims.append(avg_cluster_sim)
            bin_idx = min(int(avg_cluster_sim * 10), 9)
            sim_distribution[bin_idx] += 1

        counts = Counter(all_appearances)
        hub_images = [{"name": name, "count": count} for name, count in counts.most_common(10)]
        unique_appearing = len(counts)
        coverage_pct = (unique_appearing / n_total) * 100 if n_total > 0 else 0
        
        overlap_stats = [
            {"name": "Unique", "value": sum(1 for c in counts.values() if c == 1)},
            {"name": "Shared (2-5)", "value": sum(1 for c in counts.values() if 2 <= c <= 5)},
            {"name": "Highly Shared (6+)", "value": sum(1 for c in counts.values() if c > 5)},
        ]

        chart_data = [{"range": f"{j/10}-{(j+1)/10}", "count": sim_distribution[j]} for j in range(10)]

        result = {
            "source": filename,
            "timestamp": time.time(),
            "count": len(gt_data),
            "stats": chart_data,
            "hub_images": hub_images,
            "coverage": {
                "unique_images": unique_appearing,
                "total_images": n_total,
                "percentage": coverage_pct,
                "overlap": overlap_stats
            },
            "avg_overall_sim": sum(avg_sims) / len(avg_sims) if avg_sims else 0
        }

        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)
                logger.info(f"Saved stats cache: {cache_filename}")
        except Exception as e:
            logger.error(f"Failed to save cache {cache_filename}: {e}")

        return result

    def generate_ground_truth(self, db: Session):
        """Generate ground_truth.json and force recompute stats cache"""
        import numpy as np
        
        images = self.repository.get_all(db, limit=50000)
        if not images:
            return {"message": "No images in database", "count": 0}
            
        n = len(images)
        logger.info(f"Generating ground truth for {n} images...")
        
        vectors = []
        filenames = []
        for img in images:
            if img.dreamsim_vector is not None:
                vectors.append(img.dreamsim_vector)
                filenames.append(os.path.basename(img.file_name))
        
        if not vectors:
            return {"message": "No DreamSim vectors found", "count": 0}
            
        X = np.array(vectors, dtype=np.float32)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sim_matrix = np.dot(X_norm, X_norm.T)
        
        gt_dict = {}
        for i in range(len(filenames)):
            top_indices = np.argsort(sim_matrix[i])[::-1][:10]
            gt_dict[str(i + 1)] = [filenames[idx] for idx in top_indices]

        gt_path = self.settings.base_dir / "ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump(gt_dict, f, indent=2)
            
        # FORCE recompute stats cache
        stats = self.get_stats_for_file(db, "ground_truth.json", force_recompute=True)
        return {
            "message": f"Successfully generated ground truth and updated cache",
            **stats
        }

    def select_diverse_ground_truth(self, db: Session):
        """Select 50 clusters from ground_truth.json with minimum overlap"""
        gt_path = self.settings.base_dir / "ground_truth.json"
        if not gt_path.exists():
            return {"error": "Ground truth file not found."}
            
        with open(gt_path, "r") as f:
            full_gt = json.load(f)
            
        if len(full_gt) <= 50:
            return {"message": "Dataset too small", "count": len(full_gt)}
            
        selected_keys = []
        used_images = set()
        candidates = list(full_gt.keys())
        
        # Greedy
        first_key = candidates[0]
        selected_keys.append(first_key)
        used_images.update(full_gt[first_key])
        candidates.remove(first_key)
        
        for _ in range(49):
            best_key = None
            max_new = -1
            for key in candidates:
                new_count = sum(1 for img in full_gt[key] if img not in used_images)
                if new_count > max_new:
                    max_new = new_count
                    best_key = key
            if best_key:
                selected_keys.append(best_key)
                used_images.update(full_gt[best_key])
                candidates.remove(best_key)
            else: break
                
        test_gt = {str(i+1): full_gt[key] for i, key in enumerate(selected_keys)}
        test_path = self.settings.base_dir / "ground_truth_2.json"
        with open(test_path, "w") as f:
            json.dump(test_gt, f, indent=2)
            
        # FORCE recompute stats cache for ground_truth_2.json
        stats = self.get_stats_for_file(db, "ground_truth_2.json", force_recompute=True)
        return {
            "message": "Successfully extracted 50 diverse clusters and updated cache",
            "path": str(test_path),
            **stats
        }
