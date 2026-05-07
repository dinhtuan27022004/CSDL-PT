import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..repositories.image_repository import ImageRepository
from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.similarity_specs import get_all_feature_specs
from ..core.exceptions import ImageProcessingError
from collections import Counter

logger = get_logger(__name__)

# Global progress tracking
_progress = {"status": "idle", "current": 0, "total": 100, "message": ""}

class DataService:
    """Service for dataset management, ground truth generation, and analytics"""
    
    def __init__(self, repository: ImageRepository):
        self.repository = repository
        self.settings = get_settings()

    def get_progress(self):
        return _progress

    def _clear_folder_contents(self, folder_path: Path):
        """Delete all contents inside a folder without deleting the folder itself"""
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            return
            
        import shutil
        for item in folder_path.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.error(f"Failed to delete {item}: {e}")

    def get_stats_for_file(self, db: Session, filename: str, force_recompute: bool = False):
        return self._get_stats_internal(db, filename, force_recompute)

    def _get_stats_internal(self, db: Session, filename: str, force_recompute: bool = False):
        """Internal helper for stats calculation (NumPy optimized)"""
        import numpy as np
        import time
        
        cache_filename = filename.replace(".json", "_stats.json")
        cache_path = self.settings.base_dir / cache_filename
        
        if not force_recompute and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except: pass

        path = self.settings.base_dir / filename
        if not path.exists(): return None
        with open(path, "r") as f: gt_data = json.load(f)
        if not gt_data: return None

        all_db_images = self.repository.get_all(db, limit=50000)
        n_total = len(all_db_images)
        img_map = {os.path.basename(img.file_name): i for i, img in enumerate(all_db_images)}
        vectors = [img.dreamsim_vector for img in all_db_images if img.dreamsim_vector is not None]
        valid_indices = [i for i, img in enumerate(all_db_images) if img.dreamsim_vector is not None]
        
        if not vectors: return None
        X_norm = np.array(vectors, dtype=np.float32)
        X_norm /= (np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-9)
        orig_to_norm = {orig_idx: norm_idx for norm_idx, orig_idx in enumerate(valid_indices)}

        sim_distribution = [0] * 10
        avg_sims = []
        all_appearances = []
        
        for query_fname, matches in gt_data.items():
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
                    sim = float(np.dot(q_vec, X_norm[orig_to_norm[m_idx]]))
                else: sim = 0.5
                cluster_sims.append(sim)
            
            avg_cluster_sim = sum(cluster_sims) / len(cluster_sims)
            avg_sims.append(avg_cluster_sim)
            sim_distribution[min(int(avg_cluster_sim * 10), 9)] += 1

        counts = Counter(all_appearances)
        hub_images = [{"name": name, "count": count} for name, count in counts.most_common(10)]
        unique_appearing = len(counts)
        overlap_stats = [
            {"name": "Unique", "value": sum(1 for c in counts.values() if c == 1)},
            {"name": "Shared (2-5)", "value": sum(1 for c in counts.values() if 2 <= c <= 5)},
            {"name": "Highly Shared (6+)", "value": sum(1 for c in counts.values() if c > 5)},
        ]

        result = {
            "source": filename, "timestamp": time.time(), "count": len(gt_data),
            "stats": [{"range": f"{j/10}-{(j+1)/10}", "count": sim_distribution[j]} for j in range(10)],
            "hub_images": hub_images,
            "coverage": {"unique_images": unique_appearing, "total_images": n_total, 
                         "percentage": (unique_appearing/n_total)*100 if n_total > 0 else 0, "overlap": overlap_stats},
            "avg_overall_sim": sum(avg_sims) / len(avg_sims) if avg_sims else 0
        }
        with open(cache_path, "w") as f: json.dump(result, f, indent=2)
        return result

    def generate_ground_truth(self, db: Session):
        """Generate full ground truth and export 3236 folders to purify_data"""
        import numpy as np
        import shutil
        global _progress
        
        images = self.repository.get_all(db, limit=50000)
        if not images: return {"message": "No images"}
        
        _progress.update({"status": "processing", "current": 0, "total": 100, "message": "Analyzing visual patterns (NumPy)..."})
        
        vectors = [img.dreamsim_vector for img in images if img.dreamsim_vector is not None]
        filenames = [os.path.basename(img.file_name) for img in images if img.dreamsim_vector is not None]
        
        X_norm = np.array(vectors, dtype=np.float32)
        X_norm /= (np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-9)
        sim_matrix = np.dot(X_norm, X_norm.T)
        
        gt_dict = {}
        for i in range(len(filenames)):
            top_indices = np.argsort(sim_matrix[i])[::-1][:10]
            gt_dict[str(i + 1)] = [filenames[idx] for idx in top_indices]

        # Save JSON
        with open(self.settings.base_dir / "ground_truth.json", "w") as f:
            json.dump(gt_dict, f, indent=2)
            
        # Physical Export to purify_data (3236 folders)
        purify_dir = Path("C:/PTIT/2026/CSDL-PT/purify_data")
        self._clear_folder_contents(purify_dir)
        
        total_clusters = len(gt_dict)
        _progress.update({"status": "exporting", "current": 0, "total": total_clusters, "message": "Purifying data: Exporting 3236 clusters..."})
        
        for i, (cluster_id, cluster_files) in enumerate(gt_dict.items()):
            c_path = purify_dir / f"cluster_{int(cluster_id):04d}"
            c_path.mkdir(exist_ok=True, parents=True)
            for f in cluster_files:
                src = self.settings.uploads_dir / f
                if src.exists():
                    shutil.copy2(src, c_path / f)
            
            if (i+1) % 50 == 0 or (i+1) == total_clusters:
                _progress["current"] = i + 1
                
        _progress.update({"status": "idle", "current": 100, "total": 100, "message": "Dataset Purified Successfully"})
        return self._get_stats_internal(db, "ground_truth.json", force_recompute=True)

    def select_diverse_ground_truth(self, db: Session):
        """Extract 50 diverse clusters and export to purify_data_2"""
        import numpy as np
        import shutil
        global _progress
        
        gt_path = self.settings.base_dir / "ground_truth.json"
        if not gt_path.exists(): return {"message": "Run full generation first"}
        
        with open(gt_path, "r") as f: full_gt = json.load(f)
        
        _progress.update({"status": "processing", "current": 0, "total": 50, "message": "Identifying 50 diverse archetypes..."})
        
        selected_keys = []
        used_images = set()
        candidates = list(full_gt.keys())
        total_candidates = len(candidates)
        perfectly_unique = 0
        
        for i in range(50):
            best_key, max_new = None, -1
            for key in candidates:
                new_count = len(set(full_gt[key]) - used_images)
                if new_count > max_new:
                    max_new, best_key = new_count, key
                if new_count == 10: break
            
            if best_key:
                selected_keys.append(best_key)
                used_images.update(full_gt[best_key])
                candidates.remove(best_key)
                if max_new == 10: perfectly_unique += 1
                _progress["current"] = i + 1
            else: break
                
        # Physical Export to purify_data_2 (50 folders)
        purify_dir_2 = Path("C:/PTIT/2026/CSDL-PT/purify_data_2")
        self._clear_folder_contents(purify_dir_2)
        
        _progress.update({"status": "exporting", "current": 0, "total": 50, "message": "Purifying diverse set: Exporting 50 clusters..."})
        
        for i, key in enumerate(selected_keys):
            c_path = purify_dir_2 / f"cluster_{i+1:02d}"
            c_path.mkdir(exist_ok=True, parents=True)
            for f in full_gt[key]:
                src = self.settings.uploads_dir / f
                if src.exists():
                    shutil.copy2(src, c_path / f)
            _progress["current"] = i + 1

        # Save JSON
        test_path = self.settings.base_dir / "ground_truth_2.json"
        test_gt = {str(i+1): full_gt[key] for i, key in enumerate(selected_keys)}
        with open(test_path, "w") as f: json.dump(test_gt, f, indent=2)
        
        _progress.update({"status": "idle", "current": 100, "total": 100, "message": "Diverse Set Purified Successfully"})
        stats = self._get_stats_internal(db, "ground_truth_2.json", force_recompute=True)
        return {**stats, "analytics": {"total_candidates": total_candidates, "perfectly_unique": perfectly_unique}}
