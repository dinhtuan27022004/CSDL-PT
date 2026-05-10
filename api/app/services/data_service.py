from pathlib import Path
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
        
        # Default empty result structure to avoid frontend crashes
        default_result = {
            "source": filename,
            "timestamp": time.time(),
            "count": 0,
            "stats": [{"range": f"{j/10}-{(j+1)/10}", "count": 0} for j in range(10)],
            "hub_images": [],
            "coverage": {
                "unique_images": 0,
                "total_images": 0,
                "percentage": 0,
                "overlap": [
                    {"name": "Unique", "value": 0},
                    {"name": "Shared (2-5)", "value": 0},
                    {"name": "Highly Shared (6+)", "value": 0},
                ]
            },
            "avg_overall_sim": 0
        }

        if not path.exists():
            return default_result
            
        try:
            with open(path, "r") as f:
                gt_data = json.load(f)
        except:
            return default_result
            
        if not gt_data:
            return default_result

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
        
        if not vectors: return default_result
        
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
                filenames.append(img.file_name)
        
        if not vectors:
            return {"message": "No DreamSim vectors found", "count": 0}
            
        X = np.array(vectors, dtype=np.float32)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sim_matrix = np.dot(X_norm, X_norm.T)
        
        gt_dict = {}
        for i in range(len(filenames)):
            # Ensure we take at least 20 matches (including the query itself)
            top_indices = np.argsort(sim_matrix[i])[::-1][:20]
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
        """Select 50 clusters from ground_truth.json with highest intra-cluster similarity and max diversity"""
        import numpy as np
        import random
        import shutil
        
        n_to_select = 100
        images_per_cluster = 20
        
        gt_path = self.settings.base_dir / "ground_truth.json"
        if not gt_path.exists():
            return {"error": "Ground truth file not found. Run Full Generation first."}
            
        with open(gt_path, "r") as f:
            full_gt = json.load(f)
            
        if len(full_gt) < n_to_select:
            return {"error": f"Dataset too small ({len(full_gt)}). Need at least {n_to_select} clusters."}

        # Load DreamSim vectors for similarity scoring
        all_db_images = self.repository.get_all(db, limit=50000)
        img_map = {os.path.basename(img.file_name): i for i, img in enumerate(all_db_images)}
        vectors = []
        valid_indices = []
        for i, img in enumerate(all_db_images):
            if img.dreamsim_vector is not None:
                vectors.append(img.dreamsim_vector)
                valid_indices.append(i)

        if not vectors:
            return {"error": "No DreamSim vectors found in database."}

        X = np.array(vectors, dtype=np.float32)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        orig_to_norm = {orig_idx: norm_idx for norm_idx, orig_idx in enumerate(valid_indices)}

        # 1. Prepare candidates
        candidates = []
        for key, matches in full_gt.items():
            if not matches or len(matches) < images_per_cluster: continue
            q_name = matches[0]
            q_idx = img_map.get(q_name)
            
            if q_idx is not None and q_idx in orig_to_norm:
                q_vec = X_norm[orig_to_norm[q_idx]]
                # Calculate intra-cluster similarity (Quality)
                sims = []
                for m in matches[:images_per_cluster]:
                    m_idx = img_map.get(m)
                    if m_idx is not None and m_idx in orig_to_norm:
                        sims.append(float(np.dot(q_vec, X_norm[orig_to_norm[m_idx]])))
                
                avg_sim = sum(sims) / len(sims) if sims else 0.0
                candidates.append({
                    "key": key, "query_name": q_name, "score": avg_sim, 
                    "images": matches[:images_per_cluster], "q_vec": q_vec
                })
        
        if len(candidates) < n_to_select:
            return {"error": f"Only found {len(candidates)} valid clusters with {images_per_cluster} images. Need {n_to_select}."}

        # 2. Optimization Logic (Hill Climbing)
        def evaluate_set(indices_list):
            subset = [candidates[i] for i in indices_list]
            
            # a. Quality: Avg intra-similarity
            quality = sum(c["score"] for c in subset) / n_to_select
            
            # b. Diversity: Maximize distance between query vectors
            # Penalty for having queries that are too similar
            inter_sims = []
            for i in range(n_to_select):
                # Sample some pairs for speed if n is large, or do all for 50
                for j in range(i + 1, n_to_select):
                    inter_sims.append(float(np.dot(subset[i]["q_vec"], subset[j]["q_vec"])))
            
            avg_inter_sim = sum(inter_sims) / len(inter_sims) if inter_sims else 0
            max_inter_sim = max(inter_sims) if inter_sims else 0
            diversity_score = 1.0 - (avg_inter_sim * 0.7 + max_inter_sim * 0.3)
            
            # c. Uniqueness: Avoid overlapping images across clusters
            all_imgs = []
            for c in subset: all_imgs.extend(c["images"])
            unique_count = len(set(all_imgs))
            # Max possible unique is n_to_select * images_per_cluster
            overlap_ratio = (n_to_select * images_per_cluster - unique_count) / (n_to_select * images_per_cluster)
            
            # Final formula: Higher is better
            return quality * 1.5 + diversity_score * 1.0 - overlap_ratio * 5.0

        # Initial solution: Top N by quality
        candidates.sort(key=lambda x: x["score"], reverse=True)
        current_indices = list(range(n_to_select))
        current_score = evaluate_set(current_indices)
        
        logger.info(f"Optimizing 50 clusters (Initial Score: {current_score:.4f})...")
        n_iterations = 3000
        pool_indices = list(range(len(candidates)))
        
        for _ in range(n_iterations):
            idx_to_replace = random.randint(0, n_to_select - 1)
            new_candidate_idx = random.choice(pool_indices)
            if new_candidate_idx in current_indices: continue
            
            new_indices = current_indices.copy()
            new_indices[idx_to_replace] = new_candidate_idx
            new_score = evaluate_set(new_indices)
            
            if new_score > current_score:
                current_indices = new_indices
                current_score = new_score
        
        logger.info(f"Optimization finished. Final Score: {current_score:.4f}")
        
        # 3. Export and Save
        selected_clusters = [candidates[i] for i in current_indices]
        purify_dir = Path("C:/PTIT/2026/CSDL-PT/purify_data")
        
        if purify_dir.exists():
            shutil.rmtree(purify_dir)
        purify_dir.mkdir(parents=True, exist_ok=True)
        
        final_gt_2 = {}
        for i, cluster in enumerate(selected_clusters):
            c_id = f"cluster_{i+1:02d}"
            c_path = purify_dir / c_id
            c_path.mkdir(parents=True, exist_ok=True)
            
            final_gt_2[str(i+1)] = cluster["images"]
            
            for fname in cluster["images"]:
                src = self.settings.uploads_dir / fname
                dst = c_path / os.path.basename(fname)
                if src.exists():
                    try:
                        shutil.copy2(src, dst)
                    except: pass

        # Save ground_truth_2.json
        gt2_path = self.settings.base_dir / "ground_truth_2.json"
        with open(gt2_path, "w") as f:
            json.dump(final_gt_2, f, indent=2)
            
        # Recompute stats for the new diverse set
        stats = self.get_stats_for_file(db, "ground_truth_2.json", force_recompute=True)
        
        return {
            "message": f"Successfully extracted {n_to_select} diverse clusters (20 images each)",
            "purify_path": str(purify_dir),
            "avg_quality": round(sum(c["score"] for c in selected_clusters)/n_to_select, 4),
            **stats
        }

    def generate_ground_truth_3(self, db: Session, folder_path: str):
        """Generate ground_truth_3.json based on folder structure (subfolders = clusters)"""
        import os
        from pathlib import Path
        
        base_path = Path(folder_path)
        if not base_path.exists() or not base_path.is_dir():
            return {"error": f"Path {folder_path} is not a valid directory"}
            
        gt_dict = {}
        cluster_idx = 1
        
        # Walk through subdirectories
        for entry in os.scandir(base_path):
            if entry.is_dir():
                cluster_images = []
                for file in os.scandir(entry.path):
                    if file.is_file() and file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        cluster_images.append(file.name)
                
                if cluster_images:
                    gt_dict[str(cluster_idx)] = cluster_images
                    cluster_idx += 1
                    
        if not gt_dict:
            return {"error": "No clusters found in the specified directory structure"}
            
        gt_path = self.settings.base_dir / "ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump(gt_dict, f, indent=2)
            
        # Recompute stats for the updated GT
        stats = self.get_stats_for_file(db, "ground_truth.json", force_recompute=True)
        
        return {
            "message": f"Successfully updated main Ground Truth from {len(gt_dict)} folders",
            "path": str(gt_path),
            **stats
        }
