import os
import json
import numpy as np
import optuna
import argparse
from typing import List, Dict, Any, Tuple
import time
from app.db.session import SessionLocal
from app.models.image import ImageMetadata
from app.core.config import get_settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.logging import setup_logging, get_logger

settings = get_settings()
setup_logging() # Initialize handlers for this process
logger = get_logger(__name__)

def get_similarity_matrix(vectors: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Calculate N x N similarity matrix for a feature vector
    MATCHES ImageRepository.py SQL formulas exactly.
    """
    n = vectors.shape[0]
    if n == 0: return np.zeros((0, 0))

    if metric == "cosine":
        # Cosine Similarity = Dot Product of normalized vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-7
        norm_vecs = vectors / norms
        sim = np.dot(norm_vecs, norm_vecs.T)
        # MATCH REPOSITORY: 1.0 - distance
        return np.maximum(0, sim)
    
    elif metric == "l2_color":
        # MATCH REPOSITORY (LINEAR): 1.0 - l2_dist / max_dist
        max_dist = 255 * np.sqrt(3)
        diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]
        l2 = np.linalg.norm(diff, axis=2)
        return 1.0 - l2 / max_dist
    
    elif metric == "l2_cell_color":
        # MATCH REPOSITORY (LINEAR): 1.0 - l2_dist / max_dist
        dim = vectors.shape[1]
        max_dist = 255 * np.sqrt(dim)
        diff = vectors[:, np.newaxis, :] - vectors[np.newaxis, :, :]
        l2 = np.linalg.norm(diff, axis=2)
        return 1.0 - l2 / max_dist

    elif metric == "scalar":
        # MATCH REPOSITORY: 1.0 - abs(a - b) / max_diff
        diff = np.abs(vectors[:, np.newaxis] - vectors[np.newaxis, :])
        return 1.0 - diff
    
    elif metric == "sharpness":
        # MATCH REPOSITORY: 1.0 - abs(a-b) / (abs(a+b) + 1e-7)
        a = vectors[:, np.newaxis]
        b = vectors[np.newaxis, :]
        diff = np.abs(a - b)
        denom = np.abs(a + b) + 1e-7
        return 1.0 - diff / denom

    return np.zeros((n, n))

def get_discrete_similarity(values: List[Any], type: str = "category") -> np.ndarray:
    n = len(values)
    sim = np.zeros((n, n))
    if type == "category":
        for i in range(n):
            v1 = (values[i] or "General").lower()
            for j in range(n):
                v2 = (values[j] or "General").lower()
                sim[i, j] = 1.0 if v1 == v2 else 0.0
    elif type == "entities":
        # Simple Jaccard-ish similarity for lists
        for i in range(n):
            set1 = set([e.lower() for e in (values[i] or [])])
            for j in range(n):
                set2 = set([e.lower() for e in (values[j] or [])])
                if not set1 or not set2:
                    sim[i, j] = 0.0
                else:
                    intersect = len(set1.intersection(set2))
                    sim[i, j] = intersect / max(len(set1), len(set2))
    return sim

class WeightOptimizer:
    def __init__(self, ground_truth_model: str = "clip", train_size: float = 0.7):
        self.gt_model = ground_truth_model
        self.train_size = train_size
        self.db = SessionLocal()
        self.feature_matrices = {}
        self.image_ids = []
        self.y_true = None # Ground truth labels (N x N)
        self.train_idx = []
        self.test_idx = []

    def load_data(self):
        logger.info(f"[{self.gt_model.upper()}] Loading features for images from database...")
        images = self.db.query(ImageMetadata).all()
        if not images:
            logger.error(f"[{self.gt_model.upper()}] No images found in database.")
            return False
        
        self.image_ids = [img.id for img in images]
        n = len(images)

        # 1. Scalars
        scalars = {
            "brightness": np.array([img.brightness or 0.0 for img in images]),
            "contrast": np.array([img.contrast or 0.0 for img in images]),
            "saturation": np.array([img.saturation or 0.0 for img in images]),
            "edge_density": np.array([img.edge_density or 0.0 for img in images]),
            "sharpness": np.array([img.sharpness or 0.0 for img in images]),
        }
        for name, vec in scalars.items():
            self.feature_matrices[name] = get_similarity_matrix(vec, metric="sharpness" if name=="sharpness" else "scalar")

        # 2. Color Space Features (Dynamic)
        for space in ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"]:
            for method in ["std", "interp", "gauss"]:
                # Hist
                h_col = f"{space}_hist_{method}"
                h_vecs = []
                for img in images:
                    v = getattr(img, h_col)
                    h_vecs.append(np.array(v) if v is not None else np.zeros(24 if space != "gray" else 8))
                self.feature_matrices[h_col] = get_similarity_matrix(np.array(h_vecs), metric="cosine")
                
                # CDF
                c_col = f"{space}_cdf_{method}"
                c_vecs = []
                for img in images:
                    v = getattr(img, c_col)
                    c_vecs.append(np.array(v) if v is not None else np.zeros(24 if space != "gray" else 8))
                self.feature_matrices[c_col] = get_similarity_matrix(np.array(c_vecs), metric="cosine")
                
                # Joint (except gray)
                if space != "gray":
                    j_col = f"joint_{space}_{method}"
                    j_vecs = []
                    for img in images:
                        v = getattr(img, j_col)
                        j_vecs.append(np.array(v) if v is not None else np.zeros(64))
                    self.feature_matrices[j_col] = get_similarity_matrix(np.array(j_vecs), metric="cosine")
            
            # Cell Color (L2)
            cell_col = f"cell_{space}_vector"
            cell_vecs = []
            for img in images:
                v = getattr(img, cell_col)
                cell_vecs.append(np.array(v) if v is not None else np.zeros(48 if space != "gray" else 16))
            self.feature_matrices[f"cell_{space}"] = get_similarity_matrix(np.array(cell_vecs), metric="l2_cell_color")

        # 3. Traditional Vectors (Cosine)
        vector_cols = {
            "hog": "hog_vector",
            "hu_moments": "hu_moments_vector",
            "lbp": "lbp_vector",
            "color_moments": "color_moments_vector",
            "gabor": "gabor_vector",
            "ccv": "ccv_vector",
            "zernike": "zernike_vector",
            "geo": "geo_vector",
            "tamura": "tamura_vector",
            "edge_orientation": "edge_orientation_vector",
            "glcm": "glcm_vector",
            "wavelet": "wavelet_vector",
            "correlogram": "correlogram_vector",
            "ehd": "ehd_vector",
            "cld": "cld_vector",
            "spm": "spm_vector",
            "saliency": "saliency_vector",
            "bovw": "bovw_vector",
            "semantic": "llm_embedding",
            "clip": "clip_vector",
            "dinov2": "dinov2_vector",
            "siglip": "siglip_vector",
            "convnext": "convnext_vector",
            "efficientnet": "efficientnet_vector",
            "dreamsim": "dreamsim_vector",
            "sam": "sam_vector"
        }
        
        for name, col in vector_cols.items():
            vecs = []
            for img in images:
                v = getattr(img, col)
                if v is None:
                    # Match dimensions defined in models/image.py
                    dims = {
                        "glcm": 64, "wavelet": 12, "correlogram": 32, "ehd": 80, "cld": 64,
                        "spm": 160, "saliency": 32, "bovw": 512, "dinov2": 1536, "clip": 768,
                        "siglip": 768, "convnext": 1024, "efficientnet": 2560, "dreamsim": 1792,
                        "sam": 12544, "semantic": 1024
                    }
                    dim = dims.get(name, 512)
                    vecs.append(np.zeros(dim))
                else:
                    vecs.append(np.array(v))
            self.feature_matrices[name] = get_similarity_matrix(np.array(vecs), metric="cosine")

        dom_colors = np.array([(img.dominant_color_vector if img.dominant_color_vector is not None else [0,0,0]) for img in images])
        self.feature_matrices["dominant_color"] = get_similarity_matrix(dom_colors, metric="l2_color")

        # 4. Discrete
        categories = [img.category for img in images]
        self.feature_matrices["category"] = get_discrete_similarity(categories, type="category")
        
        entities = [img.entities for img in images]
        self.feature_matrices["entity"] = get_discrete_similarity(entities, type="entities")

        self.feature_names = list(self.feature_matrices.keys())
        
        # 5. Define Ground Truth
        gt_col = f"{self.gt_model}_vector"
        if not hasattr(images[0], gt_col) and self.gt_model != "semantic":
            print(f"Ground truth model {self.gt_model} not found. Falling back to clip.")
            gt_col = "clip_vector"
        elif self.gt_model == "semantic":
            gt_col = "llm_embedding"
        
        gt_vecs = []
        for img in images:
            v = getattr(img, gt_col)
            gt_vecs.append(np.array(v) if v is not None else np.zeros(768))
        
        # Ground Truth similarity is pure cosine for labeling
        norms = np.linalg.norm(np.array(gt_vecs), axis=1, keepdims=True) + 1e-7
        norm_vecs = np.array(gt_vecs) / norms
        gt_sim = np.dot(norm_vecs, norm_vecs.T)
        
        # Ground Truth: Strictly use Top 10 neighbors (+ itself)
        n = gt_sim.shape[0]
        self.y_true = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            # Take top 11 results (itself + 10 most similar)
            top_indices = np.argsort(gt_sim[i])[::-1][:11]
            self.y_true[i, top_indices] = 1
        # Mask diagonal (self-similarity)
        np.fill_diagonal(self.y_true, 0)
        
        avg_labels = np.mean(np.sum(self.y_true, axis=1))
        logger.info(f"Ground Truth Density ({self.gt_model}): Avg {avg_labels:.2f} matches per image (Top-10 Mode).")

        # 6. Train/Test Split
        indices = np.arange(n)
        self.train_idx, self.test_idx = train_test_split(indices, train_size=self.train_size, random_state=42)
        
        logger.info(f"[{self.gt_model.upper()}] Data split: {len(self.train_idx)} train, {len(self.test_idx)} test.")
        return True

    def calculate_map(self, weights: Dict[str, float], indices: np.ndarray, k: Optional[int] = None) -> float:
        n_total = len(self.image_ids)
        final_sim = np.zeros((len(indices), n_total))
        for name, weight in weights.items():
            if weight == 0: continue
            final_sim += weight * self.feature_matrices[name][indices, :]
        
        aps = []
        for i, idx in enumerate(indices):
            scores = final_sim[i]
            labels = self.y_true[idx]
            
            mask = np.ones(n_total, dtype=bool)
            mask[idx] = False
            
            s_masked = scores[mask]
            l_masked = labels[mask]
            
            total_positives = np.sum(l_masked)
            if total_positives == 0: continue
            
            # Sort by scores
            order = np.argsort(s_masked)[::-1]
            l_sorted = l_masked[order]
            
            if k is not None:
                # Precision at K calculation
                l_k = l_sorted[:k]
                hits = np.where(l_k == 1)[0]
                if len(hits) == 0:
                    aps.append(0.0)
                    continue
                
                precision_at_hits = []
                for hit_idx in hits:
                    # hit_idx is 0-indexed rank
                    p = np.sum(l_k[:hit_idx+1]) / (hit_idx + 1)
                    precision_at_hits.append(p)
                
                # AP@K normalized by the smaller of (total possible positives) and K
                ap_k = np.sum(precision_at_hits) / min(total_positives, k)
                aps.append(ap_k)
            else:
                # Standard MAP
                aps.append(average_precision_score(l_masked, s_masked))
            
        return np.mean(aps) if aps else 0.0

    def optimize(self, n_trials: int = 100, allow_negative: bool = False, exclude_embeddings: bool = True):
        logger.info(f"[PROCESS] Starting optimization for {self.gt_model.upper()} (Target: {n_trials} trials, Split: {self.train_size})")
        
        embedding_features = ["clip", "dinov2", "siglip", "convnext", "efficientnet", "dreamsim", "sam"]
        
        def objective(trial):
            weights = {}
            low = -1.0 if allow_negative else 0.0
            for name in self.feature_names:
                if name == self.gt_model or (exclude_embeddings and name in embedding_features):
                    weights[name] = 0.0
                else:
                    weights[name] = trial.suggest_float(name, low, 1.0)
            
            if not allow_negative:
                total_w = sum(weights.values())
                if total_w > 0:
                    weights = {k: v / total_w for k, v in weights.items()}
            
            # Only focus on mAP@5 for optimization loss
            return self.calculate_map(weights, self.train_idx, k=5)

        def callback(study, trial):
            if study.best_trial.number == trial.number:
                # New best found! Save a checkpoint
                checkpoint_weights = study.best_params
                if not allow_negative:
                    tw = sum(checkpoint_weights.values())
                    if tw > 0:
                        checkpoint_weights = {k: v / tw for k, v in checkpoint_weights.items()}
                
                # Save as partial weights so search can use it immediately if needed
                checkpoint_filename = f"weights_{self.gt_model}_checkpoint.json"
                try:
                    with open(checkpoint_filename, "w") as f:
                        json.dump(checkpoint_weights, f, indent=4)
                    
                    # Evaluate on test set for real-time metrics update in UI
                    m_test = self.calculate_map(checkpoint_weights, self.test_idx, k=None)
                    m5_test = self.calculate_map(checkpoint_weights, self.test_idx, k=5)
                    m10_test = self.calculate_map(checkpoint_weights, self.test_idx, k=10)
                    
                    # Update evaluation_results_{gt}.json so frontend polling sees new metrics
                    eval_filename = f"evaluation_results_{self.gt_model}.json"
                    results = {
                        "gt_model": self.gt_model,
                        "metrics": {
                            "test_map_after": float(m_test),
                            "test_map5_after": float(m5_test),
                            "test_map10_after": float(m10_test),
                            "improvement": 0.0, # Placeholder
                            "avg_labels_per_query": float(np.mean([np.sum(l) for l in self.y_true]))
                        },
                        "weights": checkpoint_weights,
                        "timestamp": time.time()
                    }
                    with open(eval_filename, "w") as f:
                        json.dump(results, f, indent=4)

                    # Generate temporary charts for visual feedback
                    self.generate_charts(
                        checkpoint_weights, {}, 
                        m_test, m_test, m5_test, m5_test, m10_test, m10_test, 
                        f"{self.gt_model}"
                    )
                    
                    logger.info(f"[{self.gt_model.upper()}] Trial {trial.number + 1}/{n_trials} - New Best mAP@5: {trial.value:.4f} (Test: {m5_test:.4f})")
                except Exception as e:
                    logger.error(f"[{self.gt_model.upper()}] Error saving checkpoint: {e}")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        
        best_weights = study.best_params
        if not allow_negative:
            total_w = sum(best_weights.values())
            if total_w > 0:
                best_weights = {k: v / total_w for k, v in best_weights.items()}
            
        logger.info(f"[{self.gt_model.upper()}] Optimization completed. Best Train mAP@5: {study.best_value:.4f}")
        
        # Remove checkpoint file after success
        checkpoint_filename = f"weights_{self.gt_model}_checkpoint.json"
        if os.path.exists(checkpoint_filename):
            try:
                # Copy to final weights before removing
                final_filename = f"weights_{self.gt_model}.json"
                os.replace(checkpoint_filename, final_filename)
            except:
                pass
                
        return best_weights

    def evaluate_old(self, weights: Dict[str, float]):
        logger.info(f"[{self.gt_model.upper()}] Evaluating optimized weights on Test set...")
        map_after = self.calculate_map(weights, self.test_idx)
        
        # Compare with equal weights
        n_features = len([w for w in weights.values() if w > 0]) or 1
        equal_weights = {name: (1.0/n_features if weights.get(name, 0) > 0 else 0.0) for name in self.feature_names}
        map_before = self.calculate_map(equal_weights, self.test_idx)
        
        logger.info(f"[{self.gt_model.upper()}] mAP Before: {map_before:.4f}, After: {map_after:.4f}")
        
        # Save weights
        with open(settings.weights_file, "w") as f:
            json.dump(weights, f, indent=4)
        
        # Prepare results
        results = {
            "gt_model": self.gt_model,
            "metrics": {
                "train_map": map_after,
                "test_map_before": map_before,
                "test_map_after": map_after,
                "improvement": (map_after - map_before) / (map_before + 1e-7),
                "avg_labels_per_query": float(np.mean(np.sum(self.y_true, axis=1)))
            },
            "weights": weights,
            "top_features": sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10],
            "timestamp": time.time()
        }
        
    def evaluate(self, weights: Dict[str, float], suffix: str = ""):
        """Evaluate best weights on the test set and save results with suffix"""
        logger.info(f"Evaluating best weights for {suffix} on test set...")
        
        # Calculate multiple metrics
        map_before = self.calculate_map({}, self.test_idx)
        map_after = self.calculate_map(weights, self.test_idx)
        
        map5_before = self.calculate_map({}, self.test_idx, k=5)
        map5_after = self.calculate_map(weights, self.test_idx, k=5)
        
        map10_before = self.calculate_map({}, self.test_idx, k=10)
        map10_after = self.calculate_map(weights, self.test_idx, k=10)
        
        improvement = (map_after - map_before) / (map_before + 1e-7)
        
        equal_weights = {name: 1.0/len(self.feature_matrices) for name in self.feature_matrices.keys()}
        
        results = {
            "gt_model": self.gt_model,
            "metrics": {
                "test_map_before": float(map_before),
                "test_map_after": float(map_after),
                "test_map5_before": float(map5_before),
                "test_map5_after": float(map5_after),
                "test_map10_before": float(map10_before),
                "test_map10_after": float(map10_after),
                "improvement": float(improvement),
                "avg_labels_per_query": float(np.mean([np.sum(l) for l in self.y_true]))
            },
            "weights": weights,
            "top_features": sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10],
            "timestamp": time.time()
        }
        
        # Save specific weights
        weights_filename = f"weights_{suffix}.json"
        with open(weights_filename, "w") as f:
            json.dump(weights, f, indent=4)
        
        # Always update the default weights.json with CLIP result for safety
        if suffix == "clip":
            with open("weights.json", "w") as f:
                json.dump(weights, f, indent=4)

        # Save specific evaluation results
        eval_filename = f"evaluation_results_{suffix}.json"
        with open(eval_filename, "w") as f:
            json.dump(results, f, indent=4)
            
        self.generate_charts(
            weights, equal_weights, 
            map_before, map_after, 
            map5_before, map5_after, 
            map10_before, map10_after, 
            suffix
        )
        logger.info(f"Results for {suffix} saved to {eval_filename}")

    def generate_charts(self, weights: Dict[str, float], equal_weights: Dict[str, float], 
                        map_before: float, map_after: float, 
                        map5_before: float, map5_after: float,
                        map10_before: float, map10_after: float,
                        suffix: str = ""):
        # 1. Weight Contribution
        plt.figure(figsize=(12, 6))
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_weights if x[1] > 0.01]
        vals = [x[1] for x in sorted_weights if x[1] > 0.01]
        sns.barplot(x=vals, y=names, palette="viridis")
        plt.title(f"Feature Weight Contributions (GT: {self.gt_model})")
        plt.tight_layout()
        plt.savefig(os.path.join(settings.visualizations_dir, f"weight_contribution_{suffix}.png"))
        
        # 2. mAP Comparison (Grouped Bar Chart)
        plt.figure(figsize=(10, 6))
        labels = ['mAP (All)', 'mAP@5', 'mAP@10']
        before_vals = [map_before, map5_before, map10_before]
        after_vals = [map_after, map5_after, map10_after]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, before_vals, width, label='Baseline (Equal)', color='gray')
        plt.bar(x + width/2, after_vals, width, label='Optimized', color='indigo')
        
        plt.title(f"mAP Metrics Comparison (GT: {self.gt_model})")
        plt.xticks(x, labels)
        plt.ylabel("Score")
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(settings.visualizations_dir, f"map_comparison_{suffix}.png"))
        
        # 3. PR Curve
        plt.figure(figsize=(10, 7))
        def get_mean_pr(w_dict, indices):
            n_total = len(self.image_ids)
            final_sim = np.zeros((len(indices), n_total))
            for name, weight in w_dict.items():
                if name in self.feature_matrices:
                    final_sim += weight * self.feature_matrices[name][indices, :]
            all_precisions = []
            mean_recall = np.linspace(0, 1, 100)
            for i, idx in enumerate(indices):
                scores = final_sim[i]
                labels = self.y_true[idx]
                mask = np.ones(n_total, dtype=bool); mask[idx] = False
                if np.sum(labels[mask]) == 0: continue
                precision, recall, _ = precision_recall_curve(labels[mask], scores[mask])
                interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                all_precisions.append(interp_precision)
            return mean_recall, np.mean(all_precisions, axis=0)

        r_before, p_before = get_mean_pr(equal_weights, self.test_idx)
        r_after, p_after = get_mean_pr(weights, self.test_idx)
        plt.plot(r_before, p_before, label=f'Baseline (mAP={map_before:.4f})', color='gray', ls='--')
        plt.plot(r_after, p_after, label=f'Optimized (mAP={map_after:.4f})', color='indigo', lw=2)
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Mean Precision-Recall Curve (GT: {self.gt_model})'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(settings.visualizations_dir, f"pr_curve_{suffix}.png"))

        # 4. Precision@K Curve (1 to 20)
        plt.figure(figsize=(10, 6))
        ks = np.arange(1, 21)
        pk_before = [self.calculate_map(equal_weights, self.test_idx, k=int(k)) for k in ks]
        pk_after = [self.calculate_map(weights, self.test_idx, k=int(k)) for k in ks]
        plt.plot(ks, pk_before, 'o-', label='Baseline', color='gray')
        plt.plot(ks, pk_after, 's-', label='Optimized', color='indigo')
        plt.xlabel('K (Top Results)'); plt.ylabel('Precision@K'); plt.title(f'Precision at K (1-20) - GT: {self.gt_model}')
        plt.xticks(ks); plt.ylim(0, 1.05); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(settings.visualizations_dir, f"pk_curve_{suffix}.png"))

        # 5. Score Distribution (Histogram of Similarities for Positives vs Negatives)
        # Take a sample query to visualize separation
        plt.figure(figsize=(10, 6))
        if len(self.test_idx) > 0:
            idx = self.test_idx[0]
            scores = np.zeros(len(self.image_ids))
            for name, weight in weights.items():
                if name in self.feature_matrices:
                    scores += weight * self.feature_matrices[name][idx, :]
            
            labels = self.y_true[idx]
            pos_scores = scores[labels == 1]
            neg_scores = scores[labels == 0]
            
            sns.kdeplot(pos_scores, label='Similar (GT)', fill=True, color='green')
            sns.kdeplot(neg_scores, label='Dissimilar', fill=True, color='red')
            plt.title(f"Score Separation Analysis (GT: {self.gt_model})")
            plt.xlabel("Similarity Score"); plt.ylabel("Density"); plt.legend()
            plt.savefig(os.path.join(settings.visualizations_dir, f"score_dist_{suffix}.png"))

        plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, default="all", help="Model to optimize for (clip, dinov2, siglip, dreamsim, or all)")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--allow-negative", action="store_true")
    parser.add_argument("--exclude-embeddings", action="store_true")
    args = parser.parse_args()
    
    available_models = ["clip", "dinov2", "siglip", "dreamsim"]
    gt_models = available_models if args.gt == "all" else [args.gt]
    
    for gt_name in gt_models:
        if gt_name not in available_models and args.gt != "all":
            logger.warning(f"Unknown ground truth model: {gt_name}. Skipping.")
            continue
            
        logger.info(f"\n{'='*60}\nRUNNING OPTIMIZATION FOR GROUND TRUTH: {gt_name.upper()}\n{'='*60}")
        optimizer = WeightOptimizer(ground_truth_model=gt_name, train_size=args.split)
        if optimizer.load_data():
            best_w = optimizer.optimize(
                n_trials=args.trials, 
                allow_negative=args.allow_negative, 
                exclude_embeddings=args.exclude_embeddings
            )
            optimizer.evaluate(best_w, suffix=gt_name)
    
    logger.info("\nALL REQUESTED OPTIMIZATIONS COMPLETED!")
