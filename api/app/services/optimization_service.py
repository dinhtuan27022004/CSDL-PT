import os
import json
import numpy as np
import optuna
import time
import threading
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from ..models.image import ImageMetadata
from ..core.config import get_settings
from ..core.logging import get_logger
from sklearn.metrics import average_precision_score, precision_recall_curve
from ..core.similarity_specs import get_all_feature_specs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = get_logger(__name__)
settings = get_settings()

# Mapping from spec metrics to internal numpy metrics
METRIC_MAP = {
    "scalar": "scalar",
    "sharpness": "sharpness",
    "cosine": "cosine",
    "l2_color": "l2_color",
    "l2_cell": "l2_cell_color"
}

# Mapping from feature names to actual ImageMetadata column names if different
COLUMN_MAP = {
    "hog": "hog_vector", "hu_moments": "hu_moments_vector", "lbp": "lbp_vector",
    "color_moments": "color_moments_vector", "gabor": "gabor_vector", "ccv": "ccv_vector",
    "zernike": "zernike_vector", "geo": "geo_vector", "tamura": "tamura_vector",
    "edge_orientation": "edge_orientation_vector", "glcm": "glcm_vector", "wavelet": "wavelet_vector",
    "correlogram": "correlogram_vector", "ehd": "ehd_vector", "cld": "cld_vector",
    "spm": "spm_vector", "saliency": "saliency_vector", "semantic": "llm_embedding",
    "dreamsim": "dreamsim_vector", "dominant_color": "dominant_color_vector",
    "category": "category", "entity": "entities"
}

# Add cell vector mapping
for space in ["rgb", "hsv", "lab", "ycrcb", "hls", "xyz", "gray"]:
    COLUMN_MAP[f"cell_{space}"] = f"cell_{space}_vector"

def get_similarity_matrix(vectors: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Calculate N x N similarity matrix for a feature vector"""
    n = vectors.shape[0]
    if n == 0: return np.zeros((0, 0), dtype=np.float32)

    vectors = vectors.astype(np.float32)

    if metric == "cosine":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-7
        norm_vecs = vectors / norms
        sim = np.dot(norm_vecs, norm_vecs.T)
        return np.maximum(0, sim).astype(np.float32)
    
    elif metric in ["l2_color", "l2_cell_color"]:
        if metric == "l2_color":
            max_dist = 255.0 * np.sqrt(3.0)
        else:
            dim = vectors.shape[1]
            max_dist = 255.0 * np.sqrt(dim)
            
        sq_norms = np.sum(vectors**2, axis=1)
        dot_prod = np.dot(vectors, vectors.T)
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * dot_prod
        dist_sq = np.maximum(dist_sq, 0)
        l2 = np.sqrt(dist_sq)
        return (1.0 - l2 / max_dist).astype(np.float32)

    elif metric == "scalar":
        v = vectors.flatten()
        diff = np.abs(v[:, np.newaxis] - v[np.newaxis, :])
        return (1.0 - diff).astype(np.float32)
    
    elif metric == "sharpness":
        v = vectors.flatten()
        a = v[:, np.newaxis]
        b = v[np.newaxis, :]
        diff = np.abs(a - b)
        denom = np.abs(a + b) + 1e-7
        return (1.0 - diff / denom).astype(np.float32)

    return np.zeros((n, n), dtype=np.float32)

def get_discrete_similarity(values: List[Any], type: str = "category") -> np.ndarray:
    n = len(values)
    sim = np.zeros((n, n), dtype=np.float32)
    if type == "category":
        for i in range(n):
            v1 = (values[i] or "General").lower()
            for j in range(n):
                v2 = (values[j] or "General").lower()
                sim[i, j] = 1.0 if v1 == v2 else 0.0
    elif type == "entities":
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

class SharedFeatureStore:
    def __init__(self, db_session: Session, exclude_embeddings: bool = False):
        self.db = db_session
        self.exclude_embeddings = exclude_embeddings
        self.feature_matrices = {}
        self.image_ids = []
        self.images = []
        self.feature_names = []
        self._is_loaded = False
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._is_loaded:
                return True
                
            logger.info("Loading all features into Shared Feature Store using central specs...")
            self.images = self.db.query(ImageMetadata).all()
            if not self.images:
                logger.error("No images found in database.")
                return False
            
            self.image_ids = [img.id for img in self.images]
            images = self.images
            specs = get_all_feature_specs()
            
            for name, metric_type in specs.items():
                try:
                    col_name = COLUMN_MAP.get(name, name)
                    
                    if metric_type in ["scalar", "sharpness", "cosine", "l2_color", "l2_cell"]:
                        raw_data = [getattr(img, col_name) for img in images]
                        
                        if metric_type in ["scalar", "sharpness"]:
                            vec = np.array([float(v) if v is not None else 0.0 for v in raw_data])
                            self.feature_matrices[name] = get_similarity_matrix(vec, metric=METRIC_MAP[metric_type])
                        else:
                            sample_v = next((v for v in raw_data if v is not None), None)
                            dim = len(sample_v) if sample_v is not None else 512
                            vecs = np.array([np.array(v) if v is not None else np.zeros(dim) for v in raw_data])
                            self.feature_matrices[name] = get_similarity_matrix(vecs, metric=METRIC_MAP[metric_type])
                            
                    elif metric_type == "category":
                        vals = [getattr(img, col_name) for img in images]
                        self.feature_matrices[name] = get_discrete_similarity(vals, type="category")
                        
                    elif metric_type == "entity":
                        vals = [getattr(img, col_name) for img in images]
                        self.feature_matrices[name] = get_discrete_similarity(vals, type="entities")
                        
                    # Yield control after processing each feature matrix
                    time.sleep(0.01)
                except Exception as e:
                    logger.error(f"Failed to load feature matrix for {name}: {e}")

            self.feature_names = list(self.feature_matrices.keys())
            self._is_loaded = True
            logger.info(f"Shared Feature Store loaded with {len(self.feature_names)} similarity matrices.")
            return True

class WeightOptimizer:
    def __init__(self, gt_name: str, shared_store: SharedFeatureStore):
        self.gt_model = gt_name
        self.store = shared_store
        self.feature_matrices = shared_store.feature_matrices
        self.feature_names = shared_store.feature_names
        self.image_ids = shared_store.image_ids
        self.y_true = None
        self.test_idx = []
        self.train_idx = []
        self.best_test_map5 = -1.0
        self.trial_history = [] # List of (n_features, map5)

    def prepare(self):
        logger.info("Preparing ground truth matrix for optimization...")
        # Prioritize ground_truth_2.json (Diverse subset) over full ground_truth.json
        gt_path = os.path.join(settings.base_dir, "ground_truth_2.json")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(settings.base_dir, "ground_truth.json")
            
        if not os.path.exists(gt_path):
            logger.error("No ground truth file found (tried ground_truth_2.json and ground_truth.json)")
            return False
            
        logger.info(f"Using ground truth source: {os.path.basename(gt_path)}")
        try:
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {gt_path}: {e}")
            return False
        
        file_to_label = {}
        for label, files in gt_data.items():
            for f in files:
                file_to_label[f] = label
                
        images = self.store.images
        n = len(images)
        self.y_true = np.zeros((n, n), dtype=np.uint8)
        # Use os.path.basename to match filenames in ground_truth.json
        labels = [file_to_label.get(os.path.basename(img.file_name)) for img in images]
        
        for i in range(n):
            l1 = labels[i]
            if l1 is None: continue
            for j in range(n):
                if i != j and labels[j] == l1:
                    self.y_true[i, j] = 1
                    
        self.train_idx = np.arange(n)
        self.test_idx = np.arange(n)
        self.trial_history = [] # Reset history for new optimization run
        return True

    def _extract_weights(self, params: Dict[str, Any], allow_negative: bool = False) -> Dict[str, float]:
        weights = {}
        for name in self.feature_names:
            is_active = params.get(f"use_{name}", True)
            weights[name] = params.get(name, 0.0) if is_active else 0.0
        
        if not allow_negative:
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v / total_w for k, v in weights.items()}
        return weights

    def calculate_map(self, weights: Dict[str, float], indices: np.ndarray, k: Optional[int] = None) -> float:
        n_total = len(self.image_ids)
        final_sim = np.zeros((len(indices), n_total), dtype=np.float32)
        for name, weight in weights.items():
            if weight == 0: continue
            final_sim += weight * self.feature_matrices[name][indices, :]
        
        aps = []
        for i, idx in enumerate(indices):
            scores = final_sim[i]
            labels = self.y_true[idx]
            mask = np.ones(n_total, dtype=bool); mask[idx] = False
            s_masked = scores[mask]
            l_masked = labels[mask]
            total_positives = np.sum(l_masked)
            if total_positives == 0: continue
            
            if k is not None:
                order = np.argsort(s_masked)[::-1]
                l_sorted = l_masked[order]
                l_k = l_sorted[:k]
                hits = np.where(l_k == 1)[0]
                if len(hits) == 0:
                    aps.append(0.0)
                    continue
                precision_at_hits = [np.sum(l_k[:hit_idx+1]) / (hit_idx + 1) for hit_idx in hits]
                aps.append(np.sum(precision_at_hits) / min(total_positives, k))
            else:
                aps.append(average_precision_score(l_masked, s_masked))
                
            # Periodically yield control
            if i % 100 == 0:
                time.sleep(0.001)
                
        return np.mean(aps) if aps else 0.0

    def optimize(self, n_trials: int = 100, allow_negative: bool = False):
        def objective(trial):
            alpha = 0.01 # Sparsity penalty weight
            low = -1.0 if allow_negative else 0.0
            
            # Use trial.suggest_categorical to decide which features to include
            for name in self.feature_names:
                is_active = trial.suggest_categorical(f"use_{name}", [True, False])
                if is_active:
                    trial.suggest_float(name, low, 1.0)
            
            weights = self._extract_weights(trial.params, allow_negative)
            n_active = sum(1 for v in weights.values() if abs(v) > 0.005)
            
            if n_active == 0:
                return 0.0
                
            m5 = self.calculate_map(weights, self.train_idx, k=5)
            
            # Store history for visualization
            self.trial_history.append((n_active, float(m5)))
            
            # Penalize using many features
            sparsity_penalty = (n_active / len(self.feature_names)) * alpha
            return m5 - sparsity_penalty

        def callback(study, trial):
            if study.best_trial.number == trial.number:
                checkpoint_weights = self._extract_weights(study.best_params, allow_negative)
                m5_test = self.calculate_map(checkpoint_weights, self.test_idx, k=5)
                
                if m5_test >= self.best_test_map5:
                    self.best_test_map5 = m5_test
                    self.save_results(checkpoint_weights) # Cập nhật file ảnh ngay lập tức
                    logger.info(f"Trial {trial.number + 1} - NEW BEST mAP@5: {m5_test:.4f}")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        return self._extract_weights(study.best_params, allow_negative)

    def save_results(self, weights: Dict[str, float]):
        m_test = self.calculate_map(weights, self.test_idx, k=None)
        m5_test = self.calculate_map(weights, self.test_idx, k=5)
        m10_test = self.calculate_map(weights, self.test_idx, k=10)
        
        results = {
            "gt_model": "folder",
            "metrics": {
                "test_map_after": float(m_test),
                "test_map5_after": float(m5_test),
                "test_map10_after": float(m10_test),
                "avg_labels_per_query": float(np.mean([np.sum(l) for l in self.y_true]))
            },
            "weights": weights,
            "timestamp": time.time()
        }
        
        with open(os.path.join(settings.base_dir, "weights.json"), "w") as f:
            json.dump(weights, f, indent=4)
        with open(os.path.join(settings.base_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        self.generate_charts(weights, results["metrics"])

    def generate_charts(self, weights: Dict[str, float], metrics: Dict[str, Any]):
        os.makedirs(settings.visualizations_dir, exist_ok=True)
        
        # 1. Weight Contribution
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_weights if x[1] > 0.01]
        vals = [x[1] for x in sorted_weights if x[1] > 0.01]
        if names: sns.barplot(x=vals, y=names, palette="viridis", ax=ax1)
        ax1.set_title("Feature Weight Contributions")
        fig1.tight_layout()
        fig1.savefig(os.path.join(settings.visualizations_dir, "weight_contribution_optimized.png"))
        plt.close(fig1)
        
        # 2. PR Curve
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        n_total = len(self.image_ids)
        final_sim = np.zeros((len(self.test_idx), n_total), dtype=np.float32)
        for name, weight in weights.items():
            if weight != 0: final_sim += weight * self.feature_matrices[name][self.test_idx, :]
        
        all_precisions = []
        mean_recall = np.linspace(0, 1, 100)
        
        # Data for P@K and Score Separation
        pk_values = np.zeros(20)
        gt_scores = []
        other_scores = []
        
        for i, idx in enumerate(self.test_idx):
            scores = final_sim[i]; labels = self.y_true[idx]
            mask = np.ones(n_total, dtype=bool); mask[idx] = False
            s_masked = scores[mask]
            l_masked = labels[mask]
            
            if np.sum(l_masked) == 0: continue
            
            # 2a. PR Curve data
            precision, recall, _ = precision_recall_curve(l_masked, s_masked)
            all_precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
            
            # 2b. P@K data
            order = np.argsort(s_masked)[::-1]
            l_sorted = l_masked[order]
            for k in range(1, 21):
                pk_values[k-1] += np.mean(l_sorted[:k])
                
            # 2c. Score Separation data (subsample for performance)
            pos_indices = np.where(l_masked == 1)[0]
            neg_indices = np.where(l_masked == 0)[0]
            gt_scores.extend(s_masked[pos_indices].tolist())
            if len(neg_indices) > 0:
                # Subsample negative samples to keep balanced chart
                neg_sample = np.random.choice(neg_indices, min(len(neg_indices), len(pos_indices) * 2), replace=False)
                other_scores.extend(s_masked[neg_sample].tolist())
        
        # Save PR Curve
        if all_precisions:
            ax2.plot(mean_recall, np.mean(all_precisions, axis=0), color='indigo', lw=2)
            ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
            ax2.set_title('Mean Precision-Recall Curve')
            ax2.grid(True, alpha=0.3)
        fig2.savefig(os.path.join(settings.visualizations_dir, "pr_curve_optimized.png"))
        plt.close(fig2)

        # 3. P@K Curve
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ks = np.arange(1, 21)
        pk_means = pk_values / len(all_precisions)
        ax3.plot(ks, pk_means, 'o-', color='crimson', lw=2, markersize=8)
        ax3.set_xticks(ks)
        ax3.set_xlabel('K'); ax3.set_ylabel('Precision@K')
        ax3.set_title('Mean Precision at K (1-20)')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        fig3.savefig(os.path.join(settings.visualizations_dir, "pk_curve_optimized.png"))
        plt.close(fig3)

        # 4. Score Separation
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        if gt_scores and other_scores:
            sns.kdeplot(gt_scores, fill=True, color="green", label="Ground Truth Pairs", ax=ax4)
            sns.kdeplot(other_scores, fill=True, color="red", label="Other Pairs", ax=ax4)
            ax4.set_title("Score Separation: GT vs Others")
            ax4.set_xlabel("Similarity Score")
            ax4.legend()
        fig4.savefig(os.path.join(settings.visualizations_dir, "score_dist_optimized.png"))
        plt.close(fig4)

        # 5. Pareto Frontier: Sparsity vs Accuracy
        try:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            if self.trial_history:
                n_feats = [h[0] for h in self.trial_history]
                m5_vals = [h[1] for h in self.trial_history]
                ax5.scatter(n_feats, m5_vals, alpha=0.4, color='teal', s=30, label='All Trials')
                unique_n = sorted(list(set(n_feats)))
                pareto_x, pareto_y = [], []
                current_best_y = -1
                for n in unique_n:
                    best_at_n = max([h[1] for h in self.trial_history if h[0] == n])
                    if best_at_n > current_best_y:
                        pareto_x.append(n); pareto_y.append(best_at_n)
                        current_best_y = best_at_n
                ax5.plot(pareto_x, pareto_y, 'r--', marker='x', lw=2, label='Pareto Frontier')
                ax5.set_xlabel('Number of Active Features'); ax5.set_ylabel('mAP@5')
                ax5.set_title('Sparsity vs Accuracy Trade-off (Pareto Frontier)')
                ax5.legend()
            else:
                ax5.text(0.5, 0.5, "Optimization starting... Please wait.", ha='center', va='center')
            
            ax5.grid(True, alpha=0.3)
            path5 = os.path.join(settings.visualizations_dir, "sparsity_vs_accuracy_optimized.png")
            os.makedirs(os.path.dirname(path5), exist_ok=True)
            fig5.savefig(path5)
            plt.close(fig5)
            logger.info(f"Successfully saved sparsity chart to {path5}")
        except Exception as e:
            logger.error(f"Failed to save sparsity chart: {e}")

class OptimizationService:
    def __init__(self, db: Session):
        self.db = db

    def run_optimization(self, trials: int = 50, allow_negative: bool = False):
        try:
            logger.info(f"Starting background optimization: trials={trials}, allow_negative={allow_negative}")
            store = SharedFeatureStore(self.db)
            if not store.load():
                logger.error("Failed to load features into store")
                return
                
            optimizer = WeightOptimizer("folder", store)
            if optimizer.prepare():
                # Warm-up: Create empty charts so UI doesn't show 404
                optimizer.save_results({})
                
                optimizer.optimize(n_trials=trials, allow_negative=allow_negative)
                logger.info("Optimization task completed successfully")
            else:
                logger.error("Failed to prepare optimizer")
        except Exception as e:
            logger.error(f"Error during optimization service task: {e}", exc_info=True)
