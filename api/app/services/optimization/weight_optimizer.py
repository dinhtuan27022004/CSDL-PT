import os
import json
import shutil
import numpy as np
import optuna
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import average_precision_score, precision_recall_curve
from ...core.config import get_settings
from ...core.logging import get_logger
from .feature_store import SharedFeatureStore

logger = get_logger(__name__)
settings = get_settings()

class WeightOptimizer:
    """Handles the Optuna-based optimization of feature weights using the pre-loaded FeatureStore"""
    
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
        self.gt_data = {} # Store ground truth raw data for stats

    def prepare(self, gt_path: Optional[str] = None) -> bool:
        """Load ground truth and prepare validation indices"""
        if gt_path is None:
            gt_path = os.path.join(settings.base_dir, "ground_truth.json")
            
        if not os.path.exists(gt_path):
            logger.error(f"No ground truth file found at: {gt_path}")
            return False
            
        logger.info(f"Using ground truth source: {gt_path}")
        try:
            with open(gt_path, "r") as f:
                self.gt_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {gt_path}: {e}")
            return False
        
        file_to_label = {os.path.basename(f): label for label, files in self.gt_data.items() for f in files}
                
        n = len(self.store.images)
        self.y_true = np.zeros((n, n), dtype=np.uint8)
        labels = [file_to_label.get(os.path.basename(img.file_name)) for img in self.store.images]
        
        if not any(l is not None for l in labels):
            logger.error("No images matched the ground truth labels.")
            return False
        
        for i in range(n):
            l1 = labels[i]
            if l1 is None: continue
            for j in range(n):
                if i != j and labels[j] == l1:
                    self.y_true[i, j] = 1
                    
        query_indices = []
        seen_basenames = set()
        for i, img in enumerate(self.store.images):
            basename = os.path.basename(img.file_name)
            if basename in file_to_label:
                if basename not in seen_basenames:
                    query_indices.append(i)
                    seen_basenames.add(basename)
                else:
                    logger.warning(f"Duplicate DB entry for {basename} (ID {img.id}). Skipping in evaluation.")
        
        if not query_indices:
            logger.warning("No images from ground truth found in store. Using all images as fallback.")
            self.train_idx = self.test_idx = np.arange(n)
        else:
            logger.info(f"Filtered to {len(query_indices)} unique query images for evaluation.")
            self.train_idx = self.test_idx = np.array(query_indices)
            
        self.trial_history = []
        return True

    def calculate_map(self, weights: Dict[str, float], indices: np.ndarray, k: Optional[int] = None) -> float:
        """Calculate Mean Average Precision for a given set of weights"""
        n_total = len(self.image_ids)
        final_sim = np.zeros((len(indices), n_total), dtype=np.float32)
        for name, weight in weights.items():
            if weight > 0:
                final_sim += weight * self.feature_matrices[name][indices, :]
        
        aps = []
        for i, idx in enumerate(indices):
            scores = final_sim[i]
            labels = self.y_true[idx]
            mask = np.ones(n_total, dtype=bool); mask[idx] = False
            s_masked, l_masked = scores[mask], labels[mask]
            total_positives = np.sum(l_masked)
            if total_positives == 0: continue
            
            if k is not None:
                order = np.argsort(s_masked)[::-1]
                l_k = l_masked[order][:k]
                hits = np.where(l_k == 1)[0]
                if len(hits) == 0:
                    aps.append(0.0)
                else:
                    precision_at_hits = [np.sum(l_k[:h+1]) / (h + 1) for h in hits]
                    aps.append(np.sum(precision_at_hits) / min(total_positives, k))
            else:
                aps.append(average_precision_score(l_masked, s_masked))
            
            if i % 100 == 0: time.sleep(0.001) # Yield
                
        return np.mean(aps) if aps else 0.0

    def optimize(self, n_trials: int = 100):
        """Run Optuna optimization study"""
        def objective(trial):
            weights = {}
            for name in self.feature_names:
                weights[name] = trial.suggest_float(name, 0.0, 1.0)
            
            # Normalize and handle zero total
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            else:
                return 0.0
                
            n_active = sum(1 for v in weights.values() if v > 0)
            if n_active == 0: return 0.0
                
            m5 = self.calculate_map(weights, self.train_idx, k=5)
            self.trial_history.append((n_active, float(m5)))
            return m5

        def callback(study, trial):
            if study.best_trial.number == trial.number:
                best_w = self._extract_weights(study.best_params)
                m5_test = self.calculate_map(best_w, self.test_idx, k=5)
                if m5_test >= self.best_test_map5:
                    self.best_test_map5 = m5_test
                    self.save_results(best_w)
                    logger.info(f"Trial {trial.number + 1} - NEW BEST mAP@5: {m5_test:.4f}")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        return self._extract_weights(study.best_params)

    def _extract_weights(self, params: Dict[str, Any]) -> Dict[str, float]:
        weights = {name: params.get(name, 0.0) for name in self.feature_names}
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        return weights

    def save_results(self, weights: Dict[str, float]):
        """Save results to JSON and generate visualizations"""
        m_test = self.calculate_map(weights, self.test_idx, k=None)
        m5_test = self.calculate_map(weights, self.test_idx, k=5)
        m10_test = self.calculate_map(weights, self.test_idx, k=10)
        
        # Track the absolute best score across all saves to prevent UI "flicker"
        if m5_test > self.best_test_map5:
            self.best_test_map5 = m5_test
        
        n_total = len(self.image_ids)
        final_sim = np.zeros((len(self.test_idx), n_total), dtype=np.float32)
        for name, weight in weights.items():
            if weight > 0: final_sim += weight * self.feature_matrices[name][self.test_idx, :]

        per_image_ap5 = []
        for i, idx in enumerate(self.test_idx):
            scores, labels = final_sim[i], self.y_true[idx]
            mask = np.ones(n_total, dtype=bool); mask[idx] = False
            s_masked, l_masked = scores[mask], labels[mask]
            total_positives = int(np.sum(l_masked))
            if total_positives == 0:
                per_image_ap5.append((idx, 0.0, 0))
                continue
            order = np.argsort(s_masked)[::-1]
            l_k = l_masked[order][:5]
            hits = np.where(l_k == 1)[0]
            ap5 = float(np.sum([np.sum(l_k[:h+1]) / (h + 1) for h in hits]) / min(total_positives, 5)) if len(hits) > 0 else 0.0
            per_image_ap5.append((idx, ap5, total_positives))

        sorted_queries = sorted(per_image_ap5, key=lambda x: x[1])
        
        worst_queries = [{
            "rank": r, "image_id": int(self.store.images[idx].id),
            "file_name": self.store.images[idx].file_name, "file_path": self.store.images[idx].file_path,
            "map5": round(ap, 4), "n_positives_in_gt": n_pos
        } for r, (idx, ap, n_pos) in enumerate(sorted_queries, 1) if ap < 0.5]

        best_queries = [{
            "rank": r, "image_id": int(self.store.images[idx].id),
            "file_name": self.store.images[idx].file_name, "file_path": self.store.images[idx].file_path,
            "map5": round(ap, 4), "n_positives_in_gt": n_pos
        } for r, (idx, ap, n_pos) in enumerate(reversed(sorted_queries), 1) if ap >= 0.5]

        # Add Ground Truth Stats
        gt_stats = {
            "total_images": sum(len(files) for files in self.gt_data.values()),
            "clusters": [{"name": name, "count": len(files)} for name, files in self.gt_data.items()]
        }

        results = {
            "gt_model": "folder", "weights": weights, 
            "worst_queries": worst_queries, "best_queries": best_queries, 
            "gt_stats": gt_stats,
            "timestamp": time.time(),
            "metrics": {
                "test_map_after": float(m_test), 
                "test_map5_after": float(m5_test), 
                "test_map10_after": float(m10_test),
                "avg_labels_per_query": float(np.mean([np.sum(l) for l in self.y_true]))
            }
        }
        
        with open(settings.weights_file, "w") as f: json.dump(weights, f, indent=4)
        with open(os.path.join(settings.base_dir, "evaluation_results.json"), "w") as f: json.dump(results, f, indent=4)
        
        # Export Best Data (MAP5 >= 50%)
        self.export_best_data(per_image_ap5)
        
        self.generate_charts(weights, results["metrics"])

    def export_best_data(self, per_image_ap5: List[Tuple[int, float, int]]):
        """Creates a 'best_data' folder with images having MAP5 >= 0.5, structured by clusters"""
        best_dir = os.path.join(settings.base_dir, "best_data")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.makedirs(best_dir, exist_ok=True)

        idx_to_ap5 = {idx: ap for idx, ap, _ in per_image_ap5}
        file_to_idx = {img.file_name: i for i, img in enumerate(self.store.images)}

        count = 0
        fail_count = 0
        
        logger.info(f"STARTING EXPORT TO: {best_dir}")
        logger.info(f"Total images in store: {len(self.store.images)}")
        
        # Pre-scan uploads directory for brute-force matching
        all_physical_files = {}
        if os.path.exists(settings.uploads_dir):
            for root, _, filenames in os.walk(settings.uploads_dir):
                for f in filenames:
                    all_physical_files[f] = os.path.join(root, f)

        for cluster_name, files in self.gt_data.items():
            cluster_dir = os.path.join(best_dir, cluster_name)
            
            for file_path_in_gt in files:
                fname = os.path.basename(file_path_in_gt)
                idx = file_to_idx.get(fname)
                ap5 = idx_to_ap5.get(idx, 0) if idx is not None else 0
                
                if idx is not None and ap5 >= 0.5:
                    src_path = self.store.images[idx].file_path
                    abs_src = None

                    # Strategy 1: Direct path from DB
                    if os.path.exists(src_path):
                        abs_src = src_path
                    
                    # Strategy 2: Web static resolution
                    if not abs_src and "/static/uploads/" in src_path:
                        rel_path = src_path.split("/static/uploads/")[-1]
                        p = os.path.join(settings.uploads_dir, rel_path)
                        if os.path.exists(p): abs_src = p
                    
                    # Strategy 3: Brute force filename match in uploads folder
                    if not abs_src:
                        abs_src = all_physical_files.get(fname)

                    if abs_src and os.path.exists(abs_src):
                        os.makedirs(cluster_dir, exist_ok=True)
                        shutil.copy2(abs_src, os.path.join(cluster_dir, fname))
                        count += 1
                        # logger.debug(f"COPIED: {fname} (MAP5: {ap5:.2f})")
                    else:
                        fail_count += 1
                        logger.error(f"FILE NOT FOUND: {fname} (MAP5: {ap5:.2f}). DB Path: {src_path}")
                elif idx is not None:
                    # Log images that didn't make the cut (low MAP5)
                    pass 
                else:
                    logger.warning(f"IMAGE NOT IN STORE: {fname}")
        
        logger.info(f"EXPORT FINISHED. Success: {count}, Failed: {fail_count}")

    def generate_charts(self, weights: Dict[str, float], metrics: Dict[str, Any]):
        """Generate all diagnostic charts"""
        os.makedirs(settings.visualizations_dir, exist_ok=True)
        
        # 1. Weights Bar Chart
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sorted_w = sorted([(k, v) for k, v in weights.items() if v > 0], key=lambda x: x[1], reverse=True)
        if sorted_w: sns.barplot(x=[x[1] for x in sorted_w], y=[x[0] for x in sorted_w], palette="viridis", ax=ax1)
        ax1.set_title("Feature Weight Contributions")
        fig1.tight_layout()
        fig1.savefig(os.path.join(settings.visualizations_dir, "weight_contribution_optimized.png"))
        plt.close(fig1)
        
        # 2. Performance Curves (PR, P@K, Score Dist)
        n_total = len(self.image_ids)
        final_sim = np.zeros((len(self.test_idx), n_total), dtype=np.float32)
        for name, weight in weights.items():
            if weight > 0: final_sim += weight * self.feature_matrices[name][self.test_idx, :]
        
        all_precisions = []
        mean_recall = np.linspace(0, 1, 100)
        pk_values = np.zeros(20)
        gt_scores, other_scores = [], []
        
        for i, idx in enumerate(self.test_idx):
            s, l = final_sim[i], self.y_true[idx]
            mask = np.ones(n_total, dtype=bool); mask[idx] = False
            s_m, l_m = s[mask], l[mask]
            if np.sum(l_m) == 0: continue
            
            p, r, _ = precision_recall_curve(l_m, s_m)
            all_precisions.append(np.interp(mean_recall, r[::-1], p[::-1]))
            
            order = np.argsort(s_m)[::-1]
            l_sorted = l_m[order]
            for k in range(1, 21): pk_values[k-1] += np.mean(l_sorted[:k])
                
            pos_idx = np.where(l_m == 1)[0]
            neg_idx = np.where(l_m == 0)[0]
            gt_scores.extend(s_m[pos_idx].tolist())
            if len(neg_idx) > 0:
                neg_sample = np.random.choice(neg_idx, min(len(neg_idx), len(pos_idx) * 2), replace=False)
                other_scores.extend(s_m[neg_sample].tolist())
        
        # PR Curve
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        if all_precisions:
            ax2.plot(mean_recall, np.mean(all_precisions, axis=0), color='indigo', lw=2)
            ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Mean PR Curve')
            ax2.grid(True, alpha=0.3)
        fig2.savefig(os.path.join(settings.visualizations_dir, "pr_curve_optimized.png"))
        plt.close(fig2)

        # P@K Curve
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        if all_precisions:
            ks = np.arange(1, 21)
            ax3.plot(ks, pk_values / len(all_precisions), 'o-', color='crimson', lw=2)
            ax3.set_xlabel('K'); ax3.set_ylabel('Precision@K'); ax3.set_title('Mean P@K (1-20)')
            ax3.grid(True, alpha=0.3)
        fig3.savefig(os.path.join(settings.visualizations_dir, "pk_curve_optimized.png"))
        plt.close(fig3)

        # Score Dist
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        if gt_scores and other_scores:
            sns.kdeplot(gt_scores, fill=True, color="green", label="GT Pairs", ax=ax4)
            sns.kdeplot(other_scores, fill=True, color="red", label="Other Pairs", ax=ax4)
            ax4.set_title("Score Separation"); ax4.legend()
        fig4.savefig(os.path.join(settings.visualizations_dir, "score_dist_optimized.png"))
        plt.close(fig4)

        # 3. Pareto Frontier
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        if self.trial_history:
            n_f = [h[0] for h in self.trial_history]; m5 = [h[1] for h in self.trial_history]
            ax5.scatter(n_f, m5, alpha=0.4, color='teal', s=30)
            u_n = sorted(list(set(n_f)))
            px, py = [], []
            cur_best = -1
            for n in u_n:
                best_at_n = max([h[1] for h in self.trial_history if h[0] == n])
                if best_at_n > cur_best:
                    px.append(n); py.append(best_at_n); cur_best = best_at_n
            ax5.plot(px, py, 'r--', marker='x', lw=2, label='Pareto Frontier')
            ax5.set_xlabel('Active Features'); ax5.set_ylabel('mAP@5'); ax5.legend()
            ax5.grid(True, alpha=0.3)
        fig5.savefig(os.path.join(settings.visualizations_dir, "sparsity_vs_accuracy_optimized.png"))
        plt.close(fig5)
