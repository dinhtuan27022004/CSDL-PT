import numpy as np
from typing import List, Any

class SimilarityCalculator:
    """Utility class for calculating similarity matrices using different metrics"""
    
    @staticmethod
    def _raised_cosine_sim(x: np.ndarray) -> np.ndarray:
        """Applies y = (1 + cos(pi * x)) / 2 to distance x, clamped to [0, 1]"""
        x_clamped = np.clip(x, 0, 1)
        return (1.0 + np.cos(np.pi * x_clamped)) / 2.0

    @staticmethod
    def get_matrix(vectors: np.ndarray, metric: str = "cosine") -> np.ndarray:
        """Calculate N x N similarity matrix for a feature vector set"""
        n = vectors.shape[0]
        if n == 0: return np.zeros((0, 0), dtype=np.float32)

        vectors = vectors.astype(np.float32)

        if metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-7
            norm_vecs = vectors / norms
            cos_sim = np.dot(norm_vecs, norm_vecs.T)
            # dist = 1 - sim
            return SimilarityCalculator._raised_cosine_sim(1.0 - cos_sim).astype(np.float32)
        
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
            return SimilarityCalculator._raised_cosine_sim(l2 / max_dist).astype(np.float32)

        elif metric == "scalar":
            v = vectors.flatten()
            dist = np.abs(v[:, np.newaxis] - v[np.newaxis, :])
            return SimilarityCalculator._raised_cosine_sim(dist).astype(np.float32)
        
        elif metric == "sharpness":
            v = vectors.flatten()
            a = v[:, np.newaxis]
            b = v[np.newaxis, :]
            dist = np.abs(a - b) / (np.abs(a + b) + 1e-7)
            return SimilarityCalculator._raised_cosine_sim(dist).astype(np.float32)

        return np.zeros((n, n), dtype=np.float32)

    @staticmethod
    def get_discrete_matrix(values: List[Any], type: str = "category") -> np.ndarray:
        """Calculate N x N similarity matrix for discrete values (category, entities)"""
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
                        # For entities, similarity is Jaccard. 
                        # We apply the kernel to (1 - sim)
                        jaccard = intersect / max(len(set1), len(set2))
                        sim[i, j] = SimilarityCalculator._raised_cosine_sim(1.0 - jaccard)
        return sim
