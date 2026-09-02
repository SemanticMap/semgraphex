from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import os

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False
from annoy import AnnoyIndex

class VectorIndex:
    def __init__(self, dim: int, metric: str = 'cosine'):
        self.dim = dim
        self.metric = metric
        self._built = False
        self._use_faiss = _HAS_FAISS and metric == 'cosine'
        self._ids: List[str] = []
        if self._use_faiss:
            # Use inner-product index for cosine after normalization
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = AnnoyIndex(dim, metric if metric in {'angular','euclidean'} else 'angular')

    def add(self, vec_id: str, vec: np.ndarray):
        if vec.shape[0] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim} got {vec.shape[0]}")
        if self.metric == 'cosine':  # normalize
            nrm = np.linalg.norm(vec) + 1e-12
            vec = vec / nrm
        if self._use_faiss:
            self.index.add(vec.reshape(1,-1).astype('float32'))
        else:
            self.index.add_item(len(self._ids), vec.tolist())
        self._ids.append(vec_id)

    def build(self, n_trees: int = 50):
        if self._use_faiss:
            self._built = True
        else:
            self.index.build(n_trees)
            self._built = True

    def search(self, vec: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        if not self._built:
            raise RuntimeError("Index not built")
        if self.metric == 'cosine':
            nrm = np.linalg.norm(vec) + 1e-12
            vec = vec / nrm
        if self._use_faiss:
            # Inner product equals cosine because of normalization
            D,I = self.index.search(vec.reshape(1,-1).astype('float32'), k)
            results = []
            for sim, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                results.append((self._ids[idx], float(sim)))
            return results
        else:
            idxs, dists = self.index.get_nns_by_vector(vec.tolist(), k, include_distances=True)
            # Annoy 'angular' returns distance ~ 2*(1-cosine)
            results = []
            for i, d in zip(idxs, dists):
                sim = 1 - d/2 if self.metric=='cosine' else -d
                results.append((self._ids[i], float(sim)))
            return results
