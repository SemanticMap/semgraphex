from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

@dataclass
class GraphexRepresentation:
    """Represents a global concept network as a simplified graphex.

    Attributes:
        concepts: List of concept labels.
        coords: (n, d) latent coordinates (scaled to [0,1]).
        W_grid: Discretized estimate of W(x,y) over [0,1]^2.
        S: Node-level signal (centrality / hubness) scaled to [0,1].
        I: List of prominent edges (concept_i, concept_j, weight).
        meta: Auxiliary metadata.
    """
    concepts: List[str]
    coords: np.ndarray
    W_grid: np.ndarray
    S: np.ndarray
    I: List[Tuple[str, str, float]]
    meta: Dict[str, Any]

    def top_hubs(self, k: int = 10) -> List[Tuple[str, float]]:
        order = np.argsort(-self.S)[:k]
        return [(self.concepts[i], float(self.S[i])) for i in order]


def build_graphex_from_concepts(
    concept_vectors: Dict[str, np.ndarray],
    dim: int = 2,
    grid_size: int = 64,
    similarity_threshold: float = 0.5,
    smoothing_sigma: float = 1.0,
) -> GraphexRepresentation:
    """Construct a graphex from concept descriptor vectors.

    Steps:
      1. Stack vectors and normalize for cosine similarity.
      2. Compute similarity matrix.
      3. Derive latent coordinates via PCA -> rescale to [0,1].
      4. Build edge list (I) with weights above threshold.
      5. Estimate W(x,y) by binning pairwise similarities into grid then smoothing.
      6. Compute S(x) as weighted degree (hubness) normalized.
    """
    if not concept_vectors:
        raise ValueError("No concept vectors provided")
    concepts = sorted(concept_vectors.keys())
    X = np.vstack([concept_vectors[c] for c in concepts])
    # normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    sim = Xn @ Xn.T  # cosine similarity
    np.fill_diagonal(sim, 0.0)

    # Latent coordinates via PCA
    if dim > X.shape[1]:  # clamp
        dim = min(dim, X.shape[1])
    pca = PCA(n_components=dim)
    coords = pca.fit_transform(Xn)
    # scale each dimension to [0,1]
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])  # ensure 2D shape for downstream grid mapping
    for d in range(coords.shape[1]):
        col = coords[:, d]
        mn, mx = col.min(), col.max()
        if mx - mn > 0:
            coords[:, d] = (col - mn) / (mx - mn)
        else:
            coords[:, d] = 0.5
    # If dim == 1 we added zero column already; if dim>2 keep first 2 for grid
    coords_for_grid = coords[:, :2]

    # Edge list I
    I: List[Tuple[str, str, float]] = []
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            w = sim[i, j]
            if w >= similarity_threshold:
                I.append((concepts[i], concepts[j], float(w)))

    # Weighted degree centrality (hubness S)
    S = sim.sum(axis=1)
    if S.max() > 0:
        S = S / S.max()

    # Estimate W_grid via binning similarities into grid based on latent coords
    W_grid = np.zeros((grid_size, grid_size))
    counts = np.zeros_like(W_grid)
    gx = np.clip((coords_for_grid[:, 0] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    gy = np.clip((coords_for_grid[:, 1] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    for i in range(n):
        for j in range(i + 1, n):
            w = sim[i, j]
            xi, yi = gx[i], gy[i]
            xj, yj = gx[j], gy[j]
            W_grid[xi, xj] += w
            W_grid[xj, xi] += w
            counts[xi, xj] += 1
            counts[xj, xi] += 1
    mask = counts > 0
    W_grid[mask] = W_grid[mask] / counts[mask]
    if smoothing_sigma > 0:
        W_grid = gaussian_filter(W_grid, sigma=smoothing_sigma)
    if W_grid.max() > 0:
        W_grid = W_grid / W_grid.max()

    meta = {
        "similarity_threshold": similarity_threshold,
        "dim": dim,
        "grid_size": grid_size,
        "num_edges": len(I),
    }
    return GraphexRepresentation(concepts=concepts, coords=coords_for_grid, W_grid=W_grid, S=S, I=I, meta=meta)
