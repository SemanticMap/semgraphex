from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import networkx as nx
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

@dataclass
class GraphonRepresentation:
    concept: str
    W: np.ndarray  # estimated graphon (discretized grid)
    S: np.ndarray  # node-level signal (degree-based), aligned with ordering used for W
    nodes: list    # node labels in order
    meta: Dict[str, Any]

    def descriptor(self, max_eigs: int = 8, sample_blocks: int = 16) -> np.ndarray:
        A = self.W
        # spectral features
        vals = np.linalg.svd(A, compute_uv=False)
        eig_feats = vals[:max_eigs]
        if len(eig_feats) < max_eigs:
            eig_feats = np.pad(eig_feats, (0, max_eigs-len(eig_feats)))
        # density & degree stats
        density = A.mean()
        deg = self.S
        stats = np.array([
            density,
            deg.mean() if len(deg)>0 else 0,
            deg.std() if len(deg)>0 else 0,
            np.percentile(deg, 90) if len(deg)>0 else 0,
        ])
        # sample blocks by downsampling grid
        block_side = int(np.sqrt(sample_blocks))
        if block_side > 0:
            ds = downsample(A, block_side)
            block_vec = ds.flatten()
        else:
            block_vec = np.array([])
        return np.concatenate([eig_feats, stats, block_vec])


def downsample(A: np.ndarray, blocks: int) -> np.ndarray:
    n = A.shape[0]
    size = n // blocks if n >= blocks else 1
    out = np.zeros((blocks, blocks))
    for i in range(blocks):
        for j in range(blocks):
            sub = A[i*size:(i+1)*size, j*size:(j+1)*size]
            if sub.size == 0:
                continue
            out[i,j] = sub.mean()
    return out


def estimate_graphon(G: nx.Graph, grid_size: int = 64, smooth_sigma: float = 1.0) -> GraphonRepresentation:
    if G.number_of_nodes() == 0:
        return GraphonRepresentation(concept="", W=np.zeros((1,1)), S=np.zeros(1), nodes=[], meta={"empty": True})
    # Order nodes by degree (heuristic to impose structure)
    degrees = dict(G.degree(weight='weight'))
    ordered = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    idx = {n:i for i,n in enumerate(ordered)}
    n = len(ordered)
    A = np.zeros((n,n))
    for u,v,d in G.edges(data=True):
        i,j = idx[u], idx[v]
        w = d.get('weight',1.0)
        A[i,j] = A[j,i] = w
    # normalize weights
    if A.max() > 0:
        A = A / A.max()
    # resize to grid via simple interpolation (repeat / crop)
    W = resize_to_grid(A, grid_size)
    if smooth_sigma > 0:
        W = gaussian_filter(W, sigma=smooth_sigma)
    deg_vec = np.array([degrees[n] for n in ordered], dtype=float)
    if deg_vec.max() > 0:
        deg_vec = deg_vec / deg_vec.max()
    return GraphonRepresentation(concept="", W=W, S=deg_vec, nodes=ordered, meta={"original_size": n})


def resize_to_grid(A: np.ndarray, grid: int) -> np.ndarray:
    n = A.shape[0]
    if n == grid:
        return A.copy()
    # simple nearest-neighbor expansion/compression
    xs = (np.linspace(0, n-1, grid)).astype(int)
    ys = (np.linspace(0, n-1, grid)).astype(int)
    return A[np.ix_(xs, ys)]
