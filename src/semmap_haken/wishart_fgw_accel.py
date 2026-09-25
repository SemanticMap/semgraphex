"""GPU-first, bounded approximate FGW kNN for canonical graph types.

The discovery candidate_limit (up to 25k ego candidates) is NOT the number
of FGW transport problems: FGW runs over supported canonical dictionary types.
For > exact_types, a cheap graph descriptor proposes nearest-neighbor pairs,
which are re-ranked using batched entropic FGW on CUDA. These are approximate
FGW neighbors, not exact all-pairs FGW or the CPU POT conditional-gradient
solver. All computations involving actual FGW transport stay on the selected
CUDA device. CPU threads prepare structural supports and typed features.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def _descriptor(candidate, structure: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Cheap permutation-invariant descriptor, used ONLY to propose FGW pairs."""
    deg = np.diff(candidate.adjacency.indptr)
    degree_hist = np.array([
        np.count_nonzero(deg == 0), np.count_nonzero(deg == 1),
        np.count_nonzero(deg == 2), np.count_nonzero((deg >= 3) & (deg <= 4)),
        np.count_nonzero(deg >= 5),
    ], dtype=np.float64) / max(1, len(deg))
    upper = structure[np.triu_indices(len(structure), k=1)]
    shape = np.array([
        len(deg) / 48.0,
        candidate.adjacency.nnz / max(1.0, len(deg) * max(1, len(deg) - 1)),
        float(upper.mean()) if upper.size else 0.0,
        float(upper.std()) if upper.size else 0.0,
    ])
    vector = np.concatenate((features.mean(axis=0), degree_hist, shape))
    vector[-4] = max(vector[-4], 1e-8)
    return vector


def _torch_fgw_batch(
    left: Sequence[tuple[np.ndarray, np.ndarray]],
    right: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float,
    epsilon: float,
    outer_iterations: int,
    sinkhorn_iterations: int,
    device: str,
) -> np.ndarray:
    """Batched approximate squared-loss FGW with log-domain Sinkhorn.

    Tuple is (shortest-path cost matrix, directed-relation feature matrix).
    Computes the UNREGULARIZED FGW objective at an entropically regularized
    coupling. No value is returned merely from the regularization penalty.
    """
    import torch

    if not left or len(left) != len(right):
        raise ValueError("FGW batch must have equal nonempty left/right lists")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA FGW requested but torch.cuda.is_available() is false")
    batch = len(left)
    rank = max(max(a[0].shape[0], b[0].shape[0]) for a, b in zip(left, right))
    width = max(a[1].shape[1] for a in left)
    dtype = torch.float32 if device == "cuda" else torch.float64
    c1 = np.zeros((batch, rank, rank), dtype=np.float64)
    c2 = np.zeros_like(c1)
    f1 = np.zeros((batch, rank, width), dtype=np.float64)
    f2 = np.zeros_like(f1)
    p = np.zeros((batch, rank), dtype=np.float64)
    q = np.zeros_like(p)
    for row, ((a, fa), (b, fb)) in enumerate(zip(left, right)):
        na, nb = len(a), len(b)
        c1[row, :na, :na] = a
        c2[row, :nb, :nb] = b
        f1[row, :na, :] = fa
        f2[row, :nb, :] = fb
        p[row, :na] = 1.0 / na
        q[row, :nb] = 1.0 / nb

    with torch.no_grad():
        C1 = torch.as_tensor(c1, dtype=dtype, device=device)
        C2 = torch.as_tensor(c2, dtype=dtype, device=device)
        F1 = torch.as_tensor(f1, dtype=dtype, device=device)
        F2 = torch.as_tensor(f2, dtype=dtype, device=device)
        P = torch.as_tensor(p, dtype=dtype, device=device)
        Q = torch.as_tensor(q, dtype=dtype, device=device)
        mask = (P[:, :, None] > 0) & (Q[:, None, :] > 0)
        M = torch.cdist(F1, F2).square()
        M = torch.where(mask, M, torch.zeros_like(M))
        C1_sq_p = torch.bmm(C1.square(), P.unsqueeze(-1)).squeeze(-1)
        C2_sq_q = torch.bmm(C2.square(), Q.unsqueeze(-1)).squeeze(-1)
        const = C1_sq_p[:, :, None] + C2_sq_q[:, None, :]
        T = P[:, :, None] * Q[:, None, :]
        logp = torch.where(P > 0, P.clamp_min(1e-30).log(), -1e9)
        logq = torch.where(Q > 0, Q.clamp_min(1e-30).log(), -1e9)
        for _ in range(outer_iterations):
            cross = torch.bmm(torch.bmm(C1, T), C2.transpose(1, 2))
            structure_cost = (const - 2.0 * cross).clamp_min(0.0)
            cost = (1.0 - alpha) * M + alpha * structure_cost
            logk = torch.where(mask, -cost / epsilon, -1e9)
            u = torch.zeros_like(P)
            v = torch.zeros_like(Q)
            for _ in range(sinkhorn_iterations):
                u = logp - torch.logsumexp(logk + v[:, None, :], dim=2)
                v = logq - torch.logsumexp(logk + u[:, :, None], dim=1)
            T = torch.exp(logk + u[:, :, None] + v[:, None, :])
            T = torch.where(mask, T, torch.zeros_like(T))
        marginal_error = torch.maximum(
            (T.sum(dim=2) - P).abs().amax(),
            (T.sum(dim=1) - Q).abs().amax(),
        )
        if not torch.isfinite(marginal_error) or float(marginal_error.item()) > 0.02:
            raise RuntimeError(
                "FGW Sinkhorn did not satisfy transport marginals; increase "
                "fgw_sinkhorn_iterations or fgw_epsilon"
            )
        cross = torch.bmm(torch.bmm(C1, T), C2.transpose(1, 2))
        structure_cost = (const - 2.0 * cross).clamp_min(0.0)
        value = ((1.0 - alpha) * M * T + alpha * structure_cost * T).sum(dim=(1, 2))
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError("FGW yielded a nonfinite transport cost")
        return value.clamp_min(0).sqrt().cpu().numpy().astype(np.float64)


def accelerated_fgw_neighbors(
    candidates,
    *,
    k: int,
    rank: int,
    max_candidates: int,
    alpha: float,
    exact_types: int,
    shortlist: int,
    epsilon: float,
    outer_iterations: int,
    sinkhorn_iterations: int,
    pair_batch_size: int,
    cache_pairs: int,
    cpu_workers: int,
    device: str,
    type_ids: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    checkpoint_hook: Callable[[dict[str, object]], None] | None = None,
):
    """Return top-k within an exact-small or descriptor-shortlisted FGW pair set."""
    from .wishart_metrics import (
        NeighborGraph, _knn_from_features, _node_relation_features,
        _transport_support,
    )
    from .wishart_gpu import cuda_cosine_neighbors

    n = len(candidates)
    if n > max_candidates:
        raise ValueError(
            f"supported canonical types={n} > transport_max_candidates={max_candidates}"
        )
    if n <= 1:
        return NeighborGraph(np.empty((n, 0), dtype=np.int64),
                             np.empty((n, 0), dtype=np.float64),
                             {"backend": "entropic_fgw", "candidate_count": n})
    if k < 1 or shortlist < k or rank < 1 or pair_batch_size < 1 or cache_pairs < 1:
        raise ValueError("invalid accelerated FGW k/rank/shortlist/batch limits")
    if epsilon <= 0 or outer_iterations < 1 or sinkhorn_iterations < 1:
        raise ValueError("invalid entropic FGW solver parameters")
    if device not in {"cpu", "cuda"}:
        raise ValueError("accelerated FGW device must be cpu or cuda")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for FGW but no CUDA device is available")

    relations = tuple(sorted({r for c in candidates for r in c.relation_layers}))

    def prepare(candidate):
        keep, structure = _transport_support(candidate, rank)
        features = _node_relation_features(candidate, keep, relations)
        return (structure, features), _descriptor(candidate, structure, features)

    # Sparse extraction and shortest paths are CPU tasks, independently parallel.
    if cpu_workers > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
            prepared = list(pool.map(prepare, candidates))
    else:
        prepared = [prepare(candidate) for candidate in candidates]
    supports = [item[0] for item in prepared]
    descriptors = np.asarray([item[1] for item in prepared], dtype=np.float64)

    full_pairs = n * (n - 1) // 2
    exact = n <= exact_types
    if exact:
        pairs = list(combinations(range(n), 2))
    else:
        proposed_k = min(n - 1, max(k, shortlist))
        if device == "cuda":
            near, _, _ = cuda_cosine_neighbors(
                descriptors, k=proposed_k, query_batch_size=min(128, n),
            )
        else:
            neighbors = _knn_from_features(descriptors, proposed_k, "cosine", device="cpu")
            near = neighbors.indices
        pairs = sorted({
            (min(i, int(j)), max(i, int(j)))
            for i in range(n) for j in near[i] if int(j) != i
        })
    if not pairs:
        raise RuntimeError("FGW pair proposal is empty")

    manifest = {
        "version": 1, "type_ids": list(type_ids) if type_ids is not None else None,
        "centers": [int(c.center) for c in candidates],
        "n": n, "k": k, "rank": rank, "max_candidates": max_candidates,
        "alpha": alpha, "exact_types": exact_types, "shortlist": shortlist,
        "epsilon": epsilon, "outer_iterations": outer_iterations,
        "sinkhorn_iterations": sinkhorn_iterations, "pair_batch_size": pair_batch_size,
        "cache_pairs": cache_pairs, "precision": "float32" if device == "cuda" else "float64",
        "device": device, "pairs": len(pairs),
        "proposals_sha256": hashlib.sha256(np.asarray(pairs, dtype="<i8").tobytes()).hexdigest(),
    }
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir) if cache_dir is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved.get("signature") != signature:
                raise ValueError("FGW cached pair blocks have an incompatible config/type order")
        else:
            temp = root / "manifest.tmp"
            temp.write_text(json.dumps({**manifest, "signature": signature}, indent=2), encoding="utf-8")
            os.replace(temp, manifest_path)

    neighbors: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    used_batch = pair_batch_size
    for start in range(0, len(pairs), cache_pairs):
        expected = np.asarray(pairs[start:start + cache_pairs], dtype=np.int64)
        path = root / f"pairs_{start:09d}.npz" if root is not None else None
        if path is not None and path.is_file():
            with np.load(path, allow_pickle=False) as record:
                loaded = record["pairs"]
                values = record["distances"]
            if not np.array_equal(expected, loaded) or len(values) != len(expected) or not np.isfinite(values).all():
                raise ValueError(f"invalid cached FGW block: {path}")
        else:
            values = np.empty(len(expected), dtype=np.float64)
            pos = 0
            while pos < len(expected):
                stop = min(pos + used_batch, len(expected))
                chosen = expected[pos:stop]
                try:
                    values[pos:stop] = _torch_fgw_batch(
                        [supports[int(i)] for i in chosen[:, 0]],
                        [supports[int(j)] for j in chosen[:, 1]],
                        alpha=alpha, epsilon=epsilon,
                        outer_iterations=outer_iterations,
                        sinkhorn_iterations=sinkhorn_iterations, device=device,
                    )
                    pos = stop
                except Exception as error:
                    import torch
                    if (device == "cuda"
                            and isinstance(error, torch.cuda.OutOfMemoryError)
                            and used_batch > 1):
                        used_batch = max(1, used_batch // 2)
                        continue
                    raise
            if path is not None:
                temp = root / f"pairs_{start:09d}.tmp.npz"
                np.savez_compressed(temp, pairs=expected, distances=values)
                os.replace(temp, path)
                if checkpoint_hook is not None:
                    checkpoint_hook({
                        "pairs_done": min(start + len(expected), len(pairs)),
                        "pairs_total": len(pairs), "canonical_type_count": n,
                        "device": device, "approximate": not exact,
                    })
        for (i, j), distance in zip(expected, values):
            a, b, d = int(i), int(j), float(distance)
            neighbors[a].append((d, b))
            neighbors[b].append((d, a))

    count = min(k, n - 1)
    ids = np.empty((n, count), dtype=np.int64)
    dist = np.empty((n, count), dtype=np.float64)
    for i, row in enumerate(neighbors):
        if len(row) < count:
            raise RuntimeError(f"FGW shortlist left type {i} with {len(row)} < {count} candidates")
        best = sorted(row, key=lambda item: (item[0], item[1]))[:count]
        dist[i] = [item[0] for item in best]
        ids[i] = [item[1] for item in best]
    return NeighborGraph(ids, dist, {
        "backend": "torch_batched_entropic_fgw", "device": device,
        "precision": manifest["precision"], "metric": "fgw",
        "fgw_alpha": alpha, "landmark_rank": rank,
        "entropic_epsilon": epsilon, "outer_iterations": outer_iterations,
        "sinkhorn_iterations": sinkhorn_iterations,
        "candidate_count": n, "fgw_pair_count": len(pairs),
        "full_pair_count": full_pairs, "fgw_pair_exhaustive": exact,
        "fgw_approximate": True,  # entropic OT even if pair set is exhaustive
        "pair_selection": "all_pairs" if exact else "descriptor_shortlist_union",
        "fgw_shortlist": shortlist, "gpu_pair_batch_size": used_batch,
        "cpu_workers_support_preparation": cpu_workers,
        "relations": relations,
        "feature_source": "directed relation frequency profiles (no text embeddings)",
        "structural_cost": "full-ego shortest paths between degree landmarks",
        "cache_signature": signature,
    })
