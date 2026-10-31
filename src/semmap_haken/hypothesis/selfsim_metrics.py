"""Four independent self-similarity diagnostics for typed nSBM hierarchies.

No N x N dense allocation. Units: directed assertion records; symmetrization
is used ONLY for the explicitly labelled spectral and box-covering probes.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
from math import log
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score

from .graph import TypedGraph


ARCS = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1))
TRIAD_CLASSES = tuple(sorted({
    min(sum(((mask >> ARCS.index((p[i], p[j]))) & 1) << k
            for k, (i, j) in enumerate(ARCS))
        for p in permutations(range(3)))
    for mask in range(64)
}))


def validate_levels(levels, n: int) -> list[np.ndarray]:
    """All levels map *original* vertex indices to consecutively nested IDs."""
    out = []
    for labels in levels:
        arr = np.asarray(labels, dtype=np.int32)
        if arr.shape != (n,) or np.any(arr < 0):
            raise ValueError("partition must label every original vertex")
        _, arr = np.unique(arr, return_inverse=True)
        out.append(arr.astype(np.int32))
    if not out:
        raise ValueError("at least one nSBM level required")
    for fine, coarse in zip(out, out[1:]):
        for block in np.unique(fine):
            if np.unique(coarse[fine == block]).size != 1:
                raise ValueError("partitions are not nested")
        if len(np.unique(coarse)) > len(np.unique(fine)):
            raise ValueError("coarser partition increases the block count")
    return [x for i, x in enumerate(out)
            if i == 0 or not np.array_equal(x, out[i-1])]


def typed_counts(graph: TypedGraph, labels: np.ndarray) -> np.ndarray:
    """Unit assertion multiplicities, preserving relation and direction."""
    b = int(labels.max()) + 1
    flat = (graph.relation.astype(np.int64) * b * b +
            labels[graph.source].astype(np.int64) * b +
            labels[graph.target])
    return np.bincount(flat, minlength=len(graph.relation_names)*b*b).reshape(
        len(graph.relation_names), b, b
    )


def support(graph: TypedGraph) -> sparse.csr_matrix:
    n = graph.vertex_count
    A = sparse.coo_matrix(
        (np.ones(graph.edge_count, dtype=np.int8),
         (graph.source, graph.target)), shape=(n, n)
    ).tocsr()
    A.data[:] = 1
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _undirected(directed: sparse.csr_matrix) -> sparse.csr_matrix:
    B = (directed + directed.T).tocsr()
    B.data[:] = 1
    B.setdiag(0)
    B.eliminate_zeros()
    return B


def _canonical_triad(mask: int) -> int:
    return min(
        sum(((mask >> ARCS.index((p[i], p[j]))) & 1) << k
            for k, (i, j) in enumerate(ARCS))
        for p in permutations(range(3))
    )


TRIAD_INDEX = {v: i for i, v in enumerate(TRIAD_CLASSES)}
TRIAD_LOOKUP = [_canonical_triad(m) for m in range(64)]


def motif_profile(directed: sparse.csr_matrix, *, seed: int, samples: int = 3000) -> dict:
    """Sample connected ordered wedges at every scale with identical protocol."""
    n = directed.shape[0]
    U = _undirected(directed)
    degrees = np.diff(U.indptr)
    possible = degrees.astype(np.float64) * (degrees - 1) / 2
    if possible.sum() <= 0:
        return {"wedge_count": 0, "sample_count": 0, "motif_distribution": [],
                "clustering": None, "node_count": n}
    rng = np.random.default_rng(seed)
    roots = rng.choice(n, size=samples, p=possible / possible.sum())
    row_sets = [set(map(int, directed.indices[directed.indptr[i]:directed.indptr[i+1]]))
                for i in range(n)]
    codes = Counter()
    close = 0
    for root in roots:
        neighbors = U.indices[U.indptr[root]:U.indptr[root+1]]
        other = rng.choice(neighbors, size=2, replace=False)
        triple = (int(root), int(other[0]), int(other[1]))
        close += int(triple[2] in U.indices[U.indptr[triple[1]]:U.indptr[triple[1]+1]])
        mask = sum((b in row_sets[a]) << k
                   for k, (i, j) in enumerate(ARCS)
                   for a, b in [(triple[i], triple[j])])
        codes[TRIAD_LOOKUP[mask]] += 1
    return {
        "node_count": n, "wedge_count": int(possible.sum()),
        "sample_count": samples,
        "motif_distribution": [codes[c] / samples for c in TRIAD_CLASSES],
        "clustering": close / samples,
    }


def structural(graph: TypedGraph, levels: list[np.ndarray], *, seed: int,
               samples: int) -> dict:
    rows = []
    for i, labels in enumerate([np.arange(graph.vertex_count)] + levels):
        counts = typed_counts(graph, labels)
        C = counts.sum(axis=0)
        b = C.shape[0]
        adj = sparse.csr_matrix(C > 0, dtype=np.int8)
        adj.setdiag(0)
        adj.eliminate_zeros()
        prof = motif_profile(adj, seed=seed, samples=samples)
        row = {"scale": "original" if i == 0 else f"blocks_{b}",
               "blocks": b, "internal_assertions": int(np.trace(C)),
               "internal_fraction": float(np.trace(C) / max(1, graph.edge_count)),
               "edge_density_no_diagonal": float(adj.nnz / max(1, b*(b-1))),
               "motifs": prof}
        rows.append(row)
    # Wedge-weighted sampling is a different measure on different graphs.
    distances = []
    for a, b in zip(rows, rows[1:]):
        x, y = a["motifs"]["motif_distribution"], b["motifs"]["motif_distribution"]
        if x and y:
            distances.append({"from": a["scale"], "to": b["scale"],
                              "js_divergence": float(jensenshannon(x, y, base=2)**2)})
    return {"levels": rows, "adjacent_motif_js": distances,
            "warning": "Wedge-conditioned profiles are not independent of density; 3-block motifs are low-powered."}


def _pair_rates(C: np.ndarray, counts: np.ndarray, edge_totals: np.ndarray, n: int) -> np.ndarray:
    """Relation-wise intensity / global relation density, preserving block masses."""
    potential = np.outer(counts, counts).astype(np.float64)
    global_density = np.maximum(edge_totals.astype(float), 1) / (n*n)
    return (C / np.maximum(potential, 1)[None, :, :]) / global_density[:, None, None]


def _js_hist(x: np.ndarray, y: np.ndarray) -> float:
    bins = np.linspace(0, np.log1p(200), 26)
    a = np.histogram(np.clip(x, bins[0], bins[-1]), bins=bins)[0].astype(float) + 1e-6
    b = np.histogram(np.clip(y, bins[0], bins[-1]), bins=bins)[0].astype(float) + 1e-6
    return float(jensenshannon(a, b, base=2)**2)


def graphon(graph: TypedGraph, levels: list[np.ndarray], *, seed: int,
           pairs: int = 12000, test_fraction: float = .2) -> dict:
    """Sparse typed step-kernel diagnostics plus edge-pair held-out AUC.

    The AUC is conditional on already-fitted nSBM partitions: when partitions
    were learned on the full graph, partition leakage makes it EXPLORATORY.
    """
    rng = np.random.default_rng(seed)
    n, r = graph.vertex_count, len(graph.relation_names)
    nE = graph.edge_count
    total_rel = np.bincount(graph.relation, minlength=r)
    pair_u = rng.integers(n, size=pairs)
    pair_v = rng.integers(n, size=pairs)
    pair_r = rng.choice(r, size=pairs, p=total_rel/total_rel.sum())
    logs = []
    level_rows = []

    keys = ((graph.relation.astype(np.int64)*n + graph.source)*n + graph.target)
    _, group = np.unique(keys, return_inverse=True)
    group_is_train = rng.random(int(group.max())+1) >= test_fraction
    training = group_is_train[group]
    test_ids = np.flatnonzero(~training)
    # Give each tested directed triple one positive even with repeated records.
    _, ix = np.unique(keys[test_ids], return_index=True)
    test_ids = test_ids[np.sort(ix)]
    if len(test_ids) > 5000:
        test_ids = rng.choice(test_ids, size=5000, replace=False)
    present = set(map(int, keys))
    neg_src, neg_dst, neg_rel = [], [], []
    for idx in test_ids:
        rel = int(graph.relation[idx])
        for attempt in range(10000):
            u, v = map(int, rng.integers(n, size=2))
            if ((rel*n + u)*n+v) not in present:
                neg_src.append(u); neg_dst.append(v); neg_rel.append(rel)
                break
        else:
            raise RuntimeError("could not sample absent relation triple")
    eu, ev = graph.source[test_ids], graph.target[test_ids]
    er = graph.relation[test_ids]
    neg_src, neg_dst, neg_rel = map(np.asarray, (neg_src, neg_dst, neg_rel))
    for labels in levels:
        b = int(labels.max()) + 1
        masses = np.bincount(labels, minlength=b)
        C = typed_counts(graph, labels)
        rates = _pair_rates(C, masses, total_rel, n)
        sampled = np.log1p(rates[pair_r, labels[pair_u], labels[pair_v]])
        logs.append(sampled)
        train_graph = graph.edge_subset(training)
        Ctrain = typed_counts(train_graph, labels)
        alpha = 16.0
        prior = np.bincount(train_graph.relation, minlength=r)/(n*n)
        density = (Ctrain + alpha*prior[:, None, None]) / (
            np.outer(masses, masses)[None,:,:] + alpha
        )
        score_pos = density[er, labels[eu], labels[ev]]
        score_neg = density[neg_rel, labels[neg_src], labels[neg_dst]]
        try:
            auc = float(roc_auc_score(
                np.r_[np.ones(len(score_pos)), np.zeros(len(score_neg))],
                np.r_[score_pos, score_neg]
            ))
        except ValueError:
            auc = None
        level_rows.append({
            "blocks": b, "block_masses": masses.tolist(),
            "typed_edge_totals": C.sum(axis=(1,2)).tolist(),
            "holdout_auc_conditional": auc,
            "training_assertions": int(training.sum()),
            "heldout_unique_positive_triples": len(test_ids),
        })
    adjacent = []
    for i in range(len(levels)-1):
        adjacent.append({
            "fine_blocks": level_rows[i]["blocks"],
            "coarse_blocks": level_rows[i+1]["blocks"],
            "log_intensity_js": _js_hist(logs[i], logs[i+1]),
            "log_intensity_rmse": float(np.sqrt(np.mean((logs[i]-logs[i+1])**2))),
        })
    return {"levels": level_rows, "adjacent_kernel": adjacent,
            "pairs_sampled": pairs, "holdout_semantics":
            "conditional diagnostic ONLY; full-graph nSBM partition leaks held-out edges. "
            "Independent predictive claims require nSBM re-fitting on train."}


def _symmetric_coupling(graph: TypedGraph):
    n = graph.vertex_count
    data = np.ones(graph.edge_count, dtype=np.float64)
    A = sparse.coo_matrix((data, (graph.source, graph.target)),
                           shape=(n,n)).tocsr()
    W = (A + A.T + sparse.eye(n, format="csr")).tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()
    return W, d


def _eigen(W: sparse.csr_matrix, deg: np.ndarray, *, n_modes: int, teleport: float):
    n = len(deg)
    D = 1.0/np.sqrt(deg)
    S = sparse.diags(D) @ W @ sparse.diags(D)
    q = np.sqrt(deg)/np.sqrt(deg.sum())
    def multiply(x):
        return (1-teleport)*(S@x) + teleport*q*np.dot(q,x)
    k = min(n, n_modes+1)
    if n <= max(16,k+1):
        dense = (1-teleport)*S.toarray() + teleport*np.outer(q,q)
        vals, vec = np.linalg.eigh(dense)
        return vals[-k:][::-1], vec[:, -k:][:, ::-1]
    op = LinearOperator((n,n), matvec=multiply, dtype=float)
    vals, vec = eigsh(op, k=k, which="LA", tol=1e-5, maxiter=8000)
    ix = np.argsort(vals)[::-1]
    return vals[ix], vec[:, ix]


def dynamics(graph: TypedGraph, levels: list[np.ndarray], *,
             slow_modes: int = 4, teleport: float = .03,
             alpha: float = 1.0, beta: float = .95) -> dict:
    """Linear Haken-inspired relaxation on symmetric normalized diffusion.

    The trivial stationary mode is EXCLUDED. This does not claim nonlinear
    Haken slaving or a directed/relation-aware generator.
    """
    if not 0 < teleport < 1 or slow_modes < 1:
        raise ValueError("invalid dynamics parameters")
    W, d = _symmetric_coupling(graph)
    val0, vec0 = _eigen(W, d, n_modes=slow_modes, teleport=teleport)
    n = graph.vertex_count
    rows = []
    for labels in levels:
        b = int(labels.max())+1
        # Aggregation of the SAME coupling, including diagonal self loops.
        coo = W.tocoo()
        coarse = sparse.coo_matrix(
            (coo.data, (labels[coo.row], labels[coo.col])), shape=(b,b)
        ).tocsr()
        deg = np.asarray(coarse.sum(axis=1)).ravel()
        vals, vectors = _eigen(coarse, deg, n_modes=slow_modes, teleport=teleport)
        used = min(slow_modes, len(vals)-1, len(val0)-1)
        if used:
            # L = D_f^(1/2) M D_b^(-1/2); columns of L are orthonormal.
            lifted = (np.sqrt(d) / np.sqrt(deg[labels]))[:,None] * vectors[labels,1:used+1]
            original = vec0[:,1:used+1]
            cosines = np.linalg.svd(original.T @ lifted, compute_uv=False)
            rms = float(np.sqrt(np.mean((val0[1:used+1]-vals[1:used+1])**2)))
            overlap = float(np.mean(cosines**2))
        else:
            rms, overlap = None, None
        mu = -alpha + beta*vals[1:used+1]
        rows.append({
            "blocks":b, "compared_nontrivial_modes":used,
            "nontrivial_eigenvalues": vals[1:used+1].tolist(),
            "decay_rates": mu.tolist(),
            "spectral_rmse": rms, "slow_subspace_overlap": overlap,
            "original_spectral_reference": val0[1:used+1].tolist(),
        })
    return {"levels":rows, "alpha":alpha, "beta":beta, "teleport":teleport,
            "coupling":"unit multiplicity, undirected symmetric normalized + teleport",
            "limitations":"linearized Haken proxy; compare subspaces not raw eigenvectors; "
            "no claim of nonlinear slaving or preserved relation-aware flows."}


def box_covering(graph: TypedGraph, *, radii=range(6),
                 seed: int=42, repeats: int=3,
                 q_values=(-2.,-1.,0.,1.,2.)) -> dict:
    """Greedy burning of radius-r balls; rigorous diameter <= 2r for each box.

    Not minimum box covering. N_B is a heuristic UPPER bound. No power-law
    conclusion without independent scale range, nulls and fit comparison.
    """
    A = _undirected(support(graph))
    n = graph.vertex_count
    degree = np.diff(A.indptr)
    radii = tuple(sorted(set(int(r) for r in radii)))
    if any(r < 0 for r in radii) or repeats < 1:
        raise ValueError("radii >=0 and repeats >=1 required")
    rows = []
    for i in range(repeats):
        rng = np.random.default_rng(seed + 11003*i)
        noisy = rng.uniform(.45,1.55,size=n)
        order = np.argsort(-degree*noisy, kind="stable")
        for radius in radii:
            visited = np.zeros(n, dtype=np.int32)
            covered = np.zeros(n, dtype=bool)
            stamp = 0
            mass = []
            for center in order:
                if covered[center]:
                    continue
                stamp += 1
                visited[center] = stamp
                ball = [int(center)]
                head = 0
                for depth in range(radius):
                    last = len(ball)
                    while head < last:
                        v = ball[head]
                        head += 1
                        for w in A.indices[A.indptr[v]:A.indptr[v+1]]:
                            if visited[w] != stamp:
                                visited[w] = stamp
                                ball.append(int(w))
                new = np.asarray(ball, dtype=np.int64)
                fresh = int(np.count_nonzero(~covered[new]))
                if not fresh:
                    raise AssertionError("uncovered center yielded empty ball")
                covered[new] = True
                mass.append(fresh)
            if sum(mass) != n:
                raise AssertionError("box covering failed to cover every vertex")
            mu = np.asarray(mass)/n
            z = {str(q):float(np.sum(mu**q)) for q in q_values}
            entropy = float(-np.sum(mu*np.log(mu)))
            rows.append({
                "seed":seed+11003*i, "radius":radius, "box_diameter_bound":2*radius,
                "box_scale":2*radius+1, "box_count":len(mass),
                "box_mass_entropy":entropy, "Z_q":z,
                "largest_box_mass":float(mu.max()),
            })
    # Fit on replicate-mean log-observables; avoid pseudo-replication.
    grouped = {r: [row for row in rows if row["radius"] == r] for r in radii}
    x = np.log(np.array([2*r+1 for r in radii], dtype=float))
    x_raw = np.array([2*r+1 for r in radii], dtype=float)
    y = np.array([np.mean(np.log([v["box_count"] for v in grouped[r]]))
                  for r in radii])
    def fit(x, y):
        if len(x) < 3:
            return {"slope":None,"r2":None,"aic":None}
        coefs = np.polyfit(x,y,1)
        residual = y-np.polyval(coefs,x)
        sse = max(float(np.dot(residual,residual)), 1e-15)
        total = float(np.sum((y-y.mean())**2))
        return {"slope":float(coefs[0]), "r2":float(1-sse/max(total,1e-15)),
                "aic":float(len(x)*np.log(sse/len(x))+4)}
    power=fit(x,y)
    exponential=fit(x_raw,y)
    dimensions = {}
    for q in q_values:
        if q == 1:
            yy = np.array([np.mean([v["box_mass_entropy"] for v in grouped[r]])
                           for r in radii])
            f = fit(x,yy)
            dimensions[str(q)]={"Dq":-f["slope"] if f["slope"] is not None else None,
                                "entropy_regression":f}
        else:
            yy = np.array([np.mean(np.log([v["Z_q"][str(q)] for v in grouped[r]]))
                           for r in radii])
            f=fit(x, yy)
            dimensions[str(q)]={"Dq": f["slope"]/(q-1)
                               if f["slope"] is not None else None,
                               "tau":f["slope"], "scaling_regression":f}
    return {
        "rows":rows, "heuristic_box_dimension":-power["slope"] if power["slope"] is not None else None,
        "power_logfit":power, "exponential_logfit":exponential,
        "delta_aic_exponential_minus_power":
            exponential["aic"]-power["aic"] if power["aic"] is not None else None,
        "multifractal_generalized_dimensions":dimensions,
        "warning":"Radius-ball greedy covering upper-bounds minimum N_B. "
        "Power-law fits on 3-6 scales are exploratory, not fractality proof. "
        "Graph distances ignore direction/relation; q=1 Shannon entropy fit."}
