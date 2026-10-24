"""H1 Monte Carlo significance, paired uncertainty and conservative decisions."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta


def empirical_upper_tail(observed: float, controls: list[float]) -> dict[str, object]:
    """Exact randomized-reference Monte Carlo p-value with (+1)/(N+1)."""
    arr = np.asarray(controls, dtype=float)
    if not np.isfinite(observed) or not arr.size or not np.all(np.isfinite(arr)):
        raise ValueError("observed and nonempty controls must be finite")
    n = int(arr.size)
    k = int(np.count_nonzero(arr >= observed))
    p = (k + 1) / (n + 1)
    # CP interval on the exceedance rate, distinct from the corrected p-value.
    lo = 0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {"p_value": float(p), "exceedances": k, "controls": n,
            "reference_mean": float(arr.mean()),
            "reference_sd": sd, "reference_max": float(arr.max()),
            "effect": float(observed - arr.mean()),
            "effect_z": float((observed - arr.mean()) / sd) if sd > 0 else None,
            "exceedance_rate_95ci": [lo, hi]}


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm family-wise adjustment; no optional statsmodels dependency."""
    p = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be in [0,1]")
    order = np.argsort(p)
    corrected = np.empty(len(p))
    largest = 0.
    for rank, index in enumerate(order):
        largest = max(largest, min(1., float(p[index]) * (len(p) - rank)))
        corrected[index] = largest
    return corrected.tolist()


def paired_mrr_interval(a: list[float], b: list[float], *, seed: int,
                        repetitions: int = 1000,
                        clusters: list[int] | None = None) -> dict[str, float]:
    """Paired bootstrap, clustered by source node when source IDs are supplied.

    Clustering avoids treating many same-source queries as independent draws.
    It still estimates uncertainty conditional on the particular observed graph.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size != y.size or not x.size or repetitions < 20:
        raise ValueError("requires aligned nonempty queries and at least 20 bootstrap repetitions")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("query ranking metrics must be finite")
    delta = x - y
    if clusters is None:
        index = np.arange(len(delta), dtype=np.int64)
    else:
        if len(clusters) != len(delta):
            raise ValueError("cluster labels must match the query count")
        _, index = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.bincount(index, weights=delta)
    sizes = np.bincount(index)
    k = len(sums)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions)
    for i in range(repetitions):
        draws = rng.integers(0, k, size=k)
        bootstrap[i] = sums[draws].sum() / sizes[draws].sum()
    low, high = np.quantile(bootstrap, [.025, .975])
    return {"mean_difference": float(delta.mean()), "ci_low": float(low),
            "ci_high": float(high), "queries": int(len(delta)),
            "clusters": int(k)}


def h1_verdict(
    *, mdl_gain: float, levels: int,
    p_configuration: float | None, p_flat: float | None,
    paired_mrr_delta: float | None, paired_ci_low: float | None,
    paired_ci_high: float | None = None, control_replicates: int | None = None,
    noninferiority_margin: float = 0.01, alpha: float = .05,
) -> dict[str, object]:
    """Falsifiable decision; absent validation is *not* a positive result.

    Requires positive MDL gain, >=2 nontrivial hierarchical scales, both
    control ensembles significant under Holm adjustment and independent
    held-out MRR not inferior beyond a pre-registered tolerance.
    """
    if not 0 < alpha < 1 or noninferiority_margin < 0:
        raise ValueError("invalid testing thresholds")
    if levels < 2 or mdl_gain <= 0:
        return {"status": "not_supported", "reason": "no MDL-supported nontrivial multiscale hierarchy"}
    if (p_configuration is None or p_flat is None or paired_mrr_delta is None or
        paired_ci_low is None):
        return {"status": "inconclusive", "reason": "missing calibrated controls or independent ranking"}
    # For two Holm-corrected controls, even the best possible Monte Carlo
    # result cannot reject if 2/(N+1) >= alpha.
    if control_replicates is not None and (
        control_replicates < 1 or 2.0 / (control_replicates + 1) >= alpha
    ):
        return {"status": "inconclusive", "reason": "too few null replicates for Holm-adjusted significance"}
    corrected = holm_adjust([p_configuration, p_flat])
    if any(p >= alpha for p in corrected):
        return {"status": "not_supported", "reason": "null ensembles do not reject simple explanations",
                "holm_pvalues": corrected}
    if paired_ci_low < -noninferiority_margin:
        if paired_ci_high is None or paired_ci_high >= -noninferiority_margin:
            return {"status": "inconclusive",
                    "reason": "held-out ranking uncertainty crosses the noninferiority threshold",
                    "holm_pvalues": corrected}
        return {"status": "not_supported", "reason": "held-out ranking is inferior beyond the tolerance",
                "holm_pvalues": corrected}
    return {"status": "supported", "reason": "MDL, both calibrated controls and held-out ranking agree",
            "holm_pvalues": corrected}


def nontrivial_levels(partitions: list[list[int]]) -> int:
    """Count distinct informative levels, ignoring identity and the root."""
    unique = []
    for p in partitions:
        labels = np.asarray(p, dtype=np.int64)
        n = len(labels)
        clusters = int(np.unique(labels).size)
        if clusters <= 1 or clusters >= n:
            continue
        # Canonicalize by equivalence relation, not numeric group IDs.
        _, inverse = np.unique(labels, return_inverse=True)
        members = {}
        for i, lab in enumerate(inverse):
            members.setdefault(int(lab), []).append(i)
        signature = tuple(sorted(tuple(v) for v in members.values()))
        if signature not in unique:
            unique.append(signature)
    return len(unique)
