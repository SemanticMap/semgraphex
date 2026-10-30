"""Independent recovery/stability diagnostics for nested block partitions."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score


def _informative_levels(levels: list[list[int]], n: int) -> list[tuple[int, np.ndarray]]:
    result: list[tuple[int, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for index, values in enumerate(levels):
        labels = np.asarray(values, dtype=np.int64)
        if labels.shape != (n,):
            raise ValueError("all levels must refer to exactly the same original vertices")
        _, compact = np.unique(labels, return_inverse=True)
        blocks = int(np.unique(compact).size)
        if not 1 < blocks < n:  # skip root and identity partitions
            continue
        signature = tuple(compact.tolist())
        if signature not in seen:
            result.append((index, compact))
            seen.add(signature)
    return result


def hierarchy_recovery(
    inferred_levels: list[list[int]], truth: dict[str, object],
) -> dict[str, object]:
    """Best adjusted Rand index across informative levels against planted truth.

    Assumes the original vertex order is identical to \`truth\`; never match
    levels only by ordinal index. ARI=1 implies identical partitions modulo
    block-ID permutations, ARI~0 resembles random partitions of these sizes.
    """
    if "fine" not in truth or "coarse" not in truth:
        raise ValueError("truth must contain fine and coarse memberships")
    fine = np.asarray(truth["fine"], dtype=np.int64)
    coarse = np.asarray(truth["coarse"], dtype=np.int64)
    if fine.ndim != 1 or coarse.shape != fine.shape:
        raise ValueError("truth labels must have equal one-dimensional length")
    levels = _informative_levels(inferred_levels, len(fine))

    def match(target: np.ndarray) -> dict[str, object]:
        if not levels:
            return {"ari": None, "level": None, "blocks": None}
        scored = [(float(adjusted_rand_score(target, labels)), idx, labels)
                  for idx, labels in levels]
        best, index, labels = max(scored, key=lambda x: x[0])
        return {"ari": best, "level": index, "blocks": int(np.unique(labels).size)}

    return {
        "fine": match(fine),
        "coarse": match(coarse),
        "truth_fine_blocks": int(np.unique(fine).size),
        "truth_coarse_blocks": int(np.unique(coarse).size),
        "levels_evaluated": len(levels),
        "metric": "adjusted_rand_score",
        "truth_source": "planted synthetic graph only",
    }


def hierarchy_restart_stability(
    hierarchies: list[list[list[int]]],
) -> dict[str, object]:
    """ARI between *independently fitted* hierarchies on the same graph.

    Compare finest informative levels and separately the best matching
    informative scales. Same block counts alone are not evidence of stability.
    """
    if len(hierarchies) < 2:
        return {"pair_count": 0, "finest_ari_mean": None,
                "best_scale_ari_mean": None, "pairs": []}
    if not hierarchies[0]:
        raise ValueError("empty hierarchy")
    n = len(hierarchies[0][0])
    valid = [_informative_levels(h, n) for h in hierarchies]
    pairs = []
    for i, j in combinations(range(len(hierarchies)), 2):
        if not valid[i] or not valid[j]:
            pairs.append({"left": i, "right": j, "finest_ari": None,
                          "best_scale_ari": None})
            continue
        finest = float(adjusted_rand_score(valid[i][0][1], valid[j][0][1]))
        aligned = max(
            float(adjusted_rand_score(a, b))
            for _, a in valid[i] for _, b in valid[j]
        )
        pairs.append({"left": i, "right": j,
                      "finest_ari": finest, "best_scale_ari": aligned})

    def average(name: str):
        vals = [float(x[name]) for x in pairs if x[name] is not None]
        return float(np.mean(vals)) if vals else None

    return {"pair_count": len(pairs), "finest_ari_mean": average("finest_ari"),
            "best_scale_ari_mean": average("best_scale_ari"),
            "levels_by_restart": [len(x) for x in valid],
            "pairs": pairs, "metric": "adjusted_rand_score"}
