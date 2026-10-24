"""Small seeded typed directed truth controls. Never call a fit a proof."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semmap_haken.conceptnet import Assertion
from .graph import TypedGraph


@dataclass(frozen=True)
class SyntheticTruth:
    graph: TypedGraph
    truth: dict[str, object]


def planted_typed_graph(*, seed: int, nested: bool, n_groups: int = 4,
                        per_group: int = 20) -> SyntheticTruth:
    if n_groups < 2 or per_group < 2:
        raise ValueError("need at least two groups of at least two nodes")
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    fine = np.repeat(np.arange(n_groups), per_group)
    coarse = fine // 2
    out = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if fine[i] == fine[j]:
                prob = .22
            elif nested and coarse[i] == coarse[j]:
                prob = .075
            else:
                prob = .008
            if rng.random() < prob:
                out.append(Assertion(f"/a/syn/{len(out)}",
                                     "/r/IsA" if (i+j) % 2 else "/r/RelatedTo",
                                     f"/c/en/n{i:06}", f"/c/en/n{j:06}",
                                     1., None, None, None))
    # Preserve all planted vertices, including true degree-zero controls.
    # from_assertions() would otherwise silently discard isolated vertices.
    ids = tuple(f"/c/en/n{i:06}" for i in range(n))
    index = {node: i for i, node in enumerate(ids)}
    relation_names = ("/r/IsA", "/r/RelatedTo")
    graph = TypedGraph(
        ids, relation_names,
        np.fromiter((index[a.start_uri] for a in out), dtype=np.int64),
        np.fromiter((index[a.end_uri] for a in out), dtype=np.int64),
        np.fromiter((0 if a.relation_uri == "/r/IsA" else 1 for a in out),
                    dtype=np.int32),
        np.ones(len(out), dtype=np.float64),
        tuple(a.assertion_uri for a in out),
    )
    return SyntheticTruth(
        graph,
        {"seed": seed, "nested": nested, "fine": fine.tolist(),
         "coarse": coarse.tolist(), "n_groups": n_groups, "per_group": per_group},
    )
