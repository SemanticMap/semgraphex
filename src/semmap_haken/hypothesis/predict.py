"""Comparable held-out relation-aware ranking, NOT posterior predictive likelihood."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .graph import TypedGraph


class BlockScorer:
    """Smoothed directed block propensities fitted ONLY from training edges.

    Uses the model's inferred finest partition. Degree correction can be
    enabled or disabled to match its original SBM family. This is an explicit
    plug-in predictive baseline, not graph-tool's integrated Bayesian posterior.
    """

    def __init__(self, train: TypedGraph, partition: np.ndarray, *, degree_corrected: bool,
                 smooth: float = 0.5):
        lab = np.asarray(partition, dtype=np.int64)
        if lab.shape != (train.vertex_count,) or np.any(lab < 0) or smooth <= 0:
            raise ValueError("invalid partition or smoothing")
        _, self.block = np.unique(lab, return_inverse=True)
        self.block_size = np.bincount(self.block)
        self.degree_corrected = degree_corrected
        self.smooth = smooth
        r_count, v_count = len(train.relation_names), train.vertex_count
        self.out_degree = np.zeros((r_count, v_count), dtype=np.int64)
        self.in_degree = np.zeros((r_count, v_count), dtype=np.int64)
        self.out_block = np.zeros((r_count, self.block_size.size))
        self.in_block = np.zeros((r_count, self.block_size.size))
        self.pairs: dict[tuple[int, int, int], int] = defaultdict(int)
        for r, u, v in zip(train.relation, train.source, train.target):
            r, u, v = int(r), int(u), int(v)
            self.out_degree[r, u] += 1
            self.in_degree[r, v] += 1
            self.out_block[r, self.block[u]] += 1
            self.in_block[r, self.block[v]] += 1
            self.pairs[(r, int(self.block[u]), int(self.block[v]))] += 1

    def scores(self, source: int, relation: int, targets: np.ndarray) -> np.ndarray:
        cand = np.asarray(targets, dtype=np.int64)
        if (source < 0 or source >= len(self.block) or relation < 0 or
            relation >= len(self.out_degree) or np.any(cand < 0) or
            np.any(cand >= len(self.block))):
            raise ValueError("source, relation, or candidate target out of bounds")
        sb = int(self.block[source])
        dst_blocks = self.block[cand]
        counts = np.fromiter(
            (self.pairs.get((relation, sb, int(b)), 0) for b in dst_blocks),
            dtype=float, count=len(cand),
        )
        if self.degree_corrected:
            source_share = ((self.out_degree[relation, source] + self.smooth) /
                            (self.out_block[relation, sb] + self.smooth * self.block_size[sb]))
            target_share = ((self.in_degree[relation, cand] + self.smooth) /
                            (self.in_block[relation, dst_blocks] +
                             self.smooth * self.block_size[dst_blocks]))
        else:
            source_share = 1.0 / self.block_size[sb]
            target_share = 1.0 / self.block_size[dst_blocks]
        return (counts + self.smooth) * source_share * target_share


def score_candidates(
    train: TypedGraph, partition: np.ndarray, source: int, relation: int,
    targets: np.ndarray, *, degree_corrected: bool,
) -> np.ndarray:
    return BlockScorer(train, partition, degree_corrected=degree_corrected).scores(
        source, relation, targets,
    )


def _negative_candidates(
    rng: np.random.Generator, vertex_count: int, forbidden: set[int], *, count: int,
) -> np.ndarray:
    """Draw without O(vertex_count) candidate arrays on ordinary sparse queries.

    Rejection sampling is cheap for sparse semantic graphs; a bounded full scan
    is used only for unusually dense source/relation neighbourhoods.
    """
    needed = min(count, max(0, vertex_count - len(forbidden)))
    selected: list[int] = []
    seen: set[int] = set()
    attempts = 0
    while len(selected) < needed and attempts < max(256, 12 * needed):
        batch = rng.integers(0, vertex_count, size=min(256, 4 * (needed - len(selected)) + 32))
        attempts += len(batch)
        for candidate in batch:
            k = int(candidate)
            if k not in forbidden and k not in seen:
                selected.append(k)
                seen.add(k)
                if len(selected) == needed:
                    break
    if len(selected) < needed:
        # Fallback supports near-complete neighborhoods without unbounded loops.
        for k in range(vertex_count):
            if k not in forbidden and k not in seen:
                selected.append(k)
                seen.add(k)
                if len(selected) == needed:
                    break
    return np.asarray(selected, dtype=np.int64)


def ranking_metrics(
    train: TypedGraph, heldout: TypedGraph, partition: np.ndarray, *,
    seed: int, negatives: int, degree_corrected: bool, known: TypedGraph | None = None,
) -> dict[str, object]:
    """Rank each unique true held-out triple against shared sampled unobserved targets.

    'known' may be the full graph ONLY to exclude other true statements from
    the *evaluation pool*. It never enters fitted propensities. Unobserved does
    not imply truly false. All compared models must receive identical seeds
    and the same heldout/query population.
    """
    if negatives < 1:
        raise ValueError("negatives must be positive")
    if heldout.node_ids != train.node_ids or heldout.relation_names != train.relation_names:
        raise ValueError("train and heldout must use the same indexing")
    if known is not None and (known.node_ids != train.node_ids or
                              known.relation_names != train.relation_names):
        raise ValueError("known graph indexing mismatch")
    scorer = BlockScorer(train, partition, degree_corrected=degree_corrected)
    rng = np.random.default_rng(seed)
    positive = defaultdict(set)
    for g in (train, heldout) if known is None else (known,):
        for u,v,r in zip(g.source, g.target, g.relation):
            positive[(int(u), int(r))].add(int(v))
    queries = sorted({(int(u), int(r), int(v))
                      for u,v,r in zip(heldout.source, heldout.target, heldout.relation)})
    reciprocal: list[float] = []
    query_sources: list[int] = []
    hits = []
    coverage = 0
    for u, r, v in queries:
        if u == v:
            continue
        forbidden = positive[(u, r)] | {u}
        other = _negative_candidates(rng, train.vertex_count, forbidden, count=negatives)
        if not other.size:
            continue
        scores = scorer.scores(u, r, np.concatenate(([v], other)))
        rank = 1.0 + float(np.count_nonzero(scores[1:] > scores[0])) + (
            0.5 * float(np.count_nonzero(scores[1:] == scores[0])))
        reciprocal.append(1. / rank)
        query_sources.append(u)
        hits.append(float(rank <= 10))
        coverage += 1
    return {
        "queries": coverage,
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        "hits_at_10": float(np.mean(hits)) if hits else None,
        "reciprocal_ranks": reciprocal,
        "query_sources": query_sources,
        "score_kind": "smoothed_block_propensity",
        "negative_sampling": "unobserved target candidates excluding all known positives",
    }
