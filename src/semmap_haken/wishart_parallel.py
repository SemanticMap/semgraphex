"""Ordered CPU process workers for GIL-bound Wishart dictionary phases.

ProcessPoolExecutor uses spawn explicitly, even if CUDA was initialized in the
main process. Python objects entering a worker are immutable/read-only for the
duration of one dictionary level. All registry mutations stay in the parent.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from collections.abc import Sequence
from typing import Any

from scipy import sparse

from .graph_dictionary import GraphDictionary, candidate_fingerprint
from .wishart_metrics import EgoCandidate

_WORKER_DICTIONARY: GraphDictionary | None = None


def spawn_pool(*, workers: int, initializer: Any = None, initargs: tuple[Any, ...] = ()) -> ProcessPoolExecutor:
    """Never fork a process after PyTorch may have initialized CUDA."""
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )


def fingerprint_chunk(
    task: tuple[tuple[EgoCandidate, ...], bool],
) -> tuple[str, ...]:
    batch, boundary_sensitive = task
    return tuple(
        candidate_fingerprint(candidate, boundary_sensitive=boundary_sensitive)
        for candidate in batch
    )


def ordered_fingerprints(
    candidates: Sequence[EgoCandidate],
    *,
    boundary_sensitive: bool,
    workers: int,
    batch_size: int = 64,
) -> tuple[str, ...]:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    tasks = (
        (tuple(candidates[start:start + batch_size]), boundary_sensitive)
        for start in range(0, len(candidates), batch_size)
    )
    if workers == 1:
        return tuple(item for task in tasks for item in fingerprint_chunk(task))
    with spawn_pool(workers=workers) as pool:
        return tuple(
            item for result in pool.map(fingerprint_chunk, tasks)
            for item in result
        )


def initialize_match_worker(dictionary: GraphDictionary) -> None:
    """Load one immutable dictionary snapshot per spawned worker."""
    global _WORKER_DICTIONARY
    _WORKER_DICTIONARY = dictionary


def match_chunk(
    candidates: tuple[EgoCandidate, ...],
) -> tuple[tuple[str, tuple[int, ...]] | None, ...]:
    """Return compact matches, never expensive repeated GraphType payloads."""
    if _WORKER_DICTIONARY is None:
        raise RuntimeError("matching worker dictionary is not initialized")
    output: list[tuple[str, tuple[int, ...]] | None] = []
    for candidate in candidates:
        match = _WORKER_DICTIONARY.match_with_mapping(candidate)
        if match is None:
            output.append(None)
        else:
            output.append(
                (match.graph_type.type_id, match.prototype_to_candidate)
            )
    return tuple(output)


def feature_chunk(
    task: tuple[str, tuple[EgoCandidate, ...], dict[str, int]],
) -> sparse.csr_matrix:
    """Feature rows are independent; concatenate chunks in submission order."""
    name, candidates, options = task
    from .wishart_metrics import graphlet_features, typed_wl_features

    if name == "typed_wl":
        return typed_wl_features(
            candidates,
            iterations=options["iterations"],
            dimension=options["dimension"],
            workers=1,
        )
    if name == "graphlet":
        return graphlet_features(
            candidates,
            graphlet_size=options["graphlet_size"],
            samples=options["samples"],
            dimension=options["dimension"],
            seed=options["seed"],
            workers=1,
        )
    raise ValueError(f"unknown feature job: {name}")


def ordered_feature_matrix(
    name: str,
    candidates: Sequence[EgoCandidate],
    *,
    workers: int,
    options: dict[str, int],
    batch_size: int = 64,
) -> sparse.csr_matrix:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    tasks = (
        (name, tuple(candidates[start:start + batch_size]), options)
        for start in range(0, len(candidates), batch_size)
    )
    if workers == 1:
        chunks = [feature_chunk(task) for task in tasks]
    else:
        with spawn_pool(workers=workers) as pool:
            chunks = list(pool.map(feature_chunk, tasks))
    return sparse.vstack(chunks, format="csr") if chunks else sparse.csr_matrix(
        (0, options["dimension"]), dtype=float,
    )
