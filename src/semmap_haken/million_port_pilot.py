"""Reproducible sampled ConceptNet 100k/1M prototype experiment.

CPU: bounded parallel ego extraction; spawn processes for typed VF2/WL
fingerprints; exact dictionary matching and shape VF2 stay on CPU.
CUDA-first: typed-WL feature kNN for Wishart k=4/5, only when available.
No graph contraction, full frequency scan or complete graph codec.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import resource
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .conceptnet import AssertionFilters, ParseReport, stream_assertions
from .graph_build import build_sparse_graph
from .graph_dictionary import GraphDictionary, candidate_fingerprint
from .port_factorization import snapshot
from .wishart_cluster import wishart_cluster
from .wishart_gpu import cuda_diagnostics, resolve_device
from .wishart_metrics import EgoCandidate, EgoExtractor, build_neighbor_graph, relation_layers_from_prepared

RELATIONS = ("RelatedTo", "IsA", "PartOf", "HasA", "UsedFor", "HasProperty", "CapableOf", "Causes")
VERSION = "2.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_gib() -> float:
    # On Linux ru_maxrss is KiB.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)


def _fingerprint_batch(batch: tuple[EgoCandidate, ...]) -> tuple[tuple[str, int], ...]:
    """Importable top-level process target; no CUDA initialization in workers."""
    return tuple(
        (candidate_fingerprint(candidate, boundary_sensitive=True), os.getpid())
        for candidate in batch
    )


def parallel_fingerprints(
    candidates: tuple[EgoCandidate, ...],
    *,
    workers: int,
    batch_size: int = 16,
) -> tuple[tuple[str, ...], dict]:
    """Deterministic result order and bounded process-pool work batches."""
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be positive")
    batches = (
        candidates[i:i + batch_size]
        for i in range(0, len(candidates), batch_size)
    )
    if workers == 1:
        completed = map(_fingerprint_batch, batches)
        backend = "single_process"
    else:
        # Spawn rather than fork to avoid unsafe CUDA/fork interactions.
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        )
        backend = "process_pool_spawn"
        try:
            completed = pool.map(_fingerprint_batch, batches, chunksize=1)
            flattened = [entry for batch in completed for entry in batch]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        fps = tuple(fp for fp, _ in flattened)
        return fps, {
            "fingerprint_backend": backend,
            "workers_requested": workers,
            "worker_processes_observed": len({pid for _, pid in flattened}),
        }
    flattened = [entry for batch in completed for entry in batch]
    return tuple(fp for fp, _ in flattened), {
        "fingerprint_backend": backend,
        "workers_requested": workers,
        "worker_processes_observed": len({pid for _, pid in flattened}),
    }


def wishart_diagnostics(
    dictionary: GraphDictionary,
    counts: Counter[str],
    *,
    device: str,
    workers: int,
    gpu_batch_size: int,
    seed: int,
    output_dir: Path,
) -> dict:
    """Use CUDA for true typed-WL kNN; only VF2 and feature hashing stay CPU."""
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError("device must be auto, cuda or cpu")
    diagnostics = cuda_diagnostics()
    selected = resolve_device(device)
    print(json.dumps({
        "stage": "device_selected", "requested": device,
        "selected": selected, "cuda_available": diagnostics["cuda_available"],
    }, sort_keys=True), flush=True)
    ids = tuple(sorted(counts))
    if len(ids) <= 5:
        return {"device_requested": device, "device_selected": selected, "skipped": "fewer than six types"}
    neighbors = build_neighbor_graph(
        tuple(dictionary.representative(tid) for tid in ids),
        metric="typed_wl", k=5, wl_iterations=3, feature_dim=4096,
        graphlet_size=3, graphlet_samples=256,
        transport_rank=24, transport_max_candidates=256,
        fgw_alpha=0.5, relation_js_block_size=128, seed=seed,
        device=selected, gpu_batch_size=gpu_batch_size, min_gpu_types=1,
        cpu_workers=workers,
    )
    backend = neighbors.metadata.get("backend")
    if selected == "cuda" and backend != "torch_cuda":
        raise RuntimeError(f"CUDA was selected but kNN actually used {backend!r}")
    masses = np.asarray([counts[tid] for tid in ids], dtype=np.float64)
    results: dict[str, dict] = {}
    label_arrays: dict[str, np.ndarray] = {"type_ids": np.asarray(ids, dtype=str)}
    for k in (4, 5):
        fitted = wishart_cluster(
            neighbors.indices[:, :k], neighbors.distances[:, :k],
            significance=0.7, min_cluster_size=3, sample_weights=masses,
            min_cluster_mass=10.0,
        )
        results[str(k)] = {
            "families": fitted.cluster_count,
            "noise_fraction": float(np.mean(fitted.labels < 0)),
            "zero_radius_fraction": float(np.mean(fitted.kth_radius <= 1e-8)),
            "largest_family_fraction": (
                max(fitted.cluster_sizes.values(), default=0) / len(ids)
            ),
        }
        label_arrays[f"labels_k{k}"] = fitted.labels
    np.savez_compressed(output_dir / "wishart_labels.npz", **label_arrays)
    report = {
        "device_requested": device, "device_selected": selected,
        "knn_backend": backend, "knn_metadata": neighbors.metadata,
        "cuda_available": bool(diagnostics["cuda_available"]),
        "type_count": len(ids), "families_by_k": results,
    }
    print(json.dumps({"stage": "wishart_gpu_check", **report}, sort_keys=True), flush=True)
    return report


def run_probe(
    *, dataset: Path, output_dir: Path, target_nodes: int,
    sample_centers: int, budgets: tuple[int, ...], seed: int,
    max_ego_nodes: int, workers: int, device: str = "auto",
    gpu_batch_size: int = 128, source_sha256: str | None = None,
) -> dict:
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if target_nodes < 3 or sample_centers < 2 or max_ego_nodes < 2 or workers < 1:
        raise ValueError("invalid graph or parallelism bounds")
    if gpu_batch_size < 1:
        raise ValueError("gpu_batch_size must be positive")
    if not budgets or budgets != tuple(sorted(set(budgets))) or budgets[0] < 2 or budgets[-1] > sample_centers:
        raise ValueError("budgets must be increasing, distinct, and <= sample_centers")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "COMPLETED").unlink(missing_ok=True)
    started = time.perf_counter()
    parse_report = ParseReport()
    assertions = stream_assertions(
        dataset,
        AssertionFilters(language="en", relations=frozenset(RELATIONS), min_weight=1.0),
        invalid_mode="skip_invalid", report=parse_report,
    )
    graph = build_sparse_graph(
        assertions, directed=False, weight_transform="log1p",
        component="largest", max_nodes=target_nodes, self_loop_policy="exclude",
    )
    n = int(graph.adjacency.shape[0])
    if n != target_nodes:
        raise ValueError(f"expected {target_nodes:,} nodes; got {n:,}")
    graph_seconds = round(time.perf_counter() - started, 3)
    relation_layers = relation_layers_from_prepared(graph, directed=False)
    centers = np.random.default_rng(seed).choice(
        n, size=min(sample_centers, n), replace=False
    )
    # Preserve random order: prefix budgets must not be URI-order biased.
    candidates = EgoExtractor(graph.adjacency, relation_layers).extract_centers(
        centers, radius=1, max_ego_nodes=max_ego_nodes,
        workers=workers, work_batch_size=64,
    )
    if len(candidates) < max(budgets):
        raise ValueError(
            f"only {len(candidates)} valid ego figures for budget {max(budgets)}"
        )
    extract_seconds = round(time.perf_counter() - started - graph_seconds, 3)
    fingerprints, parallel = parallel_fingerprints(
        candidates[:max(budgets)], workers=workers
    )
    if len(fingerprints) != max(budgets):
        raise AssertionError("lost candidate fingerprints")
    dictionary = GraphDictionary(boundary_sensitive=True)
    counts: Counter[str] = Counter()
    rows: list[dict] = []
    for i, (candidate, fingerprint) in enumerate(
        zip(candidates, fingerprints, strict=False), start=1
    ):
        if i > max(budgets):
            break
        graph_type = dictionary.resolve_or_create(
            candidate, level=0, fingerprint=fingerprint
        )
        counts[graph_type.type_id] += 1
        if i not in budgets:
            continue
        item = snapshot(
            {tid: entry.prototype for tid, entry in dictionary.types.items()},
            counts,
        )
        row = {
            "sample_budget": i,
            "supported_sample_types_3": sum(v >= 3 for v in counts.values()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_gib": _peak_rss_gib(),
            **item,
        }
        rows.append(row)
        (output_dir / f"budget_{i:05d}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps({"stage": "budget_complete", "nodes": n, **row}, sort_keys=True), flush=True)

    # All CPU process workers have exited before PyTorch touches CUDA.
    wishart = wishart_diagnostics(
        dictionary, counts, device=device, workers=workers,
        gpu_batch_size=gpu_batch_size, seed=seed, output_dir=output_dir,
    )
    report = {
        "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": source_sha256 or sha256_file(dataset),
        "target_nodes": target_nodes, "actual_nodes": n,
        "graph_edges": int(graph.report["edge_count"]),
        "source_assertions": int(parse_report.accepted),
        "sample_centers_requested": sample_centers,
        "candidate_figures_extracted": len(candidates),
        "budgets": list(budgets), "random_seed": seed,
        "max_ego_nodes": max_ego_nodes,
        "parallel": {
            "ego_extraction_backend": "thread_pool" if workers > 1 else "single_thread",
            "ego_workers": workers, **parallel,
        },
        "wishart": wishart,
        "timing": {
            "graph_load_seconds": graph_seconds,
            "ego_extract_seconds": extract_seconds,
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "scope": "sampled ego, exact VF2 prototypes and GPU-first Wishart k=4/5; no full scan or contraction",
        "results": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-nodes", type=int, default=1_000_000)
    parser.add_argument("--sample-centers", type=int, default=5000)
    parser.add_argument("--budgets", nargs="+", type=int, default=[500, 1000, 3000, 5000])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-ego-nodes", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-batch-size", type=int, default=128)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    run_probe(
        dataset=args.dataset, output_dir=args.output,
        target_nodes=args.target_nodes,
        sample_centers=args.sample_centers, budgets=tuple(args.budgets),
        seed=args.seed, max_ego_nodes=args.max_ego_nodes,
        workers=args.workers, device=args.device,
        gpu_batch_size=args.gpu_batch_size,
        source_sha256=args.source_sha256,
    )


if __name__ == "__main__":
    main()
