"""Small reproducible research pilot: typed data -> nulls -> GPU projection.

`--fit` enables graph-tool CPU/OpenMP model comparison when installed. The
baseline without `--fit` is deliberately NOT a test of the hierarchy hypothesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from semmap_haken.conceptnet import stream_assertions

from .graph import TypedGraph, p_sample
from .inference import fit_models
from .nulls import degree_preserving_null, typed_degrees
from .projection import estimate_block_kernel


def _null_summary(args: tuple[TypedGraph, int, int]) -> dict[str, object]:
    graph, seed, swaps = args
    null, diagnostic = degree_preserving_null(graph, seed=seed, swaps_per_edge=swaps)
    before, after = typed_degrees(graph), typed_degrees(null)
    if not all(np.array_equal(a, b) for a, b in zip(before, after)):
        raise AssertionError("null graph changed relation-specific directed degrees")
    return {"seed": seed, **diagnostic, "degree_invariants": True}


def _fit_null(args: tuple[TypedGraph, int, int, int]) -> dict[str, object]:
    graph, seed, swaps, threads = args
    null, diagnostic = degree_preserving_null(graph, seed=seed, swaps_per_edge=swaps)
    return {"seed": seed, **diagnostic,
            "fit": fit_models(null, seed=seed, threads=threads)["mdl_bits"]}


def _effective_workers(requested: str, n_tasks: int, n_vertices: int, n_edges: int) -> int:
    if requested != "auto":
        if int(requested) < 1:
            raise ValueError("workers must be positive")
        return min(int(requested), max(1, n_tasks))
    cpus = max(1, (os.cpu_count() or 1) - 1)
    # Each process holds a private graph-tool state and its graph arrays.
    available = None
    try:
        import psutil
        available = psutil.virtual_memory().available
    except ImportError:
        pass
    if available is None:
        try:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            available = 2_000_000_000
    per_worker = max(64_000_000, (n_vertices * 64 + n_edges * 160) * 6)
    memory_limit = max(1, int(0.6 * available / per_worker))
    return max(1, min(cpus, memory_limit, max(1, n_tasks)))


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SemMap: first reproducible multiscale hypothesis pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot", help="Prepare graph, control ensemble and W-only projection")
    pilot.add_argument("--input", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--nulls", type=int, default=9)
    pilot.add_argument("--swaps-per-edge", type=int, default=5)
    pilot.add_argument("--workers", default="auto")
    pilot.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    pilot.add_argument("--gpu-batch", type=int, default=250_000)
    pilot.add_argument("--fit", action="store_true", help="Fit three graph-tool SBM variants and nulls")
    args = parser.parse_args(argv)
    if args.nulls < 0 or args.swaps_per_edge < 0:
        parser.error("nulls and swaps-per-edge must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be absent or empty; use a new run name")
    graph = TypedGraph.from_assertions(stream_assertions(args.input))
    workers = _effective_workers(args.workers, args.nulls, graph.vertex_count, graph.edge_count)
    threads = max(1, ((os.cpu_count() or 1) - 1) // workers)
    seeds = [args.seed + i + 1 for i in range(args.nulls)]
    # A one-block W-only baseline. NOT an inferred nSBM hierarchy.
    kernel = estimate_block_kernel(graph, np.zeros(graph.vertex_count, dtype=np.int64),
                                   device=args.device, batch_size=args.gpu_batch)
    sampled = p_sample(graph, p=0.5, seed=args.seed)
    model_result = fit_models(graph, seed=args.seed, threads=threads) if args.fit else None
    jobs = ((graph, seed, args.swaps_per_edge, threads) if args.fit else
            (graph, seed, args.swaps_per_edge) for seed in seeds)
    worker_fn = _fit_null if args.fit else _null_summary
    if workers > 1 and args.nulls:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            controls = list(pool.map(worker_fn, jobs))
    else:
        controls = [worker_fn(job) for job in jobs]
    args.output.mkdir(parents=True, exist_ok=True)
    graphon_family = []
    if model_result is not None:
        for level, membership in enumerate(model_result["partitions"]):
            block_kernel = estimate_block_kernel(
                graph, np.asarray(membership, dtype=np.int64),
                device=args.device, batch_size=args.gpu_batch,
            )
            file_name = f"graphon_level_{level:03d}.json"
            (args.output / file_name).write_text(
                json.dumps(block_kernel, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            graphon_family.append({
                "level": level, "file": file_name,
                "blocks": len(block_kernel["block_mass"]),
                "nonempty_block_pairs": len(block_kernel["entries"]),
                "backend": block_kernel["backend"],
            })
    np.savez_compressed(args.output / "typed_graph.npz", source=graph.source,
                        target=graph.target, relation=graph.relation, weight=graph.weight,
                        node_ids=np.asarray(graph.node_ids), relation_names=np.asarray(graph.relation_names),
                        assertion_ids=np.asarray(graph.assertion_ids))
    result = {
        "input": {"path": str(args.input), "sha256": _sha256(args.input)},
        "graph": {"vertices": graph.vertex_count, "edges": graph.edge_count,
                  "relations": list(graph.relation_names)},
        "projection": kernel, "p_sample_0_5": {"vertices": sampled.vertex_count,
                                                   "edges": sampled.edge_count},
        "nulls": {"count": args.nulls, "workers": workers, "threads_per_fit": threads,
                  "replicates": controls},
        "fit": model_result,
        "graphon_family": graphon_family,
        "null_significance": (
            {"observed_mdl_gain_bits": model_result["mdl_bits"]["DC-SBM"] - model_result["mdl_bits"]["nested-DC-SBM"],
             "null_mdl_gains_bits": [rep["fit"]["DC-SBM"] - rep["fit"]["nested-DC-SBM"] for rep in controls],
             "empirical_p_upper_tail": (
                 (1 + sum((rep["fit"]["DC-SBM"] - rep["fit"]["nested-DC-SBM"]) >=
                           (model_result["mdl_bits"]["DC-SBM"] - model_result["mdl_bits"]["nested-DC-SBM"])
                          for rep in controls)) / (1 + len(controls)) if controls else None),
             "interpretation": "conditional on this relation-degree-preserving null ensemble"}
            if model_result is not None else None),
        "scientific_status": "unvalidated pilot" if args.fit else "W-only infrastructure smoke; hierarchy NOT tested",
        "seed": args.seed,
    }
    tmp = args.output / "summary.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output / "summary.json")
    (args.output / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "vertices": graph.vertex_count,
                      "edges": graph.edge_count, "projection_device": kernel["backend"],
                      "inference": "performed" if args.fit else "not requested"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
