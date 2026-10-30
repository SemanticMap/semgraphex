"""Checkpointed H1 experiment: train data -> fit restarts -> two null ensembles.

No distributed framework: process-pool on local scratch, one graph memory map
per process, task outputs committed atomically and verified on RESUME.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import shutil
from multiprocessing import get_context
from typing import Callable, Iterator

import numpy as np

from .graph import TypedGraph
from .inference import fit_one, MODEL_KINDS
from .nulls import stub_matching_null, flat_dc_sbm_null, typed_degrees
from .predict import ranking_metrics
from .hierarchy_validation import hierarchy_restart_stability
from .splits import split_graph
from .statistics import (empirical_upper_tail, paired_mrr_interval,
                         h1_verdict, nontrivial_levels)


@dataclass(frozen=True)
class H1Config:
    seed: int = 42
    restarts: int = 10
    null_replicates: int = 99
    workers: int | str = "auto"
    train_fraction: float = .8
    validation_fraction: float = .1
    negatives: int = 100
    bootstrap_repetitions: int = 500
    alpha: float = .05
    noninferiority_margin: float = .01

    def __post_init__(self):
        if self.restarts < 1 or self.null_replicates < 0 or self.negatives < 1:
            raise ValueError("restarts >=1, null_replicates >=0, negatives >=1 required")
        if self.bootstrap_repetitions < 20:
            raise ValueError("bootstrap_repetitions must be >=20")
        if self.workers != "auto" and (
            not isinstance(self.workers, int) or self.workers < 1
        ):
            raise ValueError("workers must be positive or auto")
        if not (0 < self.train_fraction < 1 and
                0 <= self.validation_fraction < 1 - self.train_fraction):
            raise ValueError("invalid data-split fractions")
        if not 0 < self.alpha < 1 or self.noninferiority_margin < 0:
            raise ValueError("invalid significance threshold or margin")


def _json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(pending, path)


def _npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".npz.tmp")
    with pending.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(pending, path)


def _sha_bytes(graph: TypedGraph) -> str:
    h = hashlib.sha256()
    for group in (graph.node_ids, graph.relation_names, graph.assertion_ids):
        h.update(json.dumps(group, ensure_ascii=False).encode("utf-8"))
    for arr in (graph.source, graph.target, graph.relation, graph.weight):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _implementation_fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _git_revision() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True,
                                capture_output=True, timeout=5)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "not-in-git"


def _save_graph(graph: TypedGraph, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for label in ("source", "target", "relation", "weight"):
        np.save(path / f"{label}.npy", getattr(graph, label), allow_pickle=False)
    _json_atomic(path / "ids.json", {
        "node_ids": graph.node_ids, "relation_names": graph.relation_names,
        "assertion_ids": graph.assertion_ids,
    })


def _load_graph(path: Path) -> TypedGraph:
    meta = json.loads((path / "ids.json").read_text(encoding="utf-8"))
    return TypedGraph(
        tuple(meta["node_ids"]), tuple(meta["relation_names"]),
        *(np.load(path / f"{col}.npy", mmap_mode="r")
          for col in ("source", "target", "relation", "weight")),
        tuple(meta["assertion_ids"]),
    )


def _available_cpus() -> int:
    """Use the most restrictive host, affinity and Linux cgroup CPU limits."""
    cpus = max(1, os.cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        try:
            cpus = min(cpus, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(np.ceil(int(quota) / int(period)))))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return max(1, cpus)


def _available_memory_bytes() -> int:
    """Respect container memory.max/current instead of only host RAM."""
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except ImportError:
        available = 2_000_000_000
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            available = min(available, max(0, int(limit) - current))
    except (OSError, ValueError):
        pass
    return available


def _worker_budget(requested: int | str, *, vertices: int, edges: int, tasks: int) -> tuple[int, int]:
    cpus = _available_cpus()
    # Conservatively allow several graph-tool state copies beyond edge arrays.
    estimate = max(250_000_000, (vertices * 128 + edges * 256) * 10)
    max_by_ram = max(1, int(_available_memory_bytes() * .5 // estimate))
    cap = min(max_by_ram, max(1, tasks), cpus)
    workers = min(cap, cpus if requested == "auto" else requested)
    threads = max(1, cpus // workers)
    return workers, threads


def _task_key(prefix: str, kind: str, index: int) -> str:
    return f"{prefix}-{kind.replace('-', '_')}-restart_{index:03d}"


def _valid_task(path: Path, partition_path: Path, *, task_key: str) -> bool:
    if not path.exists() or not partition_path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        expected_sha = data.get("partition_sha256")
        if not isinstance(expected_sha, str):
            return False
        if hashlib.sha256(partition_path.read_bytes()).hexdigest() != expected_sha:
            return False
        with np.load(partition_path, allow_pickle=False) as arrays:
            return data["task"] == task_key and len(arrays.files) == data["levels"]
    except (ValueError, OSError, KeyError, json.JSONDecodeError):
        return False


_GRAPH_CACHE: tuple[Path, TypedGraph] | None = None


def _fit_task(graph_path: Path, kind: str, seed: int, threads: int,
              task_path: Path, injected_fit=None) -> dict:
    # Cache exactly one memory-mapped graph per process; never retain all nulls.
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None or _GRAPH_CACHE[0] != graph_path:
        _GRAPH_CACHE = (graph_path, _load_graph(graph_path))
    graph = _GRAPH_CACHE[1]
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    backend = fit_one if injected_fit is None else injected_fit
    started = time.perf_counter()
    result = backend(graph, kind=kind, seed=seed, threads=threads)
    elapsed = time.perf_counter() - started
    partitions = [np.asarray(p, dtype=np.int32) for p in result["partitions"]]
    if not partitions or any(p.shape != (graph.vertex_count,) for p in partitions):
        raise RuntimeError("fit backend returned invalid original-vertex partitions")
    data = {k: v for k, v in result.items() if k != "partitions"}
    data.update(task=task_path.stem, levels=len(partitions),
                wall_seconds=float(elapsed))
    if not np.isfinite(float(data["mdl_bits"])):
        raise RuntimeError("non-finite fitted MDL")
    _npz_atomic(task_path.with_suffix(".npz"),
                {f"p{i}": p for i, p in enumerate(partitions)})
    data["partition_sha256"] = hashlib.sha256(
        task_path.with_suffix(".npz").read_bytes()
    ).hexdigest()
    _json_atomic(task_path, data)
    return data


def _parallel_fit(
    graph_path: Path, tasks: list[tuple[str, int, Path]], *,
    threads: int, workers: int, injected_fit=None,
) -> list[dict]:
    """Bound futures; do not pickle million-edge graphs for every restart."""
    answers: dict[Path, dict] = {}
    pending = [(kind, seed, path) for kind, seed, path in tasks
               if not _valid_task(path, path.with_suffix(".npz"), task_key=path.stem)]
    for kind, seed, path in tasks:
        if (kind, seed, path) not in pending:
            answers[path] = json.loads(path.read_text(encoding="utf-8"))
    if pending:
        if workers <= 1 or injected_fit is not None:
            for kind, seed, path in pending:
                answers[path] = _fit_task(graph_path, kind, seed, threads, path, injected_fit)
        else:
            iterator = iter(pending)
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                running = {}
                def submit() -> bool:
                    try:
                        kind, seed, path = next(iterator)
                    except StopIteration:
                        return False
                    future = pool.submit(_fit_task, graph_path, kind, seed, threads, path)
                    running[future] = path
                    return True
                for _ in range(min(2 * workers, len(pending))):
                    submit()
                while running:
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        path = running.pop(future)
                        answers[path] = future.result()
                        submit()
    return [answers[path] for _, _, path in tasks]


def _best_fit(tasks: list[dict], task_paths: list[Path]) -> tuple[dict, list[list[int]]]:
    winner = min(zip(tasks, task_paths), key=lambda pair: pair[0]["mdl_bits"])
    info, path = winner
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        partitions = [np.asarray(data[f"p{i}"]).tolist() for i in range(info["levels"])]
    return info, partitions


def _identity_manifest(graph: TypedGraph, config: H1Config) -> dict[str, object]:
    return {
        "schema": "h1-runnable-v1",
        "graph_sha256": _sha_bytes(graph),
        "implementation_sha256": _implementation_fingerprint(),
        "git_revision": _git_revision(),
        "config": asdict(config),
    }


def run_h1(
    graph: TypedGraph, output: Path | str, *, config: H1Config,
    resume: bool = False,
    fit_fn: Callable | None = None, flat_sampler: Callable | None = None,
) -> dict[str, object]:
    """Execute H1 with two conditional null controls and restart checkpointing.

    No claim of H1 confirmation is possible without graph-tool fits, qualified
    controls and a non-empty held-out query population. Injected callables are
    solely for small deterministic runner tests (forced synchronous execution).
    """
    output = Path(output)
    manifest = _identity_manifest(graph, config)
    manifest_path = output / "manifest.json"
    if resume:
        if not manifest_path.exists():
            raise ValueError("RESUME requires an existing H1 manifest")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("manifest graph/configuration/code identity mismatch")
        if (output / "COMPLETED").exists() and (output / "report.json").exists():
            return json.loads((output / "report.json").read_text(encoding="utf-8"))
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("NEW requires an absent or empty output directory")
        output.mkdir(parents=True, exist_ok=True)
        _json_atomic(manifest_path, manifest)
    split = split_graph(graph, seed=config.seed, train_fraction=config.train_fraction,
                        validation_fraction=config.validation_fraction)
    if split.test.edge_count == 0:
        raise ValueError("no held-out test edges: need a larger source graph")
    _save_graph(split.train, output / "data" / "train")
    _save_graph(split.validation, output / "data" / "validation")
    _save_graph(split.test, output / "data" / "test")
    train_path = output / "data" / "train"
    n_all_tasks = 3 * config.restarts
    workers, threads = _worker_budget(config.workers, vertices=graph.vertex_count,
                                       edges=split.train.edge_count, tasks=n_all_tasks)
    if fit_fn is not None:
        workers = 1
        threads = max(1, (os.cpu_count() or 1) // workers)
    kinds = list(MODEL_KINDS)
    # Use a single explicit, stable enumeration for every graph and null draw.
    def fit_all(where: Path, prefix: str) -> tuple[dict, dict]:
        work = []
        paths = {kind: [] for kind in kinds}
        for j, kind in enumerate(kinds):
            for k in range(config.restarts):
                path = output / "tasks" / f"{_task_key(prefix, kind, k)}.json"
                paths[kind].append(path)
                work.append((kind, config.seed + j * 100_000 + k, path))
        meta = _parallel_fit(where, work, threads=threads, workers=workers, injected_fit=fit_fn)
        selected = {}
        grouped = {}
        for kind in kinds:
            info = [item for item in meta if item["kind"] == kind]
            best, partitions = _best_fit(info, paths[kind])
            selected[kind] = (best, partitions)
            grouped[kind] = {
                "best_mdl_bits": float(best["mdl_bits"]),
                "median_mdl_bits": float(np.median([x["mdl_bits"] for x in info])),
                "range_mdl_bits": [float(min(x["mdl_bits"] for x in info)),
                                   float(max(x["mdl_bits"] for x in info))],
                "seed": int(best["seed"]),
                "fit_wall_seconds": [float(x["wall_seconds"]) for x in info],
                "blocks_by_level": [len(set(x)) for x in partitions],
            }
        return selected, grouped

    real, real_summary = fit_all(train_path, "real")
    # Interpretability diagnostics are separate from the MDL selection rule:
    # equal block counts do not mean equal partitions across restarts.
    restart_hierarchies = []
    for restart in range(config.restarts):
        path = output / "tasks" / f"{_task_key('real', 'nested-DC-SBM', restart)}.npz"
        with np.load(path, allow_pickle=False) as saved:
            restart_hierarchies.append([
                np.asarray(saved[f"p{level}"], dtype=np.int64).tolist()
                for level in range(len(saved.files))
            ])
    real_summary["nested-DC-SBM"]["restart_stability"] = (
        hierarchy_restart_stability(restart_hierarchies)
    )
    train_mdl_gain = (real["DC-SBM"][0]["mdl_bits"] -
                      real["nested-DC-SBM"][0]["mdl_bits"])
    best_flat = np.asarray(real["DC-SBM"][1][0], dtype=np.int64)
    controls: dict[str, list[float]] = {"configuration": [],
                                        "conditional_flat_dc_sbm": []}
    null_diagnostics: dict[str, list[dict]] = {name: [] for name in controls}
    n_ran = 0
    generator_flat = flat_sampler or flat_dc_sbm_null
    for ensemble in controls:
        for i in range(config.null_replicates):
            seed = config.seed + 1000000 + 10000 * (list(controls).index(ensemble) + 1) + i
            prefix = f"null_{ensemble}_{i:04d}"
            null_file = output / "nulls" / f"{prefix}.json"
            saved = json.loads(null_file.read_text(encoding="utf-8")) if null_file.exists() else None
            sample_path = output / "data" / prefix
            task_paths = [
                output / "tasks" / f"{_task_key(prefix, kind, k)}.json"
                for kind in kinds for k in range(config.restarts)
            ]
            fully_cached = all(_valid_task(path, path.with_suffix(".npz"),
                                           task_key=path.stem) for path in task_paths)
            if not fully_cached:
                if (sample_path / "ids.json").exists():
                    sample = _load_graph(sample_path)
                    if saved is None or _sha_bytes(sample) != saved["graph_sha256"]:
                        raise ValueError("incomplete null sample checksum/manifest mismatch")
                else:
                    # Sample once; retained on interruption, discarded after the
                    # last successful fit to avoid 2*N full-graph copies.
                    sample, diagnostic = (
                        stub_matching_null(split.train, seed=seed) if ensemble == "configuration"
                        else generator_flat(split.train, best_flat, seed=seed)
                    )
                    if any(not np.array_equal(a,b) for a,b in
                           zip(typed_degrees(sample), typed_degrees(split.train))):
                        raise AssertionError("control changed typed directed degree counts")
                    sample_hash = _sha_bytes(sample)
                    if saved is not None and sample_hash != saved["graph_sha256"]:
                        raise RuntimeError("null generator does not reproduce checkpointed sample")
                    if saved is None:
                        saved = {"seed": seed, **diagnostic, "graph_sha256": sample_hash}
                        _json_atomic(null_file, saved)
                    _save_graph(sample, sample_path)
            candidate, _ = fit_all(sample_path, prefix)
            controls[ensemble].append(
                float(candidate["DC-SBM"][0]["mdl_bits"] -
                      candidate["nested-DC-SBM"][0]["mdl_bits"]))
            null_diagnostics[ensemble].append(saved if saved is not None else
                                               json.loads(null_file.read_text(encoding="utf-8")))
            n_ran += 3 * config.restarts
            # Release cached graph before deleting the staging directory.
            global _GRAPH_CACHE
            _GRAPH_CACHE = None
            if sample_path.exists():
                shutil.rmtree(sample_path)

    # Prediction is assessed after choosing the best restart from training MDL.
    prediction = {}
    for kind in kinds:
        partitions = real[kind][1]
        dc = kind != "SBM"
        validation = ranking_metrics(split.train, split.validation,
                                     np.asarray(partitions[0]), seed=config.seed + 19,
                                     negatives=config.negatives, known=graph,
                                     degree_corrected=dc)
        test = ranking_metrics(split.train, split.test, np.asarray(partitions[0]),
                               seed=config.seed + 37, negatives=config.negatives,
                               known=graph, degree_corrected=dc)
        prediction[kind] = {"validation": validation, "test": test}
    nested_ranks = prediction["nested-DC-SBM"]["test"]["reciprocal_ranks"]
    flat_ranks = prediction["DC-SBM"]["test"]["reciprocal_ranks"]
    if nested_ranks and len(nested_ranks) == len(flat_ranks):
        pair = paired_mrr_interval(nested_ranks, flat_ranks, seed=config.seed + 83,
                                   repetitions=config.bootstrap_repetitions,
                                   clusters=prediction["nested-DC-SBM"]["test"]["query_sources"])
    else:
        pair = None
    null_report = {}
    for name in controls:
        null_report[name] = {
            "mdlgain_controls": controls[name],
            "diagnostics": null_diagnostics[name],
            "significance": (empirical_upper_tail(train_mdl_gain, controls[name])
                             if controls[name] else None),
        }
    verdict = h1_verdict(
        mdl_gain=train_mdl_gain,
        levels=nontrivial_levels(real["nested-DC-SBM"][1]),
        p_configuration=(null_report["configuration"]["significance"] or {}).get("p_value"),
        p_flat=(null_report["conditional_flat_dc_sbm"]["significance"] or {}).get("p_value"),
        paired_mrr_delta=pair["mean_difference"] if pair is not None else None,
        paired_ci_low=pair["ci_low"] if pair is not None else None,
        paired_ci_high=pair["ci_high"] if pair is not None else None,
        control_replicates=config.null_replicates,
        noninferiority_margin=config.noninferiority_margin, alpha=config.alpha,
    )
    report = {
        "status": verdict["status"], "decision": verdict,
        "hypothesis": "H1 statistical multiscale structure, not graphex limit",
        "counts": {"vertices": graph.vertex_count, "assertions": graph.edge_count,
                   "train": split.train.edge_count,
                   "validation": split.validation.edge_count, "test": split.test.edge_count,
                   "split_groups": split.group_counts},
        "fits": real_summary, "observed_mdl_gain_bits": float(train_mdl_gain),
        "relations": {
            name.rsplit("/", 1)[-1]: {
                "full": int(np.count_nonzero(graph.relation == index)),
                "train": int(np.count_nonzero(split.train.relation == index)),
                "validation": int(np.count_nonzero(split.validation.relation == index)),
                "test": int(np.count_nonzero(split.test.relation == index)),
            }
            for index, name in enumerate(graph.relation_names)
        },
        "prediction": prediction, "paired_test_mrr": pair, "nulls": null_report,
        "fit_counts": {"real": 3 * config.restarts, "null": n_ran},
        "resource_budget": {"workers": workers, "threads_per_fit": threads},
        "model_semantics": {
            "edges": "directed typed assertion multiplicities", "weights_used": False,
            "predictive": "smoothed block propensity rank; not posterior predictive likelihood",
            "configuration": "distinguishable directed typed stubs, loops/multiedges allowed",
            "flat": "conditional microcanonical flat DC-SBM per relation, loops/multiedges allowed",
        },
        "scientific_warning": (
            "Conditioned on one source network, matched null ensembles and query split. "
            "Null p-values are Monte Carlo estimates; model selection does not establish a graphon/graphex limit."
        ),
    }
    _json_atomic(output / "report.json", report)
    (output / "COMPLETED").write_text("completed\n", encoding="utf-8")
    return report
