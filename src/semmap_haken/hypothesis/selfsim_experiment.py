"""Reproducible four-method nSBM self-similarity experiment and CLI.

Research outputs are diagnostic: hierarchy alone is NOT evidence of fractality.
Controls preserve nested sizes or typed in/out degree sequences. All checkpoints
are atomically written and verified against input/config/code identity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from semmap_haken.conceptnet import stream_assertions
from .graph import TypedGraph
from .runner import (_available_cpus, _available_memory_bytes, _git_revision,
                     _load_graph, _sha_bytes)
from .scan_checkpoint import atomic_json
from .selfsim_metrics import (box_covering, dynamics, graphon, structural,
                              validate_levels)


def load_levels(path: str | Path, vertices: int) -> list[np.ndarray]:
    """Read H1 nSBM original-vertex partitions (level_000, level_001...)."""
    with np.load(path, allow_pickle=False) as data:
        keys = sorted([key for key in data.files
                       if key.startswith("level_") or
                       (key.startswith("p") and key[1:].isdigit())])
        if not keys:
            raise ValueError("nSBM NPZ contains no recognized level arrays")
        levels = [data[key].copy() for key in keys]
    return validate_levels(levels, vertices)


def typed_degree_null(graph: TypedGraph, seed: int) -> TypedGraph:
    """Relation-specific random pairing preserves both in/out degrees exactly.

    Conditional stub-matching null with independent directed target permutation
    for each type. It permits self-loops and multiedges and is NOT a uniform
    simple-graph configuration-model sampler.
    """
    rng = np.random.default_rng(seed)
    target = graph.target.copy()
    for rel in range(len(graph.relation_names)):
        idx = np.flatnonzero(graph.relation == rel)
        target[idx] = rng.permutation(target[idx])
    return TypedGraph(graph.node_ids, graph.relation_names, graph.source.copy(),
                      target, graph.relation.copy(), graph.weight.copy(),
                      graph.assertion_ids)


def nested_degree_stratified_permutation(graph: TypedGraph,
                                         levels: list[np.ndarray], seed: int):
    """Permute node-to-block association within log(in-degree),log(out-degree) bins.

    Permuting all original-vertex levels by one map preserves every block size
    and ALL nesting exactly. Degrees themselves are untouched.
    """
    n = graph.vertex_count
    rng = np.random.default_rng(seed)
    incoming = np.bincount(graph.target, minlength=n)
    outgoing = np.bincount(graph.source, minlength=n)
    # Coarse bins avoid groups with only 1 vertex at a given degree.
    key = (np.minimum(6, np.floor(np.log2(incoming+1)).astype(int))*7 +
           np.minimum(6, np.floor(np.log2(outgoing+1)).astype(int)))
    perm = np.arange(n)
    for group in np.unique(key):
        where = np.flatnonzero(key == group)
        perm[where] = rng.permutation(where)
    return validate_levels([p[perm] for p in levels], n)


def four_methods(graph: TypedGraph, levels: list[np.ndarray], *, seed: int,
                 motif_samples: int, pair_samples: int, box_repeats: int,
                 radii: tuple[int, ...], slow_modes: int,
                 compute_boxes: bool = True) -> dict:
    """One fixed scientific contract across original and coarse levels."""
    out = {
        "structural": structural(graph, levels, seed=seed,
                                 samples=motif_samples),
        "graphon": graphon(graph, levels, seed=seed,
                           pairs=pair_samples),
        "haken": dynamics(graph, levels, slow_modes=slow_modes),
        "boxes": (box_covering(graph, seed=seed, repeats=box_repeats,
                                radii=radii) if compute_boxes else None),
    }
    return out


def scores(metrics: dict) -> dict[str, float | None]:
    pairs = metrics["structural"]["adjacent_motif_js"]
    graphon_pairs = metrics["graphon"]["adjacent_kernel"]
    dynamical = [x["slow_subspace_overlap"] for x in metrics["haken"]["levels"]
                 if x["slow_subspace_overlap"] is not None]
    boxes = metrics.get("boxes")
    return {
        "motif_similarity_negative_js": (
            -float(np.mean([x["js_divergence"] for x in pairs])) if pairs else None),
        "kernel_similarity_negative_js": (
            -float(np.mean([x["log_intensity_js"] for x in graphon_pairs]))
            if graphon_pairs else None),
        "haken_slow_overlap": (float(np.mean(dynamical)) if dynamical else None),
        "box_power_over_exp_delta_aic": (
            boxes["delta_aic_exponential_minus_power"] if boxes else None),
    }


def _score_null(metric: float | None, controls: list[float | None]):
    sample = np.asarray([x for x in controls if x is not None and np.isfinite(x)])
    if metric is None or not np.isfinite(metric) or len(sample) == 0:
        return {"p_upper": None, "replicates":int(len(sample))}
    return {
        "observed":float(metric), "null_mean":float(sample.mean()),
        "null_std":float(sample.std(ddof=1)) if len(sample)>1 else None,
        "replicates":int(len(sample)),
        "p_upper":float((1+np.count_nonzero(sample >= metric))/(1+len(sample))),
        "p_value_resolution":1/(1+len(sample)),
        "direction":"larger metric implies stronger match",
    }


def _id(config: dict, input_sha: str, partitions_sha: str, git_sha: str) -> dict:
    return {"schema":"nsbm-multiscale-selfsimilarity-v1",
            "config":config,"graph_sha256":input_sha,
            "partitions_sha256":partitions_sha,"code_revision":git_sha}


def _encode(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _persist(path: Path, identity: dict, result: dict) -> None:
    payload = {"identity":identity,"result":result}
    atomic_json(path, {"payload":payload,
                       "sha256":sha256(_encode(payload)).hexdigest()})


def _recover(path: Path, identity: dict):
    if not path.is_file():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        if (payload["identity"] == identity and
                envelope["sha256"] == sha256(_encode(payload)).hexdigest()):
            return payload["result"]
    except (ValueError, KeyError, OSError, TypeError):
        pass
    return None


def _worker(kind: str, rep: int, seed: int, graph: TypedGraph,
            levels: list[np.ndarray], cfg: dict):
    """Spawn-safe top-level worker; threadpools limited before numerical work."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    if kind == "permuted_partitions":
        transformed = graph
        labels = nested_degree_stratified_permutation(graph, levels, seed)
    elif kind == "typed_degree_rewire":
        transformed = typed_degree_null(graph, seed)
        labels = levels
    else:
        raise ValueError(f"invalid control {kind}")
    metrics = four_methods(
        transformed, labels, seed=cfg["seed"],
        motif_samples=cfg["motif_samples"],
        pair_samples=cfg["pair_samples"], box_repeats=cfg["box_repeats"],
        radii=tuple(cfg["radii"]), slow_modes=cfg["slow_modes"],
        compute_boxes=kind == "typed_degree_rewire",
    )
    return {"kind":kind,"replicate":rep,"seed":seed,
            "scores":scores(metrics),"metrics":metrics}


def run(graph: TypedGraph, partitions: list[np.ndarray], output: str | Path, *,
        seed: int=42, null_repeats: int=9, jobs: int=4, motif_samples: int=2000,
        pair_samples: int=12000, box_repeats: int=3,
        radii: tuple[int,...]=(0,1,2,3,4,5), slow_modes: int=4,
        resume: bool=False) -> dict:
    if (null_repeats < 0 or jobs < 1 or motif_samples < 1 or pair_samples < 1 or
            box_repeats < 1 or slow_modes < 1 or len(radii) < 3):
        raise ValueError("invalid self-similarity experiment configuration")
    levels = validate_levels(partitions, graph.vertex_count)
    # Only nontrivial levels; at 3 blocks motif evidence is intrinsically weak.
    levels = [p for p in levels if 1 < len(set(p)) < graph.vertex_count]
    if not levels:
        raise ValueError("no nontrivial nSBM levels")
    if np.any(graph.weight < 0):
        raise ValueError("negative weights need a separately defined dynamics")
    cfg = {"seed":seed,"null_repeats":null_repeats,"jobs":jobs,
           "motif_samples":motif_samples,"pair_samples":pair_samples,
           "box_repeats":box_repeats,"radii":list(radii),"slow_modes":slow_modes,
           "unit_weight_assertions":True}
    h = sha256(b"".join(np.ascontiguousarray(p).tobytes() for p in levels)).hexdigest()
    identity = _id(cfg, _sha_bytes(graph), h, _git_revision())
    target = Path(output)
    manifest = target/"manifest.json"
    if resume:
        if not manifest.is_file() or json.loads(manifest.read_text()) != identity:
            raise ValueError("RESUME requires identical graph, partitions, configuration and revision")
        if (target/"COMPLETED").is_file():
            cached = _recover(target/"result_checkpoint.json", identity)
            if cached is not None:
                return cached
            (target/"COMPLETED").unlink()
    else:
        if target.exists() and any(target.iterdir()):
            raise ValueError("NEW cannot overwrite an existing nonempty experiment")
        target.mkdir(parents=True,exist_ok=True)
        atomic_json(manifest,identity)
    start = time.perf_counter()
    original_path = target/"original_checkpoint.json"
    original = _recover(original_path,identity) if resume else None
    if original is None:
        original = four_methods(
            graph,levels,seed=seed,motif_samples=motif_samples,
            pair_samples=pair_samples, box_repeats=box_repeats,
            radii=radii,slow_modes=slow_modes,
        )
        _persist(original_path,identity,original)
    observed = scores(original)
    nulls = []
    tasks = []
    for kind in ("permuted_partitions","typed_degree_rewire"):
        for i in range(null_repeats):
            control_path = target/f"null_{kind}_{i:03d}.json"
            task_identity = {**identity, "null_kind":kind, "replicate":i}
            cached = _recover(control_path,task_identity) if resume else None
            if cached is not None:
                nulls.append(cached)
            else:
                tasks.append((kind, i, seed+10000+1001*i+
                              (1000000 if kind=="typed_degree_rewire" else 0),
                              control_path, task_identity))
    # Real OS CPU and cgroup RAM budgets prevent overcommitting Colab.
    memory_cap = max(1, int(.55*_available_memory_bytes() //
                            max(160_000_000, graph.edge_count*300 + graph.vertex_count*400)))
    workers = min(jobs, _available_cpus(), memory_cap, max(1,len(tasks)))
    if tasks:
        if workers == 1:
            for kind,i,s,path,ident in tasks:
                row = _worker(kind,i,s,graph,levels,cfg)
                _persist(path,ident,row)
                nulls.append(row)
        else:
            with ProcessPoolExecutor(
                    max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = {pool.submit(_worker,kind,i,s,graph,levels,cfg):
                           (path,ident) for kind,i,s,path,ident in tasks}
                for fut in as_completed(futures):
                    path,ident = futures[fut]
                    result = fut.result()
                    _persist(path,ident,result)
                    nulls.append(result)
    nulls.sort(key=lambda r:(r["kind"],r["replicate"]))
    null_summary = {}
    for kind in ("permuted_partitions","typed_degree_rewire"):
        selected=[r for r in nulls if r["kind"]==kind]
        null_summary[kind] = {
            key:_score_null(value,[row["scores"][key] for row in selected])
            for key,value in observed.items()
            if not (key=="box_power_over_exp_delta_aic" and
                    kind=="permuted_partitions")
        }
    report = {
        "status":"complete",
        "identity":identity,
        "counts":{"vertices":graph.vertex_count,"assertions":graph.edge_count,
                  "relations":len(graph.relation_names)},
        "block_counts":[len(set(p)) for p in levels],
        "observed":original,"scores":observed,
        "controls":null_summary,
        "null_repeats_per_ensemble":null_repeats,
        "computed_or_reused_nulls":len(nulls),
        "cpu_workers":workers,
        "wall_seconds":time.perf_counter()-start,
        "inference_warning":"nSBM fitted on full graph: typed graphon holdout AUC is "
            "conditional and subject to partition leakage. Nulls condition on "
            "the fitted hierarchy. Refit train-only nSBM for predictive claims.",
        "scaling_warning":"3 nSBM scales do not establish power laws or multifractality; "
            "box covering is independently sampled, heuristic and tested "
            "against typed-degree rewiring. p-values are descriptive, not "
            "corrected for multiple testing.",
    }
    _persist(target/"result_checkpoint.json",identity,report)
    atomic_json(target/"report.json",report)
    (target/"COMPLETED").write_text("complete\n",encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap=argparse.ArgumentParser(description="nSBM self-similarity four tests")
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--graph-dir",type=Path,help="H1 fit_data with 4 .npy arrays and ids.json")
    group.add_argument("--input",type=Path,help="prepared 5-field ConceptNet TSV")
    ap.add_argument("--partitions",required=True,type=Path)
    ap.add_argument("--source-manifest",type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--mode",choices=("NEW","RESUME"),default="NEW")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--null-repeats",type=int,default=9)
    ap.add_argument("--jobs",type=int,default=4)
    ap.add_argument("--motif-samples",type=int,default=2000)
    ap.add_argument("--pair-samples",type=int,default=12000)
    ap.add_argument("--box-repeats",type=int,default=3)
    ap.add_argument("--box-radii",default="0,1,2,3,4,5")
    ap.add_argument("--slow-modes",type=int,default=4)
    args=ap.parse_args(argv)
    graph=(_load_graph(args.graph_dir) if args.graph_dir else
           TypedGraph.from_assertions(stream_assertions(args.input)))
    if args.source_manifest:
        meta=json.loads(args.source_manifest.read_text(encoding="utf-8"))
        if meta.get("graph_sha256") != _sha_bytes(graph):
            raise ValueError("graph and source nSBM fit manifest do not match")
    levels=load_levels(args.partitions,graph.vertex_count)
    result=run(graph,levels,args.output,seed=args.seed,
               null_repeats=args.null_repeats,jobs=args.jobs,
               motif_samples=args.motif_samples,pair_samples=args.pair_samples,
               box_repeats=args.box_repeats,slow_modes=args.slow_modes,
               radii=tuple(map(int,args.box_radii.split(","))),
               resume=args.mode=="RESUME")
    print(json.dumps({"status":result["status"],"block_counts":result["block_counts"],
                      "scores":result["scores"],"cpu_workers":result["cpu_workers"],
                      "controls":result["controls"]},indent=2,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
