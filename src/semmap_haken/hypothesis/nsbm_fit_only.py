"""Standalone checkpointable nSBM fitting for ConceptNet-100k self-similarity.

Reuses H1 graph-tool inference and task checkpoints; does NOT run VF2, graph
grammar, Wishart or Huffman. No CUDA requirement or false GPU-nSBM claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from semmap_haken.conceptnet import stream_assertions
from .graph import TypedGraph
from .grammar_experiment import _fit_partition
from .runner import _available_cpus, _git_revision, _load_graph, _save_graph, _sha_bytes
from .scan_checkpoint import atomic_json, sha256_file
from .selfsim_metrics import validate_levels


def _valid_completed(target: Path, identity: dict, graph: TypedGraph) -> dict | None:
    summary_path = target / "fit_summary.json"
    partitions = target / "nsbm_partitions.npz"
    try:
        if not ((target / "COMPLETED").is_file() and
                partitions.is_file() and summary_path.is_file()):
            return None
        meta = json.loads(summary_path.read_text(encoding="utf-8"))
        if (meta["identity"] != identity or
                meta["partitions_sha256"] != sha256_file(partitions) or
                _sha_bytes(_load_graph(target / "fit_data")) != _sha_bytes(graph)):
            return None
        with np.load(partitions, allow_pickle=False) as data:
            keys = sorted(name for name in data.files if name.startswith("level_"))
            levels = validate_levels([data[k] for k in keys], graph.vertex_count)
        if not levels or [len(set(p)) for p in levels] != meta["block_counts"]:
            return None
        return meta
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return None


def run(input_path: Path, output: Path, *, vertices: int=100_000,
        seed: int=42, restarts: int=3, cpu_budget: int=8,
        resume: bool=False) -> dict:
    if restarts < 1 or cpu_budget < 1 or vertices < 2:
        raise ValueError("invalid vertices, restarts or cpu_budget")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    cpu_budget = min(cpu_budget, _available_cpus())
    graph = TypedGraph.from_assertions(stream_assertions(input_path))
    if graph.vertex_count != vertices:
        raise ValueError(
            f"Input has {graph.vertex_count} vertices; expected exactly {vertices}. "
            "Use semmap-conceptnet-10k --nodes 100000 on a bounded genuine "
            "ConceptNet assertion sample; never take the first 100k TSV lines."
        )
    input_sha = sha256_file(input_path)
    identity = {
        "schema":"semmap-nsbm-only-v1",
        "git_revision":_git_revision(),
        "graph_sha256":_sha_bytes(graph),
        "source_tsv_sha256":input_sha,
        "config":{"vertices":vertices,"seed":seed,
                  "restarts":restarts,"cpu_budget":cpu_budget},
    }
    manifest = output / "manifest.json"
    if resume:
        if not manifest.is_file() or json.loads(manifest.read_text()) != identity:
            raise ValueError("RESUME input / graph / config / code mismatch")
        completed = _valid_completed(output,identity,graph)
        if completed is not None:
            return completed
        (output / "COMPLETED").unlink(missing_ok=True)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("NEW requires empty or absent output directory")
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(manifest,identity)
    stored = output / "fit_data"
    # Incomplete npy files can remain after an interrupted Colab VM. Never
    # mistake the presence of ids.json for a complete, correct graph fixture.
    try:
        matching = _sha_bytes(_load_graph(stored)) == identity["graph_sha256"]
    except (OSError, ValueError, KeyError, AssertionError):
        matching = False
    if not matching:
        _save_graph(graph,stored)
    levels, fit = _fit_partition(
        graph,output,restarts=restarts,seed=seed,cpu_budget=cpu_budget
    )
    levels = validate_levels(levels,graph.vertex_count)
    result = {
        "status":"complete",
        "identity":identity,
        "counts":{"vertices":graph.vertex_count,"assertions":graph.edge_count,
                  "relations":len(graph.relation_names)},
        "block_counts":[len(set(p)) for p in levels],
        "partitions_sha256":sha256_file(output/"nsbm_partitions.npz"),
        "fit":fit,
        "cpu_budget":cpu_budget,
        "note":"Full-graph nested DC-SBM fitted on CPU/OpenMP. These partitions "
               "must not be reused on a differently selected sample.",
    }
    atomic_json(output / "fit_summary.json",result)
    if _valid_completed(output,identity,graph) is not None:
        raise AssertionError("COMPLETED marker appeared before publication")
    # Final marker only after a checked full readback.
    (output / "COMPLETED").write_text("complete\n",encoding="utf-8")
    if _valid_completed(output,identity,graph) is None:
        (output / "COMPLETED").unlink(missing_ok=True)
        raise AssertionError("nSBM output integrity verification failed")
    return result


def main(argv: list[str] | None=None) -> int:
    ap=argparse.ArgumentParser(description="Fit nested DC-SBM only; save exact H1 artifacts")
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--vertices",type=int,default=100000)
    ap.add_argument("--mode",choices=("NEW","RESUME"),default="NEW")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--restarts",type=int,default=3)
    ap.add_argument("--cpu-budget",type=int,default=8)
    args=ap.parse_args(argv)
    report=run(args.input,args.output,vertices=args.vertices,
               seed=args.seed,restarts=args.restarts,
               cpu_budget=args.cpu_budget,resume=args.mode=="RESUME")
    print(json.dumps({"status":report["status"],
                      "counts":report["counts"],
                      "block_counts":report["block_counts"],
                      "fit":report["fit"],
                      "partitions_sha256":report["partitions_sha256"]},
                     ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
