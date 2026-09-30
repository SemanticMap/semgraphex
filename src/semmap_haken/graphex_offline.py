"""Offline Graphex Exact v1 experiment on immutable completed Wishart runs.

Usage:
 python -m semmap_haken.graphex_offline --run RUN --level 0 --output FILE.zip
Use --level N to inspect another saved level without modifying checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .graphex_codec import encode_graph
from .graphex_components import EdgeRecord


def encode_saved_level(run: str | Path, level: int,
                       output: str | Path, *, device: str = "auto",
                       gpu_batch_size: int = 500_000) -> dict[str, object]:
    root = Path(run)
    if not (root / "COMPLETED").is_file():
        raise ValueError("offline input must be a COMPLETED run")
    source = root / f"level_{level:03d}"
    transition = root / f"transition_{level:03d}_{level+1:03d}"
    adjacency = sparse.load_npz(source / "adjacency.npz")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("level adjacency must be square")
    vertex_count = adjacency.shape[0]
    mapping_file = transition / "figure_occurrences.jsonl"
    if not mapping_file.is_file():
        raise ValueError("no accepted occurrences for requested level")
    entries = [json.loads(line) for line in mapping_file.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    groups = [item["fine_nodes"] for item in entries]
    source_types = {i: item["dictionary_type_id"]
                    for i, item in enumerate(entries)}
    # CSR is the authoritative source for this experiment. Matrix aggregation
    # may already have lost parallel raw ConceptNet rows: do not claim TSV-level
    # losslessness. Preserve every existing per-relation directed CSR entry.
    index = json.loads((source / "relations" / "index.json").read_text(
        encoding="utf-8"))
    records: list[EdgeRecord] = []
    for relation, filename in sorted(index.items()):
        layer = sparse.load_npz(source / "relations" / filename).tocsr()
        if layer.shape != adjacency.shape:
            raise ValueError("relation layer/adjacency dimensions disagree")
        layer.sort_indices()
        # Vectorized CSR row addressing; avoid an interpreted loop per row.
        rows = np.repeat(np.arange(vertex_count, dtype=np.int64),
                         np.diff(layer.indptr))
        records.extend(EdgeRecord(offset, int(src), int(dst), relation,
                                  float(weight))
                       for offset, src, dst, weight in zip(
                           range(len(records), len(records) + len(rows)),
                           rows, layer.indices, layer.data, strict=True))
    report = encode_graph(vertex_count, records, groups, output,
                          source_type_ids=source_types, device=device,
                          gpu_batch_size=gpu_batch_size)
    report.update({"run": str(root), "level": level,
                   "exactness_scope": "directed relation-layer CSR entries",
                   "raw_parallel_edges": "unavailable if merged during preparation",
                   "checkpoint_modified": False})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-batch-size", type=int, default=500_000)
    args = parser.parse_args()
    print(json.dumps(encode_saved_level(args.run, args.level, args.output,
                                        device=args.device,
                                        gpu_batch_size=args.gpu_batch_size),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
