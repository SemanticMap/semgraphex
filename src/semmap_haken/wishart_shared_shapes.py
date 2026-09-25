"""Export a losslessly reconstructible *prototype* dictionary and figure diagnostics.

The existing level CSR matrices and checkpoints remain the authoritative full graph.
This export does not claim to be a complete lossless graph codec: internal edge
weights and all non-figure edges remain in the level matrices.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse


def _dump(row: object) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _graph(prototype: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for node in prototype["nodes"]:
        g.add_node(int(node["prototype_node"]), symbol_type=str(node["symbol_type"]))
    for edge in prototype["edges"]:
        u, v = int(edge["source"]), int(edge["target"])
        if g.has_edge(u, v):
            raise ValueError("Parallel edges need explicit multigraph support")
        g.add_edge(u, v, relation=str(edge["relation"]))
    if set(g.nodes) != set(range(len(g.nodes))):
        raise ValueError("Prototype node IDs must be consecutive, starting at zero")
    return g


def _shape(prototype: dict) -> dict:
    return {
        "nodes": [str(n["symbol_type"]) for n in sorted(prototype["nodes"], key=lambda x: x["prototype_node"])],
        "edges": [dict(e) for e in prototype["edges"]],
    }


def _restore(shape: dict, variant: dict) -> dict:
    order = variant["shape_to_prototype"]
    if order is None:
        order = list(range(len(shape["nodes"])))
    if sorted(order) != list(range(len(shape["nodes"]))):
        raise ValueError("Non-bijective prototype node mapping")
    ports = {int(n): value for n, value in variant["external_ports"]}
    nodes = [None] * len(order)
    for shape_node, original_node in enumerate(order):
        nodes[original_node] = {
            "prototype_node": original_node,
            "symbol_type": shape["nodes"][shape_node],
            "boundary_ports": ports.get(original_node, "[]"),
        }
    edges = [
        {"source": order[int(e["source"])], "target": order[int(e["target"])], "relation": e["relation"]}
        for e in shape["edges"]
    ]
    return {"nodes": nodes, "edges": sorted(edges, key=lambda e: (e["source"], e["target"], e["relation"]))}


def factor_dictionary(source: Path, target: Path) -> dict:
    """Stream all exact types; use WL only as prefilter and VF2 for exact shape reuse."""
    target.mkdir(parents=True, exist_ok=True)
    shapes: list[dict] = []
    shape_graphs: list[nx.DiGraph] = []
    buckets: dict[tuple, list[int]] = defaultdict(list)
    match_nodes = nx.algorithms.isomorphism.categorical_node_match("symbol_type", "ATOM")
    match_edges = nx.algorithms.isomorphism.categorical_edge_match("relation", "__edge__")
    baseline_bytes = 0
    count = 0
    with source.open(encoding="utf-8") as incoming, (target / "type_variants.jsonl").open("w", encoding="utf-8") as variants, (target / "type_metadata.jsonl").open("w", encoding="utf-8") as metadata, gzip.open(target / "baseline_compact.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as baseline_gz:
        for line in incoming:
            if not line.strip():
                continue
            item = json.loads(line)
            proto = item["prototype"]
            graph = _graph(proto)
            # WL collisions cannot merge non-isomorphic graphs: VF2 verifies each hit.
            wl = nx.weisfeiler_lehman_graph_hash(graph, node_attr="symbol_type", edge_attr="relation", iterations=3)
            key = (graph.number_of_nodes(), graph.number_of_edges(), wl)
            selected = None
            mapping = None
            for sid in buckets[key]:
                matcher = nx.algorithms.isomorphism.DiGraphMatcher(shape_graphs[sid], graph, node_match=match_nodes, edge_match=match_edges)
                if matcher.is_isomorphic():
                    selected, mapping = sid, matcher.mapping
                    break
            if selected is None:
                selected = len(shapes)
                shapes.append(_shape(proto))
                shape_graphs.append(graph)
                buckets[key].append(selected)
                mapping = {n: n for n in graph.nodes}
            order = [int(mapping[i]) for i in range(graph.number_of_nodes())]
            ports = [[int(n["prototype_node"]), n["boundary_ports"]] for n in proto["nodes"] if n["boundary_ports"] != "[]"]
            variant = {
                "type_id": item["type_id"], "shape_id": selected,
                "shape_to_prototype": None if order == list(range(len(order))) else order,
                "external_ports": sorted(ports),
            }
            if _restore(shapes[selected], variant) != proto:
                raise AssertionError(f"Prototype roundtrip failed: {item['type_id']}")
            variants.write(_dump(variant) + "\n")
            metadata.write(_dump({k: v for k, v in item.items() if k != "prototype"}) + "\n")
            baseline_line = _dump(item) + "\n"
            baseline_bytes += len(baseline_line.encode("utf-8"))
            baseline_gz.write(baseline_line)
            count += 1
    with (target / "internal_shapes.jsonl").open("w", encoding="utf-8") as output:
        for sid, shape in enumerate(shapes):
            output.write(_dump({"shape_id": sid, **shape}) + "\n")
    factored_bytes = sum((target / name).stat().st_size for name in ("internal_shapes.jsonl", "type_variants.jsonl", "type_metadata.jsonl"))
    with gzip.open(target / "factorized_combined.jsonl.gz", "wb", compresslevel=9) as out:
        for name in ("internal_shapes.jsonl", "type_variants.jsonl", "type_metadata.jsonl"):
            with (target / name).open("rb") as incoming:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    out.write(chunk)
    baseline_gzip = (target / "baseline_compact.jsonl.gz").stat().st_size
    factored_gzip = (target / "factorized_combined.jsonl.gz").stat().st_size
    result = {"exact_types": count, "shared_internal_shapes": len(shapes), "baseline_dictionary_jsonl_bytes": baseline_bytes, "factored_dictionary_jsonl_bytes": factored_bytes, "jsonl_saved_bytes": baseline_bytes - factored_bytes, "baseline_gzip_bytes": baseline_gzip, "factored_gzip_bytes": factored_gzip, "gzip_saved_bytes": baseline_gzip - factored_gzip, "prototype_roundtrip_exact": True}
    (target / "dictionary_report.json").write_text(_dump(result) + "\n", encoding="utf-8")
    return result


def export_transition(run: Path, level: int, target: Path) -> dict:
    transition = run / f"transition_{level:03d}_{level+1:03d}"
    source_level = run / f"level_{level:03d}"
    target_level = run / f"level_{level+1:03d}"
    occurrences = [json.loads(line) for line in (transition / "figure_occurrences.jsonl").read_text(encoding="utf-8").splitlines() if line]
    fine_to_coarse = np.load(transition / "fine_to_coarse.npy", allow_pickle=False)
    wanted_coarse = {int(fine_to_coarse[int(occ["fine_nodes"][0])]) for occ in occurrences}
    cluster_rows = {}
    with (transition / "cluster_dynamics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            coarse = int(row["coarse_node"])
            if coarse in wanted_coarse:
                cluster_rows[coarse] = row
    if len(cluster_rows) != len(wanted_coarse):
        raise ValueError("Missing cluster diagnostics for accepted figures")
    source_metrics = json.loads((source_level / "dynamic_metrics.json").read_text(encoding="utf-8"))
    target_metrics = json.loads((target_level / "dynamic_metrics.json").read_text(encoding="utf-8"))
    with np.load(target_level / "dynamic_vectors.npz", allow_pickle=False) as vectors:
        betweenness = vectors["betweenness"]
        stationary = vectors["stationary_mass"]
        slow_values = vectors["slow_eigenvalues"]
        slow_vectors = vectors["slow_eigenvectors"]
        distances = vectors["sampled_distances"]
    if slow_vectors.shape[0] != len(stationary):
        raise ValueError("Slow-mode vector count does not match target graph")
    figure_by_node = np.full(len(fine_to_coarse), -1, dtype=np.int64)
    local_node_index = np.full(len(fine_to_coarse), -1, dtype=np.int64)
    rows = []
    for occ in occurrences:
        fine_nodes = [int(n) for n in occ["fine_nodes"]]
        coarse = int(fine_to_coarse[fine_nodes[0]])
        if any(int(fine_to_coarse[n]) != coarse for n in fine_nodes):
            raise ValueError("Accepted figure is not a single coarse node")
        mapping = [int(n) for n in occ["prototype_to_fine_nodes"]]
        if set(mapping) != set(fine_nodes):
            raise ValueError("Prototype mapping does not cover accepted figure")
        for pos, fine in enumerate(mapping):
            if figure_by_node[fine] != -1:
                raise ValueError("Overlapping accepted figures")
            figure_by_node[fine] = coarse
            local_node_index[fine] = pos
        transition_row = cluster_rows[coarse]
        touched = distances[(distances[:, 0] == coarse) | (distances[:, 1] == coarse), 2].tolist() if distances.size else []
        rows.append({
            "source_level": level, "target_level": level + 1,
            "coarse_node": coarse, "dictionary_type_id": occ["dictionary_type_id"],
            "wishart_family_id": occ["wishart_family_id"], "fine_nodes": fine_nodes,
            "prototype_to_fine_nodes": mapping,
            "original_concepts": transition_row["original_concepts"],
            "concept_concat": transition_row["concept_concat"],
            "external_flow": transition_row["external_flow"],
            "external_flow_by_relation": transition_row["external_flow_by_relation"],
            "stationary_mass_source_aggregate": transition_row["stationary_mass"],
            "stationary_mass_target": float(stationary[coarse]),
            "exit_probabilities": transition_row["exit_probabilities"],
            "slow_eigenvalues_target": slow_values.tolist(),
            "slow_mode_coordinates_target": slow_vectors[coarse].tolist(),
            "betweenness_macro_node": float(betweenness[coarse]),
            "sampled_distances_touching_macro_node": touched,
            "graph_metrics_ref": "graph_metrics.json",
        })
    target.mkdir(parents=True, exist_ok=True)
    (target / "graph_metrics.json").write_text(_dump({"source_level": level, "target_level": level + 1, "source": source_metrics, "target": target_metrics}) + "\n", encoding="utf-8")
    with (target / "figure_dynamics.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(_dump(row) + "\n")
    # Each directed relation-layer entry crossing a figure boundary is emitted
    # exactly once, even when both endpoints belong to different figures.
    external_count = 0
    relation_index = json.loads((source_level / "relations" / "index.json").read_text(encoding="utf-8"))
    with gzip.open(target / "external_connections.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as out:
        for relation, filename in sorted(relation_index.items()):
            layer = sparse.load_npz(source_level / "relations" / filename).tocsr()
            if layer.shape != (len(fine_to_coarse), len(fine_to_coarse)):
                raise ValueError("Relation layer and assignment sizes differ")
            for src in range(layer.shape[0]):
                start, stop = layer.indptr[src], layer.indptr[src + 1]
                for pos in range(start, stop):
                    dst = int(layer.indices[pos])
                    if fine_to_coarse[src] == fine_to_coarse[dst] or (figure_by_node[src] < 0 and figure_by_node[dst] < 0):
                        continue
                    out.write(_dump({
                        "source_fine": src, "target_fine": dst,
                        "source_coarse": int(fine_to_coarse[src]), "target_coarse": int(fine_to_coarse[dst]),
                        "source_prototype_node": int(local_node_index[src]) if figure_by_node[src] >= 0 else None,
                        "target_prototype_node": int(local_node_index[dst]) if figure_by_node[dst] >= 0 else None,
                        "relation": relation, "weight": float(layer.data[pos]),
                    }) + "\n")
                    external_count += 1
    return {"level": level, "figures": len(rows), "external_relation_entries": external_count,
            "figure_dynamics": str(target / "figure_dynamics.jsonl"),
            "external_connections": str(target / "external_connections.jsonl.gz")}


def export_run(run: Path, output: Path) -> dict:
    if not (run / "COMPLETED").is_file():
        raise ValueError("Export requires a COMPLETED Wishart run")
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETED").unlink(missing_ok=True)
    dictionary = factor_dictionary(run / "dictionary" / "graph_types.jsonl", output / "dictionary_factorized")
    levels = sorted(run.glob("transition_???_???/figure_occurrences.jsonl"))
    transitions = [export_transition(run, int(p.parent.name.split("_")[1]), output / p.parent.name) for p in levels]
    report = {"dictionary": dictionary, "transitions": transitions,
              "scope": "exact prototype sharing + per-figure external relation entries; original CSR matrices and checkpoint remain in source run",
              "limitations": "not a complete graph codec: internal edge weights and non-figure edges remain in source level CSR matrices; global diagnostics are sample-based where specified"}
    (output / "report.json").write_text(_dump(report) + "\n", encoding="utf-8")
    (output / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(_dump(export_run(args.run, args.output)))


if __name__ == "__main__":
    main()
