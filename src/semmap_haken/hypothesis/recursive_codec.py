"""Exact multilevel nSBM grammar: shared symbols and reverse transition deltas.

The v2 envelope shares internal shapes. Existing grammar streams and the v1
reverse decoder are reused; v1 archives remain readable without conversion.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import hashlib
import io
import json
import os
import time
from pathlib import Path
import zipfile

import numpy as np

from semmap_haken.exact_edge_baseline import (
    decode_edge_baseline, encode_edge_baseline_bytes,
)
from semmap_haken.grammar_codec import encode_grammar
from semmap_haken.grammar_huffman import decode_symbols
from semmap_haken.grammar_streams import read_occurrences
from semmap_haken.grammar_types import InternalShape, ShapeEdge
from semmap_haken.graphex_components import EdgeRecord
from semmap_haken.hierarchy_codec import (
    _decode_transition_delta, _partition_from_occurrence_nodes,
)
from semmap_haken.shape_identity import CONTRACT, LEVELS, ShapeRegistry
from .scan_checkpoint import atomic_json, sha256_file


FORMAT = "semmap_nsbm_exact_hierarchy_v1"
SHARED_FORMAT = "semmap_nsbm_exact_hierarchy_v2"
DELTA_NAMES = ("manifest.json", "shapes.json", "variants.json", "rules.json",
               "occurrences.bin", "symbols.huff", "internal.bin", "ports.bin")


def _registry(z: zipfile.ZipFile):
    manifest = json.loads(z.read("manifest.json"))
    if manifest.get("format") == SHARED_FORMAT:
        return ShapeRegistry.loads(z.read("dictionary.json"))
    return None


def _shape_rows(z: zipfile.ZipFile, prefix: str, registry: ShapeRegistry | None):
    root = prefix + "/" if prefix else ""
    if registry is None:
        return json.loads(z.read(root + "shapes.json"))
    manifest = json.loads(z.read(root + "manifest.json"))
    return registry.expand(json.loads(z.read(root + "shape_refs.json")),
                           manifest["relations"])


def _occurrences(z: zipfile.ZipFile, prefix: str = "",
                 registry: ShapeRegistry | None = None):
    root = prefix + "/" if prefix else ""
    manifest = json.loads(z.read(root + "manifest.json"))
    shapes = {int(s["shape_id"]): InternalShape(
        int(s["shape_id"]), tuple(s["node_types"]),
        tuple(ShapeEdge(*map(int, edge)) for edge in s["edges"]))
        for s in _shape_rows(z, prefix, registry)}
    symbols = (decode_symbols(z.read(root + "symbols.huff"))
               if manifest.get("symbol_coding") == "huffman" else None)
    return read_occurrences(z.read(root + "occurrences.bin"), shapes, symbols=symbols)


def _next_types(groups, occurrences, node_types, step, shape_refs=None):
    """One naming rule for contraction and checkpoint reconstruction."""
    aliases = ({ref["local"]: ref["symbol"] for ref in shape_refs}
               if shape_refs is not None else None)
    symbols = {frozenset(item.shape_to_fine_nodes): item.symbol_id for item in occurrences}
    result = {}
    for coarse, children in sorted(groups.items()):
        if len(children) > 1:
            local = symbols[frozenset(children)]
            result[coarse] = f"S:{aliases[local]}" if aliases is not None else f"L{step}_S{local}"
        elif children[0] in node_types:
            result[coarse] = node_types[children[0]]
    return result


def contract_records(n: int, records: tuple[EdgeRecord, ...],
                     figures: list[tuple[int, ...]],
                     node_types: dict[int, str], step: int,
                     transition: Path, *, shape_refs: list[dict] | None = None):
    """Aggregate only changed-figure edges; untouched edge IDs remain stable."""
    assignment, _, _ = _partition_from_occurrence_nodes(n, figures)
    touched = {node for nodes in figures for node in nodes}
    groups = defaultdict(list)
    for old, new in enumerate(assignment):
        groups[int(new)].append(old)
    inherited = []
    changed = defaultdict(list)
    for edge in records:
        u, v = int(assignment[edge.source]), int(assignment[edge.target])
        if edge.source not in touched and edge.target not in touched:
            inherited.append(EdgeRecord(edge.edge_id, u, v, edge.relation, edge.weight))
        else:
            changed[(u, v, edge.relation)].append(float(edge.weight))
    next_id = max((row.edge_id for row in records), default=-1) + 1
    for (u, v, relation), weights in sorted(changed.items()):
        weight = float(sum(weights))
        if not np.isfinite(weight):
            raise ValueError("nonfinite aggregated coarse weight")
        inherited.append(EdgeRecord(next_id, u, v, relation, weight))
        next_id += 1
    inherited.sort(key=lambda row: row.edge_id)
    with zipfile.ZipFile(transition) as z:
        next_types = _next_types(groups, _occurrences(z), node_types, step, shape_refs)
    return len(groups), tuple(inherited), tuple(tuple(g) for _, g in sorted(groups.items())), next_types


def _assert_records(actual, expected):
    from semmap_haken.grammar_binary import float_to_bits
    if len(actual) != len(expected):
        raise AssertionError("recursive round-trip assertion count mismatch")
    for a, b in zip(actual, expected):
        if (a.edge_id, a.source, a.target, a.relation, float_to_bits(a.weight)) != (
            b.edge_id, b.source, b.target, b.relation, float_to_bits(b.weight)
        ):
            raise AssertionError("recursive round-trip changed an assertion")


def _get_parts(z: zipfile.ZipFile, prefix: str, registry: ShapeRegistry | None = None):
    manifest = json.loads(z.read(prefix + "/manifest.json"))
    return {
        "manifest": manifest,
        "shapes_raw": _shape_rows(z, prefix, registry),
        "variants_raw": json.loads(z.read(prefix + "/variants.json")),
        "occurrences_data": z.read(prefix + "/occurrences.bin"),
        "internal_data": z.read(prefix + "/internal.bin"),
        "ports_data": z.read(prefix + "/ports.bin"),
        "symbols_data": (z.read(prefix + "/symbols.huff")
                         if manifest.get("symbol_coding") == "huffman" else None),
    }


def decode_recursive(path: Path):
    """Decode every reverse delta, not merely the last transition."""
    with zipfile.ZipFile(path) as outer:
        manifest = json.loads(outer.read("manifest.json"))
        if manifest.get("format") not in (FORMAT, SHARED_FORMAT):
            raise ValueError("unsupported nSBM hierarchy version")
        registry = _registry(outer)
        current_n, current = decode_edge_baseline(io.BytesIO(outer.read("final_graph.zip")))
        ids = json.loads(outer.read("source_ids.json"))
        for prefix in reversed(manifest["transitions"]):
            current_n, current = _decode_transition_delta(
                target_records=current, preserve_ids=True,
                **_get_parts(outer, prefix, registry)
            )
        if current_n != int(manifest["original_vertices"]):
            raise ValueError("recursive hierarchy did not reach level zero")
        if len(current) != int(manifest["original_assertions"]):
            raise ValueError("recursive hierarchy lost assertions")
        return current_n, current, ids


def _state(path: Path):
    """Reconstruct memberships/types directly, without building temporary ZIPs."""
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        registry = _registry(z)
        n, edges = decode_edge_baseline(io.BytesIO(z.read("final_graph.zip")))
        members = [(i,) for i in range(int(manifest["original_vertices"]))]
        types: dict[int, str] = {}
        for level, prefix in enumerate(manifest["transitions"]):
            occurrences = _occurrences(z, prefix, registry)
            assignments, _, _ = _partition_from_occurrence_nodes(
                len(members), [tuple(o.shape_to_fine_nodes) for o in occurrences])
            grouped, children = defaultdict(list), defaultdict(list)
            for old, new in enumerate(assignments):
                grouped[int(new)].extend(members[old])
                children[int(new)].append(old)
            refs = json.loads(z.read(prefix + "/shape_refs.json")) if registry is not None else None
            types = _next_types(children, occurrences, types, level, refs)
            members = [tuple(grouped[i]) for i in range(len(grouped))]
        if len(members) != n:
            raise AssertionError("recursive membership does not fit final graph")
        return n, edges, members, types, len(manifest["transitions"])


def _register_transition(path: Path, registry: ShapeRegistry) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        return registry.references(json.loads(z.read("shapes.json")), manifest["relations"])


def _write_archive(previous: Path | None, *, original_ids: dict,
                   n: int, edges: tuple[EdgeRecord, ...],
                   original_count: int, original_assertions: int,
                   output: Path, transition: Path | None = None,
                   registry: ShapeRegistry | None = None,
                   shape_refs: list[dict] | None = None):
    """Atomic envelope; the supplied registry is published with its transitions.

    No registry keeps the legacy writer contract for existing callers. The
    recursive experiment supplies one registry for the entire new hierarchy.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    prefixes = []
    with zipfile.ZipFile(previous) if previous is not None else nullcontext() as prior:
        if previous is not None:
            old_manifest = json.loads(prior.read("manifest.json"))
            expected = SHARED_FORMAT if registry is not None else FORMAT
            if old_manifest["format"] != expected:
                raise ValueError("do not mix legacy and shared-symbol hierarchies")
            if registry is not None:
                old = json.loads(prior.read("dictionary.json"))
                new = json.loads(registry.dumps())
                if (new["namespace"] != old["namespace"] or
                    new["shapes"][:len(old["shapes"])] != old["shapes"]):
                    raise ValueError("shared dictionary must be append-only")
            prefixes = list(old_manifest["transitions"])
        prefix = None
        if transition is not None:
            prefix = f"transition_{len(prefixes):03d}_{len(prefixes)+1:03d}"
            prefixes.append(prefix)
            if registry is not None and shape_refs is None:
                shape_refs = _register_transition(transition, registry)
        header = {
            "format": SHARED_FORMAT if registry is not None else FORMAT,
            "transitions": prefixes,
            "original_vertices": original_count,
            "original_assertions": original_assertions,
        }
        pending = output.with_name(output.name + ".tmp")
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as outer:
            outer.writestr("manifest.json", json.dumps(header, sort_keys=True))
            if registry is not None:
                outer.writestr("dictionary.json", registry.dumps())
            outer.writestr("source_ids.json",
                (prior.read("source_ids.json") if previous is not None
                 else json.dumps(original_ids, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")))
            outer.writestr("final_graph.zip", encode_edge_baseline_bytes(n, edges),
                           compress_type=zipfile.ZIP_STORED)
            if previous is not None:
                for name in prior.namelist():
                    if name.startswith("transition_"):
                        outer.writestr(name, prior.read(name))
            if transition is not None:
                with zipfile.ZipFile(transition) as delta:
                    if registry is not None:
                        rows = json.loads(delta.read("shapes.json"))
                        relations = json.loads(delta.read("manifest.json"))["relations"]
                        if registry.expand(shape_refs, relations) != rows:
                            raise ValueError("shared references do not reconstruct local shape slots")
                        outer.writestr(prefix + "/shape_refs.json",
                                       json.dumps(shape_refs, separators=(",", ":")))
                    for name in DELTA_NAMES:
                        # Rules are derivable; the exact shared shape is stored once.
                        if registry is not None and name in ("shapes.json", "rules.json"):
                            continue
                        if name in delta.namelist():
                            outer.writestr(prefix + "/" + name, delta.read(name))
        os.replace(pending, output)
    return output.stat().st_size


def _valid_checkpoint(path: Path, meta: Path, *, identity: dict,
                      expected_n: int, expected_edges: tuple[EdgeRecord, ...]):
    if not path.is_file() or not meta.is_file():
        return False
    try:
        data = json.loads(meta.read_text())
        if data.get("identity") != identity or data.get("sha256") != sha256_file(path):
            return False
        n, edges, ids = decode_recursive(path)
        if ids.get("graph_sha256") != identity["graph_sha256"]:
            return False
        if n != expected_n:
            return False
        _assert_records(edges, expected_edges)
        return True
    except (OSError, KeyError, ValueError, AssertionError, zipfile.BadZipFile):
        return False


def run_recursive(
    graph, partitions: list[np.ndarray], directory: Path, *,
    seed: int, cpu_budget: int, discovery: int, max_figures: int,
    max_ego_nodes: int, min_figure_nodes: int, min_support: int,
    batch: int, device: str, graph_hash: str, code_revision: str,
    max_levels: int = 3, resume: bool = False,
    isomorphism_level: str = "recursive", progress_hook=None,
) -> dict:
    """Every accepted level must shorten the *whole* exact hierarchy ZIP."""
    from .grammar_experiment import (
        _degree, _discover, _scan, _select, _views, edge_records,
        _vf2_cpu_plan,
    )
    from .grammar_selection import prune_by_archive_bytes
    from .graph import TypedGraph

    if isomorphism_level not in LEVELS:
        raise ValueError(f"unknown isomorphism level: {isomorphism_level!r}")
    directory.mkdir(parents=True, exist_ok=True)
    original = edge_records(graph)
    names = graph.relation_names
    ids = {"node_ids": list(graph.node_ids),
           "assertion_ids": list(graph.assertion_ids),
           "relation_names": list(names), "graph_sha256": graph_hash}
    identity = {
        "graph_sha256": graph_hash, "code_revision": code_revision,
        "symbol_identity": CONTRACT, "hierarchy_format": SHARED_FORMAT,
        "isomorphism_level": isomorphism_level,
        "seed": seed, "max_levels": max_levels,
        "discovery": discovery, "max_figures": max_figures,
        "max_ego_nodes": max_ego_nodes, "min_figure_nodes": min_figure_nodes,
        "min_support": min_support, "partition_hash": hashlib.sha256(
            b"".join(np.asarray(p, dtype=np.int32).tobytes() for p in partitions)
        ).hexdigest(),
    }
    valid = None
    if resume:
        for archive in sorted(directory.glob("level_???.zip"), reverse=True):
            try:
                with zipfile.ZipFile(archive) as z:
                    archive_format = json.loads(z.read("manifest.json")).get("format")
            except (OSError, KeyError, ValueError, zipfile.BadZipFile):
                archive_format = None
            if archive_format == FORMAT:
                raise ValueError("legacy checkpoint: use a separate NEW directory for shared symbols")
            if _valid_checkpoint(archive, archive.with_suffix(".json"),
                                 identity=identity,
                                 expected_n=graph.vertex_count,
                                 expected_edges=original):
                valid = archive
                break
    if valid is None:
        valid = directory / "level_000.zip"
        _write_archive(None, original_ids=ids, n=graph.vertex_count,
                       edges=original, original_count=graph.vertex_count,
                       original_assertions=graph.edge_count, output=valid,
                       registry=ShapeRegistry())
        atomic_json(valid.with_suffix(".json"), {
            "identity": identity, "sha256": sha256_file(valid),
            "bytes": valid.stat().st_size,
        })
    n, edges, members, node_types, offset = _state(valid)
    with zipfile.ZipFile(valid) as z:
        registry = _registry(z)
    if registry is None:
        raise ValueError("shared-symbol runs require a v2 checkpoint; start a NEW run")
    attempted = []
    if progress_hook is not None:
        for accepted in range(offset):
            progress_hook(accepted, "level_done", 1, 1, 1.0)
    stop_reason = "max_levels"
    for level in range(offset, min(max_levels, len(partitions))):
        level_started = time.perf_counter()
        level_timing = {}
        if progress_hook is not None:
            progress_hook(level, "prepare", 0, 1, 0.0)
        labels = np.asarray(partitions[level], dtype=np.int64)
        if labels.shape != (graph.vertex_count,):
            raise ValueError("nSBM partitions must cover original vertices")
        current_labels = np.array(
            [int(labels[g[0]]) if len({int(labels[v]) for v in g}) == 1 else -1
             for g in members], dtype=np.int32,
        )
        relation_index = {r: i for i, r in enumerate(names)}
        current = TypedGraph(
            tuple(f"c{i}" for i in range(n)), names,
            np.array([e.source for e in edges], dtype=np.int64),
            np.array([e.target for e in edges], dtype=np.int64),
            np.array([relation_index[e.relation] for e in edges], dtype=np.int32),
            np.array([e.weight for e in edges], dtype=np.float64),
            tuple(str(e.edge_id) for e in edges),
        )
        from .runner import _sha_bytes
        current_hash = _sha_bytes(current)
        discovery_started = time.perf_counter()
        degree, _ = _degree(current, device)
        views = {"full": _views(current, None)["full"]}
        extract = max(1, cpu_budget // 3)
        dictionary = _discover(
            current, views, current_labels, degree, budget=discovery,
            max_ego_nodes=max_ego_nodes, min_figure_nodes=min_figure_nodes,
            seed=seed + level, extraction_workers=extract,
            node_types=node_types,
        )
        level_timing["discovery"] = time.perf_counter() - discovery_started
        if progress_hook is not None:
            progress_hook(level, "discovery", 1, 1, 0.12)
        extract, match = _vf2_cpu_plan(cpu_budget, dictionary)
        scan_rows, counts, scan_meta = _scan(
            views, dictionary, n=n, batch=batch, max_ego_nodes=max_ego_nodes,
            min_figure_nodes=min_figure_nodes, extraction_workers=extract,
            match_workers=match, node_types=node_types,
            progress_hook=(lambda done, total: progress_hook(
                level, "VF2", done, total, 0.12 + 0.58 * done / max(1, total))
                if progress_hook is not None else None),
            checkpoint_dir=directory / f"scan_{level:03d}",
            identity={**identity, "level": level,
                      "symbol_registry_sha256": hashlib.sha256(registry.dumps().encode()).hexdigest(),
                      "source_n": n, "current_graph_sha256": current_hash},
        )
        level_timing.update(scan_wall=scan_meta["wall_seconds"],
                            extraction=scan_meta["extraction_seconds"],
                            vf2=scan_meta["vf2_seconds"])
        selection_started = time.perf_counter()
        figures, selection = _select(
            scan_rows, counts, dictionary, graph=current,
            min_support=min_support, max_figures=max_figures,
            records=edges,
        )
        types_by_nodes = {nodes: type_id for type_id, nodes in scan_rows}
        candidates = [
            (types_by_nodes[tuple(group)], tuple(group))
            for group in figures
            if current_labels[group[0]] >= 0 and
            all(current_labels[v] == current_labels[group[0]] for v in group)
        ]
        level_timing["selection"] = time.perf_counter() - selection_started
        if progress_hook is not None:
            progress_hook(level, "selection", 1, 1, 0.78)
        pruning_started = time.perf_counter()
        figures, measured = prune_by_archive_bytes(
            vertex_count=n, records=edges, candidates=candidates,
            workspace=directory, node_types=node_types,
        )
        level_timing["measured_pruning"] = time.perf_counter() - pruning_started
        if progress_hook is not None:
            progress_hook(level, "pruning", 1, 1, 0.88)
        if not figures:
            stop_reason = "no_profitable_symbols"
            attempted.append({"scale": level, "figures": 0,
                              "decision": stop_reason, "selection": selection,
                              "measured": measured,
                              "scan": scan_meta,
                              "cpu": {"extract": extract, "vf2": match},
                              "phase_timing_seconds": level_timing})
            if progress_hook is not None:
                progress_hook(level, "level_done", 1, 1, 1.0)
            break
        archive_started = time.perf_counter()
        transition = directory / f"transition_{level:03d}.zip"
        encode_grammar(n, edges, figures, transition,
                       node_types=node_types, symbol_coding="huffman")
        shape_refs = _register_transition(transition, registry)
        coarse_n, coarse_edges, coarse_groups, next_types = contract_records(
            n, edges, figures, node_types, level, transition, shape_refs=shape_refs)
        new_members = [
            tuple(x for node in group for x in members[node])
            for group in coarse_groups
        ]
        candidate = directory / f"level_{level+1:03d}.zip"
        _write_archive(valid, original_ids=ids, n=coarse_n,
                       edges=coarse_edges, original_count=graph.vertex_count,
                       original_assertions=graph.edge_count,
                       output=candidate, transition=transition,
                       registry=registry, shape_refs=shape_refs)
        restored_n, restored, restored_ids = decode_recursive(candidate)
        if restored_n != graph.vertex_count or restored_ids != ids:
            raise AssertionError("recursive archive changed original identifiers")
        _assert_records(restored, original)
        transition.unlink(missing_ok=True)
        level_timing["archive_and_validation"] = time.perf_counter() - archive_started
        level_timing["total_wall"] = time.perf_counter() - level_started
        if progress_hook is not None:
            progress_hook(level, "archive_verified", 1, 1, 0.98)
        gain = valid.stat().st_size - candidate.stat().st_size
        attempted.append({"scale": level, "figures": len(figures),
                          "decision": "accepted" if gain > 0 else "not_smaller",
                          "saved_bytes": gain, "coarse_vertices": coarse_n,
                          "selection": selection, "measured": measured,
                          "scan": scan_meta,
                          "cpu": {"extract": extract, "vf2": match},
                          "phase_timing_seconds": level_timing})
        if gain <= 0:
            candidate.unlink()
            stop_reason = "no_actual_hierarchy_gain"
            if progress_hook is not None:
                progress_hook(level, "level_done", 1, 1, 1.0)
            break
        atomic_json(candidate.with_suffix(".json"), {
            "identity": identity, "sha256": sha256_file(candidate),
            "bytes": candidate.stat().st_size,
        })
        valid, n, edges, members, node_types = (
            candidate, coarse_n, coarse_edges, new_members, next_types
        )
        if progress_hook is not None:
            progress_hook(level, "level_done", 1, 1, 1.0)
    # Reload the accepted snapshot: a rejected proposal must not leak symbols.
    with zipfile.ZipFile(valid) as z:
        registry = _registry(z)
    summary = {"format": SHARED_FORMAT, "archive": str(valid),
               "archive_sha256": sha256_file(valid),
               "archive_bytes": valid.stat().st_size,
               "dictionary_namespace": registry.namespace,
               "global_symbols": len(registry.shapes),
               "isomorphism_level": isomorphism_level,
               "similarity_classes": registry.classes(isomorphism_level),
               "accepted_levels": offset + sum(
                   row["decision"] == "accepted" for row in attempted),
               "attempted": attempted, "stop_reason": stop_reason,
               "roundtrip_exact": True,
               "fidelity": "all original directed typed assertion records and binary64 weights"}
    atomic_json(directory / "recursive_summary.json", summary)
    return summary
