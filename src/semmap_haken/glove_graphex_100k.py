"""ConceptNet 100k / GloVe / Wishart / finite Graphex codebook experiment.

Losslessness refers to *saved relation-layer CSR records and node memberships*.
W/S/I are a finite descriptive partition, NOT an estimated exchangeable limit;
R is the explicit exact residual. The statistical block model is evaluated
separately, and no compression or dynamical gain is assumed in advance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import shutil
import struct
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from .graph_mdl import build_canonical_huffman_codes
from .graphex_components import EdgeRecord, classify_edges, validate_partition
from .wishart_cluster import wishart_cluster

FORMAT = "semmap_glove_graphex_exact_v2"


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read_members(level):
    raw = json.loads((Path(level) / "membership.json").read_text(encoding="utf8"))
    if set(map(int, raw)) != set(range(len(raw))):
        raise ValueError("membership node IDs must be contiguous from zero")
    members = []
    for i in range(len(raw)):
        value = raw[str(i)]
        concepts = value.get("original_concepts", []) if isinstance(value, dict) else value
        if isinstance(concepts, str): concepts = [concepts]
        if not concepts or not all(isinstance(x, str) for x in concepts):
            raise ValueError(f"missing ConceptNet URI for node {i}")
        members.append(list(concepts))
    return members


def read_graph(level, expected_nodes=None):
    level = Path(level)
    adjacency = sparse.load_npz(level / "adjacency.npz").tocsr()
    n = adjacency.shape[0]
    if adjacency.shape != (n, n) or (expected_nodes is not None and n != expected_nodes):
        raise ValueError(f"expected {expected_nodes} prepared concepts, got {adjacency.shape}")
    members = read_members(level)
    if len(members) != n: raise ValueError("membership / graph size mismatch")
    index = json.loads((level / "relations" / "index.json").read_text(encoding="utf8"))
    edges = []
    for relation, filename in sorted(index.items()):
        matrix = sparse.load_npz(level / "relations" / filename).tocsr()
        if matrix.shape != (n, n): raise ValueError("relation shape mismatch")
        matrix.sum_duplicates(); matrix.sort_indices()
        if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0):
            raise ValueError("nonfinite or negative edge weights")
        rows = np.repeat(np.arange(n), np.diff(matrix.indptr))
        for u, v, w in zip(rows, matrix.indices, matrix.data):
            edges.append(EdgeRecord(len(edges), int(u), int(v), relation, float(w)))
    return adjacency, members, tuple(edges)


def node_tokens(uri):
    parts = urllib.parse.unquote(uri).split("/")
    if len(parts) < 4 or parts[1:3] != ["c", "en"]: return []
    return re.findall(r"[a-z]+(?:'[a-z]+)?", parts[3].replace("_", " ").lower())


def align_glove(members, glove_text, output, dimension=100):
    """Stream ONLY vocabulary needed for this graph, not a 400k-word Python dict."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    tokens = [node_tokens(uris[0]) for uris in members]
    wanted = {token for row in tokens for token in row}
    found = {}
    with open(glove_text, "r", encoding="utf8") as source:
        for line in source:
            word, sep, values = line.partition(" ")
            if sep and word in wanted and word not in found:
                vector = np.fromstring(values, dtype=np.float32, sep=" ")
                if vector.size == dimension and np.all(np.isfinite(vector)):
                    norm = float(np.linalg.norm(vector))
                    if norm > 0: found[word] = vector / norm
    vectors = np.zeros((len(members), dimension), np.float32)
    coverage = np.zeros(len(members), np.uint8)  # 0=OOV, 1=partial, 2=full
    for i, words in enumerate(tokens):
        matched = [found[w] for w in words if w in found]
        if not matched: continue
        vec = np.mean(matched, axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vectors[i] = vec / norm
            coverage[i] = 2 if len(matched) == len(words) else 1
    np.save(output / "vectors.npy", vectors)
    np.save(output / "coverage.npy", coverage)
    report = {"nodes": len(members), "dimension": dimension, "vocab_needed": len(wanted),
              "vocab_found": len(found), "full": int((coverage == 2).sum()),
              "partial": int((coverage == 1).sum()), "oov": int((coverage == 0).sum()),
              "policy": "mean of normalized English word vectors; no synthetic OOV vector",
              "glove_sha256": sha(glove_text)}
    write_json(output / "report.json", report)
    return report


def cuda_knn(vectors, coverage, directory, k=12, batch=256, shard_rows=4096, seed=1729, checkpoint_drive=None):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required for 100k kNN")
    valid = np.flatnonzero(coverage > 0)
    if len(valid) <= k: raise ValueError("not enough GloVe-covered concepts")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    # A tile is persisted individually so an interrupted session resumes by shard.
    torch.manual_seed(seed)
    base = torch.as_tensor(np.asarray(vectors[valid]), dtype=torch.float32, device="cuda")
    base = torch.nn.functional.normalize(base, dim=1)
    all_idx, all_dist = [], []
    tile = max(1, batch)
    shard_rows = max(tile, shard_rows)
    for start in range(0, len(valid), shard_rows):
        stop = min(start + shard_rows, len(valid))
        shard = directory / f"shard_{start:07d}_{stop:07d}.npz"
        remote = Path(checkpoint_drive) / "knn" / shard.name if checkpoint_drive else None
        if not shard.is_file() and remote is not None and remote.is_file():
            shard.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(remote, shard)
        if shard.is_file():
            with np.load(shard) as obj:
                ix, ds = obj["indices"], obj["distances"]
            if ix.shape != (stop - start, k) or ds.shape != ix.shape:
                raise ValueError(f"invalid existing kNN shard {shard}")
        else:
            indices_parts, distance_parts = [], []
            offset = start
            while offset < stop:
                end = min(offset + tile, stop)
                try:
                    with torch.no_grad():
                        similarity = base[offset:end] @ base.T
                        distance = (1 - similarity.clamp(-1, 1)).clamp_min(0)
                        distance[torch.arange(end - offset, device="cuda"),
                                 torch.arange(offset, end, device="cuda")] = float("inf")
                        ds_gpu, ix_gpu = torch.topk(distance, k, dim=1, largest=False, sorted=True)
                        indices_parts.append(ix_gpu.cpu().numpy().astype(np.int32))
                        distance_parts.append(ds_gpu.cpu().numpy().astype(np.float32))
                    offset = end
                except torch.cuda.OutOfMemoryError:
                    if tile <= 1: raise
                    tile = max(1, tile // 2)
            ix = np.concatenate(indices_parts); ds = np.concatenate(distance_parts)
            np.savez_compressed(shard, indices=ix, distances=ds)
            if remote is not None:
                remote.parent.mkdir(parents=True, exist_ok=True)
                temporary = remote.with_suffix(remote.suffix + ".part")
                shutil.copyfile(shard, temporary); os.replace(temporary, remote)
        all_idx.append(ix); all_dist.append(ds)
    np.save(directory / "valid.npy", valid)
    np.savez_compressed(directory / "knn.npz", indices=np.concatenate(all_idx),
                        distances=np.concatenate(all_dist))
    report = {"gpu": torch.cuda.get_device_name(0), "backend": "torch.float32 tiled exact top-k",
              "k": k, "queries": len(valid), "query_batch": tile, "shard_rows": shard_rows, "shards": len(all_idx),
              "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "note": "exact top-k for fp32 similarities; near-ties can vary across kernels"}
    write_json(directory / "report.json", report)
    return report


def cluster_wishart(directory, n, k=12, significance=0.7, min_cluster_size=3):
    directory = Path(directory)
    with np.load(directory / "knn.npz") as data:
        w = wishart_cluster(data["indices"], data["distances"],
                            significance=significance, min_cluster_size=min_cluster_size)
    valid = np.load(directory / "valid.npy")
    labels = np.full(n, -1, np.int32)
    labels[valid] = w.labels.astype(np.int32)
    np.save(directory / "labels.npy", labels)
    result = {"clusters": w.cluster_count, "noise_and_oov": int((labels < 0).sum()),
              "sizes": {str(key): int(value) for key, value in w.cluster_sizes.items()},
              "significance": significance, "min_cluster_size": min_cluster_size}
    write_json(directory / "wishart.json", result)
    return result


def varint(num):
    if num < 0: raise ValueError("negative varint")
    out = bytearray()
    while num >= 128:
        out.append((num & 127) | 128); num >>= 7
    out.append(num)
    return bytes(out)


def read_varint(stream):
    value = shift = 0
    for _ in range(10):
        x = stream.read(1)
        if not x: raise ValueError("truncated varint")
        b = x[0]; value |= (b & 127) << shift
        if b < 128: return value
        shift += 7
    raise ValueError("invalid varint")


def binary_edges(records, relations):
    """Source-delta sorted exact directed/typed records including edge IDs."""
    from io import BytesIO
    out = BytesIO(); last_source = 0
    ids = {relation: i for i, relation in enumerate(relations)}
    for e in sorted(records, key=lambda e: (e.source, e.target, e.edge_id)):
        out.write(varint(e.source - last_source)); last_source = e.source
        out.write(varint(e.target)); out.write(varint(ids[e.relation]))
        out.write(varint(e.edge_id)); out.write(struct.pack("<d", e.weight))
    return out.getvalue()


def unbinary_edges(data, relations, count):
    from io import BytesIO
    stream = BytesIO(data); last_source = 0; result = []
    for _ in range(count):
        last_source += read_varint(stream)
        target = read_varint(stream); relation = read_varint(stream)
        eid = read_varint(stream)
        weight = struct.unpack("<d", stream.read(8))[0]
        result.append(EdgeRecord(eid, last_source, target, relations[relation], weight))
    if stream.read(1): raise ValueError("trailing edge bytes")
    return result


def huffman_bits(symbols, codes):
    buffer = bytearray(); current = used = count = 0
    for symbol in symbols:
        for bit in codes[symbol]:
            current = (current << 1) | (bit == "1"); used += 1; count += 1
            if used == 8: buffer.append(current); current = used = 0
    if used: buffer.append(current << (8 - used))
    return bytes(buffer), count


def huffman_symbols(data, bits, n, codes):
    reverse = {v: k for k, v in codes.items()}; result = []; prefix = ""
    if len(reverse) != len(codes) or bits > 8 * len(data): raise ValueError("bad Huffman codebook")
    for i in range(bits):
        prefix += "1" if data[i >> 3] & (128 >> (i & 7)) else "0"
        if prefix in reverse: result.append(reverse[prefix]); prefix = ""
    if prefix or len(result) != n: raise ValueError("bad Huffman stream")
    return result


def discover_pairs(
    records,
    labels,
    max_figures=4096,
    min_support=5,
    semantic_prior="ranking",
):
    """Exact two-port motifs with an optional GloVe/Wishart semantic prior.

    ranking (default) never forbids a structural motif: Wishart agreement
    only breaks ties between equally useful shapes. filter reproduces the
    legacy same-cluster restriction. disabled ignores Wishart labels.
    """
    if semantic_prior not in {"ranking", "filter", "disabled"}:
        raise ValueError("semantic_prior must be ranking, filter or disabled")
    pairs = defaultdict(list)
    pair_semantic = {}
    loop_vertices = {e.source for e in records if e.source == e.target}
    for e in records:
        if e.source == e.target or e.source in loop_vertices or e.target in loop_vertices:
            continue
        same_family = (
            int(labels[e.source]) >= 0
            and int(labels[e.source]) == int(labels[e.target])
        )
        if semantic_prior == "filter" and not same_family:
            continue
        key = (min(e.source, e.target), max(e.source, e.target))
        pairs[key].append(e)
        pair_semantic[key] = bool(same_family)

    groups = defaultdict(list)
    semantic_support = Counter()
    for (u, v), edges in pairs.items():
        shape = tuple(sorted((int(e.source == v), e.relation) for e in edges))
        groups[shape].append((u, v))
        if pair_semantic[(u, v)]:
            semantic_support[shape] += 1

    allowed = [shape for shape, occ in groups.items() if len(occ) >= min_support]
    if semantic_prior == "ranking":
        allowed.sort(
            key=lambda shape: (
                -len(groups[shape]) * len(shape),
                -semantic_support[shape],
                shape,
            )
        )
    else:
        allowed.sort(key=lambda shape: (-len(groups[shape]) * len(shape), shape))

    selected = []
    used = set()
    for shape in allowed:
        occurrences = sorted(
            groups[shape],
            key=lambda pair: (
                -int(pair_semantic.get(pair, False))
                if semantic_prior == "ranking" else 0,
                pair,
            ),
        )
        for u, v in occurrences:
            if u not in used and v not in used:
                selected.append((u, v, shape))
                used.add(u)
                used.add(v)
                if len(selected) == max_figures:
                    return selected
    return selected


def block_model(records, labels, max_blocks=128, max_pairs=20000):
    freqs = Counter(int(x) for x in labels if x >= 0)
    allowed = set(k for k, _ in freqs.most_common(max_blocks))
    counts = Counter()
    for e in records:
        a, b = int(labels[e.source]), int(labels[e.target])
        if a in allowed and b in allowed: counts[(a, b, e.relation)] += 1
    # Count/(|a||b|) is a directed multi-edge INTENSITY, not necessarily a probability.
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_pairs]
    return {"scope": "finite empirical typed directed block intensity; not a fitted limit graphex",
            "blocks": {str(k): v for k, v in freqs.items() if k in allowed},
            "edges": [[a, b, rel, cnt, cnt / (freqs[a] * freqs[b])]
                      for (a, b, rel), cnt in items],
            "truncated_pairs": max(0, len(counts) - len(items))}


def connector_profiles(records, labels):
    """Exact directed/typed external interface of each GloVe-Wishart family."""
    profile = defaultdict(lambda: {"internal_weight": 0.0, "out_weight": 0.0,
        "in_weight": 0.0, "out_by_relation": defaultdict(float),
        "in_by_relation": defaultdict(float)})
    for e in records:
        a, b = int(labels[e.source]), int(labels[e.target])
        if a >= 0:
            if a == b: profile[a]["internal_weight"] += e.weight
            else:
                profile[a]["out_weight"] += e.weight
                profile[a]["out_by_relation"][e.relation] += e.weight
        if b >= 0 and a != b:
            profile[b]["in_weight"] += e.weight
            profile[b]["in_by_relation"][e.relation] += e.weight
    return {str(k): {**{field: value for field, value in data.items()
                        if not isinstance(value, defaultdict)},
                      "out_by_relation": dict(data["out_by_relation"]),
                      "in_by_relation": dict(data["in_by_relation"]),
                      "exit_probability_proxy": (data["out_weight"] /
                        (data["out_weight"] + data["internal_weight"])
                        if data["out_weight"] + data["internal_weight"] else 0.0)}
            for k, data in sorted(profile.items())}


def encode_exact(path, adjacency_bytes, members, records, labels, selected, block=None, connectors=None):
    """Encode W prototypes + exact S/I/R, with measurable complete ZIP size."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rels = sorted({e.relation for e in records})
    groups = [(u, v) for u, v, _ in selected]
    assigned = classify_edges(len(members), records, groups)
    validate_partition(records, assigned)
    shapes = sorted(set(shape for _, _, shape in selected))
    shape_ids = {shape: str(i) for i, shape in enumerate(shapes)}
    occurrence = [[u, v] for u, v, _ in selected]
    occurrence_symbols = [shape_ids[shape] for _, _, shape in selected]
    codes = build_canonical_huffman_codes(Counter(occurrence_symbols))
    bits, nbits = huffman_bits(occurrence_symbols, codes)
    label_symbols = ["OOV_OR_NOISE" if int(value) < 0 else str(int(value)) for value in labels]
    label_codes = build_canonical_huffman_codes(Counter(label_symbols))
    label_bits, label_nbits = huffman_bits(label_symbols, label_codes)
    owner_edges = defaultdict(list); parts = {key: [] for key in ("S", "I", "R")}
    for a in assigned:
        if a.part == "W": owner_edges[a.owner].append(a.edge)
        else: parts[a.part].append(a.edge)
    # Slot order is uniquely determined by direction, relation, edge_id.
    from io import BytesIO
    wdata = BytesIO()
    for oid, (_, v, shape) in enumerate(selected):
        es = sorted(owner_edges[oid], key=lambda e: (int(e.source == v), e.relation, e.edge_id))
        if tuple((int(e.source == v), e.relation) for e in es) != shape:
            raise AssertionError("prototype / edge ownership mismatch")
        for e in es: wdata.write(varint(e.edge_id)); wdata.write(struct.pack("<d", e.weight))
    meta = {"format": FORMAT, "vertex_count": len(members), "edge_count": len(records),
            "relations": rels, "shape_count": len(shapes), "occurrence_count": len(selected),
            "shape_codes": codes, "shape_bits": nbits,
            "label_codes": label_codes, "label_bits": label_nbits,
            "partition": {k: len(owner_edges) if k == "W_owners" else len(parts[k])
                          for k in ("S", "I", "R")}, "w_records": sum(map(len, owner_edges.values())),
            "exactness_scope": "saved directed typed CSR records, original membership JSON and adjacency.npz bytes"}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("manifest.json", json.dumps(meta, sort_keys=True))
        z.writestr("block_model.json", json.dumps(block or {}, sort_keys=True, separators=(",", ":")))
        z.writestr("connectors.json", json.dumps(connectors or {}, sort_keys=True, separators=(",", ":")))
        z.writestr("members.json", json.dumps(members, ensure_ascii=False, separators=(",", ":")))
        z.writestr("adjacency.npz", adjacency_bytes, compress_type=zipfile.ZIP_STORED)
        z.writestr("shapes.json", json.dumps(shapes, separators=(",", ":")))
        z.writestr("occurrences.json", json.dumps(occurrence, separators=(",", ":")))
        z.writestr("shapes.bin", bits)
        z.writestr("labels.bin", label_bits)
        z.writestr("W.bin", wdata.getvalue())
        for part in ("S", "I", "R"):
            z.writestr(part + ".bin", binary_edges(parts[part], rels))
    n, restored_members, restored, adj = decode_exact(path)
    if (n != len(members) or restored_members != members or restored != tuple(records)
            or adj != adjacency_bytes or not np.array_equal(decode_labels(path), labels)):
        path.unlink(missing_ok=True)
        raise AssertionError("failed lossless graph / URI / adjacency roundtrip")
    return {"archive_bytes": path.stat().st_size, "roundtrip_exact": True,
            "shapes": len(shapes), "occurrences": len(selected),
            "W": meta["w_records"], "S": len(parts["S"]), "I": len(parts["I"]), "R": len(parts["R"])}


def decode_labels(path):
    """Decode the canonical Huffman node-to-Wishart-class map from the archive."""
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("manifest.json"))
        symbols = huffman_symbols(z.read("labels.bin"), meta["label_bits"],
                                  meta["vertex_count"], meta["label_codes"])
    return np.array([-1 if symbol == "OOV_OR_NOISE" else int(symbol) for symbol in symbols],
                    dtype=np.int32)

def decode_exact(path):
    from io import BytesIO
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("manifest.json"))
        if meta["format"] != FORMAT: raise ValueError("wrong codec version")
        n = meta["vertex_count"]; members = json.loads(z.read("members.json"))
        adj = z.read("adjacency.npz")
        _ = huffman_symbols(z.read("labels.bin"), meta["label_bits"],
                            n, meta["label_codes"])
        shapes = json.loads(z.read("shapes.json"))
        occurrence = json.loads(z.read("occurrences.json"))
        shape_symbols = huffman_symbols(z.read("shapes.bin"), meta["shape_bits"],
                                        meta["occurrence_count"], meta["shape_codes"])
        wstream = BytesIO(z.read("W.bin")); restored = []
        for (u, v), sym in zip(occurrence, shape_symbols):
            for direction, relation in shapes[int(sym)]:
                eid = read_varint(wstream)
                raw = wstream.read(8)
                if len(raw) != 8: raise ValueError("truncated W weight")
                restored.append(EdgeRecord(eid, v if direction else u,
                                           u if direction else v, relation, struct.unpack("<d", raw)[0]))
        if wstream.read(1): raise ValueError("trailing W bytes")
        for part in ("S", "I", "R"):
            count = meta["partition"][part]
            restored.extend(unbinary_edges(z.read(part + ".bin"), meta["relations"], count))
    restored.sort(key=lambda e: e.edge_id)
    if len(restored) != meta["edge_count"] or [e.edge_id for e in restored] != list(range(len(restored))):
        raise ValueError("decoded record IDs not exact")
    if len(members) != n or any(not 0 <= e.source < n or not 0 <= e.target < n for e in restored):
        raise ValueError("decoded indices invalid")
    return n, members, tuple(restored), adj


def baseline_zip(path, adjacency_bytes, members, records):
    """Independent DEFLATE JSON baseline with IDENTICAL reconstruction scope."""
    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("members.json", json.dumps(members, ensure_ascii=False, separators=(",", ":")))
        z.writestr("adjacency.npz", adjacency_bytes, compress_type=zipfile.ZIP_STORED)
        with z.open("records.jsonl", "w") as stream:
            for e in records:
                stream.write((json.dumps([e.edge_id, e.source, e.target, e.relation, e.weight],
                                         separators=(",", ":")) + "\n").encode("utf8"))
    return path.stat().st_size


def coarse_graph(adjacency, labels):
    n = len(labels); _, comp = connected_components((adjacency + adjacency.T).tocsr(), directed=False)
    mapping = np.empty(n, dtype=np.int32); dictionary = {}; next_id = 0
    for i, (label, component) in enumerate(zip(labels, comp)):
        key = (int(label), int(component)) if label >= 0 else ("singleton", i)
        if key not in dictionary: dictionary[key] = next_id; next_id += 1
        mapping[i] = dictionary[key]
    p = sparse.csr_matrix((np.ones(n), (np.arange(n), mapping)), shape=(n, next_id))
    q = (p.T @ adjacency @ p).tocsr()  # NEVER symmetrize the encoded graph.
    if not np.isclose(float(adjacency.sum()), float(q.sum()), rtol=1e-10, atol=1e-7):
        raise AssertionError("lost graph weight in quotient")
    return mapping, q


def slow_spectrum(adjacency, k=8):
    g = (adjacency + adjacency.T).tocsr().astype(np.float64) * 0.5
    g.data = np.maximum(g.data, 0)
    degree = np.asarray(g.sum(axis=1)).ravel()
    inv = np.zeros(len(degree)); inv[degree > 0] = 1 / np.sqrt(degree[degree > 0])
    normalized = (sparse.diags(inv) @ g @ sparse.diags(inv)).tocsr()
    top = min(k + 8, len(degree) - 2)
    if top < 1: return [], np.empty((len(degree), 0))
    vals, vecs = eigsh(normalized, k=top, which="LA", tol=1e-5, maxiter=5000)
    idx = np.argsort(vals)[::-1]
    return vals[idx].tolist(), vecs[:, idx]


def haken_report(adjacency, labels, output, modes=8, alpha=1.0, beta=0.95):
    from scipy.linalg import subspace_angles
    mapping, quotient = coarse_graph(adjacency, labels)
    original_vals, u = slow_spectrum(adjacency, modes)
    coarse_vals, v = slow_spectrum(quotient, modes)
    # Compare only nontrivial eigenmodes; eigenvectors have arbitrary signs.
    ui = [i for i, eig in enumerate(original_vals) if eig < 1 - 1e-7][:modes]
    vi = [i for i, eig in enumerate(coarse_vals) if eig < 1 - 1e-7][:modes]
    count = min(len(ui), len(vi))
    angles = []
    if count:
        a = u[:, ui[:count]]
        b = v[:, vi[:count]][mapping]
        b /= np.sqrt(np.bincount(mapping, minlength=quotient.shape[0])[mapping])[:, None]
        a, _ = np.linalg.qr(a); b, _ = np.linalg.qr(b)
        angles = np.degrees(subspace_angles(a, b)).tolist()
    pi = np.asarray(((adjacency + adjacency.T) * 0.5).sum(axis=1)).ravel()
    pq = np.asarray(((quotient + quotient.T) * 0.5).sum(axis=1)).ravel()
    pi = pi / pi.sum() if pi.sum() else pi
    pq = pq / pq.sum() if pq.sum() else pq
    agg = np.bincount(mapping, weights=pi, minlength=len(pq))
    report = {"original_nodes": adjacency.shape[0], "quotient_nodes": quotient.shape[0],
              "original_eigenvalues": original_vals, "quotient_eigenvalues": coarse_vals,
              "original_haken_rates": [beta * x - alpha for x in original_vals],
              "quotient_haken_rates": [beta * x - alpha for x in coarse_vals],
              "nontrivial_principal_angles_degrees": angles,
              "stationary_mass_total_variation": float(0.5 * np.abs(agg - pq).sum()),
              "model": "linearized Haken on symmetrized normalized adjacency; beta is shared",
              "note": "no nonlinear center-manifold assertion; large angles mean dynamic distortion"}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.save(output / "fine_to_coarse.npy", mapping)
    sparse.save_npz(output / "quotient.npz", quotient)
    write_json(output / "haken.json", report)
    return report


def run_identity(source_level, glove, args):
    level = Path(source_level)
    rel_dir = level / "relations"
    paths = [level / "adjacency.npz", level / "membership.json", rel_dir / "index.json"]
    index = json.loads(paths[2].read_text(encoding="utf8"))
    paths += [rel_dir / index[key] for key in sorted(index)]
    revision = "unknown"
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"format": FORMAT, "code_revision": revision,
            "source": {x.relative_to(level).as_posix(): sha(x) for x in paths},
            "glove_sha256": sha(glove), "expected_nodes": args.expected_nodes,
            "k": args.k, "batch": args.batch, "max_figures": args.max_figures,
            "min_support": args.min_support,
            "semantic_pair_mode": args.semantic_pair_mode}


STAGE_FILES = {
    "align": ["embedding/vectors.npy", "embedding/coverage.npy", "embedding/report.json"],
    "knn": ["knn/valid.npy", "knn/knn.npz", "knn/report.json"],
    "wishart": ["knn/labels.npy", "knn/wishart.json"],
    "codec": ["graph_exact.zip", "codec_report.json", "block_model.json", "connectors.json", "baseline.zip"],
    "haken": ["dynamics/quotient.npz", "dynamics/fine_to_coarse.npy", "dynamics/haken.json"],
}


def stage_valid(run, phase):
    check = Path(run) / "checkpoints" / f"{phase}.json"
    if not check.exists(): return False
    stored = json.loads(check.read_text(encoding="utf8"))
    current = {rel: sha(Path(run) / rel) for rel in STAGE_FILES[phase]}
    if current != stored: raise ValueError(f"corrupted {phase} checkpoint; do not silently resume")
    return True


def stage_commit(run, phase):
    write_json(Path(run) / "checkpoints" / f"{phase}.json",
               {rel: sha(Path(run) / rel) for rel in STAGE_FILES[phase]})


def verify_completed(source_level, run, expected_nodes):
    for phase in STAGE_FILES:
        if not stage_valid(run, phase): raise ValueError(f"missing {phase} checkpoint")
    adjacency, members, records = read_graph(source_level, expected_nodes)
    n, restored_members, restored, adjacency_bytes = decode_exact(Path(run) / "graph_exact.zip")
    if n != expected_nodes or restored_members != members or restored != records:
        raise AssertionError("full-run reconstruction failed")
    if adjacency_bytes != (Path(source_level) / "adjacency.npz").read_bytes():
        raise AssertionError("original adjacency bytes differ")
    report = json.loads((Path(run) / "codec_report.json").read_text(encoding="utf8"))
    write_json(Path(run) / "verification.json",
               {"lossless": True, "n": n, "edge_records": len(records),
                "archive_bytes": report["selected_archive_bytes"],
                "baseline_bytes": report["baseline_bytes"],
                "net_saved_bytes": report["net_saved_bytes"]})
    (Path(run) / "COMPLETED").write_text("validated\n", encoding="ascii")
    return report


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-level", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("align", "knn", "wishart", "codec", "haken", "verify", "all"), default="all")
    parser.add_argument("--expected-nodes", type=int, default=100000)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max-figures", type=int, default=4096)
    parser.add_argument("--checkpoint-drive", type=Path, default=None)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument(
        "--semantic-pair-mode",
        choices=("ranking", "filter", "disabled"),
        default="ranking",
        help="Use GloVe/Wishart as ranking prior, legacy hard filter, or ignore it.",
    )
    args = parser.parse_args(argv)
    run = args.run; run.mkdir(parents=True, exist_ok=True)
    identity = run_identity(args.source_level, args.glove, args)
    identity_path = run / "input.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf8")) != identity:
            raise ValueError("source, GloVe, code revision or configuration changed; use a NEW run")
    else:
        if any(run.iterdir()): raise ValueError("existing run has no input manifest")
        write_json(identity_path, identity)
    phases = ["align", "knn", "wishart", "codec", "haken", "verify"] if args.stage == "all" else [args.stage]
    adjacency = members = records = None
    if "codec" in phases:
        adjacency, members, records = read_graph(args.source_level, args.expected_nodes)
    elif "align" in phases:
        members = read_members(args.source_level)
        if len(members) != args.expected_nodes: raise ValueError("wrong prepared graph size")
    for phase in phases:
        if phase == "verify":
            print(json.dumps(verify_completed(args.source_level, run, args.expected_nodes)), flush=True)
            continue
        if stage_valid(run, phase):
            print(json.dumps({"stage": phase, "resume": "verified checkpoint"}), flush=True)
            continue
        if phase == "align":
            print(json.dumps(align_glove(members, args.glove, run / "embedding")), flush=True)
        elif phase == "knn":
            print(json.dumps(cuda_knn(np.load(run / "embedding" / "vectors.npy", mmap_mode="r"),
                        np.load(run / "embedding" / "coverage.npy"), run / "knn",
                        k=args.k, batch=args.batch, checkpoint_drive=args.checkpoint_drive)), flush=True)
        elif phase == "wishart":
            print(json.dumps(cluster_wishart(run / "knn", args.expected_nodes, k=args.k)), flush=True)
        elif phase == "codec":
            labels = np.load(run / "knn" / "labels.npy")
            adj_bytes = (args.source_level / "adjacency.npz").read_bytes()
            baseline = baseline_zip(run / "baseline.zip", adj_bytes, members, records)
            report = {"baseline_bytes": baseline, "models": {}}
            model = block_model(records, labels)
            write_json(run / "block_model.json", model)
            connectors = connector_profiles(records, labels)
            write_json(run / "connectors.json", connectors)
            figures = discover_pairs(
                records,
                labels,
                args.max_figures,
                args.min_support,
                semantic_prior=args.semantic_pair_mode,
            )
            options = (0, min(1024, len(figures)), len(figures))
            best = None
            for cap in sorted(set(options)):
                path = run / f"codec_figures_{cap}.zip"
                metrics = encode_exact(path, adj_bytes, members, records, labels, figures[:cap], block=model, connectors=connectors)
                report["models"][str(cap)] = metrics
                if best is None or metrics["archive_bytes"] < best[0]: best = (metrics["archive_bytes"], path)
            shutil.copy2(best[1], run / "graph_exact.zip")
            report.update({"selected_archive_bytes": best[0],
                           "net_saved_bytes": baseline - best[0],
                           "ratio": best[0] / baseline,
                           "win": best[0] < baseline,
                           "scope": "prepared CSR, not original unsymmetrized ConceptNet assertions"})
            write_json(run / "codec_report.json", report)
            print(json.dumps(report), flush=True)
        elif phase == "haken":
            if adjacency is None: adjacency = sparse.load_npz(args.source_level / "adjacency.npz")
            labels = np.load(run / "knn" / "labels.npy")
            print(json.dumps(haken_report(adjacency, labels, run / "dynamics")), flush=True)
        stage_commit(run, phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
