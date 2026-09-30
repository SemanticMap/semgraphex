"""Compact exact EdgeRecord baseline for grammar-compression experiments.

This is intentionally *not* a grammar. It stores the same directed, typed,
weighted record stream with integer/varint fields and exact IEEE-754 binary64
weights. Use it as a stronger storage baseline than DEFLATE-compressed JSON.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from typing import Iterable

from .grammar_binary import (
    bits_to_float,
    float_to_bits,
    read_u64,
    read_uvarint,
    write_u64,
    write_uvarint,
)
from .graphex_components import EdgeRecord

FORMAT = "semmap_edge_exact_baseline_v1"


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate(vertex_count: int, records: tuple[EdgeRecord, ...]) -> None:
    if vertex_count < 0:
        raise ValueError("vertex_count must be non-negative")
    if len({int(row.edge_id) for row in records}) != len(records):
        raise ValueError("edge_id must be unique")
    for row in records:
        if not 0 <= int(row.source) < vertex_count:
            raise ValueError("edge source outside graph")
        if not 0 <= int(row.target) < vertex_count:
            raise ValueError("edge target outside graph")
        if not math.isfinite(float(row.weight)):
            raise ValueError("non-finite edge weight")


def _write_edges(
    records: tuple[EdgeRecord, ...],
    relation_to_id: dict[str, int],
    *,
    implicit_edge_ids: bool,
) -> bytes:
    stream = io.BytesIO()
    write_uvarint(stream, len(records))
    for row in records:
        if not implicit_edge_ids:
            write_uvarint(stream, int(row.edge_id))
        write_uvarint(stream, int(row.source))
        write_uvarint(stream, int(row.target))
        write_uvarint(stream, relation_to_id[str(row.relation)])
        write_u64(stream, float_to_bits(float(row.weight)))
    return stream.getvalue()


def _read_edges(
    data: bytes,
    relations: tuple[str, ...],
    *,
    implicit_edge_ids: bool,
) -> tuple[EdgeRecord, ...]:
    stream = io.BytesIO(data)
    count = read_uvarint(stream)
    result: list[EdgeRecord] = []
    for index in range(count):
        edge_id = index if implicit_edge_ids else read_uvarint(stream)
        source = read_uvarint(stream)
        target = read_uvarint(stream)
        relation_id = read_uvarint(stream)
        if relation_id >= len(relations):
            raise ValueError("relation ID outside dictionary")
        result.append(
            EdgeRecord(
                int(edge_id),
                int(source),
                int(target),
                relations[relation_id],
                bits_to_float(read_u64(stream)),
            )
        )
    if stream.read(1):
        raise ValueError("trailing edge baseline payload bytes")
    return tuple(result)


def encode_edge_baseline(
    vertex_count: int,
    edges: Iterable[EdgeRecord],
    output: str | Path,
) -> dict[str, object]:
    records = tuple(edges)
    _validate(vertex_count, records)
    relations = tuple(sorted({str(row.relation) for row in records}))
    relation_to_id = {name: index for index, name in enumerate(relations)}
    implicit_edge_ids = all(
        int(row.edge_id) == index for index, row in enumerate(records)
    )
    manifest = {
        "format": FORMAT,
        "vertex_count": int(vertex_count),
        "edge_count": len(records),
        "relations": list(relations),
        "implicit_edge_ids": bool(implicit_edge_ids),
        "exactness_scope": (
            "directed typed weighted EdgeRecord stream; binary64 weights "
            "preserved by exact IEEE bit pattern"
        ),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr("manifest.json", _json(manifest))
        archive.writestr(
            "edges.bin",
            _write_edges(
                records,
                relation_to_id,
                implicit_edge_ids=implicit_edge_ids,
            ),
        )

    decoded_n, decoded = decode_edge_baseline(target)
    if decoded_n != vertex_count or decoded != records:
        target.unlink(missing_ok=True)
        raise AssertionError("exact edge baseline roundtrip failed")

    with zipfile.ZipFile(target) as archive:
        entry_bytes = {
            info.filename: int(info.compress_size)
            for info in archive.infolist()
        }
    return {
        "format": FORMAT,
        "archive_bytes": target.stat().st_size,
        "edge_records": len(records),
        "relations": len(relations),
        "implicit_edge_ids": implicit_edge_ids,
        "entry_compressed_bytes": entry_bytes,
        "roundtrip_exact": True,
    }


def decode_edge_baseline(
    archive_path: str | Path,
) -> tuple[int, tuple[EdgeRecord, ...]]:
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported edge baseline format")
        relations = tuple(str(value) for value in manifest["relations"])
        records = _read_edges(
            archive.read("edges.bin"),
            relations,
            implicit_edge_ids=bool(manifest.get("implicit_edge_ids", False)),
        )
    if len(records) != int(manifest["edge_count"]):
        raise ValueError("decoded baseline edge count differs")
    return int(manifest["vertex_count"]), records
