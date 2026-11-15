"""Pinned Hugging Face ConceptNet 5.5 SQLite download for Colab local SSD.

The dataset uses node_norm/rel_norm/edge_norm, NOT ConceptNet Local's
concepts/relations schema. Download only; never modify the Hub cache.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3

REPO_ID = "cstr/conceptnet-normalized-multi"
FILENAME = "conceptnet_normalized.db"
# Observed public dataset HEAD commit; pin for reproducible experiments.
REVISION = "4d3fd17e80543e88d9823e2f4fe47901dcf9a4e6"
REQUIRED_TABLES = {
    "node_norm": {"node_pk", "node_url", "language"},
    "rel_norm": {"rel_pk", "rel_url"},
    "edge_norm": {"start_fk", "end_fk", "rel_fk", "weight"},
}


def inspect_database(path: str | Path) -> dict:
    """Check on-disk schema and read a sample, without re-indexing source."""
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as f:
        if f.read(16) != b"SQLite format 3\\x00":
            raise ValueError("Downloaded file is not SQLite")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as conn:
        for table, wanted in REQUIRED_TABLES.items():
            found = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not wanted <= found:
                raise ValueError(f"{table}: missing columns {sorted(wanted - found)}")
        sample = conn.execute(
            "SELECT node_url FROM node_norm WHERE language='en' LIMIT 1").fetchone()
        if sample is None or "/c/en/" not in sample[0]:
            raise ValueError("The normalized database has no English concepts")
        # Indexes on start_fk/end_fk matter for multi-document random access.
        indices = [row[1] for row in conn.execute(
            "PRAGMA index_list(edge_norm)")]
    return {"path": str(source), "size_bytes": source.stat().st_size,
            "format": "hf-normalized-5.5", "dataset": REPO_ID,
            "dataset_revision": REVISION, "edge_indices": indices}


def download_database(*, cache_dir: str | Path,
                      revision: str = REVISION, min_bytes: int = 1_000_000_000):
    """Resumable Hub-cache download into local SSD; no manual Drive DB required.

    The Hub client owns temporary files/chunk caching; on Colab reconnect it
    downloads missing chunks. A returned cached snapshot is not copied twice.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install extra 'text' for huggingface_hub") from exc
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Avoid starting a known 1.8 GB transfer on an almost full local disk.
    free = shutil.disk_usage(cache).free
    if free < min_bytes:
        raise OSError(f"Insufficient Colab local disk for ConceptNet: {free} "
                      f"bytes free, at least {min_bytes} required")
    target = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                                  repo_type="dataset", revision=revision,
                                  cache_dir=str(cache)))
    if target.stat().st_size < min_bytes:
        raise ValueError(f"ConceptNet SQLite file is unexpectedly small: {target}")
    result = inspect_database(target)
    result["dataset_revision"] = revision
    return result
