"""Stage and verify an already trained nSBM snapshot from Google Drive.

A COMPLETED file is only a marker, never proof that its companion files were
materialized in the ephemeral Colab VM. No refitting or graph-tool dependency.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np

from .runner import _load_graph, _sha_bytes
from .scan_checkpoint import sha256_file
from .selfsim_metrics import validate_levels


FIT_FILES = (
    "fit_data/ids.json", "fit_data/source.npy", "fit_data/target.npy",
    "fit_data/relation.npy", "fit_data/weight.npy",
    "nsbm_partitions.npz", "manifest.json", "fit_summary.json",
)


def _replace_from(source: Path, destination: Path) -> bool:
    """Stage a Drive file atomically and verify every byte; repair partial copies."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_file(source)
    if destination.is_file() and sha256_file(destination) == expected:
        return False
    pending = destination.with_name(destination.name + ".copying")
    try:
        shutil.copy2(source, pending)
        if sha256_file(pending) != expected:
            raise IOError(f"Drive file changed during staging: {source}")
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)
    return True


def stage_fit_snapshot(
    remote: str | Path, local: str | Path, *,
    expected_vertices: int = 100_000,
) -> dict:
    """Recover files and independently validate the fit before declaring READY.

    A partial remote run is staged for checkpointed RESUME, not accepted as
    complete. A contradictory completed snapshot raises a clear error rather
    than causing silent retraining against a possibly different graph.
    """
    remote, local = Path(remote), Path(local)
    local.mkdir(parents=True, exist_ok=True)
    copied = 0
    if remote.is_dir():
        for source in sorted(remote.rglob("*")):
            if (not source.is_file() or source.name == "COMPLETED"
                    or source.name.endswith((".tmp", ".partial", ".copying"))):
                continue
            copied += _replace_from(source, local/source.relative_to(remote))
    # A local COMPLETED marker from a previous, incomplete cell execution
    # must not cause fitting to be skipped.
    (local/"COMPLETED").unlink(missing_ok=True)
    missing = [relative for relative in FIT_FILES
               if not (local/relative).is_file()
               or not (remote/relative).is_file()]
    if missing:
        return {"ready": False, "staged_files": copied,
                "reason": "missing required nSBM files", "missing": missing}
    # Do not infer completion from merely having all files: the fitting
    # process may have been interrupted while writing a final checkpoint.
    if not (remote/"COMPLETED").is_file():
        return {"ready": False, "staged_files": copied,
                "reason": "nSBM fitting not published as COMPLETED", "missing": []}
    try:
        identity = json.loads((local/"manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((local/"fit_summary.json").read_text(encoding="utf-8"))
        if summary.get("identity") != identity:
            raise ValueError("manifest identity differs from fit_summary")
        counts = summary.get("counts", {})
        if counts.get("vertices") != expected_vertices:
            raise ValueError(
                f"fit has {counts.get('vertices')} vertices, expected {expected_vertices}"
            )
        actual_sha = sha256_file(local/"nsbm_partitions.npz")
        if actual_sha != summary.get("partitions_sha256"):
            raise ValueError("nsbm_partitions.npz checksum mismatch")
        graph = _load_graph(local/"fit_data")
        if graph.vertex_count != expected_vertices or (
            _sha_bytes(graph) != identity.get("graph_sha256")
        ):
            raise ValueError("fit_data graph SHA-256 differs from manifest")
        with np.load(local/"nsbm_partitions.npz", allow_pickle=False) as archive:
            keys = sorted(k for k in archive.files if k.startswith("level_"))
            if not keys:
                raise ValueError("no nSBM level_* arrays in partition archive")
            levels = validate_levels([archive[k] for k in keys], graph.vertex_count)
        blocks = [len(set(level)) for level in levels]
        if blocks != summary.get("block_counts"):
            raise ValueError("nSBM block counts differ from fit_summary")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"nSBM snapshot exists but failed integrity validation at {remote}: "
            f"{exc}. Do not discard saved tasks; inspect Drive artifacts."
        ) from exc
    _replace_from(remote/"COMPLETED", local/"COMPLETED")
    return {"ready": True, "staged_files": copied, "missing": [],
            "blocks": blocks, "vertices": expected_vertices,
            "reason": "all fit files recovered and validated"}


def ensure_verified_fit_snapshot(remote: str | Path, local: str | Path, *,
                                 expected_vertices: int=100_000) -> dict:
    status = stage_fit_snapshot(remote, local,
                                expected_vertices=expected_vertices)
    if not status["ready"]:
        raise RuntimeError(
            "nSBM fit is not ready for self-similarity tests: "
            + status["reason"] + "; missing=" + ", ".join(status["missing"])
            + ". Run the nSBM fitting cell first, or restore a completed "
            "fit from the correct Google Drive folder."
        )
    return status



def load_verified_fit_bundle(remote: str | Path, local: str | Path, *,
                             expected_vertices: int = 100_000):
    """Stage and load the *same* snapshot, not stale notebook GRAPH_DIR state.

    Return the verified graph, original-vertex partitions, source manifest,
    fit summary, and staging status in one call. An incomplete snapshot raises
    before metrics are attempted. nSBM is never refitted by this function.
    """
    status = ensure_verified_fit_snapshot(remote, local,
                                          expected_vertices=expected_vertices)
    root = Path(local)
    graph_dir = root/"fit_data"
    id_path = graph_dir/"ids.json"
    if not id_path.is_file():
        raise FileNotFoundError(
            f"After staging, expected IDs at {id_path}. "
            "The local scratch directory may have been cleared by another "
            "cell or execution. Do not rerun nSBM; restore Drive artifacts."
        )
    graph = _load_graph(graph_dir)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root/"fit_summary.json").read_text(encoding="utf-8"))
    with np.load(root/"nsbm_partitions.npz", allow_pickle=False) as archive:
        keys = sorted(k for k in archive.files if k.startswith("level_"))
        levels = validate_levels([archive[k] for k in keys],
                                 graph.vertex_count)
    if _sha_bytes(graph) != manifest.get("graph_sha256"):
        raise ValueError("Input graph differs from nSBM fit manifest")
    if [len(set(x)) for x in levels] != summary.get("block_counts"):
        raise ValueError("Loaded nSBM partition hierarchy differs from summary")
    return graph, levels, manifest, summary, status
