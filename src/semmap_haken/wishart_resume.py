"""Versioned, checksum-verified semantic-boundary checkpoints for Wishart.

Only load state.pkl.gz from a run you created and trust: pickle is not a safe
format for untrusted third-party data. Every checkpoint is immutable and its
manifest is written after the payload. A partial latest checkpoint is ignored
in favor of the preceding verified checkpoint.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

CHECKPOINT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config_path: str | Path) -> str:
    return _sha256(Path(config_path))


def write_level_checkpoint(
    run_dir: str | Path,
    *,
    next_level: int,
    current: Any,
    relation_layers: Mapping[str, Any],
    memberships: Mapping[int, tuple[str, ...]],
    symbol_types: Mapping[int, str],
    dictionary: Any,
    family_registry: Any,
    current_huffman: Mapping[str, str],
    level_summaries: list[dict[str, object]],
    transition_summaries: list[dict[str, object]],
    config_hash: str,
    input_hash: str,
    code_revision: str,
) -> Path:
    """Atomically publish an immutable local checkpoint for the NEXT level."""
    if next_level < 0:
        raise ValueError("next_level must be non-negative")
    root = Path(run_dir) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"level_{next_level:03d}"
    if final.exists():
        manifest = final / "manifest.json"
        if manifest.is_file() and (final / "state.pkl.gz").is_file():
            return final
        shutil.rmtree(final)

    stage = Path(tempfile.mkdtemp(prefix=f".level_{next_level:03d}.", dir=root))
    state = {
        "version": CHECKPOINT_VERSION,
        "next_level": int(next_level),
        "current": current,
        "relation_layers": dict(relation_layers),
        "memberships": dict(memberships),
        "symbol_types": dict(symbol_types),
        "dictionary": dictionary,
        "family_registry": family_registry,
        "current_huffman": dict(current_huffman),
        "level_summaries": list(level_summaries),
        "transition_summaries": list(transition_summaries),
    }
    try:
        payload = stage / "state.pkl.gz"
        with gzip.open(payload, mode="wb", compresslevel=1) as stream:
            pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with payload.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {
            "format_version": CHECKPOINT_VERSION,
            "next_level": int(next_level),
            "config_sha256": str(config_hash),
            "input_sha256": str(input_hash),
            "code_revision": str(code_revision),
            "state_file": payload.name,
            "state_size": payload.stat().st_size,
            "state_sha256": _sha256(payload),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, final)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return final


def load_latest_checkpoint(
    run_dir: str | Path,
    *,
    config_hash: str,
    input_hash: str,
    code_revision: str,
) -> dict[str, Any] | None:
    """Restore the newest checksum-valid checkpoint; reject incompatible runs."""
    root = Path(run_dir) / "checkpoints"
    if not root.exists():
        return None
    candidates = sorted(
        (item for item in root.glob("level_*") if item.is_dir()),
        key=lambda item: item.name,
        reverse=True,
    )
    if not candidates:
        return None
    for item in candidates:
        try:
            manifest = json.loads((item / "manifest.json").read_text(encoding="utf-8"))
            if int(manifest["format_version"]) != CHECKPOINT_VERSION:
                continue
            state_file = manifest["state_file"]
            if state_file != "state.pkl.gz":
                continue
            payload = item / state_file
            if payload.stat().st_size != int(manifest["state_size"]):
                continue
            if _sha256(payload) != manifest["state_sha256"]:
                continue
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue

        if manifest["config_sha256"] != config_hash:
            raise ValueError("checkpoint configuration hash differs from requested configuration")
        if manifest["input_sha256"] != input_hash:
            raise ValueError("checkpoint input graph hash differs from supplied input")
        if manifest["code_revision"] != code_revision:
            raise ValueError("checkpoint code revision differs from installed code")
        with gzip.open(payload, "rb") as stream:
            state = pickle.load(stream)
        if (
            state.get("version") != CHECKPOINT_VERSION
            or state.get("next_level") != manifest["next_level"]
        ):
            continue
        return state
    raise ValueError("checkpoints exist but none passed integrity validation")


def truncate_after_checkpoint(run_dir: str | Path, *, next_level: int) -> None:
    """Remove incomplete/recomputed outputs after a verified checkpoint."""
    run = Path(run_dir)
    if not run.exists():
        return
    for item in run.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if name.startswith("level_") and name[6:].isdigit():
            if int(name[6:]) >= next_level:
                shutil.rmtree(item)
        elif name.startswith("transition_"):
            parts = name.split("_")
            if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) >= next_level:
                shutil.rmtree(item)
    checkpoints = run / "checkpoints"
    if checkpoints.is_dir():
        for item in checkpoints.glob("level_*"):
            if item.is_dir() and item.name[6:].isdigit() and int(item.name[6:]) > next_level:
                shutil.rmtree(item)
    for name in ("COMPLETED", "hierarchy.json", "COLAB_RUN.json", "FAILED.json"):
        (run / name).unlink(missing_ok=True)
