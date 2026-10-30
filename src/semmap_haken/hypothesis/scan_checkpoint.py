"""Small checksum-verified atomic checkpoints for deterministic graph scans."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def save_shard(path: Path, *, identity: dict, rows: list) -> None:
    """Write payload completely before publishing the shard path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"identity": identity, "rows": rows}
    digest = hashlib.sha256(_encode(payload)).hexdigest()
    pending = path.with_name(path.name + ".tmp")
    pending.write_bytes(_encode({"payload": payload, "sha256": digest}))
    os.replace(pending, path)


def load_shard(path: Path, *, identity: dict) -> list | None:
    """A torn, corrupt, stale or incomplete shard is a cache miss, not a hit."""
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_bytes())
        payload = record["payload"]
        if payload["identity"] != identity or not isinstance(payload["rows"], list):
            return None
        expected = hashlib.sha256(_encode(payload)).hexdigest()
        if record["sha256"] != expected:
            return None
        for item in payload["rows"]:
            if (not isinstance(item, list) or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], list)
                or any(not isinstance(n, int) for n in item[1])):
                return None
        return payload["rows"]
    except (ValueError, OSError, KeyError, TypeError, UnicodeError):
        return None


def verify_artifacts(report: dict, base: Path) -> bool:
    """Check both archive hashes; filesize alone cannot detect corruption."""
    for name, values in report.get("coding", {}).items():
        for prefix, path in (
            ("archive", base / f"grammar_{report['method']}_{name}.zip"),
            ("bundle", base / f"bundle_{report['method']}_{name}.zip"),
        ):
            if not path.is_file():
                return False
            if values.get(prefix + "_sha256") != sha256_file(path):
                return False
            if values.get("archive_bytes" if prefix == "archive" else "full_bundle_bytes") != path.stat().st_size:
                return False
    return bool(report.get("coding"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_bytes(json.dumps(value, indent=2, sort_keys=True,
                                   ensure_ascii=False).encode("utf-8") + b"\n")
    os.replace(pending, path)
