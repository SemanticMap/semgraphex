"""CLI for inspecting and decoding recursive lossless graph dictionaries."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .hierarchy_codec import (
    FORMAT as HIERARCHY_FORMAT,
    decode_hierarchy_bundle,
    write_decoded_hierarchy,
)


def _inspect(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != HIERARCHY_FORMAT:
            raise ValueError("unsupported hierarchy archive format")
        entries = {
            info.filename: {
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
            }
            for info in archive.infolist()
        }
    return {
        "archive": str(path),
        "archive_bytes": path.stat().st_size,
        "manifest": manifest,
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect, verify, or decode hierarchy_exact_v1 archives."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--archive", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--archive", type=Path, required=True)

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--archive", type=Path, required=True)
    decode_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(_inspect(args.archive), indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        n, records, memberships, raw_adjacency = decode_hierarchy_bundle(
            args.archive
        )
        print(json.dumps({
            "archive": str(args.archive),
            "verified": True,
            "nodes": n,
            "edge_records": len(records),
            "membership_nodes": len(memberships),
            "stored_adjacency_fallback": raw_adjacency is not None,
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "decode":
        report = write_decoded_hierarchy(args.archive, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
