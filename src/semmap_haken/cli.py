"""Stable CLI surface; workflow implementations attach through registered handlers."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections.abc import Callable, Sequence

CommandHandler = Callable[[argparse.Namespace], int]
_HANDLERS: dict[str, CommandHandler] = {}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_handler(command: str, handler: CommandHandler) -> None:
    """Register a workstream implementation without changing argument parsing."""
    _HANDLERS[command] = handler


def _not_implemented(command: str) -> CommandHandler:
    def handler(_: argparse.Namespace) -> int:
        print(f"semmap-haken {command}: not yet implemented; this command is owned by a later workstream.", file=__import__("sys").stderr)
        return 2

    return handler


def _add_config_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], command: str) -> None:
    parser = subparsers.add_parser(command, help=f"{command} workflow")
    parser.add_argument("--config", required=True, help="Path to the YAML experiment configuration.")
    parser.set_defaults(command=command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semmap-haken", description="Reproducible Haken-coarsening research CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_config_command(subparsers, "download")
    _add_config_command(subparsers, "prepare")
    _add_config_command(subparsers, "run")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate an existing run.")
    evaluate.add_argument("--run", required=True, help="Path to a run directory.")
    evaluate.set_defaults(command="evaluate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    handler = _HANDLERS.get(arguments.command, _not_implemented(arguments.command))
    return handler(arguments)


def _data_handlers() -> None:
    from .conceptnet import AssertionFilters, ParseReport, stream_assertions
    from .config import load_config
    from .data_manager import DatasetDescriptor, acquire_dataset, sha256_file
    from .graph_build import build_sparse_graph, save_prepared_graph
    from .manifest import RunManifest
    import yaml

    def validate_resource_profile(config: object) -> None:
        profile_path = config.runtime.resource_profile
        if profile_path is None:
            return
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict) or profile.get("name") != config.runtime.profile:
            raise ValueError("runtime.profile must match a valid resource profile name")
        if int(profile.get("expected_max_nodes", -1)) != config.dataset.max_nodes:
            raise ValueError("resource profile expected_max_nodes must equal dataset.max_nodes")

    def download(args: argparse.Namespace) -> int:
        config = load_config(args.config)
        validate_resource_profile(config)
        descriptor = DatasetDescriptor(dataset_id=config.dataset.source)
        result = acquire_dataset(descriptor, cache_root=config.paths.cache_root, manual_path=config.dataset.path)
        print(json.dumps({"path": str(result.path), "sha256": result.sha256, "cache_hit": result.cache_hit}, sort_keys=True))
        return 0

    def prepare(args: argparse.Namespace) -> int:
        config = load_config(args.config)
        if config.dataset.path is None:
            raise ValueError("dataset.path is required for offline prepare; run download or provide a fixture/manual path")
        if config.dataset.source == "conceptnet-5.7" and config.dataset.expected_sha256 is None:
            raise ValueError("production ConceptNet prepare requires dataset.expected_sha256; a placeholder is not verification")
        report = ParseReport()
        records = stream_assertions(config.dataset.path, AssertionFilters(config.dataset.language, frozenset(config.dataset.relations), config.dataset.min_weight), report=report)
        graph = build_sparse_graph(records, directed=config.graph.directed, weight_transform=config.graph.weight_transform, component=config.dataset.component, max_nodes=config.dataset.max_nodes, self_loop_policy=config.graph.self_loop_policy)
        run_id = f"prepare-{uuid.uuid4().hex[:12]}"
        run_dir = config.paths.runs_root / run_id
        raw_digest = sha256_file(config.dataset.path)
        source_identity = {"path": str(config.dataset.path), "sha256": raw_digest, "size_bytes": config.dataset.path.stat().st_size, "dataset_name": config.dataset.source, "dataset_version": config.dataset.version, "source_url": config.dataset.source_url, "acquisition_status": "manual", "verification_status": "verified" if config.dataset.expected_sha256 == raw_digest else "manual_unverified"}
        paths = save_prepared_graph(graph, run_dir, metadata={"parser_report": report.to_dict(), "source_identity": source_identity}, resolved_config=config.resolved)
        checksums = {path.name: _sha256(str(path)) for path in paths.values()}
        artifacts = {
            name: {"path": str(path), "sha256": checksums[path.name]}
            for name, path in paths.items()
        }
        manifest = RunManifest.create(run_id=run_id, resolved_config=config.resolved, execution_environment="cli")
        manifest = manifest.__class__(**{
            **manifest.to_dict(),
            "checksums": checksums,
            "stages": {"prepare": "completed"},
            "artifacts": artifacts,
            "cli_replay": True,
            "random_seeds": {"runtime": config.runtime.random_seed},
            "resumability": {"source_identity": source_identity, "environment_constraints": "requirements/constraints.txt"},
        })
        manifest.write_json(run_dir / "manifest.json")
        (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
        print(str(run_dir))
        return 0

    register_handler("download", download)
    register_handler("prepare", prepare)


_data_handlers()
