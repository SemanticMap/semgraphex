"""Stable CLI surface; workflow implementations attach through registered handlers."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Callable, Sequence

CommandHandler = Callable[[argparse.Namespace], int]
_HANDLERS: dict[str, CommandHandler] = {}


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
    from .data_manager import DatasetDescriptor, acquire_dataset
    from .graph_build import build_sparse_graph, save_prepared_graph
    from .manifest import RunManifest

    def download(args: argparse.Namespace) -> int:
        config = load_config(args.config)
        descriptor = DatasetDescriptor(dataset_id=config.dataset.source)
        result = acquire_dataset(descriptor, cache_root=config.paths.cache_root, manual_path=config.dataset.path)
        print(json.dumps({"path": str(result.path), "sha256": result.sha256, "cache_hit": result.cache_hit}, sort_keys=True))
        return 0

    def prepare(args: argparse.Namespace) -> int:
        config = load_config(args.config)
        if config.dataset.path is None:
            raise ValueError("dataset.path is required for offline prepare; run download or provide a fixture/manual path")
        report = ParseReport()
        records = stream_assertions(config.dataset.path, AssertionFilters(config.dataset.language, frozenset(config.dataset.relations), config.dataset.min_weight), report=report)
        graph = build_sparse_graph(records, directed=config.graph.directed, weight_transform=config.graph.weight_transform, component=config.dataset.component, max_nodes=config.dataset.max_nodes)
        run_id = f"prepare-{uuid.uuid4().hex[:12]}"
        run_dir = config.paths.runs_root / run_id
        paths = save_prepared_graph(graph, run_dir, metadata={"parser_report": report.to_dict(), "dataset_path": str(config.dataset.path)}, resolved_config=config.resolved)
        manifest = RunManifest.create(run_id=run_id, resolved_config=config.resolved, execution_environment="cli")
        manifest = manifest.__class__(**{**manifest.to_dict(), "stages": {"prepare": "completed"}, "artifacts": {name: {"path": str(path)} for name, path in paths.items()}})
        manifest.write_json(run_dir / "manifest.json")
        print(str(run_dir))
        return 0

    register_handler("download", download)
    register_handler("prepare", prepare)


_data_handlers()
