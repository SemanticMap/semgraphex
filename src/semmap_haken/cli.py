"""Stable CLI surface; workflow implementations attach through registered handlers."""

from __future__ import annotations

import argparse
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
