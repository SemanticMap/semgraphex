"""CLI for the M4.1 falsification/sensitivity experiment."""

from __future__ import annotations

import argparse
import json

from .m41_sweep import run_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semmap-haken-m41")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep = subparsers.add_parser("sweep", help="Run M4.1 synthetic parameter sweep")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "sweep":
        summary = run_sweep(args.config, args.output)
        print(json.dumps({
            "status": summary["status"],
            "positive_run_count": summary["positive_run_count"],
            "negative_run_count": summary["negative_run_count"],
            "robust_detector_setting_count": summary["robust_detector_setting_count"],
            "best_by_metric": summary["best_by_metric"],
        }, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
