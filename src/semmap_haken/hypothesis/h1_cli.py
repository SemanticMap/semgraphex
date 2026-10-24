"""Scientific H1 driver; unlike the earlier pilot, does not admit prefix sampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from semmap_haken.conceptnet import stream_assertions

from .graph import TypedGraph
from .runner import H1Config, run_h1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="H1: typed directed SBM vs DC-SBM vs nested DC-SBM, "
                    "restarts, grouped held-out queries and two calibrated null ensembles")
    parser.add_argument("--input", type=Path, required=True,
                        help="prepared, filtered assertion TSV; no physical-prefix scientific sampling")
    parser.add_argument("--output", type=Path, required=True,
                        help="run directory on fast local disk (stage completed results to Drive)")
    parser.add_argument("--mode", choices=("NEW", "RESUME"), default="NEW")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--nulls", type=int, default=99,
                        help="replicates PER ensemble (configuration and flat DC-SBM)")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--train-fraction", type=float, default=.8)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--noninferiority-margin", type=float, default=.01)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args(argv)
    if args.workers != "auto":
        try:
            args.workers = int(args.workers)
        except ValueError:
            parser.error("workers must be 'auto' or a positive integer")
    graph = TypedGraph.from_assertions(stream_assertions(args.input))
    config = H1Config(
        seed=args.seed, restarts=args.restarts, null_replicates=args.nulls,
        workers=args.workers, negatives=args.negatives,
        bootstrap_repetitions=args.bootstrap, train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        alpha=args.alpha, noninferiority_margin=args.noninferiority_margin,
    )
    report = run_h1(graph, args.output, config=config, resume=args.mode == "RESUME")
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "output": str(args.output),
        "observed_mdl_gain_bits": report["observed_mdl_gain_bits"],
        "counts": report["counts"],
        "workers": report["resource_budget"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
