"""Empirical H1 calibration: type-I error on flat truth, power on nested truth.

Expensive by design: each synthetic replicate performs the same inference and
two full null ensemble comparisons as a real experiment. Synthetic positives
are not a demonstration of true hierarchy in ConceptNet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from semmap_haken.hypothesis.runner import H1Config, run_h1
from semmap_haken.hypothesis.synthetic_benchmarks import planted_typed_graph
from semmap_haken.hypothesis.hierarchy_validation import hierarchy_recovery


def _rate(successes: int, n: int) -> dict[str, object]:
    lo = 0.0 if successes == 0 else float(beta.ppf(.025, successes, n-successes+1))
    hi = 1.0 if successes == n else float(beta.ppf(.975, successes+1, n-successes))
    return {"count": successes, "trials": n, "estimate": successes/n,
            "binomial_95ci": [lo, hi]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="H1 false-positive/power calibration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=5,
                        help="independent planted graph seeds per condition")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--nulls", type=int, default=99)
    parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.simulations < 1 or args.per_group < 4:
        parser.error("simulations must be positive and per-group >= 4")
    workers = args.workers if args.workers == "auto" else int(args.workers)
    results = []
    for condition in (False, True):
        for index in range(args.simulations):
            seed = args.seed + index + (0 if not condition else 100_000)
            sample = planted_typed_graph(seed=seed, nested=condition, per_group=args.per_group)
            directory = args.output / ("nested" if condition else "flat") / f"run-{index:03d}"
            config = H1Config(seed=seed, restarts=args.restarts,
                              null_replicates=args.nulls, workers=workers,
                              bootstrap_repetitions=500)
            report = run_h1(sample.graph, directory, config=config,
                            resume=args.resume and (directory / "manifest.json").exists())
            # Evaluate real inferred partitions against planted ground truth,
            # never only the count of inferred blocks/levels.
            task_files = list((directory / "tasks").glob("real-nested_DC_SBM-restart_*.json"))
            if not task_files:
                raise RuntimeError("no stored real nSBM fits for synthetic recovery check")
            winner = min(task_files,
                         key=lambda f: json.loads(f.read_text(encoding="utf-8"))["mdl_bits"])
            with np.load(winner.with_suffix(".npz"), allow_pickle=False) as saved:
                inferred = [saved[f"p{i}"].astype(np.int64).tolist()
                            for i in range(len(saved.files))]
            recovery = hierarchy_recovery(inferred, sample.truth)
            results.append({"condition": "nested" if condition else "flat",
                            "index": index, "seed": seed,
                            "detected": report["status"] == "supported",
                            "status": report["status"],
                            "mdlgain_bits": report["observed_mdl_gain_bits"],
                            "vertices": sample.graph.vertex_count,
                            "assertions": sample.graph.edge_count,
                            "recovery": recovery})
    flat = [x for x in results if x["condition"] == "flat"]
    nested = [x for x in results if x["condition"] == "nested"]
    summary = {
        "conditions": ["flat", "nested"],
        "null_replicates_per_ensemble": args.nulls,
        "restarts_per_model": args.restarts,
        "false_positive": _rate(sum(x["detected"] for x in flat), len(flat)),
        "detection_power": _rate(sum(x["detected"] for x in nested), len(nested)),
        "runs": results,
        "nested_ground_truth_recovery": {
            name: {
                "observations": [entry["recovery"][name]["ari"] for entry in nested],
                "mean_ari": (
                    float(np.mean([entry["recovery"][name]["ari"] for entry in nested
                                   if entry["recovery"][name]["ari"] is not None]))
                    if any(entry["recovery"][name]["ari"] is not None
                           for entry in nested) else None
                ),
            }
            for name in ("fine", "coarse")
        },
        "scientific_caveat": (
            "Rates and intervals apply to this synthetic graph generator only. "
            "Small numbers of replications do not establish nominal type-I error "
            "control. No ConceptNet hierarchy or graphex limit is inferred."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"false_positive": summary["false_positive"],
                      "detection_power": summary["detection_power"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
