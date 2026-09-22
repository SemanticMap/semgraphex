"""Post-hoc diagnostics for a completed M4.1 sweep.

This module does not change thresholds or detection decisions. It explains the
already-computed result by showing which runs passed and which individual gates
blocked each transition under the best pre-existing setting for each trajectory
metric.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _trajectory_value(transition: dict, metric: str) -> float:
    return float(transition[{"full": "full_trajectory_error", "slow": "slow_order_parameter_error", "post_transient": "post_transient_error"}[metric]])


def _transition_reasons(transition: dict, detector: dict) -> list[str]:
    reasons: list[str] = []
    if abs(int(transition["source_r"]) - int(transition["target_r"])) > int(detector["r_tolerance"]):
        reasons.append("r_instability")
    if float(transition["achieved_reduction"]) < 0.10:
        reasons.append("insufficient_reduction")
    if float(transition["subspace_projection_distance"]) > float(detector["subspace_threshold"]):
        reasons.append("subspace_distortion")
    if _trajectory_value(transition, detector["trajectory_metric"]) > float(detector["trajectory_threshold"]):
        reasons.append("trajectory_distortion")
    if float(transition["slow_eigenvalue_max_abs_error"]) > float(detector["eigenvalue_threshold"]):
        reasons.append("eigenvalue_distortion")
    return reasons


def _detect(transitions: list[dict], detector: dict, min_consecutive: int) -> tuple[bool, list[list[int]], Counter]:
    reasons = Counter()
    mask: list[bool] = []
    for item in transitions:
        failed = _transition_reasons(item, detector)
        reasons.update(failed)
        mask.append(not failed)
    ranges: list[list[int]] = []
    start = None
    for index in range(len(mask) + 1):
        active = index < len(mask) and mask[index]
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index - 1
            if end - start + 1 >= min_consecutive:
                ranges.append([int(transitions[start]["source_level"]), int(transitions[end]["target_level"])])
            start = None
    return bool(ranges), ranges, reasons


def diagnose(path: str | Path, output: str | Path) -> dict:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    min_consecutive = int(summary["options"]["min_consecutive"])
    best = summary["best_by_metric"]
    rows = []
    failure_totals = {metric: {"positive": Counter(), "negative": Counter()} for metric in best}
    grouped = {metric: defaultdict(lambda: [0, 0]) for metric in best}
    for run in summary["runs"]:
        transitions = run["transitions"]
        r_values = [int(transitions[0]["source_r"])] + [int(item["target_r"]) for item in transitions] if transitions else []
        row = {
            "label": run["label"], "control": run["control"], "bridge_weight": run["bridge_weight"],
            "target_reduction": run["target_reduction"], "seed": run["seed"], "r_values": r_values,
            "mean_full_error": sum(float(x["full_trajectory_error"]) for x in transitions) / len(transitions),
            "mean_slow_error": sum(float(x["slow_order_parameter_error"]) for x in transitions) / len(transitions),
            "mean_post_transient_error": sum(float(x["post_transient_error"]) for x in transitions) / len(transitions),
            "mean_subspace_distance": sum(float(x["subspace_projection_distance"]) for x in transitions) / len(transitions),
            "mean_eigenvalue_error": sum(float(x["slow_eigenvalue_max_abs_error"]) for x in transitions) / len(transitions),
        }
        for metric, detector in best.items():
            detected, ranges, reasons = _detect(transitions, detector, min_consecutive)
            row[f"detected_{metric}"] = detected
            row[f"ranges_{metric}"] = ranges
            row[f"failure_counts_{metric}"] = dict(reasons)
            failure_totals[metric][run["control"]].update(reasons)
            if run["control"] == "positive":
                for key, value in (("reduction", run["target_reduction"]), ("seed", run["seed"]), ("bridge", run["bridge_weight"])):
                    group_key = f"{key}={value}"
                    grouped[metric][group_key][1] += 1
                    grouped[metric][group_key][0] += int(detected)
        rows.append(row)
    grouped_rates = {
        metric: {key: hits / total for key, (hits, total) in groups.items()}
        for metric, groups in grouped.items()
    }
    result = {
        "best_detectors": best,
        "grouped_positive_detection_rates": grouped_rates,
        "failure_gate_counts": {
            metric: {control: dict(counts) for control, counts in controls.items()}
            for metric, controls in failure_totals.items()
        },
        "runs": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# M4.1 diagnostic report", "", "No thresholds were changed in this diagnostic step.", "", "## Positive detection rate by experimental factor", ""]
    for metric, groups in grouped_rates.items():
        lines.append(f"### {metric}")
        for key, value in sorted(groups.items()):
            lines.append(f"- {key}: {value:.3f}")
        lines.append("")
    lines += ["## Failed transition gates under each best detector", ""]
    for metric, controls in result["failure_gate_counts"].items():
        lines.append(f"### {metric}")
        lines.append(f"- positive: {controls['positive']}")
        lines.append(f"- negative: {controls['negative']}")
        lines.append("")
    lines += ["## Per-run summary", "", "| run | bridge | reduction | seed | r sequence | full | slow | tail | Dsub | Dlambda | detected(full/slow/tail) |", "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['label']} | {row['bridge_weight']} | {row['target_reduction']} | {row['seed']} | {row['r_values']} | {row['mean_full_error']:.3f} | {row['mean_slow_error']:.3f} | {row['mean_post_transient_error']:.3f} | {row['mean_subspace_distance']:.3f} | {row['mean_eigenvalue_error']:.3f} | {int(row['detected_full'])}/{int(row['detected_slow'])}/{int(row['detected_post_transient'])} |")
    (output.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    result = diagnose(args.input, args.output)
    print(json.dumps({"grouped_positive_detection_rates": result["grouped_positive_detection_rates"], "failure_gate_counts": result["failure_gate_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
