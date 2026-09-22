"""M4.1 synthetic sensitivity/specificity sweep.

The sweep is deliberately broader than a single hand-picked plateau threshold.
It varies planted bridge strength, coarsening ratio, random seed, detector
thresholds, and the trajectory metric used by the plateau gate.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import yaml

from .compute import ComputeContext
from .config import load_config
from .hierarchy import run_hierarchy, save_hierarchy_result
from .m41_metrics import M41TransitionMetrics, derive_fine_to_coarse, evaluate_transition_m41
from .multiscale_config import load_research_extensions
from .synthetic import hierarchical_positive, homogeneous_negative

TrajectoryMetric = Literal["full", "slow", "post_transient"]


@dataclass(frozen=True)
class M41SweepOptions:
    post_transient_start_fraction: float
    bridge_weights: tuple[float, ...]
    target_reductions: tuple[float, ...]
    seeds: tuple[int, ...]
    trajectory_metrics: tuple[TrajectoryMetric, ...]
    r_tolerances: tuple[int, ...]
    subspace_thresholds: tuple[float, ...]
    trajectory_thresholds: tuple[float, ...]
    eigenvalue_thresholds: tuple[float, ...]
    min_consecutive: int
    min_achieved_reduction: float
    robust_sensitivity: float
    robust_specificity: float
    robust_setting_count: int


def load_m41_options(path: str | Path) -> M41SweepOptions:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = raw.get("m41", {})
    if not isinstance(section, dict):
        raise ValueError("m41 must be a mapping")
    metrics = tuple(section.get("trajectory_metrics", ["full", "slow", "post_transient"]))
    if any(item not in {"full", "slow", "post_transient"} for item in metrics):
        raise ValueError("unsupported M4.1 trajectory metric")
    options = M41SweepOptions(
        post_transient_start_fraction=float(section.get("post_transient_start_fraction", 0.5)),
        bridge_weights=tuple(float(x) for x in section.get("bridge_weights", [0.02, 0.05, 0.10])),
        target_reductions=tuple(float(x) for x in section.get("target_reductions", [0.35, 0.5])),
        seeds=tuple(int(x) for x in section.get("seeds", [1729, 1730])),
        trajectory_metrics=metrics,
        r_tolerances=tuple(int(x) for x in section.get("r_tolerances", [0, 1])),
        subspace_thresholds=tuple(float(x) for x in section.get("subspace_thresholds", [0.2, 0.45, 0.8])),
        trajectory_thresholds=tuple(float(x) for x in section.get("trajectory_thresholds", [0.25, 0.5, 0.8, 1.2])),
        eigenvalue_thresholds=tuple(float(x) for x in section.get("eigenvalue_thresholds", [0.15, 0.3, 0.6])),
        min_consecutive=int(section.get("min_consecutive", 2)),
        min_achieved_reduction=float(section.get("min_achieved_reduction", 0.10)),
        robust_sensitivity=float(section.get("robust_sensitivity", 0.75)),
        robust_specificity=float(section.get("robust_specificity", 0.75)),
        robust_setting_count=int(section.get("robust_setting_count", 3)),
    )
    if not 0 <= options.post_transient_start_fraction < 1:
        raise ValueError("post_transient_start_fraction must be in [0,1)")
    if not options.bridge_weights or not options.target_reductions or not options.seeds:
        raise ValueError("M4.1 sweep grids must be non-empty")
    if any(not 0 < x <= 0.5 for x in options.target_reductions):
        raise ValueError("target reductions must be in (0,0.5]")
    return options


def _cpu_context(config: object) -> ComputeContext:
    return ComputeContext.create(
        backend="cpu", device=0, dtype="float64", workers=1,
        reserved_cpu_cores=1, threads_per_worker=1, gpu_memory_fraction=0.8,
        batch_size="auto", deterministic=True, allow_auto_fallback=True,
    )


def augment_hierarchy(hierarchy: object, *, config: object, options: M41SweepOptions, compute: object) -> tuple[M41TransitionMetrics, ...]:
    metrics: list[M41TransitionMetrics] = []
    for level in range(len(hierarchy.adjacencies) - 1):
        assignment = derive_fine_to_coarse(hierarchy.memberships[level], hierarchy.memberships[level + 1])
        metrics.append(evaluate_transition_m41(
            hierarchy.adjacencies[level], hierarchy.adjacencies[level + 1], assignment,
            config=config, source_level=level,
            post_transient_start_fraction=options.post_transient_start_fraction,
            compute=compute,
        ))
    return tuple(metrics)


def _metric_value(item: M41TransitionMetrics, metric: TrajectoryMetric) -> float:
    if metric == "full":
        return item.full_trajectory_error
    if metric == "slow":
        return item.slow_order_parameter_error
    return item.post_transient_error


def detect_plateau(metrics: tuple[M41TransitionMetrics, ...], *, trajectory_metric: TrajectoryMetric, r_tolerance: int, subspace_threshold: float, trajectory_threshold: float, eigenvalue_threshold: float, min_consecutive: int, min_achieved_reduction: float) -> tuple[bool, tuple[tuple[int, int], ...]]:
    """Apply one explicitly specified detector to an M4.1 transition sequence."""
    mask = [
        abs(item.source_r - item.target_r) <= r_tolerance
        and item.achieved_reduction >= min_achieved_reduction
        and item.subspace_projection_distance <= subspace_threshold
        and _metric_value(item, trajectory_metric) <= trajectory_threshold
        and item.slow_eigenvalue_max_abs_error <= eigenvalue_threshold
        for item in metrics
    ]
    ranges: list[tuple[int, int]] = []
    start = None
    for index in range(len(mask) + 1):
        active = index < len(mask) and mask[index]
        if active and start is None:
            start = index
        if not active and start is not None:
            end = index - 1
            if end - start + 1 >= min_consecutive:
                ranges.append((metrics[start].source_level, metrics[end].target_level))
            start = None
    return bool(ranges), tuple(ranges)


def _variant(config: object, *, reduction: float, seed: int) -> object:
    return replace(
        config,
        coarsening=replace(config.coarsening, target_reduction=reduction, seed=seed),
        dynamics=replace(config.dynamics, perturbation_seed=seed),
    )


def _run_record(graph: object, *, config: object, hierarchy_options: object, options: M41SweepOptions, compute: object, output: Path, label: str, bridge_weight: float | None, reduction: float, seed: int) -> dict[str, object]:
    hierarchy = run_hierarchy(graph.adjacency, graph.node_ids, config=config, options=hierarchy_options, compute=compute)
    metrics = augment_hierarchy(hierarchy, config=config, options=options, compute=compute)
    run_dir = output / "runs" / label
    save_hierarchy_result(hierarchy, run_dir / "hierarchy")
    (run_dir / "m41_metrics.json").write_text(json.dumps([item.to_dict() for item in metrics], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "label": label,
        "control": "positive" if graph.name == "hierarchical_positive" else "negative",
        "condition": graph.ground_truth["condition"],
        "bridge_weight": bridge_weight,
        "target_reduction": reduction,
        "seed": seed,
        "stop_reason": hierarchy.stop_reason,
        "levels": len(hierarchy.levels),
        "transitions": [item.to_dict() for item in metrics],
        "_metrics": metrics,
    }


def _detector_rows(records: list[dict[str, object]], options: M41SweepOptions) -> list[dict[str, object]]:
    positives = [item for item in records if item["control"] == "positive"]
    negatives = [item for item in records if item["control"] == "negative"]
    rows: list[dict[str, object]] = []
    for metric, r_tol, sub_thr, traj_thr, eig_thr in itertools.product(
        options.trajectory_metrics, options.r_tolerances, options.subspace_thresholds,
        options.trajectory_thresholds, options.eigenvalue_thresholds,
    ):
        kwargs = dict(
            trajectory_metric=metric, r_tolerance=r_tol, subspace_threshold=sub_thr,
            trajectory_threshold=traj_thr, eigenvalue_threshold=eig_thr,
            min_consecutive=options.min_consecutive,
            min_achieved_reduction=options.min_achieved_reduction,
        )
        pos_hits = [detect_plateau(item["_metrics"], **kwargs)[0] for item in positives]
        neg_hits = [detect_plateau(item["_metrics"], **kwargs)[0] for item in negatives]
        sensitivity = sum(pos_hits) / len(pos_hits) if pos_hits else 0.0
        false_positive_rate = sum(neg_hits) / len(neg_hits) if neg_hits else 0.0
        specificity = 1.0 - false_positive_rate
        bridge_sensitivity = {}
        for bridge in options.bridge_weights:
            subset = [item for item in positives if item["bridge_weight"] == bridge]
            hits = [detect_plateau(item["_metrics"], **kwargs)[0] for item in subset]
            bridge_sensitivity[str(bridge)] = sum(hits) / len(hits) if hits else 0.0
        rows.append({
            "trajectory_metric": metric,
            "r_tolerance": r_tol,
            "subspace_threshold": sub_thr,
            "trajectory_threshold": traj_thr,
            "eigenvalue_threshold": eig_thr,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "youden_j": sensitivity - false_positive_rate,
            "positive_hits": int(sum(pos_hits)),
            "positive_total": len(pos_hits),
            "negative_hits": int(sum(neg_hits)),
            "negative_total": len(neg_hits),
            "positive_sensitivity_by_bridge": bridge_sensitivity,
        })
    rows.sort(key=lambda row: (row["youden_j"], row["sensitivity"], row["specificity"]), reverse=True)
    return rows


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# M4.1 synthetic validation report", "",
        "## Terms", "",
        "- **Slow mode**: a collective pattern that decays much more slowly than fast microscopic variations.",
        "- **Order parameter candidate**: the amplitude of a selected slow mode; a compact macroscopic coordinate, not yet a proven Haken order parameter.",
        "- **Transient**: the short initial response immediately after a perturbation.",
        "- **Sensitivity**: fraction of positive controls in which a plateau is detected.",
        "- **Specificity**: fraction of negative controls correctly rejected as having no plateau.",
        "- **False-positive rate**: fraction of negative controls incorrectly called a plateau.",
        "- **Youden J**: sensitivity - false-positive-rate; 1 is perfect separation, 0 is no discrimination.",
        "- **Sweep**: systematic repetition over a grid of parameter values instead of choosing one convenient setting.", "",
        "## Pre-registered decision rule", "",
        f"Candidate robust discrimination requires sensitivity >= {summary['options']['robust_sensitivity']} and specificity >= {summary['options']['robust_specificity']} for at least {summary['options']['robust_setting_count']} detector settings.", "",
        "## Result", "",
        f"Status: **{summary['status']}**", "",
        f"Positive runs: {summary['positive_run_count']}; negative runs: {summary['negative_run_count']}.", "",
        "## Best detector by trajectory metric", "",
        "| metric | sensitivity | specificity | Youden J | r tol | subspace thr | trajectory thr | eigenvalue thr |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, row in summary["best_by_metric"].items():
        lines.append(f"| {metric} | {row['sensitivity']:.3f} | {row['specificity']:.3f} | {row['youden_j']:.3f} | {row['r_tolerance']} | {row['subspace_threshold']} | {row['trajectory_threshold']} | {row['eigenvalue_threshold']} |")
    lines += ["", "## Interpretation boundary", "", "A better slow-mode metric supports the claim only if it separates positive and negative controls across multiple parameter settings. A single successful threshold is treated as tuning, not evidence.", ""]
    return "\n".join(lines)


def run_sweep(config_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    extensions = load_research_extensions(config.source_path)
    options = load_m41_options(config.source_path)
    compute = _cpu_context(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    # Positive controls vary the weak bridge strength because that directly
    # changes planted timescale separation.
    for bridge, reduction, seed in itertools.product(options.bridge_weights, options.target_reductions, options.seeds):
        variant = _variant(config, reduction=reduction, seed=seed)
        synthetic_options = replace(extensions.synthetic, bridge_weight=bridge, seed=seed)
        graph = hierarchical_positive(synthetic_options)
        label = f"positive_bridge-{bridge:g}_reduction-{reduction:g}_seed-{seed}"
        records.append(_run_record(graph, config=variant, hierarchy_options=extensions.hierarchy, options=options, compute=compute, output=output, label=label, bridge_weight=bridge, reduction=reduction, seed=seed))

    # The homogeneous negative graph has no weak inter-block bridge parameter,
    # so it is evaluated once per compression/seed pair rather than duplicated
    # across the positive bridge grid.
    for reduction, seed in itertools.product(options.target_reductions, options.seeds):
        variant = _variant(config, reduction=reduction, seed=seed)
        synthetic_options = replace(extensions.synthetic, seed=seed)
        graph = homogeneous_negative(synthetic_options)
        label = f"negative_reduction-{reduction:g}_seed-{seed}"
        records.append(_run_record(graph, config=variant, hierarchy_options=extensions.hierarchy, options=options, compute=compute, output=output, label=label, bridge_weight=None, reduction=reduction, seed=seed))

    rows = _detector_rows(records, options)
    best_by_metric = {metric: next(row for row in rows if row["trajectory_metric"] == metric) for metric in options.trajectory_metrics}
    robust_rows = [row for row in rows if row["sensitivity"] >= options.robust_sensitivity and row["specificity"] >= options.robust_specificity]
    status = "candidate_discriminative_region" if len(robust_rows) >= options.robust_setting_count else ("isolated_discriminative_settings" if robust_rows else "no_discriminative_setting")

    public_records = [{key: value for key, value in item.items() if key != "_metrics"} for item in records]
    summary = {
        "schema_version": 1,
        "status": status,
        "options": asdict(options),
        "positive_run_count": sum(item["control"] == "positive" for item in records),
        "negative_run_count": sum(item["control"] == "negative" for item in records),
        "robust_detector_setting_count": len(robust_rows),
        "best_by_metric": best_by_metric,
        "top_detector_settings": rows[:20],
        "runs": public_records,
    }
    (output / "m41_sweep.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "m41_report.md").write_text(_markdown(summary), encoding="utf-8")
    return summary
