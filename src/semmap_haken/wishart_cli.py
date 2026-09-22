"""CLI for ConceptNet -> recursive Wishart compression-figure discovery."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from semmap_haken.config import load_config
from semmap_haken.conceptnet import AssertionFilters, ParseReport, stream_assertions
from semmap_haken.graph_build import build_sparse_graph, load_prepared_graph
from semmap_haken.wishart_config import load_wishart_options
from semmap_haken.wishart_hierarchy import run_wishart_hierarchy


def _load_or_build(config_path: Path, prepared: Path | None):
    config = load_config(config_path)
    if prepared is not None:
        return config, load_prepared_graph(prepared), {
            "mode": "prepared",
            "prepared_graph_dir": str(prepared.resolve()),
        }

    if config.dataset.path is None:
        raise SystemExit(
            "dataset.path is required when --prepared is not supplied"
        )
    if config.graph.weight_transform not in {"binary", "raw", "log1p"}:
        raise SystemExit(
            "Wishart raw preparation currently supports binary/raw/log1p weight_transform"
        )
    report = ParseReport()
    filters = AssertionFilters(
        language=config.dataset.language or None,
        relations=frozenset(config.dataset.relations),
        min_weight=config.dataset.min_weight,
    )
    assertions = stream_assertions(
        config.dataset.path,
        filters,
        invalid_mode="skip_invalid",
        report=report,
        max_rows=config.dataset.max_rows,
    )
    graph = build_sparse_graph(
        assertions,
        directed=config.graph.directed,
        weight_transform=config.graph.weight_transform,
        component=config.dataset.component,
        max_nodes=config.dataset.max_nodes,
        self_loop_policy=config.graph.self_loop_policy,
    )
    return config, graph, {
        "mode": "raw_conceptnet",
        "dataset_path": str(config.dataset.path),
        "parse_report": report.to_dict(),
        "graph_report": graph.report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover unknown recurring compression figures with Wishart clustering."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Optional completed semmap-haken prepare directory. If omitted, read dataset.path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory. Defaults to <runs_root>/wishart-<UTC timestamp>.",
    )
    args = parser.parse_args()

    config, graph, source = _load_or_build(args.config.resolve(), args.prepared)
    options = load_wishart_options(args.config.resolve())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output is not None
        else (config.paths.runs_root / f"wishart-{stamp}").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "input.json").write_text(
        json.dumps(
            {
                "source": source,
                "config": str(args.config.resolve()),
                "metric": options.metric,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_wishart_hierarchy(
        graph,
        directed=config.graph.directed,
        options=options,
        output_dir=output,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
