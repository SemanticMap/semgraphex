"""CLI handlers for multiscale, synthetic, and fixed-hierarchy dynamics studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_research_handlers() -> None:
    from .cli import register_handler
    from .compute import ComputeContext
    from .config import load_config
    from .dynamics_compare import compare_checkpoint_dynamics, materialize_dynamics_checkpoints
    from .graph_build import load_prepared_graph
    from .hierarchy import run_hierarchy, save_hierarchy_result
    from .manifest import RunManifest
    from .multiscale_config import load_research_extensions
    from .plateau import detect_plateaus, save_plateau_report
    from .run_contracts import validate_prepared_graph_contract
    from .synthetic import build_synthetic_controls, save_synthetic_graph

    def _cpu_context(config: object) -> ComputeContext:
        return ComputeContext.create(
            backend="cpu", device=0, dtype="float64", workers=config.execution.workers,
            reserved_cpu_cores=config.execution.reserved_cpu_cores,
            threads_per_worker=config.execution.threads_per_worker,
            gpu_memory_fraction=config.execution.gpu_memory_fraction,
            batch_size=config.execution.batch_size, deterministic=True, allow_auto_fallback=True,
        )

    def multiscale(args: object) -> int:
        config = load_config(args.config)
        extensions = load_research_extensions(config.source_path)
        if not extensions.hierarchy.enabled:
            raise ValueError("hierarchy.enabled must be true for multiscale")
        prepared_graph_dir = Path(args.prepared_graph).expanduser().resolve() if args.prepared_graph else config.spectral.prepared_graph_dir
        if prepared_graph_dir is None:
            raise ValueError("provide --prepared-graph or spectral.prepared_graph_dir")
        preparation_contract = validate_prepared_graph_contract(prepared_graph_dir, config)
        graph = load_prepared_graph(prepared_graph_dir)
        compute = _cpu_context(config)
        result = run_hierarchy(graph.adjacency, graph.node_ids, config=config, options=extensions.hierarchy, compute=compute)
        run_id = f"hierarchy-{uuid.uuid4().hex[:12]}"
        run_dir = config.paths.runs_root / run_id
        run_dir.mkdir(parents=True)
        hierarchy_paths = save_hierarchy_result(result, run_dir / "hierarchy")
        plateau_path: Path | None = None
        report = None
        if extensions.plateau.enabled:
            report = detect_plateaus(result, extensions.plateau)
            plateau_path = save_plateau_report(report, run_dir / "plateau.json")
        paths = dict(hierarchy_paths)
        if plateau_path is not None:
            paths["plateau"] = plateau_path
        checksums = {name: _sha256(path) for name, path in paths.items()}
        resolved = json.loads(json.dumps(config.resolved))
        resolved["spectral"]["prepared_graph_dir"] = str(prepared_graph_dir)
        resolved["hierarchy"] = extensions.hierarchy.__dict__
        resolved["plateau"] = extensions.plateau.__dict__
        manifest = RunManifest.create(run_id=run_id, resolved_config=resolved, execution_environment="cli")
        manifest = manifest.__class__(**{
            **manifest.to_dict(),
            "checksums": checksums,
            "stages": {"hierarchy": "completed", "plateau": "completed" if report is not None else "disabled"},
            "artifacts": {name: {"path": str(path), "sha256": checksums[name]} for name, path in paths.items()},
            "random_seeds": {"runtime": config.runtime.random_seed, "perturbations": config.dynamics.perturbation_seed, "coarsening": config.coarsening.seed},
            "cli_replay": True,
            "execution_telemetry": compute.telemetry(),
            "resumability": {"prepared_graph_dir": str(prepared_graph_dir), "preparation_contract": preparation_contract},
        })
        manifest.write_json(run_dir / "manifest.json")
        (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
        print(json.dumps({
            "run_id": run_id, "run_dir": str(run_dir), "levels": len(result.levels),
            "transitions": len(result.transitions), "stop_reason": result.stop_reason,
            "plateau_candidates": len(report.candidates) if report is not None else None,
        }, sort_keys=True))
        return 0

    def synthetic(args: object) -> int:
        config = load_config(args.config)
        extensions = load_research_extensions(config.source_path)
        if not extensions.hierarchy.enabled:
            raise ValueError("hierarchy.enabled must be true for synthetic")
        compute = _cpu_context(config)
        run_id = f"synthetic-{uuid.uuid4().hex[:12]}"
        run_dir = config.paths.runs_root / run_id
        run_dir.mkdir(parents=True)
        summary: dict[str, object] = {}
        paths: dict[str, Path] = {}
        for graph in build_synthetic_controls(extensions.synthetic):
            graph_dir = run_dir / graph.name
            graph_paths = save_synthetic_graph(graph, graph_dir / "input")
            hierarchy = run_hierarchy(graph.adjacency, graph.node_ids, config=config, options=extensions.hierarchy, compute=compute)
            hierarchy_paths = save_hierarchy_result(hierarchy, graph_dir / "hierarchy")
            plateau = detect_plateaus(hierarchy, extensions.plateau)
            plateau_path = save_plateau_report(plateau, graph_dir / "plateau.json")
            for name, path in graph_paths.items():
                paths[f"{graph.name}_input_{name}"] = path
            for name, path in hierarchy_paths.items():
                paths[f"{graph.name}_hierarchy_{name}"] = path
            paths[f"{graph.name}_plateau"] = plateau_path
            summary[graph.name] = {
                "condition": graph.ground_truth["condition"],
                "levels": len(hierarchy.levels),
                "transitions": len(hierarchy.transitions),
                "stop_reason": hierarchy.stop_reason,
                "plateau_candidates": len(plateau.candidates),
                "candidate_ranges": [[item.start_level, item.end_level] for item in plateau.candidates],
            }
        summary_path = run_dir / "synthetic_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["summary"] = summary_path
        checksums = {name: _sha256(path) for name, path in paths.items()}
        resolved = json.loads(json.dumps(config.resolved))
        resolved["hierarchy"] = extensions.hierarchy.__dict__
        resolved["plateau"] = extensions.plateau.__dict__
        resolved["synthetic"] = extensions.synthetic.__dict__
        manifest = RunManifest.create(run_id=run_id, resolved_config=resolved, execution_environment="cli")
        manifest = manifest.__class__(**{
            **manifest.to_dict(),
            "checksums": checksums,
            "stages": {"synthetic_controls": "completed", "hierarchy": "completed", "plateau": "completed"},
            "artifacts": {name: {"path": str(path), "sha256": checksums[name]} for name, path in paths.items()},
            "random_seeds": {"runtime": config.runtime.random_seed, "synthetic": extensions.synthetic.seed, "coarsening": config.coarsening.seed},
            "cli_replay": True,
            "execution_telemetry": compute.telemetry(),
            "resumability": {"control_names": list(summary)},
        })
        manifest.write_json(run_dir / "manifest.json")
        (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
        print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "controls": summary}, sort_keys=True))
        return 0

    def dynamics_checkpoints(args: object) -> int:
        config = load_config(args.config)
        extensions = load_research_extensions(config.source_path)
        if not extensions.dynamics_comparison.enabled:
            raise ValueError("dynamics_comparison.enabled must be true")
        index = materialize_dynamics_checkpoints(args.hierarchy_run, extensions.dynamics_comparison)
        print(json.dumps({
            "checkpoint_root": str(Path(args.hierarchy_run).expanduser().resolve() / "dynamics_checkpoints"),
            "checkpoints": len(index["checkpoints"]),
            "unavailable_targets": index["unavailable_targets"],
        }, sort_keys=True))
        return 0

    def compare_dynamics(args: object) -> int:
        config = load_config(args.config)
        extensions = load_research_extensions(config.source_path)
        if not extensions.dynamics_comparison.enabled:
            raise ValueError("dynamics_comparison.enabled must be true")
        report = compare_checkpoint_dynamics(
            args.hierarchy_run,
            config=config,
            options=extensions.dynamics_comparison,
        )
        print(json.dumps({
            "models": report["models"],
            "rows": len(report["rows"]),
            "unavailable_targets": report["unavailable_targets"],
            "output_dir": str(Path(args.hierarchy_run).expanduser().resolve() / "dynamics_comparison"),
        }, sort_keys=True))
        return 0

    register_handler("multiscale", multiscale)
    register_handler("synthetic", synthetic)
    register_handler("dynamics-checkpoints", dynamics_checkpoints)
    register_handler("compare-dynamics", compare_dynamics)


def main(argv: list[str] | None = None) -> int:
    from .cli import _HANDLERS

    register_research_handlers()
    parser = argparse.ArgumentParser(prog="semmap-haken-research")
    subparsers = parser.add_subparsers(dest="command", required=True)
    multiscale = subparsers.add_parser("multiscale")
    multiscale.add_argument("--config", required=True)
    multiscale.add_argument("--prepared-graph")
    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--config", required=True)
    checkpoints = subparsers.add_parser("dynamics-checkpoints")
    checkpoints.add_argument("--config", required=True)
    checkpoints.add_argument("--hierarchy-run", required=True)
    comparison = subparsers.add_parser("compare-dynamics")
    comparison.add_argument("--config", required=True)
    comparison.add_argument("--hierarchy-run", required=True)
    args = parser.parse_args(argv)
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
