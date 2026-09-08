"""Stable CLI surface; workflow implementations attach through registered handlers."""

from __future__ import annotations

import argparse
import hashlib
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
            raise ValueError("dataset.path is required for prepare and must reference a local decompressed assertions file")
        report = ParseReport()
        records = stream_assertions(
            config.dataset.path,
            AssertionFilters(config.dataset.language, frozenset(config.dataset.relations), config.dataset.min_weight),
            report=report,
            max_rows=config.dataset.max_rows,
        )
        graph = build_sparse_graph(records, directed=config.graph.directed, weight_transform=config.graph.weight_transform, component=config.dataset.component, max_nodes=config.dataset.max_nodes, self_loop_policy=config.graph.self_loop_policy)
        run_id = f"prepare-{uuid.uuid4().hex[:12]}"
        run_dir = config.paths.runs_root / run_id
        source_identity = {
            "path": str(config.dataset.path),
            "dataset_name": config.dataset.source,
            "dataset_version": config.dataset.version,
            "source_url": config.dataset.source_url,
            "input_format": "decompressed_conceptnet_assertions_tsv",
            "compression": "none",
            "max_rows": config.dataset.max_rows,
        }
        paths = save_prepared_graph(graph, run_dir, metadata={"parser_report": report.to_dict(), "source_identity": source_identity}, resolved_config=config.resolved)
        checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
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

    def run(args: argparse.Namespace) -> int:
        from .compute import ComputeContext
        from .dynamics import build_perturbations, run_linear_dynamics, save_dynamics_result, save_m1_plots
        from .graph_build import load_prepared_graph
        from .modes import analyze_normalized_adjacency, save_mode_result
        from .operators import normalized_adjacency

        config = load_config(args.config)
        # M2 compares both its embedding source and trajectory reference across
        # scales.  Therefore an enabled one-step run overrides the requested
        # execution settings before *any* M1 evidence is calculated, rather than
        # merely relabelling a potentially accelerated/float32 computation.
        m2_cpu_reference_override = config.coarsening.enabled and (
            config.execution.backend != "cpu" or config.execution.dtype != "float64"
        )
        if config.coarsening.enabled:
            compute = ComputeContext.create(
                backend="cpu", device=0, dtype="float64", workers=config.execution.workers,
                reserved_cpu_cores=config.execution.reserved_cpu_cores,
                threads_per_worker=config.execution.threads_per_worker,
                gpu_memory_fraction=config.execution.gpu_memory_fraction, batch_size=config.execution.batch_size,
                deterministic=True, allow_auto_fallback=True,
            )
        else:
            compute = ComputeContext.create(
                backend=config.execution.backend, device=config.execution.device, dtype=config.execution.dtype,
                workers=config.execution.workers, reserved_cpu_cores=config.execution.reserved_cpu_cores,
                threads_per_worker=config.execution.threads_per_worker,
                gpu_memory_fraction=config.execution.gpu_memory_fraction, batch_size=config.execution.batch_size,
                deterministic=config.execution.deterministic, allow_auto_fallback=config.execution.allow_auto_fallback,
            )
        print(json.dumps({"execution": compute.telemetry()["execution"], "fallback_reason": compute.fallback_reason, "m2_cpu_reference_override": m2_cpu_reference_override}, sort_keys=True))
        if config.spectral.prepared_graph_dir is None:
            raise ValueError("spectral.prepared_graph_dir is required for run")
        graph = load_prepared_graph(config.spectral.prepared_graph_dir)
        operator, degrees = normalized_adjacency(graph.adjacency, symmetry_tolerance=config.spectral.symmetry_tolerance)
        modes = analyze_normalized_adjacency(operator, degrees=degrees, alpha=config.dynamics.alpha, beta=config.dynamics.beta, margin=config.dynamics.auto_critical_margin, top_k=config.spectral.top_k, max_r=config.spectral.max_r, tolerance=config.spectral.tolerance, maxiter=config.spectral.maxiter, compute=compute)
        batch = build_perturbations(node_count=operator.shape[0], degrees=degrees, seed=config.dynamics.perturbation_seed, amplitude=config.dynamics.perturbation_amplitude, per_kind=config.dynamics.perturbations_per_kind, sparse_fraction=config.dynamics.random_sparse_fraction)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        run_dir = config.paths.runs_root / run_id
        run_dir.mkdir(parents=True)
        input_checksums = {"prepared_adjacency": hashlib.sha256((config.spectral.prepared_graph_dir / "adjacency.npz").read_bytes()).hexdigest()}
        spectral_paths = save_mode_result(modes, run_dir / "spectral", resolved_config=config.resolved, input_checksums=input_checksums)
        time_grid = __import__("numpy").linspace(config.dynamics.time_start, config.dynamics.time_stop, config.dynamics.time_steps)
        dynamics = run_linear_dynamics(operator, modes, initial_states=batch.initial_states, labels=batch.labels, time_grid=time_grid, storage_policy=config.dynamics.storage_policy, max_storage_bytes=config.dynamics.max_storage_mb * 1024 * 1024, compute=compute)
        dynamics_paths = save_dynamics_result(dynamics, run_dir / "dynamics", resolved_config=config.resolved, input_artifacts={"prepared_graph": str(config.spectral.prepared_graph_dir), "spectral_numeric": hashlib.sha256(spectral_paths["numeric"].read_bytes()).hexdigest()})
        plot_paths = save_m1_plots(run_dir / "plots", modes, dynamics)
        all_paths = {**{f"spectral_{key}": value for key, value in spectral_paths.items()}, **{f"dynamics_{key}": value for key, value in dynamics_paths.items()}, **{f"plot_{key}": value for key, value in plot_paths.items()}}
        stages = {"spectral": "completed", "dynamics": "completed"}
        coarsening_summary: dict[str, object] | None = None
        if config.coarsening.enabled:
            # M2 evidence runs on the trusted strict CPU float64 reference path per the delivery plan.
            from .coarsen import build_partition
            from .haken_embedding import build_haken_embedding
            from .metrics import evaluate_one_step, save_m2_result
            from .quotient import build_quotient

            embedding = build_haken_embedding(modes, weighting=config.coarsening.embedding_weighting)
            fine_trajectories = dynamics.trajectories
            if fine_trajectories is None:
                # Recompute the same perturbations with full storage only for the distortion comparison.
                fine_reference = run_linear_dynamics(operator, modes, initial_states=batch.initial_states, labels=batch.labels, time_grid=time_grid, storage_policy="all", max_storage_bytes=config.dynamics.max_storage_mb * 1024 * 1024, compute=compute)
                fine_trajectories = fine_reference.trajectories
            beta_value = float(modes.beta_selection.beta)
            method_results: dict[str, object] = {}
            for method in config.coarsening.methods:
                partition = build_partition(
                    graph.adjacency, embedding.coordinates, method=method,
                    target_reduction=config.coarsening.target_reduction, seed=config.coarsening.seed,
                    distance_threshold=config.coarsening.distance_threshold,
                )
                quotient = build_quotient(graph.adjacency, partition.fine_to_coarse, aggregation=config.coarsening.aggregation, node_ids=graph.node_ids, level=0, parent_run_id=run_id)
                result = evaluate_one_step(
                    graph.adjacency, modes, partition, quotient,
                    fine_trajectories=fine_trajectories, trajectory_labels=batch.labels, time_grid=time_grid,
                    alpha=float(modes.beta_selection.alpha), beta=beta_value, max_r=config.spectral.max_r,
                    symmetry_tolerance=config.spectral.symmetry_tolerance,
                )
                method_dir = run_dir / "coarsening" / method
                m2_paths = save_m2_result(
                    result, partition, quotient, embedding, method_dir,
                    resolved_config=config.resolved,
                    input_checksums={"prepared_adjacency": input_checksums["prepared_adjacency"], "spectral_numeric": hashlib.sha256(spectral_paths["numeric"].read_bytes()).hexdigest(), "dynamics_numeric": hashlib.sha256(dynamics_paths["numeric"].read_bytes()).hexdigest()},
                    execution_telemetry={"execution": {**compute.telemetry()["execution"], "cpu_reference_enforced": True, "dtype": "float64", "backend": "cpu"}},
                )
                for key, path in m2_paths.items():
                    all_paths[f"coarsening_{method}_{key}"] = path
                method_results[method] = result.to_dict()
            stages["coarsening"] = "completed"
            coarsening_summary = {"methods": list(config.coarsening.methods), "results": method_results}
        checksums = {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest() for path in all_paths.values()}
        manifest = RunManifest.create(run_id=run_id, resolved_config=config.resolved, execution_environment="cli")
        manifest = manifest.__class__(**{**manifest.to_dict(), "checksums": checksums, "stages": stages, "artifacts": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in all_paths.items()}, "random_seeds": {"runtime": config.runtime.random_seed, "perturbations": config.dynamics.perturbation_seed, "coarsening": config.coarsening.seed}, "cli_replay": True, "execution_telemetry": compute.telemetry(), "resumability": {"prepared_graph_dir": str(config.spectral.prepared_graph_dir), "environment_constraints": "requirements/constraints.txt"}})
        manifest.write_json(run_dir / "manifest.json")
        (run_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
        print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "spectral": str(run_dir / "spectral"), "dynamics": str(run_dir / "dynamics"), "coarsening": coarsening_summary}, sort_keys=True))
        return 0

    register_handler("download", download)
    register_handler("prepare", prepare)
    register_handler("run", run)


_data_handlers()
