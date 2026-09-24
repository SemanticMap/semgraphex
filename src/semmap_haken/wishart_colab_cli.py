"""Colab-first CLI for Drive-backed recursive Wishart experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .wishart_cli import _load_or_build
from .wishart_colab import (
    ColabLayout,
    DriveCheckpointSync,
    ensure_drive_root,
    limit_native_threads,
    mount_google_drive,
    preflight_colab_storage,
    resolve_drive_path,
    stage_from_drive,
    sync_tree,
)
from .wishart_config import (
    load_colab_options,
    load_dictionary_options,
    load_wishart_options,
)
from .wishart_gpu import resolve_device
from .wishart_resume import (
    config_sha256, load_latest_checkpoint, truncate_after_checkpoint,
)
from .wishart_hierarchy import run_wishart_hierarchy


def _nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir()) if path.is_dir() else path.exists()


def _prepare_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        elif _nonempty(path):
            marker = " (COMPLETED)" if (path / "COMPLETED").is_file() else ""
            raise ValueError(
                f"destination already contains data{marker}: {path}; "
                "use a new --run-name or pass --overwrite"
            )
    path.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semmap-wishart-colab",
        description=(
            "Run recursive Wishart coarsening on fast Colab scratch while "
            "staging inputs from and checkpointing results to Google Drive."
        ),
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config", type=Path, help="Local YAML configuration path.",
    )
    config_group.add_argument(
        "--config-drive", type=Path,
        help="YAML stored under --drive-root; staged to fast local scratch.",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/SemanticMap/colab/wishart"),
        help="Durable project root on mounted Google Drive.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/content/semmap-wishart"),
        help="Fast ephemeral local root used for all compute-intensive I/O.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--prepared-drive",
        type=Path,
        help=(
            "Completed prepared graph on Drive. Relative paths are resolved "
            "under --drive-root."
        ),
    )
    source.add_argument(
        "--dataset-drive",
        type=Path,
        help=(
            "ConceptNet assertion file on Drive. Relative paths are resolved "
            "under --drive-root."
        ),
    )
    parser.add_argument(
        "--run-name",
        help="Durable run directory name under <drive-root>/runs.",
    )
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help=(
            "Attempt google.colab.drive.mount('/content/drive'). In notebooks, "
            "mounting explicitly in a cell before invoking the CLI is preferred."
        ),
    )
    parser.add_argument("--force-remount", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Restore latest verified checkpoint from this named Drive run.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"),
        help="Override colab.device from the YAML. CUDA applies to feature kNN.",
    )
    parser.add_argument(
        "--cpu-workers", type=int,
        help="Override bounded parallel ego-extraction workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing scratch and Drive run directories with this run name.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("size_mtime", "sha256"),
        default="size_mtime",
        help="Checkpoint file-change detection. sha256 is safer but rereads files.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="Limit BLAS/OpenMP pools to avoid CPU oversubscription in Colab.",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep local /content run after successful Drive synchronization.",
    )
    return parser


def _source_sha256(path: Path) -> str:
    """Fingerprint input bytes and relative names, independently of Drive mtimes."""
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file() and not item.is_symlink()
    )
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _device_details(selected: str, requested: str) -> dict[str, object]:
    details: dict[str, object] = {
        "requested": requested,
        "selected": selected,
        "cpu_count": os.cpu_count(),
    }
    if selected == "cuda":
        import torch
        properties = torch.cuda.get_device_properties(0)
        details.update({
            "gpu_name": properties.name,
            "gpu_memory_bytes": int(properties.total_memory),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        })
    return details


def _assert_resume_device_consistent(previous: object, selected: str) -> None:
    """Keep one run's CPU float64 or GPU float32 nearest-neighbor lineage."""
    if previous == selected:
        return
    raise ValueError(
        f"resume compute device changed from {previous} to {selected}; "
        "CPU (float64 sklearn) and CUDA (float32 torch) kNN numerics differ. "
        "Use a new run name for the other device (recommended -gpu / -cpu), "
        "or finish this run on its original device."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.blas_threads < 1:
        raise ValueError("--blas-threads must be >= 1")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be combined")
    if args.cpu_workers is not None and args.cpu_workers < 1:
        raise ValueError("--cpu-workers must be >= 1")

    if args.mount_drive:
        mount_google_drive("/content/drive", force_remount=args.force_remount)

    drive_root = ensure_drive_root(args.drive_root)
    scratch_root = args.scratch_root.expanduser().resolve()
    layout = ColabLayout(drive_root=drive_root, scratch_root=scratch_root)
    layout.scratch_inputs.mkdir(parents=True, exist_ok=True)
    layout.scratch_runs.mkdir(parents=True, exist_ok=True)
    layout.drive_runs.mkdir(parents=True, exist_ok=True)

    if args.config_drive is not None:
        config_source = resolve_drive_path(drive_root, args.config_drive)
        if not config_source.is_file():
            raise FileNotFoundError(config_source)
        # Configuration is always staged before parsing; never repeatedly
        # parse YAML through the Drive FUSE mount during heavy computation.
        config_path = layout.scratch_root / "config" / config_source.name
        stage_from_drive(config_source, config_path, compare_mode="sha256")
    else:
        config_source = args.config.expanduser().resolve()
        config_path = config_source
    scientific_config_hash = config_sha256(config_path)
    config = load_config(config_path)
    options = load_wishart_options(config_path)
    dictionary_options = load_dictionary_options(config_path)
    execution_options = load_colab_options(config_path)
    if args.device is not None:
        execution_options = replace(execution_options, device=args.device)
    if args.cpu_workers is not None:
        execution_options = replace(
            execution_options, cpu_workers=args.cpu_workers,
        )
    execution_options.validate()
    available_cores = max(1, os.cpu_count() or 1)
    if execution_options.cpu_workers > available_cores:
        print(
            f"WARNING: requested {execution_options.cpu_workers} CPU workers but "
            f"this runtime exposes only {available_cores}; clamping to "
            f"{available_cores} to avoid oversubscription.",
            file=sys.stderr, flush=True,
        )
        execution_options = replace(
            execution_options, cpu_workers=available_cores,
        )
    requested_device = execution_options.device
    selected_device = resolve_device(requested_device)
    device = _device_details(selected_device, requested_device)
    # Resolve auto exactly once: no accidental CPU/GPU backend change between
    # hierarchy levels after a CUDA driver or resource-state change.
    if options.metric in {"typed_wl", "graphlet"}:
        execution_options = replace(execution_options, device=selected_device)
    elif selected_device == "cuda" and requested_device == "cuda":
        raise ValueError(
            f"metric={options.metric} has no CUDA backend; select --device cpu "
            "or choose typed_wl/graphlet"
        )
    elif requested_device == "auto" and selected_device == "cuda":
        execution_options = replace(execution_options, device="cpu")
        device["selected"] = "cpu"
        selected_device = "cpu"
    device["cpu_workers"] = execution_options.cpu_workers
    device["gpu_eligible"] = options.metric in {"typed_wl", "graphlet"}
    print(json.dumps(
        {"stage": "device_selected", "device": device}, sort_keys=True,
    ), flush=True)
    if not device["gpu_eligible"] and requested_device == "auto":
        print(
            f"WARNING: metric={options.metric} does not support GPU kNN; "
            "using CPU for this scientific metric.",
            file=sys.stderr, flush=True,
        )
    elif requested_device == "auto" and selected_device == "cpu":
        from .wishart_gpu import cuda_diagnostics
        print(
            "WARNING: colab.device=auto resolved to CPU; GPU kNN will not be "
            f"used. Diagnostics: {json.dumps(cuda_diagnostics(), sort_keys=True)}. "
            "Select a Colab GPU runtime and install the [gpu] extra.",
            file=sys.stderr, flush=True,
        )
    elif selected_device == "cuda" and options.metric not in {"typed_wl", "graphlet"}:
        print(
            f"WARNING: metric={options.metric} has no CUDA kNN backend; "
            "this experiment will use CPU for metric computation.",
            file=sys.stderr, flush=True,
        )
    code_revision = _git_revision()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"wishart-{options.metric}-{stamp}"
    local_run = layout.scratch_runs / run_name
    drive_run = layout.drive_runs / run_name
    if args.resume:
        if not drive_run.is_dir():
            raise FileNotFoundError(
                f"no existing Drive run to resume: {drive_run}"
            )
        local_run.mkdir(parents=True, exist_ok=True)
        # Restage all immutable checkpoints plus output artifacts once.
        stage_from_drive(drive_run, local_run, compare_mode=args.compare_mode)
    else:
        _prepare_destination(local_run, overwrite=args.overwrite)
        _prepare_destination(drive_run, overwrite=args.overwrite)

    local_prepared: Path | None = None
    local_dataset: Path | None = None
    source_on_drive: Path | None = None

    if args.prepared_drive is not None:
        source_on_drive = resolve_drive_path(drive_root, args.prepared_drive)
        local_prepared = layout.scratch_inputs / "prepared" / source_on_drive.name
        preflight = preflight_colab_storage(
            scratch_root=scratch_root,
            source_path=source_on_drive,
            drive_root=drive_root,
        )
        if not preflight.ok:
            raise RuntimeError(
                "Colab scratch preflight failed: " + "; ".join(preflight.warnings)
            )
        stage_stats = stage_from_drive(
            source_on_drive,
            local_prepared,
            compare_mode=args.compare_mode,
            require_completed=True,
        )
        staged_source = local_prepared
    elif args.dataset_drive is not None:
        source_on_drive = resolve_drive_path(drive_root, args.dataset_drive)
        local_dataset = layout.scratch_inputs / "datasets" / source_on_drive.name
        preflight = preflight_colab_storage(
            scratch_root=scratch_root,
            source_path=source_on_drive,
            drive_root=drive_root,
        )
        if not preflight.ok:
            raise RuntimeError(
                "Colab scratch preflight failed: " + "; ".join(preflight.warnings)
            )
        stage_stats = stage_from_drive(
            source_on_drive, local_dataset, compare_mode=args.compare_mode,
        )
        staged_source = local_dataset
    else:
        if config.dataset.path is None:
            raise ValueError(
                "provide --prepared-drive or --dataset-drive, or configure dataset.path"
            )
        source_on_drive = config.dataset.path
        local_dataset = layout.scratch_inputs / "datasets" / source_on_drive.name
        preflight = preflight_colab_storage(
            scratch_root=scratch_root,
            source_path=source_on_drive,
            drive_root=drive_root,
        )
        if not preflight.ok:
            raise RuntimeError(
                "Colab scratch preflight failed: " + "; ".join(preflight.warnings)
            )
        stage_stats = stage_from_drive(
            source_on_drive, local_dataset, compare_mode=args.compare_mode,
        )
        staged_source = local_dataset

    input_hash = _source_sha256(staged_source)

    if args.resume:
        original_input = local_run / "input.json"
        if not original_input.is_file():
            raise ValueError("resume requires the original input.json manifest")
        original = json.loads(original_input.read_text(encoding="utf-8"))
        for field, current_value in (
            ("config_sha256", scientific_config_hash),
            ("input_sha256", input_hash),
            ("code_revision", code_revision),
        ):
            if original.get(field) != current_value:
                raise ValueError(
                    f"resume {field} mismatch: keep the exact config, input and Git revision"
                )
        previous_device = original.get("device", {}).get("selected")
        _assert_resume_device_consistent(previous_device, selected_device)
        if (drive_run / "COMPLETED").is_file():
            print(json.dumps({
                "status": "already_completed",
                "drive_run": str(drive_run),
            }, indent=2))
            return 0
        state = load_latest_checkpoint(
            local_run,
            config_hash=scientific_config_hash,
            input_hash=input_hash,
            code_revision=code_revision,
        )
        if state is None:
            raise ValueError("resume requested but no valid level checkpoint exists")
        next_level = int(state["next_level"])
        truncate_after_checkpoint(local_run, next_level=next_level)
        truncate_after_checkpoint(drive_run, next_level=next_level)
        with (local_run / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "next_level": next_level,
                "code_revision": code_revision,
            }, sort_keys=True) + "\n")
    else:
        input_payload = {
            "execution_environment": "colab",
            "config": str(config_path),
            "config_source": str(config_source),
            "config_sha256": scientific_config_hash,
            "input_sha256": input_hash,
            "code_revision": code_revision,
            "metric": options.metric,
            "dictionary_enabled": dictionary_options.enabled,
            "source": str(source_on_drive),
            "drive_source": str(source_on_drive),
            "drive_run": str(drive_run),
            "local_run": str(local_run),
            "staging": stage_stats.to_dict(),
            "preflight": preflight.to_dict(),
            "device": device,
            "execution_options": {
                "device": execution_options.device,
                "cpu_workers": execution_options.cpu_workers,
                "gpu_batch_size": execution_options.gpu_batch_size,
                "min_gpu_types": execution_options.min_gpu_types,
                "checkpoint_every_levels": execution_options.checkpoint_every_levels,
            },
            "storage_strategy": "drive_to_local_scratch_compute_to_incremental_drive_checkpoint",
            "blas_threads": args.blas_threads,
        }
        (local_run / "input.json").write_text(
            json.dumps(input_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # A prepared graph on Drive is preferred: no repeated ConceptNet parsing
    # after runtime interruption. A raw dataset remains supported as fallback.
    resolved_config = load_config(config_path)
    _, graph, source_meta = _load_or_build(
        config_path, local_prepared, dataset_path=local_dataset,
    )

    checkpoint = DriveCheckpointSync(
        drive_run_dir=drive_run, compare_mode=args.compare_mode,
    )
    checkpoint(local_run, {
        "stage": "restaged" if args.resume else "initialized",
        "metric": options.metric,
        "source": str(source_on_drive),
    })

    try:
        with limit_native_threads(args.blas_threads):
            summary = run_wishart_hierarchy(
                graph,
                directed=resolved_config.graph.directed,
                options=options,
                dictionary_options=dictionary_options,
                execution_options=execution_options,
                output_dir=local_run,
                checkpoint_hook=checkpoint,
                resume=args.resume,
                checkpoint_config_hash=scientific_config_hash,
                checkpoint_input_hash=input_hash,
                checkpoint_code_revision=code_revision,
            )
    except BaseException as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (local_run / "FAILED.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sync_tree(local_run, drive_run, compare_mode=args.compare_mode, final=False)
        print(json.dumps({
            "status": "failed",
            "drive_run": str(drive_run),
            "local_run": str(local_run),
            "failure": failure,
        }, indent=2, sort_keys=True), file=sys.stderr)
        raise

    colab_summary = {
        "status": "completed",
        "summary": summary.to_dict(),
        "drive_run": str(drive_run),
        "local_run": str(local_run),
        "preflight": preflight.to_dict(),
        "staging": stage_stats.to_dict(),
        "device": device,
        "resumed": bool(args.resume),
        "scratch_removed_after_sync": not args.keep_scratch,
    }
    (local_run / "COLAB_RUN.json").write_text(
        json.dumps(colab_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint(local_run, {
        "stage": "published",
        "levels": summary.levels,
        "final_nodes": summary.final_nodes,
        "stop_reason": summary.stop_reason,
    })
    if not args.keep_scratch:
        shutil.rmtree(local_run)
    print(json.dumps(colab_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
