"""Colab-first CLI for Drive-backed recursive Wishart experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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
from .wishart_config import load_dictionary_options, load_wishart_options
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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/SemanticMap/semgraphex"),
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if args.blas_threads < 1:
        raise ValueError("--blas-threads must be >= 1")

    if args.mount_drive:
        mount_google_drive("/content/drive", force_remount=args.force_remount)

    config = load_config(config_path)
    options = load_wishart_options(config_path)
    dictionary_options = load_dictionary_options(config_path)
    drive_root = ensure_drive_root(args.drive_root)
    scratch_root = args.scratch_root.expanduser().resolve()
    layout = ColabLayout(drive_root=drive_root, scratch_root=scratch_root)
    layout.scratch_inputs.mkdir(parents=True, exist_ok=True)
    layout.scratch_runs.mkdir(parents=True, exist_ok=True)
    layout.drive_runs.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"wishart-{options.metric}-{stamp}"
    local_run = layout.scratch_runs / run_name
    drive_run = layout.drive_runs / run_name
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
                "Colab scratch preflight failed: "
                + "; ".join(preflight.warnings)
            )
        stage_stats = stage_from_drive(
            source_on_drive,
            local_prepared,
            compare_mode=args.compare_mode,
            require_completed=True,
        )
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
                "Colab scratch preflight failed: "
                + "; ".join(preflight.warnings)
            )
        stage_stats = stage_from_drive(
            source_on_drive,
            local_dataset,
            compare_mode=args.compare_mode,
        )
    else:
        # Local dataset.path remains supported for offline testing and for users
        # who download/copy data to /content themselves.
        if config.dataset.path is None:
            raise ValueError(
                "provide --prepared-drive or --dataset-drive, or configure dataset.path"
            )
        source_on_drive = config.dataset.path
        preflight = preflight_colab_storage(
            scratch_root=scratch_root,
            source_path=source_on_drive,
            drive_root=drive_root,
        )
        if not preflight.ok:
            raise RuntimeError(
                "Colab scratch preflight failed: "
                + "; ".join(preflight.warnings)
            )
        local_dataset = layout.scratch_inputs / "datasets" / source_on_drive.name
        stage_stats = stage_from_drive(
            source_on_drive,
            local_dataset,
            compare_mode=args.compare_mode,
        )

    resolved_config = load_config(config_path)
    _, graph, source_meta = _load_or_build(
        config_path,
        local_prepared,
        dataset_path=local_dataset,
    )

    input_payload = {
        "execution_environment": "colab",
        "config": str(config_path),
        "metric": options.metric,
        "dictionary_enabled": dictionary_options.enabled,
        "source": source_meta,
        "drive_source": str(source_on_drive),
        "drive_run": str(drive_run),
        "local_run": str(local_run),
        "staging": stage_stats.to_dict(),
        "preflight": preflight.to_dict(),
        "storage_strategy": "drive_to_local_scratch_compute_to_incremental_drive_checkpoint",
        "blas_threads": args.blas_threads,
    }
    (local_run / "input.json").write_text(
        json.dumps(input_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checkpoint = DriveCheckpointSync(
        drive_run_dir=drive_run,
        compare_mode=args.compare_mode,
    )
    checkpoint(
        local_run,
        {
            "stage": "initialized",
            "metric": options.metric,
            "source": str(source_on_drive),
        },
    )

    try:
        with limit_native_threads(args.blas_threads):
            summary = run_wishart_hierarchy(
                graph,
                directed=resolved_config.graph.directed,
                options=options,
                dictionary_options=dictionary_options,
                output_dir=local_run,
                checkpoint_hook=checkpoint,
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
        sync_tree(
            local_run,
            drive_run,
            compare_mode=args.compare_mode,
            final=False,
        )
        print(
            json.dumps(
                {
                    "status": "failed",
                    "drive_run": str(drive_run),
                    "local_run": str(local_run),
                    "failure": failure,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise

    colab_summary = {
        "status": "completed",
        "summary": summary.to_dict(),
        "drive_run": str(drive_run),
        "local_run": str(local_run),
        "preflight": preflight.to_dict(),
        "staging": stage_stats.to_dict(),
        "scratch_removed_after_sync": not args.keep_scratch,
    }
    (local_run / "COLAB_RUN.json").write_text(
        json.dumps(colab_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint(
        local_run,
        {
            "stage": "published",
            "levels": summary.levels,
            "final_nodes": summary.final_nodes,
            "stop_reason": summary.stop_reason,
        },
    )

    if not args.keep_scratch:
        shutil.rmtree(local_run)

    print(json.dumps(colab_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
