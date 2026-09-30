"""Colab/Google Drive adapters for Wishart experiments.

The scientific pipeline remains local-disk first.  Google Drive is used only as
durable input/output storage because repeated sparse/random I/O through the
Drive FUSE mount is substantially slower than /content local storage.

This module intentionally imports google.colab only inside mount_google_drive().
It is therefore import-safe in ordinary CLI/test environments.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping


CompareMode = Literal["size_mtime", "sha256"]


@dataclass(frozen=True)
class ColabLayout:
    drive_root: Path
    scratch_root: Path

    @property
    def scratch_inputs(self) -> Path:
        return self.scratch_root / "inputs"

    @property
    def scratch_runs(self) -> Path:
        return self.scratch_root / "runs"

    @property
    def drive_runs(self) -> Path:
        return self.drive_root / "runs"


@dataclass(frozen=True)
class ColabPreflight:
    available_ram_bytes: int
    scratch_free_bytes: int
    drive_free_bytes: int | None
    source_bytes: int
    recommended_scratch_bytes: int
    ok: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SyncStats:
    source: str
    destination: str
    copied_files: int
    skipped_files: int
    copied_bytes: int
    elapsed_seconds: float
    final: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def mount_google_drive(
    mount_point: str | Path = "/content/drive",
    *,
    force_remount: bool = False,
) -> Path:
    """Mount Google Drive in Colab and return the mount point.

    Importing this function's module outside Colab is safe. Calling this
    function outside Colab raises a clear RuntimeError.
    """
    try:
        from google.colab import drive  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "google.colab is unavailable; --mount-drive must be used inside Google Colab"
        ) from error

    target = Path(mount_point)
    drive.mount(str(target), force_remount=force_remount)
    return target


def ensure_drive_root(path: str | Path) -> Path:
    """Validate a Colab Drive mount before creating a project directory."""
    root = Path(path).expanduser().resolve()
    colab_mount = Path("/content/drive")
    try:
        under_colab_mount = root.is_relative_to(colab_mount)
    except AttributeError:  # pragma: no cover - Python >= 3.10 in this project.
        under_colab_mount = str(root).startswith(str(colab_mount))
    if under_colab_mount and not (colab_mount / "MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive does not appear mounted at /content/drive. "
            "Run drive.mount('/content/drive') in a Colab cell first."
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _available_memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return 0


def tree_size_bytes(path: str | Path) -> int:
    source = Path(path)
    if source.is_file():
        return source.stat().st_size
    if not source.exists():
        raise FileNotFoundError(source)
    total = 0
    for item in source.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def preflight_colab_storage(
    *,
    scratch_root: str | Path,
    source_path: str | Path,
    drive_root: str | Path | None = None,
    scratch_multiplier: float = 3.0,
    minimum_extra_bytes: int = 2 * 1024**3,
) -> ColabPreflight:
    """Check local scratch capacity before staging a Drive-backed experiment.

    The estimate intentionally errs high: source bytes plus room for CSR
    intermediates, relation layers and output artifacts.
    """
    scratch = Path(scratch_root)
    scratch.mkdir(parents=True, exist_ok=True)
    source_bytes = tree_size_bytes(source_path)
    recommended = max(
        int(source_bytes * scratch_multiplier),
        source_bytes + minimum_extra_bytes,
    )
    scratch_free = shutil.disk_usage(scratch).free
    drive_free: int | None = None
    warnings: list[str] = []
    if drive_root is not None:
        drive = Path(drive_root)
        drive.mkdir(parents=True, exist_ok=True)
        try:
            drive_free = shutil.disk_usage(drive).free
        except OSError:
            warnings.append("could not determine Google Drive free space")

    ram = _available_memory_bytes()
    if ram and ram < 6 * 1024**3:
        warnings.append(
            "available RAM is below 6 GiB; reduce candidate_limit / diagnostics sampling"
        )
    ok = scratch_free >= recommended
    if not ok:
        warnings.append(
            f"insufficient local scratch: need about {recommended} bytes, have {scratch_free}"
        )
    return ColabPreflight(
        available_ram_bytes=ram,
        scratch_free_bytes=scratch_free,
        drive_free_bytes=drive_free,
        source_bytes=source_bytes,
        recommended_scratch_bytes=recommended,
        ok=ok,
        warnings=tuple(warnings),
    )


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_file(source: Path, destination: Path, compare_mode: CompareMode) -> bool:
    if not destination.is_file():
        return False
    src_stat = source.stat()
    dst_stat = destination.stat()
    if src_stat.st_size != dst_stat.st_size:
        return False
    if compare_mode == "sha256":
        return _sha256(source) == _sha256(destination)

    # copy2 preserves mtime, which makes this cheap for repeated checkpoint syncs.
    # Drive FUSE can round timestamp precision, so allow a small tolerance.
    return abs(src_stat.st_mtime - dst_stat.st_mtime) <= 2.0


def _atomic_copy(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
        delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, temporary, length=4 * 1024 * 1024)
    try:
        shutil.copystat(source, temp_path)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return destination.stat().st_size


def stage_from_drive(
    source: str | Path,
    destination: str | Path,
    *,
    compare_mode: CompareMode = "size_mtime",
    require_completed: bool = False,
) -> SyncStats:
    """Stage a Drive file/tree to fast local scratch, incrementally."""
    return sync_tree(
        source,
        destination,
        compare_mode=compare_mode,
        final=True,
        require_completed=require_completed,
    )


def sync_tree(
    source: str | Path,
    destination: str | Path,
    *,
    compare_mode: CompareMode = "size_mtime",
    final: bool,
    require_completed: bool = False,
) -> SyncStats:
    """Incrementally copy a file or directory.

    When final=False, a source COMPLETED marker is deliberately skipped.  This
    guarantees that a Drive run is never advertised as complete before all
    checkpoint files have been copied.
    """
    started = time.perf_counter()
    src = Path(source)
    dst = Path(destination)
    if not src.exists():
        raise FileNotFoundError(src)
    if require_completed and src.is_dir() and not (src / "COMPLETED").is_file():
        raise ValueError(f"source directory is not completed: {src}")

    copied = 0
    skipped = 0
    copied_bytes = 0

    if src.is_file():
        target = dst
        if dst.exists() and dst.is_dir():
            target = dst / src.name
        if _same_file(src, target, compare_mode):
            skipped += 1
        else:
            copied_bytes += _atomic_copy(src, target)
            copied += 1
    else:
        dst.mkdir(parents=True, exist_ok=True)
        completed_source = src / "COMPLETED"
        for item in sorted(src.rglob("*")):
            if item.is_symlink() or not item.is_file():
                continue
            if item.name.endswith(".part"):
                continue
            relative = item.relative_to(src)
            # The completion marker is always handled explicitly after every
            # other file, never as part of lexical directory traversal.
            if relative == Path("COMPLETED"):
                continue
            target = dst / relative
            if _same_file(item, target, compare_mode):
                skipped += 1
                continue
            copied_bytes += _atomic_copy(item, target)
            copied += 1
        if final and completed_source.is_file():
            completed_target = dst / "COMPLETED"
            if _same_file(completed_source, completed_target, compare_mode):
                skipped += 1
            else:
                copied_bytes += _atomic_copy(completed_source, completed_target)
                copied += 1

    return SyncStats(
        source=str(src),
        destination=str(dst),
        copied_files=copied,
        skipped_files=skipped,
        copied_bytes=copied_bytes,
        elapsed_seconds=time.perf_counter() - started,
        final=final,
    )


class DriveCheckpointSync:
    """Callable checkpoint hook for run_wishart_hierarchy()."""

    def __init__(
        self,
        *,
        drive_run_dir: str | Path,
        compare_mode: CompareMode = "size_mtime",
    ) -> None:
        self.drive_run_dir = Path(drive_run_dir)
        self.compare_mode = compare_mode
        self.drive_run_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        local_run_dir: Path,
        event: Mapping[str, object],
    ) -> None:
        # The hierarchy's local "completed" event remains checkpoint-only.
        # The Colab CLI emits "published" only after COLAB_RUN.json exists.
        final = str(event.get("stage")) == "published"

        # Phase 1: synchronize every payload except COMPLETED.
        stats = sync_tree(
            local_run_dir,
            self.drive_run_dir,
            compare_mode=self.compare_mode,
            final=False,
        )

        # Phase 2: persist checkpoint metadata before the authoritative marker.
        checkpoint = {
            "event": dict(event),
            "sync": {**stats.to_dict(), "final": final},
            "updated_at_unix": time.time(),
            "completion_marker_policy": "written_after_checkpoint_when_stage_is_published",
        }
        local_checkpoint = local_run_dir / "DRIVE_CHECKPOINT.json"
        with tempfile.NamedTemporaryFile(
            dir=local_run_dir,
            prefix=".DRIVE_CHECKPOINT.",
            suffix=".part",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            json.dump(checkpoint, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        try:
            os.replace(temp_path, local_checkpoint)
        finally:
            temp_path.unlink(missing_ok=True)
        _atomic_copy(local_checkpoint, self.drive_run_dir / "DRIVE_CHECKPOINT.json")

        # Phase 3: publish COMPLETED as the final durable write.
        if final:
            completed = local_run_dir / "COMPLETED"
            if not completed.is_file():
                raise FileNotFoundError(
                    f"cannot publish completed run without local marker: {completed}"
                )
            _atomic_copy(completed, self.drive_run_dir / "COMPLETED")


def resolve_drive_path(drive_root: str | Path, value: str | Path) -> Path:
    """Resolve a relative Drive path under drive_root; preserve absolute paths."""
    value_path = Path(value).expanduser()
    if value_path.is_absolute():
        return value_path
    return (Path(drive_root).expanduser() / value_path).resolve()


@contextmanager
def limit_native_threads(threads: int) -> Iterator[None]:
    """Limit BLAS/OpenMP pools to avoid Colab CPU oversubscription."""
    if threads < 1:
        raise ValueError("threads must be >= 1")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=threads):
        yield
