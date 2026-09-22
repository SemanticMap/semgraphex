"""Verified, cache-aware acquisition of versioned ConceptNet assertion dumps."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OFFICIAL_CONCEPTNET_57_URL = "https://conceptnet.s3.amazonaws.com/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be acquired and verified safely."""


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_id: str = "conceptnet-5.7"
    version: str = "5.7.0"
    source_url: str = OFFICIAL_CONCEPTNET_57_URL
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None
    filename: str = "conceptnet-assertions-5.7.0.csv.gz"


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    size_bytes: int
    cache_hit: bool
    source: str
    verification_status: str
    restarted_download: bool = False


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versioned_cache_path(cache_root: str | Path, descriptor: DatasetDescriptor) -> Path:
    key = descriptor.expected_sha256 or "unverified"
    return Path(cache_root) / "datasets" / "conceptnet" / descriptor.version / key / descriptor.filename


def _verify(path: Path, descriptor: DatasetDescriptor) -> tuple[str, int]:
    size = path.stat().st_size
    if descriptor.expected_size_bytes is not None and size != descriptor.expected_size_bytes:
        raise DownloadError(f"size mismatch for {path}: expected {descriptor.expected_size_bytes}, got {size}")
    digest = sha256_file(path)
    if descriptor.expected_sha256 is not None and digest != descriptor.expected_sha256:
        raise DownloadError(f"SHA-256 mismatch for {path}: expected {descriptor.expected_sha256}, got {digest}")
    return digest, size


def _preflight(destination: Path, required_bytes: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if required_bytes is not None and shutil.disk_usage(destination.parent).free < required_bytes:
        raise DownloadError(f"insufficient free disk for {required_bytes} bytes at {destination.parent}")


def acquire_dataset(
    descriptor: DatasetDescriptor,
    *,
    cache_root: str | Path,
    manual_path: str | Path | None = None,
    source_url: str | None = None,
    retries: int = 3,
    timeout_seconds: float = 60.0,
) -> DownloadResult:
    """Acquire a dump atomically.

    Existing cache/manual files are verified before use. Downloads deliberately restart
    from zero after interruption: no partial content is ever treated as a cache hit.
    """
    if descriptor.expected_sha256 is None and manual_path is None:
        raise DownloadError("official acquisition requires a pinned expected SHA-256; refusing unverified cache/download")
    destination = versioned_cache_path(cache_root, descriptor)
    for candidate, source in ((destination, "cache"), (Path(manual_path) if manual_path else None, "manual")):
        if candidate is not None and candidate.is_file():
            try:
                digest, size = _verify(candidate, descriptor)
            except DownloadError:
                if candidate == destination:
                    candidate.unlink()
                else:
                    raise
            else:
                status = "verified" if descriptor.expected_sha256 else "manual_unverified"
                if source == "cache" and status != "verified":
                    raise DownloadError("unverified official cache must not be reused")
                return DownloadResult(candidate, digest, size, True, source, status)

    _preflight(destination, descriptor.expected_size_bytes)
    url = source_url or descriptor.source_url
    part = destination.with_suffix(destination.suffix + ".part")
    # Restart rather than attempt Range: this is deterministic across servers/proxies.
    part.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response, part.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            digest, size = _verify(part, descriptor)
            os.replace(part, destination)
            return DownloadResult(destination, digest, size, False, "http", "verified", restarted_download=attempt > 0)
        except Exception as error:  # urllib errors differ by transport implementation.
            last_error = error
            part.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise DownloadError(f"download failed after {retries} attempts: {last_error}")
