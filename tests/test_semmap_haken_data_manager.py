from __future__ import annotations

import hashlib
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from semmap_haken.data_manager import DatasetDescriptor, DownloadError, acquire_dataset, versioned_cache_path


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def test_http_download_is_atomic_and_subsequent_request_is_cache_hit(tmp_path: Path) -> None:
    raw = tmp_path / "remote.gz"
    raw.write_bytes(b"remote fixture")
    handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(tmp_path), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        descriptor = DatasetDescriptor(expected_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), expected_size_bytes=raw.stat().st_size, filename=raw.name)
        url = f"http://127.0.0.1:{server.server_port}/{raw.name}"
        first = acquire_dataset(descriptor, cache_root=tmp_path / "cache", source_url=url)
        second = acquire_dataset(descriptor, cache_root=tmp_path / "cache", source_url=url)
    finally:
        server.shutdown()
        thread.join()
    assert first.source == "http" and not first.path.with_suffix(".gz.part").exists()
    assert second.source == "cache" and second.cache_hit


def test_manual_acquisition_is_verified_and_then_cached(tmp_path: Path) -> None:
    raw = tmp_path / "fixture.tsv.gz"
    raw.write_bytes(b"tiny fixture")
    descriptor = DatasetDescriptor(expected_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), expected_size_bytes=raw.stat().st_size, filename=raw.name)
    first = acquire_dataset(descriptor, cache_root=tmp_path / "cache", manual_path=raw)
    second = acquire_dataset(descriptor, cache_root=tmp_path / "cache", manual_path=raw)
    assert first.source == "manual" and first.cache_hit
    assert second.path == raw and second.sha256 == first.sha256


def test_bad_cached_file_is_removed_before_http_failure(tmp_path: Path) -> None:
    descriptor = DatasetDescriptor(expected_sha256="a" * 64, filename="bad.gz")
    cached = versioned_cache_path(tmp_path, descriptor)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"bad")
    with pytest.raises(DownloadError):
        acquire_dataset(descriptor, cache_root=tmp_path, source_url="http://127.0.0.1:1/unavailable", retries=1, timeout_seconds=0.01)
    assert not cached.exists()
    assert not cached.with_suffix(".gz.part").exists()
