"""Colab runner tests: no Google account, cloud mounts or production database."""
from __future__ import annotations
import json
from pathlib import Path
import zipfile

import pytest

from semmap_haken.text_colab import (
    checkpoint_document, prepare_zip, restore_documents, safe_label
)
from semmap_haken.text_corpus import sha256


def test_zip_selection_and_stable_order(tmp_path):
    archive = tmp_path / "texts.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("z/b.txt", "Cat animal")
        z.writestr("a/a.txt", "Dog tool")
        z.writestr("notes.csv", "ignored")
    summary = prepare_zip(archive, tmp_path / "input", max_docs=1,
                          max_doc_bytes=100, max_total_bytes=200)
    assert summary["selected_count"] == 1
    assert (tmp_path / "input/a/a.txt").read_text() == "Dog tool"
    assert not (tmp_path / "input/z/b.txt").exists()


@pytest.mark.parametrize("member", ["../escape.txt", "/root.txt", "a/../escape.txt"])
def test_zip_traversal_rejected(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(member, "oops")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        prepare_zip(archive, tmp_path / "out", max_docs=None,
                    max_doc_bytes=100, max_total_bytes=200)


def test_zip_size_guard(tmp_path):
    archive = tmp_path / "huge.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a.txt", "abc" * 100)
    with pytest.raises(ValueError, match="too large"):
        prepare_zip(archive, tmp_path / "out", max_docs=None,
                    max_doc_bytes=10, max_total_bytes=300)


def test_label_validation():
    assert safe_label("pilot-01") == "pilot-01"
    for text in ("", "../bad", "space label", "_prefix" * 20):
        with pytest.raises(ValueError):
            safe_label(text)


def test_incremental_checkpoint_and_restore(tmp_path):
    key = "c" * 64
    ident = {"doc_id": "doc-a", "key": key, "text_sha256": "fixture", "language": "en", "topic": None}
    src = tmp_path / "local" / "documents" / key
    src.mkdir(parents=True)
    (src / "edges.parquet").write_bytes(b"fixture")
    report = {"identity": ident, "checksums": {"edges.parquet": sha256(src / "edges.parquet")}}
    (src / "manifest.json").write_text(json.dumps(report))
    (src / "COMPLETED").write_text("complete\n")
    remote = tmp_path / "drive"
    (remote / "documents").mkdir(parents=True)
    assert checkpoint_document(tmp_path / "local", remote, report)
    assert not checkpoint_document(tmp_path / "local", remote, report)
    output = tmp_path / "session2"
    assert restore_documents(output, remote) == 1
    assert (output / "documents" / key / "edges.parquet").read_bytes() == b"fixture"
    # A corrupted durable checkpoint is not silently trusted or overwritten.
    (remote / "documents" / key / "edges.parquet").write_bytes(b"damage")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        restore_documents(tmp_path / "session3", remote)
