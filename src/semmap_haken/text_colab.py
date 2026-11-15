"""Colab/Drive runner with durable per-document checkpoints for text_corpus.

Run from a mounted Drive folder; calculate on /content SSD, copy each verified
document to Drive after its worker finishes. Never run SQLite queries on Drive FUSE.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import zipfile

from .text_corpus import Options, build_corpus, read_documents, sha256, _verify_document


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", label):
        raise ValueError("Experiment label: 1–64 ASCII letters/digits/_/-, starts with a letter or digit")
    return label


def save_json(path: Path, data):
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def prepare_zip(archive: Path, destination: Path, *, max_docs: int | None,
                max_doc_bytes: int, max_total_bytes: int):
    """Select deterministic TXT files. No zip traversal/symlinks, no zip bombs."""
    destination.mkdir(parents=True, exist_ok=True)
    selected = []
    with zipfile.ZipFile(archive) as z:
        for item in sorted(z.infolist(), key=lambda x: x.filename):
            if item.is_dir() or not item.filename.lower().endswith(".txt"):
                continue
            name = item.filename.replace("\\", "/")
            parts = name.split("/")
            if (name.startswith("/") or any(p in {"", ".", ".."} for p in parts)
                    or ":" in parts[0] or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unsafe ZIP member: {name!r}")
            if item.file_size > max_doc_bytes:
                raise ValueError(f"Document too large: {name}: {item.file_size} bytes")
            selected.append(item)
        if not selected:
            raise ValueError("Archive has no .txt members; inspect its structure")
        if max_docs is not None:
            if max_docs < 1:
                raise ValueError("max_docs must be positive or None")
            selected = selected[:max_docs]
        total = sum(i.file_size for i in selected)
        if total > max_total_bytes:
            raise ValueError(f"Selected archive data {total} bytes exceeds max_total_bytes={max_total_bytes}")
        names = [i.filename for i in selected]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate TXT member names in ZIP")
        for item in selected:
            target = destination.joinpath(*item.filename.replace("\\", "/").split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # Staging is created afresh per Colab session.
                raise FileExistsError(target)
            with z.open(item) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            if target.stat().st_size != item.file_size:
                raise ValueError(f"ZIP member length differs: {item.filename}")
    return {"selected_count": len(selected), "selected_bytes": total,
            "selected_names_sha256": hashlib.sha256(
                "\n".join(names).encode("utf-8")).hexdigest()}


def preflight_english(documents):
    """Prevent silent invalid English mapping of a Cyrillic-dominant corpus."""
    checked_letters = 0
    checked_cyrillic = 0
    mixed = []
    for document in documents:
        sample = document.text[:20000]
        letters = sum(c.isalpha() for c in sample)
        cyrillic = sum("\u0400" <= c <= "\u052f" for c in sample)
        checked_letters += letters
        checked_cyrillic += cyrillic
        if letters >= 10 and cyrillic / letters > .05:
            mixed.append(document.doc_id)
    fraction = checked_cyrillic / max(checked_letters, 1)
    print(f"[preflight] Cyrillic fraction across sampled documents: {fraction:.2%}")
    if mixed:
        raise ValueError("Detected Cyrillic text in documents (first 8): "
                         + repr(mixed[:8]) + ". ConceptNet Local is English-only; "
                         "Russian documents require a different adapter.")
    return fraction


def validate_remote(dst: Path, identity):
    return _verify_document(dst, identity)


def checkpoint_document(local_root: Path, drive_root: Path, report: dict):
    identity = report["identity"]
    key = identity["key"]
    src = local_root / "documents" / key
    dst = drive_root / "documents" / key
    _verify_document(src, identity)
    if dst.exists():
        remote = validate_remote(dst, identity)
        if remote["checksums"] != report["checksums"]:
            raise ValueError("Remote checkpoint conflicts with local result")
        return False
    scratch = drive_root / "documents" / (".partial-" + key)
    if scratch.exists():
        shutil.rmtree(scratch)  # only incomplete owned staging, not published files
    scratch.mkdir(parents=True)
    try:
        for file in sorted(src.iterdir()):
            if file.is_file() and file.name != "COMPLETED":
                shutil.copy2(file, scratch / file.name)
        _verify_document(src, identity)  # source unchanged during copy
        # Verify payload on Drive before publishing the completed marker.
        for name, checksum in report["checksums"].items():
            if sha256(scratch / name) != checksum:
                raise ValueError(f"Drive copy checksum mismatch: {name}")
        shutil.copy2(src / "COMPLETED", scratch / "COMPLETED")
        validate_remote(scratch, identity)
        os.replace(scratch, dst)
        validate_remote(dst, identity)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    return True


def restore_documents(local_root: Path, drive_root: Path):
    remote = drive_root / "documents"
    if not remote.exists():
        return 0
    target = local_root / "documents"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for folder in sorted(remote.iterdir()):
        if not folder.is_dir() or folder.name.startswith(".partial-"):
            continue
        if not (folder / "COMPLETED").is_file():
            raise ValueError(f"Remote document checkpoint is incomplete: {folder}")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        _verify_document(folder, manifest["identity"])
        dst = target / folder.name
        if dst.exists():
            raise FileExistsError(dst)
        shutil.copytree(folder, dst)
        _verify_document(dst, manifest["identity"])
        count += 1
    return count


def execute(archive: str | Path, database: str | Path, drive_parent: str | Path,
            label: str, *, workspace: str | Path = "/content/semmap-text-corpus",
            workers: int = 4, threads_per_worker: int = 1,
            memory_limit: str = "1GB", max_edges_per_document: int = 20_000_000,
            max_docs: int | None = None, max_doc_bytes: int = 64_000_000,
            max_total_bytes: int = 4_000_000_000,
            spacy_model: str | None = None):
    label = safe_label(label)
    source = Path(archive).resolve(strict=True)
    database = Path(database).resolve(strict=True)
    if source.suffix.lower() != ".zip":
        raise ValueError("Expected a ZIP of UTF-8 TXT files")
    drive_parent = Path(drive_parent).resolve(strict=True)
    if not drive_parent.is_dir():
        raise NotADirectoryError(drive_parent)
    drive_root = drive_parent / ("text-corpus-" + label)
    drive_root.mkdir(exist_ok=True)
    (drive_root / "documents").mkdir(exist_ok=True)
    scratch = Path(workspace).resolve() / label
    if drive_root.is_relative_to(scratch) or scratch.is_relative_to(drive_root):
        raise ValueError("Scratch and Drive checkpoint folders must differ")
    if scratch.exists():
        shutil.rmtree(scratch)  # ephemeral /content scratch only
    scratch.mkdir(parents=True)
    local_archive = scratch / "source.zip"
    shutil.copyfile(source, local_archive)
    if sha256(source) != sha256(local_archive):
        raise ValueError("ZIP failed staging integrity check")
    archive_digest = sha256(local_archive)
    selection = prepare_zip(local_archive, scratch / "texts", max_docs=max_docs,
                            max_doc_bytes=max_doc_bytes, max_total_bytes=max_total_bytes)
    documents = read_documents(scratch / "texts", language="en")
    preflight_english(documents)
    run_descriptor = {
        "schema": 1, "experiment_label": label,
        "source_drive_file_id": "1I8dYdI4eLVmlhZjNewxl63EGgJKLmDqR",
        "source_archive_sha256": archive_digest,
        "selection": selection, "conceptnet_database_sha256": sha256(database),
        "workers_are_runtime_only": True,
        "document_language": "en",
        "spacy_model": spacy_model,
        "max_edges_per_document": max_edges_per_document,
        "data_model": "all direct ConceptNet-connected word occurrences, absolute start offsets"
    }
    descriptor_path = drive_root / "experiment.json"
    if descriptor_path.exists():
        previous = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if previous != run_descriptor:
            raise ValueError("Experiment label exists but input/database/config changed. "
                             "Choose a new experiment label.")
    else:
        save_json(descriptor_path, run_descriptor)

    local_root = scratch / "run"
    local_root.mkdir()
    if (drive_root / "run.json").exists():
        shutil.copy2(drive_root / "run.json", local_root / "run.json")
        print("[resume] Restoring verified completed document checkpoints")
        restored = restore_documents(local_root, drive_root)
        print(f"[resume] Restored {restored} documents")
        resume = True
    else:
        if any(x.name != "experiment.json" for x in drive_root.iterdir()
               if x.name != "documents"):
            raise ValueError("Drive output has unexpected files without run.json")
        resume = False

    options = Options(database=str(database), output=str(local_root),
                      workers=workers, threads_per_worker=threads_per_worker,
                      memory_limit=memory_limit,
                      max_edges_per_document=max_edges_per_document,
                      spacy_model=spacy_model)
    start = time.monotonic()
    published = 0
    completed = 0
    def on_done(report):
        nonlocal published, completed
        completed += 1
        if checkpoint_document(local_root, drive_root, report):
            published += 1
        save_json(drive_root / "progress.json", {
            "completed": completed, "total": len(documents),
            "published_new_this_session": published,
            "last_doc_id": report["identity"]["doc_id"],
            "updated_at": utcnow(), "state": "running"
        })
        print(f"[checkpoint] {completed}/{len(documents)}: "
              f"{report['identity']['doc_id']} edges={report['edge_count']}", flush=True)

    # Copy immutable run contract as soon as the first worker has created it.
    # We initialize via the package's build_corpus, and callback sees run.json.
    def callback(report):
        remote_run = drive_root / "run.json"
        if not remote_run.exists():
            shutil.copy2(local_root / "run.json", remote_run)
        on_done(report)

    try:
        manifest = build_corpus(documents, options, resume=resume,
                                on_document_completed=callback)
        shutil.copy2(local_root / "run.json", drive_root / "run.json")
        for filename in ("manifest.json",):
            shutil.copy2(local_root / filename, drive_root / filename)
        # Verify before publishing whole run on Drive.
        remote_manifest = json.loads((drive_root / "manifest.json").read_text(encoding="utf-8"))
        if len(remote_manifest["documents"]) != len(documents):
            raise RuntimeError("Drive manifest missing documents")
        for report in remote_manifest["documents"]:
            validate_remote(drive_root / "documents" / report["identity"]["key"],
                            report["identity"])
        shutil.copy2(local_root / "COMPLETED", drive_root / "COMPLETED")
        save_json(drive_root / "progress.json", {
            "state": "complete", "completed": len(documents),
            "total": len(documents), "updated_at": utcnow(),
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "word_count": manifest["word_count"], "edge_count": manifest["edge_count"]
        })
        print("[complete]", drive_root)
        return manifest
    except BaseException as exc:
        save_json(drive_root / "progress.json", {
            "state": "interrupted", "completed_this_session": completed,
            "updated_at": utcnow(), "error_type": type(exc).__name__,
            "error": str(exc)[:1000]})
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--drive-parent", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--workspace", default="/content/semmap-text-corpus")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--memory-limit", default="1GB")
    p.add_argument("--max-edges-per-document", type=int, default=20_000_000)
    p.add_argument("--max-docs", type=int)
    p.add_argument("--spacy-model")
    a = p.parse_args(argv)
    result = execute(a.archive, a.database, a.drive_parent, a.label,
                     workspace=a.workspace, workers=a.workers,
                     threads_per_worker=a.threads_per_worker,
                     memory_limit=a.memory_limit,
                     max_edges_per_document=a.max_edges_per_document,
                     max_docs=a.max_docs, spacy_model=a.spacy_model)
    print(json.dumps({"documents": len(result["documents"]),
                      "edges": result["edge_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
