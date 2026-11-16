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


def validate_fresh_experiment_folder(drive_root: Path):
    """Resume a failed setup without run.json only when no graph was published.

    progress.json is written by our own exception handler, even before the
    initial run.json is created. It is harmless diagnostic state, not a
    checkpoint. Published documents without their run contract must instead
    remain untouched: we cannot prove they came from this code and database.
    """
    allowed = {"experiment.json", "progress.json", "documents"}
    unexpected = sorted(p.name for p in drive_root.iterdir() if p.name not in allowed)
    if unexpected:
        raise ValueError("Drive output has unexpected files without run.json: "
                         + repr(unexpected) + ". Existing data was not modified.")
    docs = drive_root / "documents"
    if docs.exists():
        for child in docs.iterdir():
            if child.is_dir() and child.name.startswith(".partial-"):
                continue  # incomplete staging from killed FUSE copy, never trusted
            raise ValueError(
                "Drive has document data but no run.json; cannot validate "
                "the original run contract. Preserve the folder and choose "
                "a new experiment label or recover a matching run.json: "
                + str(child))


_STAGES_RU = {
    'initializing': 'запуск',
    'mapping_chunks': 'разбор и привязка частей текста',
    'tokenizing': 'разбор и лемматизация',
    'mapping_concepts': 'поиск понятий ConceptNet',
    'querying_relations': 'поиск отношений ConceptNet',
    'preparing_tables': 'подготовка DuckDB',
    'counting_edges': 'подсчёт рёбер',
    'writing_occurrences': 'запись слов в Parquet',
    'writing_assertions': 'запись отношений',
    'writing_edges': 'запись рёбер в Parquet',
    'verifying_edges': 'проверка рёбер',
    'writing_concept_edges': 'агрегация понятий',
    'finalizing': 'проверка и сохранение',
    'ready_for_checkpoint': 'копирование в Drive',
    'already_completed': 'восстановлен',
    'failed': 'ошибка',
}


def corpus_progress_snapshot(total, done_ids, active_documents,
                             *, state='running'):
    """Exact document count; *estimated* percent by stage for active documents."""
    completed = len(done_ids)
    if total < 0 or completed > total:
        raise ValueError('Invalid corpus document progress counters')
    active = []
    for doc_id, status in sorted(active_documents.items()):
        if doc_id in done_ids:
            continue
        active.append({
            'doc_id': doc_id,
            'percent': int(status['percent']),
            'stage': status['stage'],
            'stage_label': _STAGES_RU.get(status['stage'], status['stage']),
            'current': status.get('current'),
            'total': status.get('total'),
            'percent_kind': 'stage_estimate',
        })
    return {
        'state': state,
        'completed': completed, 'total': total,
        'completed_percent': round(100 * completed / total, 1) if total else 100.0,
        'active_documents': active,
        'percent_explanation': ('Active document percent is a weighted stage '
                                'estimate, not elapsed time or an exact SQL '
                                'execution percentage; current/total are exact '
                                'for tokenization and word mapping.'),
        'updated_at': utcnow(),
    }


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
            spacy_model: str | None = None,
            database_version: str = 'user-supplied',
            chunked_graph: bool = False,
            chunk_chars: int = 100_000,
            edge_partition_rows: int = 1_000_000,
            max_total_edges_per_document: int | None = 150_000_000):
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
        "conceptnet_database_version": database_version,
        "workers_are_runtime_only": True,
        "document_language": "en",
        "spacy_model": spacy_model,
        "max_edges_per_document": max_edges_per_document,
        "chunked_graph": chunked_graph, "chunk_chars": chunk_chars,
        "edge_partition_rows": edge_partition_rows,
        "max_total_edges_per_document": max_total_edges_per_document,
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
        validate_fresh_experiment_folder(drive_root)
        resume = False

    options = Options(database=str(database), output=str(local_root),
                      workers=workers, threads_per_worker=threads_per_worker,
                      memory_limit=memory_limit,
                      max_edges_per_document=max_edges_per_document,
                      spacy_model=spacy_model,
                      database_version=database_version,
                      chunked_graph=chunked_graph,
                      chunk_chars=chunk_chars,
                      edge_partition_rows=edge_partition_rows,
                      max_total_edges_per_document=max_total_edges_per_document)
    start = time.monotonic()
    total_documents = len(documents)
    # Restored documents count as completed BEFORE the first new computation.
    from .text_corpus import _identity
    done_ids = {
        doc.doc_id for doc in documents
        if (local_root / 'documents' / _identity(doc)['key'] / 'COMPLETED').is_file()
    }
    active_documents = {}
    published = 0
    last_persisted = 0.0

    def persist_progress(*, force=False):
        nonlocal last_persisted
        now = time.monotonic()
        if not force and now - last_persisted < 10.0:
            return
        snapshot = corpus_progress_snapshot(
            total_documents, done_ids, active_documents)
        snapshot['published_new_this_session'] = published
        snapshot['elapsed_seconds'] = round(now - start, 1)
        save_json(drive_root / 'progress.json', snapshot)
        last_persisted = now
        segments = []
        for status in snapshot['active_documents']:
            item = (f"{status['doc_id']}: {status['percent']}% "
                    f"({status['stage_label']}")
            if status['current'] is not None and status['total']:
                item += f", {status['current']}/{status['total']}"
            segments.append(item + ')')
        activity = ' | '.join(segments) if segments else 'нет активных документов'
        print(f"[progress] Документы: {snapshot['completed']}/{total_documents} "
              f"({snapshot['completed_percent']:.1f}%) | {activity}",
              flush=True)

    def on_progress(status):
        # status=None is a heartbeat sent while DuckDB is running a long
        # single SQL statement with no measurable intra-query percentage.
        if status is not None and status['doc_id'] not in done_ids:
            active_documents[status['doc_id']] = status
        persist_progress()

    def on_done(report):
        nonlocal published
        doc_id = report['identity']['doc_id']
        was_done = doc_id in done_ids
        if checkpoint_document(local_root, drive_root, report):
            published += 1
        done_ids.add(doc_id)
        active_documents.pop(doc_id, None)
        persist_progress(force=not was_done)
        if not was_done:
            print(f"[checkpoint] {len(done_ids)}/{total_documents}: "
                  f"{doc_id} — 100% (опубликован), "
                  f"edges={report['edge_count']}", flush=True)

    # Copy immutable run contract as soon as the first worker has created it.
    # We initialize via the package's build_corpus, and callback sees run.json.
    def callback(report):
        remote_run = drive_root / "run.json"
        if not remote_run.exists():
            shutil.copy2(local_root / "run.json", remote_run)
        on_done(report)

    try:
        persist_progress(force=True)
        manifest = build_corpus(documents, options, resume=resume,
                                on_document_completed=callback,
                                on_document_progress=on_progress)
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
        finished = corpus_progress_snapshot(total_documents, done_ids, {},
                                            state='complete')
        finished.update({
            'elapsed_seconds': round(time.monotonic() - start, 2),
            'word_count': manifest['word_count'], 'edge_count': manifest['edge_count'],
        })
        save_json(drive_root / 'progress.json', finished)
        print("[complete]", drive_root)
        return manifest
    except BaseException as exc:
        failed = corpus_progress_snapshot(total_documents, done_ids,
                                          active_documents, state='interrupted')
        failed.update({'completed_this_session': published,
                       'error_type': type(exc).__name__,
                       'error': str(exc)[:1000]})
        save_json(drive_root / 'progress.json', failed)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--database", help="Existing ConceptNet SQLite path")
    p.add_argument("--hf-db", action="store_true",
                   help="Download pinned cstr/conceptnet-normalized-multi to Colab local SSD")
    p.add_argument("--hf-revision", default=None,
                   help="Optional pinned Hugging Face dataset revision")
    p.add_argument("--database-version", default="user-supplied")
    p.add_argument("--drive-parent", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--workspace", default="/content/semmap-text-corpus")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--memory-limit", default="1GB")
    p.add_argument("--max-edges-per-document", type=int, default=20_000_000,
                   help="Hard total cap; 0 disables it when chunked graph is enabled")
    p.add_argument("--chunked-graph", action="store_true",
                   help="Process each text sequentially by chunks; store one full graph")
    p.add_argument("--chunk-chars", type=int, default=100_000,
                   help="Text chunk size, retaining absolute original offsets")
    p.add_argument("--edge-partition-rows", type=int, default=1_000_000,
                   help="Maximum number of edges in any output Parquet part")
    p.add_argument("--max-total-edges-per-document", type=int, default=150_000_000,
                   help="Independent whole-graph disk safety cap, never truncates")
    p.add_argument("--max-docs", type=int)
    p.add_argument("--spacy-model")
    a = p.parse_args(argv)
    if a.max_edges_per_document == 0 and not a.chunked_graph:
        p.error("--max-edges-per-document 0 requires --chunked-graph")
    if a.max_edges_per_document < 0:
        p.error("--max-edges-per-document cannot be negative")
    max_document_edges = (None if a.max_edges_per_document == 0
                          else a.max_edges_per_document)
    if a.hf_db and a.database:
        p.error("Choose either --hf-db or --database, not both")
    if not a.hf_db and not a.database:
        p.error("Supply --hf-db for automatic download or --database")
    database, dbversion = a.database, a.database_version
    if a.hf_db:
        from .text_hf import REVISION, download_database
        revision = a.hf_revision or REVISION
        info = download_database(cache_dir=Path(a.workspace) / "hf-cache",
                                 revision=revision)
        database = info["path"]
        dbversion = "cstr/conceptnet-normalized-multi@" + revision
        print("[hf] Local SQLite:", database, flush=True)
    result = execute(a.archive, database, a.drive_parent, a.label,
                     workspace=a.workspace, workers=a.workers,
                     threads_per_worker=a.threads_per_worker,
                     memory_limit=a.memory_limit,
                     max_edges_per_document=max_document_edges,
                     max_docs=a.max_docs, spacy_model=a.spacy_model,
                     database_version=dbversion,
                     chunked_graph=a.chunked_graph, chunk_chars=a.chunk_chars,
                     edge_partition_rows=a.edge_partition_rows,
                     max_total_edges_per_document=a.max_total_edges_per_document)
    print(json.dumps({"documents": len(result["documents"]),
                      "edges": result["edge_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
