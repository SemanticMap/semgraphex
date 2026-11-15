"""Colab runner tests: no Google account, cloud mounts or production database."""
from __future__ import annotations
import json
from pathlib import Path
import zipfile

import pytest

from semmap_haken.text_colab import (
    checkpoint_document, prepare_zip, restore_documents, safe_label,
    validate_fresh_experiment_folder
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


def test_notebook_installs_spacy_before_corpus_run():
    """Notebook has a working optional spaCy setup, not a manual missing step."""
    import ast
    notebook_path = (Path(__file__).resolve().parents[1] / "notebooks" /
                     "10_text_corpus_conceptnet_local_drive.ipynb")
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for cell in code:
        ast.parse(cell)
    settings = next(i for i, cell in enumerate(code) if "SPACY_MODEL = " in cell)
    install = next(i for i, cell in enumerate(code) if '"[text]"' in cell or 'str(repo)+"[text]"' in cell)
    model_setup = next(i for i, cell in enumerate(code) if "spacy.load(SPACY_MODEL)" in cell)
    run = next(i for i, cell in enumerate(code) if '"semmap_haken.text_colab"' in cell)
    assert settings < install < model_setup < run
    assert 'SPACY_MODEL = "en_core_web_sm"' in code[settings]
    assert '"spacy", "download", SPACY_MODEL' in code[model_setup]
    assert '"--spacy-model", SPACY_MODEL' in code[run]


def test_resume_accepts_own_progress_file_before_run_contract(tmp_path):
    """Failure before first checkpoint must not poison the next NEW/RESUME."""
    destination = tmp_path / "text-corpus-example"
    (destination / "documents").mkdir(parents=True)
    (destination / "experiment.json").write_text('{"schema": 1}', encoding="utf-8")
    (destination / "progress.json").write_text(
        '{"state": "interrupted"}', encoding="utf-8"
    )
    validate_fresh_experiment_folder(destination)
    # Killed FUSE staging is ignored, not treated as a valid checkpoint.
    (destination / "documents" / ".partial-stopped").mkdir()
    validate_fresh_experiment_folder(destination)


def test_resume_rejects_orphan_documents_without_run_contract(tmp_path):
    destination = tmp_path / "text-corpus-example"
    (destination / "documents" / "a-completed-file").mkdir(parents=True)
    (destination / "experiment.json").write_text("{}", encoding="utf-8")
    (destination / "progress.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no run.json"):
        validate_fresh_experiment_folder(destination)
    assert (destination / "documents" / "a-completed-file").exists()


def test_resume_rejects_unknown_metadata_without_run_contract(tmp_path):
    destination = tmp_path / "text-corpus-example"
    destination.mkdir()
    (destination / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        validate_fresh_experiment_folder(destination)
    assert (destination / "manifest.json").exists()


def test_clean_notebook_has_subprocess_import_and_no_monkeypatches():
    """All cells parse; the new notebook is idempotent across kernel reruns."""
    import ast
    path = (Path(__file__).resolve().parents[1] / "notebooks" /
            "11_text_corpus_hf55_colab_clean.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"]
            if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    all_code = "\n".join(code)
    assert "SPACY_MODEL = \"en_core_web_sm\"" in all_code
    assert "semmap_haken.text_colab" in all_code
    assert '"--hf-db"' in all_code
    assert 'subprocess.run(cmd, check=True, cwd=repo)' in all_code
    assert 'from semmap_haken.text_hf import download_database' in all_code
    for suspect in (
        "patched_inspect_database", "patched_init", "patched_iterdir",
        "custom_execute", "_is_patched_for_schema", "_is_patched_for_resume",
        "Path.iterdir =", "sqlite3.connect =", "sys.path.insert",
    ):
        assert suspect not in all_code
    # Install inside the same Python executable invoked for processing.
    assert 'sys.executable, "-m", "pip", "install"' in all_code
    assert 'sys.executable, "-u", "-m", "semmap_haken.text_colab"' in all_code
