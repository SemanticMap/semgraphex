"""Full-graph semantics for sequential text chunks and bounded Parquet edges."""
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from semmap_haken.text_corpus import Document, Options, build_corpus, open_corpus
from semmap_haken.text_chunk_graph import edge_partitions
from semmap_haken.text_export import export_prepared
from semmap_haken.graph_build import load_prepared_graph


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "cn.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE node_norm (
                node_pk INTEGER PRIMARY KEY, node_url TEXT UNIQUE NOT NULL
            );
            CREATE TABLE rel_norm (rel_pk INTEGER PRIMARY KEY, rel_url TEXT UNIQUE);
            CREATE TABLE edge_norm (
                start_fk INTEGER, end_fk INTEGER, rel_fk INTEGER, weight REAL
            );
        """)
        db.executemany("INSERT INTO node_norm VALUES (?,?)", [
            (1, "http://conceptnet.io/c/en/cat"),
            (2, "http://conceptnet.io/c/en/animal"),
            (3, "http://conceptnet.io/c/en/dog"),
        ])
        db.executemany("INSERT INTO rel_norm VALUES (?,?)", [
            (1, "http://conceptnet.io/r/IsA"),
            (2, "http://conceptnet.io/r/RelatedTo"),
        ])
        db.executemany("INSERT INTO edge_norm VALUES (?,?,?,?)", [
            (1, 2, 1, 2.0), (2, 1, 2, 1.0), (1, 1, 2, 1.4),
        ])
    return path


def options(db, path, chunked):
    return Options(database=str(db), output=str(path),
                   chunked_graph=chunked, chunk_chars=20,
                   edge_partition_rows=2,
                   max_edges_per_document=None if chunked else 50_000,
                   max_total_edges_per_document=500_000)


def all_rows(corpus):
    db = open_corpus(corpus)
    try:
        return {
            name: db.execute(f"SELECT * FROM {name} ORDER BY ALL").fetchall()
            for name in ("occurrences", "edges", "concept_edges")
        }
    finally:
        db.close()


def test_split_document_one_full_graph_and_all_cross_part_edges(database, tmp_path):
    text = "cat dog. animal dog cat. animal cat. dog animal"
    original = options(database, tmp_path / "original", False)
    split = options(database, tmp_path / "split", True)
    build_corpus([Document("x", text)], original)
    split_report = build_corpus([Document("x", text)], split)
    original_rows = all_rows(original.output)
    split_rows = all_rows(split.output)
    assert original_rows["occurrences"] == split_rows["occurrences"]
    assert original_rows["edges"] == split_rows["edges"]
    assert len(original_rows["concept_edges"]) == len(split_rows["concept_edges"])
    for a, b in zip(original_rows["concept_edges"], split_rows["concept_edges"]):
        for left, right in zip(a, b):
            if isinstance(left, float):
                assert left == pytest.approx(right)
            else:
                assert left == right
    report = split_report["documents"][0]
    assert report["chunk_count"] >= 2
    assert report["edge_part_count"] > 1
    assert report["edge_count"] > split.edge_partition_rows
    graph_dir = next((tmp_path / "split" / "documents").iterdir())
    parts = sorted(graph_dir.glob("edges_part_*.parquet"))
    assert len(parts) == report["edge_part_count"]
    import duckdb
    for part in parts:
        with duckdb.connect(":memory:") as con:
            assert con.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(part)]
            ).fetchone()[0] <= split.edge_partition_rows
    # First 'cat' and LAST 'animal' are in different chunks, but connected.
    conn = open_corpus(split.output)
    try:
        first = conn.execute(
            "SELECT occurrence_id FROM occurrences WHERE surface='cat'"
            " ORDER BY occurrence_id LIMIT 1"
        ).fetchone()[0]
        last = conn.execute(
            "SELECT occurrence_id FROM occurrences WHERE surface='animal'"
            " ORDER BY occurrence_id DESC LIMIT 1"
        ).fetchone()[0]
        delta = conn.execute(
            "SELECT distance_chars FROM edges WHERE source_occurrence_id=? "
            "AND target_occurrence_id=? AND relation_uri='/r/IsA'",
            [first, last],
        ).fetchone()
        assert delta is not None
        assert delta[0] == text.rfind("animal") - text.find("cat")
    finally:
        conn.close()


def test_oversized_single_source_splits_by_target(database, tmp_path):
    text = "cat animal animal animal animal animal"
    opt = options(database, tmp_path / "huge-src", True)
    report = build_corpus([Document("s", text)], opt)
    assert report["edge_count"] >= 5
    assert report["documents"][0]["edge_part_count"] >= 3
    c = open_corpus(opt.output)
    try:
        edges = c.execute("SELECT count(*) FROM edges").fetchone()[0]
        assert edges == report["edge_count"]
    finally:
        c.close()


def test_graph_cap_notifies_and_continues_with_next_document(database, tmp_path, capsys):
    from dataclasses import replace
    opt = replace(
        options(database, tmp_path / "guard", True),
        max_total_edges_per_document=3,
        edge_partition_rows=2,
    )
    docs = [
        Document("large", "cat animal cat animal cat animal"),
        Document("small", "dog dog"),
    ]
    summary = build_corpus(docs, opt)
    captured = capsys.readouterr().out
    assert "[WARNING]" in captured
    assert "large" in captured
    assert "пропущено" in captured
    assert (Path(opt.output) / "COMPLETED").is_file()
    assert summary["truncated_document_count"] == 1
    assert summary["edges_omitted"] > 0
    assert summary["edge_count"] == 3
    large, small = summary["documents"]
    assert large["identity"]["doc_id"] == "large"
    assert large["edge_count"] == 3
    assert large["graph_complete"] is False
    assert large["edge_count_total_expected"] == (
        large["edge_count"] + large["edges_omitted"]
    )
    assert large["edge_selection_policy"].startswith("first_N")
    assert small["identity"]["doc_id"] == "small"
    assert small["edge_count"] == 0
    assert small["graph_complete"] is True
    assert all_rows(opt.output)["edges"]
    assert build_corpus(docs, opt, resume=True) == summary


def test_capped_edges_are_exact_deterministic_prefix(database, tmp_path):
    from dataclasses import replace
    base = options(database, tmp_path / "full", True)
    cap = replace(base, output=str(tmp_path / "capped"),
                  max_total_edges_per_document=3)
    doc = Document("document", "cat animal cat animal cat")
    build_corpus([doc], base)
    limited_report = build_corpus([doc], cap)
    full_edges = all_rows(base.output)["edges"]
    kept_edges = all_rows(cap.output)["edges"]
    assert len(kept_edges) == 3
    assert kept_edges == full_edges[:3]
    assert len(full_edges) > len(kept_edges)
    report = limited_report["documents"][0]
    assert report["edges_omitted"] == len(full_edges) - len(kept_edges)
    assert report["edge_count_total_expected"] == len(full_edges)
    assert report["edge_cap"] == 3
    assert not report["graph_complete"]
    folder = next((Path(cap.output) / "documents").iterdir())
    assert sum(1 for _ in folder.glob("edges_part_*.parquet")) >= 2


def test_stricter_of_two_caps_used_without_abort(database, tmp_path):
    from dataclasses import replace
    opt = replace(options(database, tmp_path / "twocaps", True),
                  max_edges_per_document=2, max_total_edges_per_document=20)
    summary = build_corpus([Document("x", "cat animal cat animal")], opt)
    assert summary["edge_count"] == 2
    assert summary["documents"][0]["edge_cap"] == 2
    assert summary["truncated_document_count"] == 1


def test_empty_and_unmapped_chunked_documents_and_resume(database, tmp_path):
    opt = options(database, tmp_path / "empty", True)
    docs = [Document("empty", ""), Document("unknown", "qwerty zzz")]
    first = build_corpus(docs, opt)
    second = build_corpus(docs, opt, resume=True)
    assert first == second
    rows = all_rows(opt.output)
    assert rows["edges"] == []
    assert len(rows["occurrences"]) == 2


def test_legacy_and_chunked_docs_in_single_corpus_query(database, tmp_path):
    # Check direct union of both output variants by copying document folders into
    # a shared corpus (the production reader should accept a mixed layout).
    import json
    import shutil
    from semmap_haken.text_corpus import sha256
    legacy = options(database, tmp_path / "legacy", False)
    chunks = options(database, tmp_path / "chunks", True)
    build_corpus([Document("one", "cat animal")], legacy)
    build_corpus([Document("two", "cat animal cat animal")], chunks)
    source = tmp_path / "mixed"
    source.mkdir()
    (source / "documents").mkdir()
    (source / "run.json").write_bytes((Path(legacy.output) / "run.json").read_bytes())
    originals = []
    for folder in (Path(legacy.output), Path(chunks.output)):
        manifest = json.loads((folder / "manifest.json").read_text())
        report = manifest["documents"][0]
        key = report["identity"]["key"]
        shutil.copytree(folder / "documents" / key, source / "documents" / key)
        originals.append(report)
    # Only test the reader's published structure; mixed-layout semantic contracts
    # are not a legal RESUME contract for a real experiment.
    manifest = {"schema_version": 1, "documents": originals,
                "word_count": sum(i["word_count"] for i in originals),
                "edge_count": sum(i["edge_count"] for i in originals),
                "run_sha256": sha256(source / "run.json")}
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "COMPLETED").write_text("complete\n")
    con = open_corpus(source)
    try:
        assert con.execute("SELECT count(DISTINCT doc_id) FROM occurrences").fetchone()[0] == 2
        assert con.execute("SELECT count(*) FROM edges").fetchone()[0] == manifest["edge_count"]
    finally:
        con.close()


def test_export_retains_capped_graph_warning(database, tmp_path):
    from dataclasses import replace
    opt = replace(options(database, tmp_path / "capped-export", True),
                  max_total_edges_per_document=2)
    corpus = build_corpus([Document("partial", "cat animal cat animal")], opt)
    export_dir = tmp_path / "exported-partial"
    report = export_prepared(opt.output, "partial", export_dir)
    assert not report["source_graph_complete"]
    assert report["source_graph_edges_omitted"] == corpus["edges_omitted"]
    assert report["source_graph_edge_count_total_expected"] > 2
    prepared = load_prepared_graph(export_dir)
    assert not prepared.report["source_graph_complete"]


def test_bounded_prepared_export_from_partitioned_graph(database, tmp_path):
    opt = options(database, tmp_path / "parts", True)
    build_corpus([Document("doc", "cat animal cat animal")], opt)
    result = tmp_path / "prepared"
    export_prepared(opt.output, "doc", result, directed=False)
    graph = load_prepared_graph(result)
    assert graph.adjacency.shape == (4, 4)
    assert len(graph.node_ids) == 4


def test_multifile_document_drive_checkpoint_and_recovery(database, tmp_path):
    """Every edge part is checksummed/published/restored as ONE document."""
    from semmap_haken.text_colab import checkpoint_document, restore_documents
    opt = options(database, tmp_path / "stage", True)
    run = build_corpus([Document("id", "cat animal cat animal cat")], opt)
    report = run["documents"][0]
    assert report["edge_part_count"] > 1
    durable = tmp_path / "durable"
    (durable / "documents").mkdir(parents=True)
    assert checkpoint_document(Path(opt.output), durable, report)
    assert not checkpoint_document(Path(opt.output), durable, report)
    restored = tmp_path / "restored"
    assert restore_documents(restored, durable) == 1
    key = report["identity"]["key"]
    parts = sorted((restored / "documents" / key).glob("edges_part_*.parquet"))
    assert len(parts) == report["edge_part_count"]
    from semmap_haken.text_corpus import _verify_document
    assert _verify_document(restored / "documents" / key, report["identity"]) == report


def test_cross_chunk_unicode_original_character_coordinates(database, tmp_path):
    opt = options(database, tmp_path / "unicode", True)
    text = "😀cat.\\r\\n" + ("dog. " * 12) + "animal"
    build_corpus([Document("unicode", text)], opt)
    con = open_corpus(opt.output)
    try:
        value = con.execute("""
            SELECT e.distance_chars, e.signed_offset
            FROM edges e
            JOIN occurrences s ON s.doc_id=e.doc_id
                AND s.occurrence_id=e.source_occurrence_id
            JOIN occurrences t ON t.doc_id=e.doc_id
                AND t.occurrence_id=e.target_occurrence_id
            WHERE s.surface='cat' AND t.surface='animal'
              AND e.relation_uri='/r/IsA'
        """).fetchone()
        assert value == (text.index("animal") - text.index("cat"),) * 2
    finally:
        con.close()
