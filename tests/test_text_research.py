"""P0 research evidence tests: caps must never corrupt full implicit analysis."""
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from semmap_haken.text_corpus import Document, Options, build_corpus
from semmap_haken.text_research import audit_corpus, open_research_views
from semmap_haken.text_research_adapter import load_typed_text_graph


@pytest.fixture
def database(tmp_path):
    source = tmp_path / "research-cn.db"
    with sqlite3.connect(source) as db:
        db.executescript("""
            CREATE TABLE node_norm (node_pk INTEGER PRIMARY KEY,
                node_url TEXT UNIQUE NOT NULL);
            CREATE TABLE rel_norm (rel_pk INTEGER PRIMARY KEY,
                rel_url TEXT UNIQUE NOT NULL);
            CREATE TABLE edge_norm (start_fk INTEGER, end_fk INTEGER,
                rel_fk INTEGER, weight REAL);
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
    return source


@pytest.fixture
def corpus(database, tmp_path):
    root = tmp_path / "research-corpus"
    options = Options(
        database=str(database), output=str(root), chunked_graph=True,
        chunk_chars=8, edge_partition_rows=2,
        max_edges_per_document=None, max_total_edges_per_document=3,
    )
    docs = [Document("main", "cat animal cat"),
            Document("other", "dog dog")]
    manifest = build_corpus(docs, options)
    assert manifest["edges_omitted"] == 3
    return root


def test_audit_identifies_capped_and_complete_and_preserves_provenance(corpus):
    report = audit_corpus(corpus)
    assert report["integrity"] == "sha256_verified"
    assert report["total_documents"] == 2
    assert report["capped_documents"] == 1
    assert report["full_documents"] == 1
    assert report["primary_h1_eligible_count"] == 1
    by_id = {d["doc_id"]: d for d in report["documents"]}
    assert by_id["main"]["stored_edges"] == 3
    assert by_id["main"]["expected_edges"] == 6
    assert by_id["main"]["omitted_edges"] == 3
    assert not by_id["main"]["primary_h1_eligible"]
    assert by_id["main"]["implicit_reconstruction_available"]
    assert by_id["other"]["primary_h1_eligible"]
    assert not audit_corpus(corpus, verify=False)["documents"][1]["primary_h1_eligible"]


def test_full_views_independent_of_capped_stored_edges(corpus):
    with open_research_views(corpus, "main") as view:
        db = view.db
        assert view.audit_row["state"] == "capped"
        assert view.full_edge_count() == 6
        assert db.execute("SELECT count(*) FROM research_expanded_edges").fetchone()[0] == 6
        assert db.execute("SELECT count(*) FROM research_assertions").fetchone()[0] == 3
        assert db.execute(
            "SELECT sum(edge_count_full) FROM research_concept_projection"
        ).fetchone()[0] == 6
        assert db.execute(
            "SELECT sum(edge_count_full) FROM research_concept_projection "
            "WHERE source_concept='/c/en/cat' AND "
            "target_concept='/c/en/cat'"
        ).fetchone()[0] == 2
        assert db.execute(
            "SELECT count(*) FROM " + view.distance_view(4)
        ).fetchone()[0] == 2
        assert db.execute(
            "SELECT count(*) FROM " + view.distance_view(4, shuffled=True)
        ).fetchone()[0] >= 0
        assert db.execute(
            "SELECT count(*) FROM research_shuffled_edges"
        ).fetchone()[0] == 6
        original_positions = db.execute(
            "SELECT char_start FROM research_occurrences ORDER BY char_start"
        ).fetchall()
        shuffled_positions = db.execute(
            "SELECT shuffled_start FROM research_shuffled_occurrences "
            "ORDER BY shuffled_start"
        ).fetchall()
        assert original_positions == shuffled_positions
        assert view.metadata()["implicit_full_edges"] == 6


def test_typed_graph_concept_and_occurrence_are_exact_and_groupable(corpus):
    concept = load_typed_text_graph(corpus, "main", projection="concept",
                                    max_vertices=10, max_edges=10)
    assert concept.graph.vertex_count == 2
    assert concept.graph.edge_count == 3
    assert concept.provenance_complete
    assert concept.stored_graph_state == "capped"
    assert set(concept.source_assertion_ids) == set(concept.graph.assertion_ids)
    assert concept.weight_semantics == "original_conceptnet_assertion_weight"

    occurrence = load_typed_text_graph(corpus, "main", projection="occurrence",
                                       max_vertices=10, max_edges=10)
    assert occurrence.graph.vertex_count == 3
    assert occurrence.graph.edge_count == 6  # NOT the capped three
    assert len(set(occurrence.graph.assertion_ids)) == 6
    assert len(occurrence.holdout_groups) == 6
    assert len(set(occurrence.holdout_groups)) == 3
    assert max(occurrence.distance_chars) == 11
    window = load_typed_text_graph(
        corpus, "main", projection="occurrence",
        max_distance_chars=4, max_vertices=10, max_edges=10,
    )
    assert window.graph.edge_count == 2
    assert all(d <= 4 for d in window.distance_chars)


def test_explicit_budgets_refuse_no_unannounced_prefix(corpus):
    with pytest.raises(ValueError, match="max_edges"):
        load_typed_text_graph(corpus, "main", projection="occurrence",
                              max_edges=3)
    with pytest.raises(ValueError, match="max_vertices"):
        load_typed_text_graph(corpus, "main", projection="concept",
                              max_vertices=1)
    with pytest.raises(ValueError, match="Distance"):
        open_research_views(corpus, "main").distance_view(-3)
    with pytest.raises(ValueError, match="Distance window"):
        load_typed_text_graph(corpus, "main", projection="concept",
                              max_distance_chars=20)


def test_corrupted_files_never_become_admissible(corpus):
    report = audit_corpus(corpus)
    row = next(x for x in report["documents"] if x["doc_id"] == "other")
    target = corpus / "documents" / row["key"] / "text.txt"
    target.write_text("tampered")
    with pytest.raises(ValueError, match="Checksum"):
        audit_corpus(corpus)
    with pytest.raises(ValueError, match="Checksum"):
        load_typed_text_graph(corpus, "other")


def test_legacy_missing_proof_is_unknown_not_full(corpus):
    report = audit_corpus(corpus)
    key = next(x["key"] for x in report["documents"] if x["doc_id"] == "other")
    doc_manifest = corpus / "documents" / key / "manifest.json"
    top_manifest = corpus / "manifest.json"
    for path in [doc_manifest, top_manifest]:
        record = json.loads(path.read_text())
        if path == doc_manifest:
            record.pop("graph_complete", None)
            record.pop("edge_count_total_expected", None)
            record.pop("edges_omitted", None)
        else:
            for entry in record["documents"]:
                if entry["identity"]["doc_id"] == "other":
                    for name in ("graph_complete", "edge_count_total_expected",
                                 "edges_omitted"):
                        entry.pop(name, None)
        path.write_text(json.dumps(record))
    revised = audit_corpus(corpus)
    assert revised["unknown_documents"] == 1
    assert revised["primary_h1_eligible_count"] == 0


def test_single_document_verification_does_not_hash_unrelated_150m_edge_files(corpus):
    docs = audit_corpus(corpus)["documents"]
    other = next(d for d in docs if d["doc_id"] == "other")
    (corpus / "documents" / other["key"] / "text.txt").write_text("altered")
    # Whole corpus must reject the tampering, while a separate verified
    # research view on 'main' can be opened without scanning other documents.
    with pytest.raises(ValueError, match="Checksum"):
        audit_corpus(corpus)
    targeted = audit_corpus(corpus, doc_id="main")
    assert targeted["total_documents"] == 1
    assert targeted["corpus_document_count"] == 2
    assert targeted["integrity"] == "selected_document_sha256_verified"
    with open_research_views(corpus, "main") as selected:
        assert selected.full_edge_count() == 6
