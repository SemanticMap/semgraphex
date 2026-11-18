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



def test_h1_grouped_split_never_separates_repeated_assertion_families(corpus):
    from semmap_haken.hypothesis.splits import split_graph
    data = load_typed_text_graph(corpus, "main", projection="occurrence",
                                 max_vertices=10, max_edges=10)
    assert len(data.holdout_groups) == data.graph.edge_count
    split = split_graph(data.graph, seed=37, group_keys=data.holdout_groups,
                        train_fraction=.5, validation_fraction=.0)
    families = dict(zip(data.graph.assertion_ids, data.holdout_groups))
    tr = {families[i] for i in split.train.assertion_ids}
    va = {families[i] for i in split.validation.assertion_ids}
    te = {families[i] for i in split.test.assertion_ids}
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert split.train.edge_count + split.validation.edge_count + split.test.edge_count == 6
    assert sum(split.group_counts.values()) == len(set(data.holdout_groups))
    # All duplicate cat→animal occurrences of the same assertion stay together.
    pairs = [(aid, grp) for aid, grp in zip(data.source_assertion_ids, data.holdout_groups)]
    from collections import defaultdict
    indices = defaultdict(set)
    for selected, which in ((split.train, "train"), (split.test, "test"),
                             (split.validation, "validation")):
        for item in selected.assertion_ids:
            indices[families[item]].add(which)
    assert all(len(v) == 1 for v in indices.values())


def test_text_position_null_preserves_unweighted_edges_and_changes_only_distances(corpus):
    a = load_typed_text_graph(corpus, "main", projection="occurrence",
                              max_vertices=10, max_edges=10)
    b = load_typed_text_graph(corpus, "main", projection="occurrence",
                              position_shuffle_seed=5,
                              max_vertices=10, max_edges=10)
    c = load_typed_text_graph(corpus, "main", projection="occurrence",
                              position_shuffle_seed=5,
                              max_vertices=10, max_edges=10)
    assert a.graph.assertion_ids == b.graph.assertion_ids
    assert a.source_assertion_ids == b.source_assertion_ids
    assert a.holdout_groups == b.holdout_groups
    assert (a.graph.source == b.graph.source).all()
    assert (a.graph.target == b.graph.target).all()
    assert (a.graph.relation == b.graph.relation).all()
    assert b.distance_chars == c.distance_chars
    assert b.position_shuffle_seed == 5
    with pytest.raises(ValueError, match="only occurrence"):
        load_typed_text_graph(corpus, "main", projection="concept",
                              position_shuffle_seed=10)


def test_h1_resume_fingerprints_split_families_and_source_provenance(corpus):
    from semmap_haken.hypothesis.runner import _identity_manifest, H1Config
    g = load_typed_text_graph(corpus, "main", projection="occurrence",
                              max_vertices=10, max_edges=10)
    config = H1Config(restarts=1, null_replicates=0, bootstrap_repetitions=20)
    base = _identity_manifest(g.graph, config, group_keys=g.holdout_groups,
                              source_contract={"doc_id": "main", "schema": "text-v1"})
    changed = _identity_manifest(
        g.graph, config, group_keys=g.holdout_groups,
        source_contract={"doc_id": "another", "schema": "text-v1"})
    different_groups = _identity_manifest(
        g.graph, config, group_keys=tuple(x + "_other" for x in g.holdout_groups),
        source_contract={"doc_id": "main", "schema": "text-v1"})
    assert base["source_contract_sha256"] != changed["source_contract_sha256"]
    assert base["split_group_sha256"] != different_groups["split_group_sha256"]
    with pytest.raises(ValueError, match="group"):
        _identity_manifest(g.graph, config, group_keys=("bad",))


def test_text_h1_runner_receives_complete_graph_and_grouped_split(corpus, tmp_path, monkeypatch):
    from semmap_haken import text_h1
    seen = {}

    def fake_runner(graph, output, *, config, resume, group_keys,
                    source_contract, fit_fn, flat_sampler):
        seen.update({"graph": graph, "groups": group_keys,
                     "source": source_contract, "resume": resume})
        return {"status": "inconclusive", "decision": {"status": "inconclusive"}}

    monkeypatch.setattr(text_h1, "run_h1", fake_runner)
    result = text_h1.run_text_h1(
        corpus, "main", tmp_path / "h1",
        projection="occurrence", max_vertices=10, max_edges=10,
    )
    assert result["status"] == "inconclusive"
    assert seen["graph"].edge_count == 6  # 3 were archived; 6 are implicit
    assert len(seen["groups"]) == 6
    assert seen["source"]["stored_graph_state"] == "capped"
    assert seen["source"]["implicit_full_graph_reconstructed"] is True
