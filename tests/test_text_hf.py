"""Offline integration tests for HF ConceptNet 5.5 normalized SQLite adapter.

Tests use a tiny normalized SQLite fixture. No network, no production HF DB.
"""
from pathlib import Path
import sqlite3
import sys
import types

import pytest

pytest.importorskip("duckdb")

from semmap_haken.text_conceptnet import Word, NormalizedConceptNet, open_conceptnet
from semmap_haken.text_corpus import Document, Options, build_corpus, open_corpus, sha256
from semmap_haken.text_export import export_prepared
from semmap_haken.graph_build import load_prepared_graph
from semmap_haken.text_hf import REVISION, inspect_database, download_database


@pytest.fixture
def normalized_db(tmp_path):
    p = tmp_path / "conceptnet_normalized.db"
    with sqlite3.connect(p) as con:
        con.executescript("""
          CREATE TABLE node_norm (
            node_pk INTEGER PRIMARY KEY,
            node_url TEXT UNIQUE NOT NULL,
            language TEXT NOT NULL
          );
          CREATE TABLE rel_norm (
            rel_pk INTEGER PRIMARY KEY, rel_url TEXT UNIQUE NOT NULL
          );
          CREATE TABLE edge_norm (
            start_fk INTEGER NOT NULL, end_fk INTEGER NOT NULL,
            rel_fk INTEGER NOT NULL, weight REAL NOT NULL
          );
          CREATE INDEX src_ix ON edge_norm(start_fk);
          CREATE INDEX dst_ix ON edge_norm(end_fk);
        """)
        con.executemany("INSERT INTO node_norm VALUES (?,?,?)", [
          (1, "http://conceptnet.io/c/en/cat", "en"),
          (2, "http://conceptnet.io/c/en/animal", "en"),
          (3, "http://conceptnet.io/c/en/dog", "en"),
          (4, "http://conceptnet.io/c/en/the", "en"),
          (5, "http://conceptnet.io/c/ru/кот", "ru"),
        ])
        con.executemany("INSERT INTO rel_norm VALUES (?,?)", [
          (1, "http://conceptnet.io/r/IsA"),
          (2, "http://conceptnet.io/r/RelatedTo"),
        ])
        con.executemany("INSERT INTO edge_norm VALUES (?,?,?,?)", [
          (1,2,1,2.0), (1,2,2,1.0), (2,1,2,1.5), (1,3,2,0.3),
          # Russian target must never be included in an English-only induced graph:
          (1,5,2,0.4),
        ])
    return p


def test_schema_detection_and_exact_mapping(normalized_db):
    d = inspect_database(normalized_db)
    assert d["format"] == "hf-normalized-5.5"
    assert "src_ix" in d["edge_indices"]
    initial = sha256(normalized_db)
    with open_conceptnet(normalized_db) as cn:
        assert isinstance(cn, NormalizedConceptNet)
        assert cn.map_word(Word(0,3,"Cat")) == ("/c/en/cat", "exact")
        assert cn.map_word(Word(0,4,"cats", "cat")) == ("/c/en/cat","lemma")
        assert cn.map_word(Word(0,3,"zzz")) == (None,"unmapped")
        all_rows = cn.induced_assertions({"/c/en/cat","/c/en/animal"})
        assert [v[0] for v in all_rows] == [
            "hf55:edge_norm:1","hf55:edge_norm:2","hf55:edge_norm:3"
        ]
        assert all(v[1].startswith("/c/en") and v[2].startswith("/c/en") for v in all_rows)
        assert [row[3] for row in cn.induced_assertions(
            {"/c/en/cat","/c/en/animal"},("IsA",))] == ["/r/IsA"]
        with pytest.raises(sqlite3.OperationalError):
            cn.connection.execute("DELETE FROM node_norm")
    assert sha256(normalized_db) == initial


def test_hf_pipeline_all_pairs_and_export(normalized_db, tmp_path):
    opt = Options(database=str(normalized_db), output=str(tmp_path / "run"),
                  database_version="cstr/conceptnet-normalized-multi@fixture")
    result = build_corpus([Document("doc", "Cat animal cat unknown", topic="animals")], opt)
    assert result["edge_count"] == 6
    con = open_corpus(opt.output)
    try:
        assert con.execute("SELECT count(*) FROM edges").fetchone()[0] == 6
        rows = con.execute("""SELECT source_occurrence_id,target_occurrence_id,
                                   distance_chars,signed_offset
                            FROM edges WHERE assertion_id='hf55:edge_norm:1'
                            ORDER BY source_occurrence_id""").fetchall()
        assert rows == [(0,1,4,4), (2,1,7,-7)]
        assert con.execute("SELECT count(*) FROM occurrences").fetchone()[0] == 4
        assert con.execute("SELECT count(DISTINCT relation_uri) FROM edges").fetchone()[0] == 2
    finally:
        con.close()
    dst = tmp_path / "export"
    export_prepared(opt.output, "doc", dst, level="occurrence", directed=False)
    graph = load_prepared_graph(dst)
    assert len(graph.node_ids) == 4
    assert len(graph.selected_edges) == 6
    assert graph.adjacency.nnz > 0
    assert (graph.adjacency - graph.adjacency.T).nnz == 0
    assert build_corpus([Document("doc", "Cat animal cat unknown",topic="animals")],
                        opt, resume=True)["edge_count"] == 6


def test_hf_download_uses_dataset_commit_and_local_cache(normalized_db, tmp_path, monkeypatch):
    expected = normalized_db
    def fake(**kwargs):
        assert kwargs["repo_id"] == "cstr/conceptnet-normalized-multi"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["filename"] == "conceptnet_normalized.db"
        assert kwargs["revision"] == REVISION
        assert str(tmp_path / "cache") == kwargs["cache_dir"]
        return str(expected)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    data = download_database(cache_dir=tmp_path / "cache", min_bytes=0)
    assert data["path"] == str(normalized_db.resolve())
    assert data["dataset_revision"] == REVISION


def test_hf_schema_mismatch(tmp_path):
    p = tmp_path/"bad.db"
    with sqlite3.connect(p) as con:
        con.execute("CREATE TABLE concepts(id TEXT)")
    with pytest.raises(ValueError,match="node_norm"):
        inspect_database(p)
