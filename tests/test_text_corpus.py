"""Real ConceptNet Local + DuckDB, tiny synthetic SQLite; no production data.

Setup installs the text extra and NLTK stopwords. No network is used by tests.
"""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
pytest.importorskip('duckdb')
pytest.importorskip('conceptnet_local')

from semmap_haken.graph_build import load_prepared_graph
from semmap_haken.text_conceptnet import ConceptNetLocal, Word
from semmap_haken.text_corpus import Document, Options, build_corpus, open_corpus
from semmap_haken.text_export import export_prepared


@pytest.fixture
def database(tmp_path):
    path = tmp_path / 'cn.db'
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE concepts(id TEXT PRIMARY KEY, degree INT);
            CREATE TABLE relations(id TEXT PRIMARY KEY, start TEXT, end TEXT, rel TEXT, weight REAL);
            CREATE INDEX relation_start ON relations(start);
            CREATE INDEX relation_end ON relations(end);''')
        db.executemany('INSERT INTO concepts VALUES (?,?)',
                       [('/c/en/' + w, 2) for w in ('cat', 'animal', 'the', 'isolated', 'dog')])
        db.executemany('INSERT INTO relations VALUES (?,?,?,?,?)', [
            ('fixture:1', '/c/en/cat', '/c/en/animal', '/r/IsA', 2.0),
            ('fixture:2', '/c/en/cat', '/c/en/animal', '/r/RelatedTo', 1.5),
            ('fixture:3', '/c/en/animal', '/c/en/cat', '/r/RelatedTo', 1.0),
            ('fixture:4', '/c/en/cat', '/c/en/dog', '/r/Antonym', 0.5)])
    return path


def options(database, tmp_path, name='corpus', **kwargs):
    return Options(database=str(database), output=str(tmp_path / name), **kwargs)


def test_real_library_readonly_and_mapping(database):
    with ConceptNetLocal(database) as cn:
        assert cn.map_word(Word(0, 3, 'Cat')) == ('/c/en/cat', 'exact')
        assert cn.map_word(Word(0, 4, 'cats', 'cat')) == ('/c/en/cat', 'lemma')
        assert cn.map_word(Word(0, 3, 'the')) == ('/c/en/the', 'exact')
        assert cn.map_word(Word(0, 3, 'zzz')) == (None, 'unmapped')
        assert len(cn.induced_assertions({'/c/en/cat', '/c/en/animal'})) == 3
        with pytest.raises(sqlite3.OperationalError):
            cn.cursor.execute("DELETE FROM relations")


def test_all_pairs_distance_direction_isolates(database, tmp_path):
    opt = options(database, tmp_path)
    # cat starts 0 and 12; animal starts 4. 2 pairs * 3 assertions = 6 edges.
    text = 'Cat animal. cat zzz isolated the'
    report = build_corpus([Document('a', text, topic='animals'), Document('b', 'cat')], opt)
    assert report['edge_count'] == 6
    db = open_corpus(opt.output)
    rows = db.execute("SELECT source_occurrence_id,target_occurrence_id,distance_chars,"
                      "signed_offset FROM edges WHERE assertion_id='fixture:1' "
                      "ORDER BY source_occurrence_id").fetchall()
    assert rows == [(0, 1, 4, 4), (2, 1, 8, -8)]
    assert db.execute('SELECT count(*) FROM occurrences').fetchone()[0] == 7
    assert db.execute("SELECT count(*) FROM edges WHERE doc_id='b'").fetchone()[0] == 0
    assert db.execute("SELECT mapping_status FROM occurrences WHERE surface='isolated'").fetchone()[0] == 'exact'
    db.close()
    out = tmp_path / 'prepared'
    export_prepared(opt.output, 'a', out, directed=False)
    graph = load_prepared_graph(out)
    assert graph.adjacency.shape == (6, 6)
    assert (graph.adjacency - graph.adjacency.T).nnz == 0
    assert len(graph.selected_edges) == 6
    assert graph.adjacency[0, 1] == pytest.approx(3 / 5)
    from semmap_haken.wishart_metrics import relation_layers_from_prepared
    layers = relation_layers_from_prepared(graph, directed=False)
    total = sum(layers.values())
    assert (total - graph.adjacency).nnz == 0
    concept = tmp_path / 'concept'
    export_prepared(opt.output, 'a', concept, level='concept')
    assert load_prepared_graph(concept).node_ids == ('/c/en/animal', '/c/en/cat', '/c/en/isolated', '/c/en/the')


def test_empty_unmapped_and_empty_edges(database, tmp_path):
    opt = options(database, tmp_path)
    report = build_corpus([Document('empty', ''), Document('unknown', 'zzyyy'),
                           Document('single', 'cat')], opt)
    assert report['edge_count'] == 0
    db = open_corpus(opt.output)
    assert db.execute('SELECT count(*) FROM edges').fetchone()[0] == 0
    assert db.execute('SELECT count(*) FROM concept_edges').fetchone()[0] == 0
    db.close()
    export_prepared(opt.output, 'empty', tmp_path / 'empty-prepared')
    assert load_prepared_graph(tmp_path / 'empty-prepared').adjacency.shape == (0, 0)


def test_resume_integrity_and_changed_inputs(database, tmp_path):
    opt = options(database, tmp_path)
    docs = [Document('a', 'cat animal')]
    first = build_corpus(docs, opt)
    second = build_corpus(docs, opt, resume=True)
    assert first == second
    with pytest.raises(ValueError, match='contract mismatch'):
        build_corpus([Document('a', 'animal cat')], opt, resume=True)
    edge_file = next(Path(opt.output).glob('documents/*/edges.parquet'))
    edge_file.write_bytes(edge_file.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        build_corpus(docs, opt, resume=True)


def test_parallel_matches_serial(database, tmp_path):
    docs = [Document('a', 'cat animal cat'), Document('b', 'animal cat'), Document('c', '')]
    opt = options(database, tmp_path)
    parallel = replace(opt, output=str(tmp_path / 'parallel'), workers=2)
    build_corpus(docs, opt)
    build_corpus(docs, parallel)
    a, b = open_corpus(opt.output), open_corpus(parallel.output)
    try:
        for table in ('occurrences', 'edges', 'concept_edges'):
            assert a.execute(f'SELECT * FROM {table} ORDER BY ALL').fetchall() == b.execute(
                f'SELECT * FROM {table} ORDER BY ALL').fetchall()
    finally:
        a.close()
        b.close()


def test_edge_budget_fails_without_truncation(database, tmp_path):
    opt = options(database, tmp_path, max_edges_per_document=1)
    with pytest.raises(ValueError, match='nothing truncated'):
        build_corpus([Document('a', 'cat animal')], opt)
    assert not (Path(opt.output) / 'COMPLETED').exists()
    assert not list(Path(opt.output).glob('documents/*/COMPLETED'))


def test_filter_and_same_concept_distinct_occurrences(database, tmp_path):
    with sqlite3.connect(database) as db:
        db.execute('INSERT INTO relations VALUES (?,?,?,?,?)',
                   ('fixture:loop', '/c/en/cat', '/c/en/cat', '/r/RelatedTo', 1.0))
    opt = options(database, tmp_path, relations=('RelatedTo',))
    build_corpus([Document('a', 'cat cat')], opt)
    db = open_corpus(opt.output)
    assert db.execute('SELECT source_occurrence_id,target_occurrence_id,distance_chars '
                      'FROM edges ORDER BY ALL').fetchall() == [(0, 1, 4), (1, 0, 4)]
    db.close()


def test_cli_and_relocated_corpus(database, tmp_path):
    import shutil
    (tmp_path / 'a.txt').write_bytes(b'cat\tcat\r\nanimal')
    config = tmp_path / 'config.yaml'
    config.write_text('input: a.txt\ndatabase: cn.db\noutput: cli-result\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'semmap_haken.text_corpus', 'build',
                    '--config', str(config)], check=True, capture_output=True, text=True)
    relocated = tmp_path / 'moved'
    shutil.move(str(tmp_path / 'cli-result'), relocated)
    db = open_corpus(relocated)
    assert db.execute('SELECT count(*) FROM edges').fetchone()[0] == 6
    db.close()


def test_parallel_document_progress_events(database, tmp_path):
    """With spawn workers, the coordinator receives progress and heartbeats."""
    opt = options(database, tmp_path, name="with-progress", workers=2)
    seen = []
    completed = []
    report = build_corpus(
        [Document("a", "cat animal cat"), Document("b", "cat animal")],
        opt, on_document_progress=lambda status: seen.append(status),
        on_document_completed=lambda status: completed.append(status['identity']['doc_id']),
    )
    assert report["edge_count"] == 9
    assert sorted(completed) == ["a", "b"]
    stages = [s for s in seen if s is not None]
    assert {s["doc_id"] for s in stages} == {"a", "b"}
    assert all(0 <= s["percent"] <= 99 for s in stages)
    assert all(s["percent_kind"] == "stage_estimate" for s in stages)
    assert (Path(opt.output) / ".document-progress").is_dir()
    assert (Path(opt.output) / "COMPLETED").is_file()
