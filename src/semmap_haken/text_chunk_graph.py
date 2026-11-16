"""Sequential text chunks, one complete document occurrence graph.

A chunk is an *execution/storage partition*, not a graph boundary. All source
chunks are joined to ALL target occurrences, preserving cross-chunk assertions.
Parquet edge parts are one logical relation, unioned by open_corpus().
"""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import time

from .text_conceptnet import _spacy_chunks, open_conceptnet, words


_OCCURRENCE_FIELDS = ("doc_id", "occurrence_id", "surface", "lemma",
                      "char_start", "char_end", "concept_uri", "mapping_status")


def _insert_batch(connection, batch):
    """Use an in-memory pandas relation for fast bounded inserts if available."""
    if not batch:
        return
    try:
        import pandas as pd
    except ImportError:
        connection.executemany("INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?)",
                               batch)
        return
    frame = pd.DataFrame.from_records(batch, columns=_OCCURRENCE_FIELDS)
    connection.register("_occurrence_batch", frame)
    try:
        connection.execute("INSERT INTO occurrences SELECT * FROM _occurrence_batch")
    finally:
        connection.unregister("_occurrence_batch")


def _edge_sql(source_start, source_stop, target_start, target_stop):
    """Directed assertion edge with global occurrence IDs and absolute offsets."""
    for value in (source_start, source_stop, target_start, target_stop):
        if type(value) is not int or value < 0:
            raise ValueError("Invalid occurrence partition bounds")
    return f"""
        SELECT s.doc_id, s.occurrence_id AS source_occurrence_id,
               t.occurrence_id AS target_occurrence_id, a.*,
               abs(t.char_start-s.char_start) AS distance_chars,
               t.char_start-s.char_start AS signed_offset,
               1.0/(1+abs(t.char_start-s.char_start)) AS proximity_weight
        FROM assertions a
        JOIN occurrences s ON s.concept_uri=a.source_concept
        JOIN occurrences t ON t.concept_uri=a.target_concept
        WHERE s.occurrence_id >= {source_start}
          AND s.occurrence_id < {source_stop}
          AND t.occurrence_id >= {target_start}
          AND t.occurrence_id < {target_stop}
          AND s.occurrence_id <> t.occurrence_id
        ORDER BY source_occurrence_id, target_occurrence_id, assertion_id
    """


def _source_degrees(connection):
    """Return streaming cursor: exact edge count for each *mapped* source word."""
    return connection.execute("""
        WITH target_counts AS (
            SELECT concept_uri, count(*) AS n
            FROM occurrences WHERE concept_uri IS NOT NULL
            GROUP BY concept_uri
        )
        SELECT s.occurrence_id,
               sum(t.n - CASE WHEN a.source_concept = a.target_concept
                              THEN 1 ELSE 0 END)::BIGINT AS edge_count
        FROM occurrences s
        JOIN assertions a ON a.source_concept = s.concept_uri
        JOIN target_counts t ON t.concept_uri = a.target_concept
        GROUP BY s.occurrence_id
        ORDER BY s.occurrence_id
    """)


def _target_windows(connection, source_id, budget):
    """Fallback when one word emits >budget edges: split by target IDs."""
    rows = connection.execute("""
        SELECT t.occurrence_id, count(*) AS edge_count
        FROM occurrences s
        JOIN assertions a ON a.source_concept=s.concept_uri
        JOIN occurrences t ON t.concept_uri=a.target_concept
        WHERE s.occurrence_id=? AND t.occurrence_id<>s.occurrence_id
        GROUP BY t.occurrence_id ORDER BY t.occurrence_id
    """, [source_id])
    begin = None
    previous = None
    count = 0
    result = []
    while True:
        batch = rows.fetchmany(8192)
        if not batch:
            break
        for target_id, edge_count in batch:
            target_id, edge_count = int(target_id), int(edge_count)
            if edge_count > budget:
                raise ValueError(
                    f"One source-target occurrence pair has {edge_count} assertions, "
                    f"exceeding edge_partition_rows={budget}; no edges truncated"
                )
            if begin is not None and count + edge_count > budget:
                result.append((source_id, source_id + 1, begin, previous + 1, count))
                begin, count = None, 0
            if begin is None:
                begin = target_id
            previous = target_id
            count += edge_count
    if begin is not None:
        result.append((source_id, source_id + 1, begin, previous + 1, count))
    return result


def edge_partitions(connection, total_occurrences, budget):
    """Exact row-bounded parts: contiguous source ranges, target fallback.

    The output union covers all directed occurrence/assertion pairs exactly
    once; it does not require loading the full edge list into Python memory.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("edge_partition_rows must be positive")
    source = _source_degrees(connection)
    parts, oversized = [], []
    start = None
    last = None
    count = 0
    total = 0
    while True:
        batch = source.fetchmany(8192)
        if not batch:
            break
        for source_id, degree in batch:
            source_id, degree = int(source_id), int(degree)
            if degree <= 0:
                continue
            total += degree
            if degree > budget:
                if start is not None:
                    parts.append((start, last + 1, 0, total_occurrences, count))
                    start, count = None, 0
                oversized.append((source_id, degree))
                continue
            if start is not None and count + degree > budget:
                parts.append((start, last + 1, 0, total_occurrences, count))
                start, count = None, 0
            if start is None:
                start = source_id
            last = source_id
            count += degree
    if start is not None:
        parts.append((start, last + 1, 0, total_occurrences, count))
    # Query for exceptional high-degree source occurrences after the first
    # result cursor is exhausted. Each fallback part is still fully bounded.
    for source_id, degree in oversized:
        windows = _target_windows(connection, source_id, budget)
        if sum(p[4] for p in windows) != degree:
            raise RuntimeError("Oversized source target partitions lost edges")
        parts.extend(windows)
    if sum(p[4] for p in parts) != total:
        raise RuntimeError("Source degree partition count mismatch")
    return parts, total


def build_chunked_document(doc, options, staging, final, identity,
                           progress, started):
    """Build one logical graph from sequential text and edge Parquet chunks."""
    from .text_corpus import (
        OCCURRENCE_SCHEMA, ASSERTION_SCHEMA, _connect, _copy, _literal,
        _json, sha256,
    )

    connection = _connect(options, staging / "spill")
    nlp = None
    try:
        if options.spacy_model:
            import spacy
            nlp = spacy.load(options.spacy_model, disable=["ner", "parser"])
        connection.execute("CREATE TABLE occurrences (" + OCCURRENCE_SCHEMA + ")")
        connection.execute("CREATE TABLE assertions (" + ASSERTION_SCHEMA + ")")
        observed = 0
        mapped = 0
        chunk_count = 0
        concepts = set()
        total_chars = len(doc.text)
        progress("mapping_chunks", 0, 0, total_chars)
        with open_conceptnet(options.database, doc.language) as cn:
            # Only ONE chunk's tokens and rows are held at any time. The
            # complete union of vertices lives in the DuckDB occurrences table.
            for offset, chunk in _spacy_chunks(
                    doc.text, nlp, target_chars=options.chunk_chars):
                batch = []
                for word in words(chunk, nlp):
                    uri, status = cn.map_word(word)
                    start, end = offset + word.start, offset + word.end
                    if doc.text[start:end] != word.surface:
                        raise RuntimeError("Original document offsets changed")
                    batch.append((doc.doc_id, observed, word.surface, word.lemma,
                                  start, end, uri, status))
                    observed += 1
                    if uri is not None:
                        mapped += 1
                        concepts.add(uri)
                _insert_batch(connection, batch)
                chunk_count += 1
                done = offset + len(chunk)
                progress("mapping_chunks", int(60 * done / max(total_chars, 1)),
                         done, total_chars)
            progress("querying_relations", 60, 0, len(concepts))
            assertions = cn.induced_assertions(concepts, options.relations)
        # The assertion table connects concepts from ALL text chunks.
        if assertions:
            connection.executemany(
                "INSERT INTO assertions VALUES (?,?,?,?,?)", assertions)
        progress("counting_edges", 68)
        partitions, expected = edge_partitions(
            connection, observed, options.edge_partition_rows)
        if options.max_edges_per_document is not None and (
                expected > options.max_edges_per_document):
            raise ValueError(
                f"{doc.doc_id}: {expected} edges exceed explicitly configured "
                f"max_edges_per_document={options.max_edges_per_document}; "
                "set it to null to enable complete partitioned output"
            )
        if (options.max_total_edges_per_document is not None and
                expected > options.max_total_edges_per_document):
            raise ValueError(
                f"{doc.doc_id}: {expected} edges exceed global disk-safety limit "
                f"max_total_edges_per_document="
                f"{options.max_total_edges_per_document}; no edges truncated"
            )

        progress("writing_occurrences", 72, observed, observed)
        _copy(connection, "SELECT * FROM occurrences ORDER BY occurrence_id",
              staging / "occurrences.parquet")
        progress("writing_assertions", 75, len(assertions), len(assertions))
        _copy(connection, "SELECT * FROM assertions ORDER BY assertion_id",
              staging / "assertions.parquet")

        # Backward-compatible one-file layout when one part fits. Otherwise
        # write multiple edge_part_*.parquet files into the SAME graph folder.
        # Neither source nor target occurrences are restricted to the same
        # *text* chunk, so cross-chunk links are included.
        if expected == 0:
            partitions = [(0, observed, 0, observed, 0)]
        multipart = len(partitions) > 1
        written = 0
        edge_files = []
        for index, (sl, sr, tl, tr, predicted) in enumerate(partitions):
            name = (f"edges_part_{index:06d}.parquet" if multipart
                    else "edges.parquet")
            path = staging / name
            _copy(connection, _edge_sql(sl, sr, tl, tr), path)
            file_expr = _literal(str(path))
            actual = int(connection.execute(
                f"SELECT count(*) FROM read_parquet({file_expr})").fetchone()[0])
            if actual != predicted:
                raise RuntimeError(
                    f"Edge part {index}: {actual} rows, expected {predicted}")
            written += actual
            edge_files.append(path)
            progress("writing_edges", 76 + int(19 * written / max(expected, 1)),
                     written, expected)
        if written != expected:
            raise RuntimeError(
                f"Lost/duplicate cross-chunk edges: {written} != {expected}")

        paths_sql = "[" + ",".join(_literal(str(p)) for p in edge_files) + "]"
        progress("writing_concept_edges", 96)
        _copy(connection, f"""
            SELECT doc_id,source_concept,target_concept,relation_uri,
                count(*) AS occurrence_pair_count,
                count(DISTINCT assertion_id) AS assertion_count,
                min(distance_chars) AS min_distance_chars,
                avg(distance_chars) AS mean_distance_chars,
                max(distance_chars) AS max_distance_chars,
                sum(proximity_weight) AS proximity_weight_sum
            FROM read_parquet({paths_sql})
            GROUP BY doc_id,source_concept,target_concept,relation_uri
            ORDER BY source_concept,target_concept,relation_uri
        """, staging / "concept_edges.parquet")
        progress("finalizing", 98)
        (staging / "text.txt").write_bytes(doc.text.encode("utf-8"))
        checksums = {p.name: sha256(p) for p in sorted(staging.glob("*.parquet"))}
        checksums["text.txt"] = sha256(staging / "text.txt")
        report = {
            "identity": identity, "source": doc.source,
            "word_count": observed, "mapped_words": mapped,
            "unmapped_words": observed - mapped,
            "coverage": mapped / observed if observed else 0.0,
            "concept_count": len(concepts),
            "assertion_count": len(assertions), "edge_count": expected,
            "chunk_count": chunk_count, "edge_part_count": len(edge_files),
            "graph_representation": "one_doc_all_pairs_partitioned",
            "elapsed_seconds": time.perf_counter() - started,
            "checksums": checksums,
        }
        connection.close()
        connection = None
        shutil.rmtree(staging / "spill", ignore_errors=True)
        _json(staging / "manifest.json", report)
        (staging / "COMPLETED").write_text("complete\n", encoding="utf-8")
        os.replace(staging, final)
        progress("ready_for_checkpoint", 99, expected, expected)
        return report
    finally:
        if connection is not None:
            connection.close()
