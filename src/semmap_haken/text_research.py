"""Research-safe views over a published text/ConceptNet corpus.

The corpus's Parquet edge limit applies to STORAGE, not to these implicit
views. All occurrences/assertions for each document remain available. No
all-pairs edge relation is materialized unless the caller explicitly asks.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
import json
from pathlib import Path

from .text_corpus import _literal, _verify_document, sha256


def audit_corpus(corpus: str | Path, *, verify: bool = True,
                 doc_id: str | None = None) -> dict:
    """Admit only integrity-checked, demonstrably full graphs to primary H1.

    Capped documents remain suitable for reconstructing the full *implicit*
    ConceptNet-induced graph from occurrences+assertions, but their published
    edges/concept_edges must not be treated as complete. Older manifests with
    no completeness evidence are classified as unknown, not silently promoted.
    """
    root = Path(corpus).resolve()
    if not (root / "COMPLETED").is_file():
        raise ValueError(f"Corpus not published (missing COMPLETED): {root}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    run_path = root / "run.json"
    if sha256(run_path) != manifest.get("run_sha256"):
        raise ValueError("Corpus run.json hash mismatch")
    rows = manifest.get("documents")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Corpus has no published documents")
    # Inspect all *manifest identities* cheaply even for a single-document
    # audit. Hashing enormous edge Parquet parts of OTHER documents is neither
    # necessary nor affordable when creating just one research view.
    selected_id = doc_id
    seen_ids, seen_keys = set(), set()
    results = []
    for entry in rows:
        identity = entry["identity"]
        row_id, key = identity["doc_id"], identity["key"]
        if (not isinstance(row_id, str) or not row_id or
                key != hashlib.sha256(row_id.encode("utf-8")).hexdigest()):
            raise ValueError("Invalid or unsafe document key in corpus manifest")
        if row_id in seen_ids or key in seen_keys:
            raise ValueError("Duplicate document identity in corpus manifest")
        seen_ids.add(row_id)
        seen_keys.add(key)
        if selected_id is not None and row_id != selected_id:
            continue
        doc_id = row_id
        folder = root / "documents" / key
        local = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if local != entry:
            raise ValueError(f"Published manifest differs for document {doc_id}")
        if verify:
            _verify_document(folder, identity)
            if local["checksums"].get("text.txt") != identity["text_sha256"]:
                raise ValueError(f"Source text checksum mismatch for {doc_id}")
        stored = entry["edge_count"]
        expected = entry.get("edge_count_total_expected")
        omitted = entry.get("edges_omitted")
        claimed_complete = entry.get("graph_complete")
        if not isinstance(stored, int) or stored < 0:
            raise ValueError(f"Invalid stored edge count for {doc_id}")
        if claimed_complete is None or expected is None or omitted is None:
            state = "unknown_legacy"
        elif (not isinstance(expected, int) or not isinstance(omitted, int) or
              expected < stored or omitted != expected - stored or
              not isinstance(claimed_complete, bool) or
              claimed_complete != (omitted == 0)):
            raise ValueError(f"Inconsistent edge completeness metadata: {doc_id}")
        else:
            state = "full" if claimed_complete else "capped"
        for required in ("occurrences.parquet", "assertions.parquet"):
            if required not in local["checksums"] or not (folder / required).is_file():
                raise ValueError(f"Missing full research source {required}: {doc_id}")
        results.append({
            "doc_id": doc_id, "key": key, "state": state,
            "word_count": entry["word_count"], "mapped_words": entry["mapped_words"],
            "stored_edges": stored, "expected_edges": expected,
            "omitted_edges": omitted, "primary_h1_eligible": verify and state == "full",
            "implicit_reconstruction_available": verify,
        })
    if selected_id is not None and not results:
        raise ValueError(f"Unknown document ID: {selected_id!r}")
    # Validate whole-manifest arithmetic without touching all other files.
    if sum(r["edge_count"] for r in rows) != manifest["edge_count"]:
        raise ValueError("Corpus edge count disagrees with document manifests")
    return {
        "schema": "text-research-audit-v1",
        "root": str(root),
        "integrity": ("sha256_verified" if selected_id is None
                      else "selected_document_sha256_verified") if verify else "unchecked",
        "corpus_document_count": len(rows),
        "selected_document_id": selected_id,
        "run_sha256": manifest["run_sha256"],
        "total_documents": len(results),
        "full_documents": sum(x["state"] == "full" for x in results),
        "capped_documents": sum(x["state"] == "capped" for x in results),
        "unknown_documents": sum(x["state"] == "unknown_legacy" for x in results),
        "primary_h1_eligible_count": sum(x["primary_h1_eligible"] for x in results),
        "documents": results,
    }


@dataclass
class ResearchViews:
    """Context manager owning one transient DuckDB connection and one doc ID."""
    db: object
    document_id: str
    audit_row: dict
    seed: int

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def distance_view(self, max_distance: int, *, shuffled: bool = False) -> str:
        """Create a lazy scale view without changing the underlying graph."""
        if type(max_distance) is not int or max_distance < 0:
            raise ValueError("Distance scale must be a nonnegative integer")
        name = f"research_distance_{max_distance}_{'shuffled' if shuffled else 'observed'}"
        source = "research_shuffled_edges" if shuffled else "research_expanded_edges"
        self.db.execute(f"CREATE OR REPLACE VIEW {name} AS "
                        f"SELECT * FROM {source} WHERE distance_chars <= {max_distance}")
        return name

    def full_edge_count(self) -> int:
        """Exact count from concept multiplicities; no Cartesian edge expansion."""
        return int(self.db.execute(
            "SELECT coalesce(sum(edge_count_full),0) FROM research_concept_projection"
        ).fetchone()[0])

    def metadata(self) -> dict:
        return {
            "doc_id": self.document_id,
            "published_graph_state": self.audit_row["state"],
            "published_edges": self.audit_row["stored_edges"],
            "implicit_full_edges": self.full_edge_count(),
            "edge_semantics": "ConceptNet-induced, not extracted claims from text",
            "position_shuffle_seed": self.seed,
            "verification": "original occurrences and assertions checksummed",
        }


def open_research_views(corpus: str | Path, doc_id: str, *,
                        seed: int = 42, verify: bool = True) -> ResearchViews:
    """Expose finite concept projection, full implicit occurrence graph, and null.

    The view's complete graph is reconstructed on demand from the checksum-
    verified occurrences and assertions, *not* from capped edges.parquet.
    This guarantees that the full concept projection is unaffected by any
    storage edge cap; callers must still bound expensive all-pairs queries.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not verify:
        raise ValueError("Research projections require checksum verification")
    audit = audit_corpus(corpus, verify=True, doc_id=doc_id)
    match = [row for row in audit["documents"] if row["doc_id"] == doc_id]
    if len(match) != 1:
        raise ValueError(f"Unknown document ID: {doc_id!r}")
    row = match[0]
    root = Path(corpus).resolve()
    folder = root / "documents" / row["key"]
    import duckdb
    db = duckdb.connect(":memory:")
    try:
        occ_path = _literal(str(folder / "occurrences.parquet"))
        assertions_path = _literal(str(folder / "assertions.parquet"))
        db.execute("CREATE VIEW research_occurrences AS "
                   f"SELECT * FROM read_parquet({occ_path}) "
                   f"WHERE doc_id = {_literal(doc_id)}")
        db.execute("CREATE VIEW research_assertions AS "
                   f"SELECT * FROM read_parquet({assertions_path})")
        db.execute("""
            CREATE VIEW research_concept_counts AS
            SELECT concept_uri, count(*)::BIGINT AS n
            FROM research_occurrences
            WHERE concept_uri IS NOT NULL
            GROUP BY concept_uri
        """)
        db.execute("""
            CREATE VIEW research_concept_projection AS
            SELECT a.source_concept, a.target_concept, a.relation_uri,
                   count(*)::BIGINT AS assertion_count,
                   sum(a.conceptnet_weight) AS assertion_weight_sum,
                   (count(*)::BIGINT *
                    (src.n * dst.n -
                     CASE WHEN a.source_concept=a.target_concept
                          THEN src.n ELSE 0 END))::BIGINT AS edge_count_full
            FROM research_assertions a
            JOIN research_concept_counts src ON src.concept_uri=a.source_concept
            JOIN research_concept_counts dst ON dst.concept_uri=a.target_concept
            GROUP BY a.source_concept, a.target_concept, a.relation_uri,
                     src.n, dst.n
        """)
        db.execute("""
            CREATE VIEW research_expanded_edges AS
            SELECT s.doc_id,
                s.occurrence_id AS source_occurrence_id,
                t.occurrence_id AS target_occurrence_id,
                a.assertion_id, a.source_concept, a.target_concept,
                a.relation_uri, a.conceptnet_weight,
                abs(t.char_start-s.char_start) AS distance_chars,
                t.char_start-s.char_start AS signed_offset,
                1.0/(1+abs(t.char_start-s.char_start)) AS proximity_weight
            FROM research_assertions a
            JOIN research_occurrences s ON s.concept_uri=a.source_concept
            JOIN research_occurrences t ON t.concept_uri=a.target_concept
            WHERE s.occurrence_id<>t.occurrence_id
        """)
        # Permute *only* positions: preserve concept frequencies, identities,
        # relations, and the multiset of original word start coordinates.
        db.execute(f"""
            CREATE VIEW research_shuffled_occurrences AS
            WITH randomized AS (
                SELECT *, row_number() OVER (
                    ORDER BY hash(occurrence_id, {seed}), occurrence_id) AS seq
                FROM research_occurrences
            ), positions AS (
                SELECT char_start AS shuffled_start,
                    row_number() OVER (ORDER BY char_start, occurrence_id) AS seq
                FROM research_occurrences
            )
            SELECT r.*, p.shuffled_start
            FROM randomized r JOIN positions p USING(seq)
        """)
        db.execute("""
            CREATE VIEW research_shuffled_edges AS
            SELECT s.doc_id,
                s.occurrence_id AS source_occurrence_id,
                t.occurrence_id AS target_occurrence_id,
                a.assertion_id, a.source_concept, a.target_concept,
                a.relation_uri, a.conceptnet_weight,
                abs(t.shuffled_start-s.shuffled_start) AS distance_chars,
                t.shuffled_start-s.shuffled_start AS signed_offset,
                1.0/(1+abs(t.shuffled_start-s.shuffled_start)) AS proximity_weight
            FROM research_assertions a
            JOIN research_shuffled_occurrences s ON s.concept_uri=a.source_concept
            JOIN research_shuffled_occurrences t ON t.concept_uri=a.target_concept
            WHERE s.occurrence_id<>t.occurrence_id
        """)
        opened = ResearchViews(db, doc_id, row, seed)
        actual = opened.full_edge_count()
        if row["expected_edges"] is not None and actual != row["expected_edges"]:
            raise ValueError(f"Implicit expansion disagrees with manifest: {doc_id}")
        return opened
    except Exception:
        db.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit and inspect a text research corpus")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--doc-id", help="Also inspect one document's implicit graph")
    parser.add_argument("--scale", type=int, action="append", default=[],
                        help="Optional text-distance window to count (may be costly)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verify", action="store_true",
                        help="Fast metadata inspection only; NEVER research eligible")
    args = parser.parse_args(argv)
    outcome = audit_corpus(args.corpus, verify=not args.no_verify)
    if args.doc_id:
        with open_research_views(args.corpus, args.doc_id, seed=args.seed) as views:
            outcome["selected_document"] = views.metadata()
            outcome["selected_document"]["distances"] = {
                str(scale): int(views.db.execute(
                    f"SELECT count(*) FROM {views.distance_view(scale)}"
                ).fetchone()[0]) for scale in args.scale
            }
    print(json.dumps(outcome, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
