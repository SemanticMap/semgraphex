"""Bounded bridge from immutable text Parquet to H1 directed typed graphs.

This adapter never silently takes the first K edges. It either materializes the
ENTIRE requested view, within explicit node/edge budgets, or refuses. A separate
origin/group mapping makes provenance-aware P1 holdouts possible.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

from .hypothesis.graph import TypedGraph
from .text_research import open_research_views


@dataclass(frozen=True)
class TypedTextGraph:
    graph: TypedGraph
    doc_id: str
    projection: str
    distance_limit: int | None
    stored_graph_state: str
    source_assertion_ids: tuple[str, ...]
    source_concepts: tuple[str, ...]
    target_concepts: tuple[str, ...]
    source_occurrence_ids: tuple[int | None, ...]
    target_occurrence_ids: tuple[int | None, ...]
    distance_chars: tuple[int | None, ...]
    holdout_groups: tuple[str, ...]
    weight_semantics: str
    position_shuffle_seed: int | None = None

    @property
    def provenance_complete(self) -> bool:
        return len(self.source_assertion_ids) == self.graph.edge_count

    def metadata(self) -> dict:
        return {
            "doc_id": self.doc_id, "projection": self.projection,
            "distance_limit": self.distance_limit,
            "position_shuffle_seed": self.position_shuffle_seed,
            "stored_graph_state": self.stored_graph_state,
            "reconstructed_from": ["occurrences.parquet", "assertions.parquet"],
            "typed_vertices": self.graph.vertex_count,
            "typed_edges": self.graph.edge_count,
            "provenance_complete": self.provenance_complete,
            "holdout_group_policy": "relation + undirected concept pair (all occurrences)",
            "weight_semantics": self.weight_semantics,
            "warning": ("Graph edges are ConceptNet-induced; they are not "
                        "independent text-extracted assertions."),
        }


def load_typed_text_graph(
    corpus: str | Path, doc_id: str, *,
    projection: Literal["concept", "occurrence"] = "concept",
    max_distance_chars: int | None = None,
    max_vertices: int = 10_000,
    max_edges: int = 200_000,
    position_shuffle_seed: int | None = None,
) -> TypedTextGraph:
    """Return exact H1 TypedGraph plus source attribution for a bounded view.

    The concept view contains one record per underlying ConceptNet assertion
    and *all* observed mapped concepts, not a multiplicity-blown-up network.
    The occurrence view reconstructs all eligible edges across all text chunks.
    A capped published edge file DOES NOT cap the implicit source. Both modes
    refuse rather than truncate when the requested research graph is too big.
    """
    if projection not in ("concept", "occurrence"):
        raise ValueError("Unknown projection")
    if type(max_vertices) is not int or max_vertices < 1:
        raise ValueError("max_vertices must be positive")
    if type(max_edges) is not int or max_edges < 1:
        raise ValueError("max_edges must be positive")
    if max_distance_chars is not None and (
            type(max_distance_chars) is not int or max_distance_chars < 0):
        raise ValueError("max_distance_chars must be nonnegative or None")
    if projection == "concept" and max_distance_chars is not None:
        raise ValueError("Distance window is defined only for occurrence vertices")
    if position_shuffle_seed is not None and type(position_shuffle_seed) is not int:
        raise ValueError("position_shuffle_seed must be an integer or None")
    if projection == "concept" and position_shuffle_seed is not None:
        raise ValueError("Position shuffling affects only occurrence projection")

    with open_research_views(
        corpus, doc_id, seed=position_shuffle_seed if position_shuffle_seed is not None else 42
    ) as view:
        db = view.db
        if projection == "concept":
            n = db.execute(
                "SELECT count(DISTINCT concept_uri) FROM research_occurrences "
                "WHERE concept_uri IS NOT NULL").fetchone()[0]
            node_query = (
                "SELECT DISTINCT concept_uri FROM research_occurrences "
                "WHERE concept_uri IS NOT NULL ORDER BY concept_uri"
            )
            edge_query = (
                "SELECT source_concept, target_concept, relation_uri, "
                "conceptnet_weight, assertion_id, "
                "CAST(NULL AS BIGINT), CAST(NULL AS BIGINT), "
                "CAST(NULL AS BIGINT), source_concept, target_concept "
                "FROM research_assertions "
                "ORDER BY source_concept, target_concept, relation_uri, assertion_id"
            )
            known_edges = db.execute(
                "SELECT count(*) FROM research_assertions").fetchone()[0]
            weight_kind = "original_conceptnet_assertion_weight"
        else:
            n = db.execute(
                "SELECT count(*) FROM research_occurrences").fetchone()[0]
            node_query = (
                "SELECT occurrence_id FROM research_occurrences "
                "ORDER BY occurrence_id"
            )
            where = ""
            if max_distance_chars is not None:
                where = f"WHERE distance_chars <= {max_distance_chars}"
            selected_source = ("research_shuffled_edges" if position_shuffle_seed is not None
                               else "research_expanded_edges")
            edge_query = (
                "SELECT source_occurrence_id, target_occurrence_id, "
                "relation_uri, proximity_weight, assertion_id, "
                "source_occurrence_id, target_occurrence_id, distance_chars, "
                "source_concept, target_concept "
                f"FROM {selected_source} {where} "
                "ORDER BY source_occurrence_id, target_occurrence_id, "
                "assertion_id"
            )
            known_edges = (view.full_edge_count()
                           if max_distance_chars is None else None)
            weight_kind = ("1/(1+shuffled_character_distance)"
                           if position_shuffle_seed is not None else
                           "1/(1+absolute_character_distance)"
        if n > max_vertices:
            raise ValueError(
                f"Research view has {n} nodes > max_vertices={max_vertices}; "
                "reduce the VIEW first, never silently cut the node set"
            )
        if known_edges is not None and known_edges > max_edges:
            raise ValueError(
                f"Research view has {known_edges} edges > max_edges={max_edges}; "
                "reduce the view or increase the budget explicitly"
            )
        raw_nodes = tuple(row[0] for row in db.execute(node_query).fetchall())
        if projection == "occurrence":
            prefix = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
            node_ids = tuple(f"urn:semmap:{prefix}:word:{int(i):012d}"
                             for i in raw_nodes)
        else:
            node_ids = tuple(raw_nodes)
        # Bounded result: LIMIT +1 rejects oversize data *before* fetchall of
        # a potentially huge occurrence Cartesian product.
        rows = db.execute(edge_query + f" LIMIT {max_edges + 1}").fetchall()
        if len(rows) > max_edges:
            raise ValueError(
                f"Research view exceeds max_edges={max_edges}; "
                "no positional/prefix truncation allowed in scientific H1"
            )
        labels = tuple(sorted({str(r[2]) for r in rows}))
        node_index = {key: i for i, key in enumerate(raw_nodes)}
        relation_index = {key: i for i, key in enumerate(labels)}
        source, target, relations, weights = [], [], [], []
        composed_ids, origin_ids = [], []
        source_concepts, target_concepts = [], []
        source_occ, target_occ, distances, holdout = [], [], [], []
        for src, dst, relation, weight, assertion_id, so, to, distance, cs, ct in rows:
            source.append(node_index[src])
            target.append(node_index[dst])
            relations.append(relation_index[str(relation)])
            weights.append(float(weight))
            origin_ids.append(str(assertion_id))
            if projection == "occurrence":
                identity = [doc_id, int(so), int(to), str(assertion_id)]
                composed_ids.append(
                    "urn:semmap:edge:" + hashlib.sha256(
                        json.dumps(identity, ensure_ascii=False).encode("utf-8")
                    ).hexdigest()
                )
            else:
                composed_ids.append(str(assertion_id))
            source_concepts.append(cs)
            target_concepts.append(ct)
            source_occ.append(None if so is None else int(so))
            target_occ.append(None if to is None else int(to))
            distances.append(None if distance is None else int(distance))
            # Prevent false held-out gains from duplicate ConceptNet assertions,
            # reverse edges, and repeated word occurrences.
            holdout.append(json.dumps(
                [str(relation), *sorted((str(cs), str(ct)))],
                ensure_ascii=False, separators=(",", ":"),
            ))
        graph = TypedGraph(
            node_ids=node_ids, relation_names=labels,
            source=np.asarray(source, dtype=np.int64),
            target=np.asarray(target, dtype=np.int64),
            relation=np.asarray(relations, dtype=np.int32),
            weight=np.asarray(weights, dtype=np.float64),
            assertion_ids=tuple(composed_ids),
        )
        return TypedTextGraph(
            graph, doc_id, projection, max_distance_chars,
            view.audit_row["state"], tuple(origin_ids),
            tuple(source_concepts), tuple(target_concepts),
            tuple(source_occ), tuple(target_occ), tuple(distances),
            tuple(holdout), weight_kind, position_shuffle_seed,
        )
