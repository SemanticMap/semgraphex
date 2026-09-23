"""Persistent exact graph types and cross-scale Wishart family registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import networkx as nx

from .wishart_metrics import EgoCandidate


@dataclass
class GraphType:
    """One exact relation-aware structural symbol in the learned dictionary."""

    type_id: str
    fingerprint: str
    node_count: int
    edge_count: int
    relation_signature: dict[str, int]
    boundary_signature: tuple[tuple[str, str, int], ...]
    child_types: tuple[str, ...]
    first_level: int
    levels_seen: set[int] = field(default_factory=set)
    candidate_frequency: int = 0
    accepted_frequency: int = 0
    total_mdl_gain_bits: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["levels_seen"] = sorted(self.levels_seen)
        return payload


@dataclass(frozen=True)
class GraphFamily:
    """Persistent family of exact graph types discovered by Wishart."""

    family_id: str
    member_types: tuple[str, ...]
    levels_seen: tuple[int, ...]
    observations: int


class WishartFamilyRegistry:
    """Match level-local Wishart modes to persistent families by type overlap."""

    def __init__(self, *, match_jaccard: float = 0.5) -> None:
        if not 0.0 <= match_jaccard <= 1.0:
            raise ValueError("match_jaccard must be in [0, 1]")
        self.match_jaccard = float(match_jaccard)
        self._families: dict[str, GraphFamily] = {}
        self._next_id = 1

    @property
    def families(self) -> Mapping[str, GraphFamily]:
        return self._families

    def resolve(self, *, level: int, member_types: set[str]) -> GraphFamily:
        if not member_types:
            raise ValueError("member_types must not be empty")
        best_id: str | None = None
        best_score = -1.0
        for family_id, family in self._families.items():
            existing = set(family.member_types)
            union = existing | member_types
            score = len(existing & member_types) / len(union)
            if score > best_score or (
                score == best_score and (best_id is None or family_id < best_id)
            ):
                best_score = score
                best_id = family_id

        if best_id is None or best_score < self.match_jaccard:
            family_id = f"WF_{self._next_id:06d}"
            self._next_id += 1
            family = GraphFamily(
                family_id=family_id,
                member_types=tuple(sorted(member_types)),
                levels_seen=(int(level),),
                observations=1,
            )
            self._families[family_id] = family
            return family

        previous = self._families[best_id]
        updated = GraphFamily(
            family_id=best_id,
            member_types=tuple(sorted(set(previous.member_types) | member_types)),
            levels_seen=tuple(sorted(set(previous.levels_seen) | {int(level)})),
            observations=previous.observations + 1,
        )
        self._families[best_id] = updated
        return updated

    def write(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "families.jsonl").write_text(
            "".join(
                json.dumps(asdict(self._families[key]), sort_keys=True) + "\n"
                for key in sorted(self._families)
            ),
            encoding="utf-8",
        )


def _relation_signature(candidate: EgoCandidate) -> dict[str, int]:
    return {
        relation: int(layer.nnz)
        for relation, layer in sorted(candidate.relation_layers.items())
        if layer.nnz
    }


def _candidate_graph(candidate: EgoCandidate) -> nx.DiGraph:
    """Build an exact topology/edge-type graph for VF2 matching.

    Edge weights are deliberately omitted from type identity. Relation labels
    and recursive child-symbol labels are retained.
    """
    n = int(candidate.adjacency.shape[0])
    graph = nx.DiGraph()
    node_types = candidate.node_types or tuple(None for _ in range(n))
    if len(node_types) != n:
        raise ValueError("candidate.node_types must match candidate node count")
    for node, symbol_type in enumerate(node_types):
        graph.add_node(node, symbol_type=symbol_type or "ATOM")

    relation_by_edge: dict[tuple[int, int], list[str]] = {}
    for relation, layer in sorted(candidate.relation_layers.items()):
        coo = layer.tocoo()
        for row, col in zip(coo.row, coo.col, strict=True):
            relation_by_edge.setdefault((int(row), int(col)), []).append(relation)

    coo = candidate.adjacency.tocoo()
    for row, col in zip(coo.row, coo.col, strict=True):
        key = (int(row), int(col))
        relations = tuple(sorted(relation_by_edge.get(key, ("__edge__",))))
        graph.add_edge(key[0], key[1], relation="|".join(relations))
    return graph


def candidate_fingerprint(
    candidate: EgoCandidate,
    *,
    boundary_sensitive: bool,
) -> str:
    """Permutation-invariant WL bucket used before exact VF2 verification."""
    graph = _candidate_graph(candidate)
    wl_hash = nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="symbol_type",
        edge_attr="relation",
        iterations=3,
        digest_size=16,
    )
    payload = {
        "wl": wl_hash,
        "n": graph.number_of_nodes(),
        "e": graph.number_of_edges(),
        "relations": _relation_signature(candidate),
        "boundary": list(candidate.boundary_signature) if boundary_sensitive else [],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def candidates_are_isomorphic(
    left: EgoCandidate,
    right: EgoCandidate,
    *,
    boundary_sensitive: bool,
) -> bool:
    """Exact relation-aware identity check after the WL bucket filter."""
    if left.adjacency.shape != right.adjacency.shape:
        return False
    if boundary_sensitive and left.boundary_signature != right.boundary_signature:
        return False
    if _relation_signature(left) != _relation_signature(right):
        return False
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "symbol_type", "ATOM"
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        "relation", "__edge__"
    )
    return nx.is_isomorphic(
        _candidate_graph(left),
        _candidate_graph(right),
        node_match=node_match,
        edge_match=edge_match,
    )


class GraphDictionary:
    """Persistent exact graph vocabulary reused across recursive levels."""

    def __init__(self, *, boundary_sensitive: bool = True) -> None:
        self.boundary_sensitive = bool(boundary_sensitive)
        self.types: dict[str, GraphType] = {}
        self._representatives: dict[str, EgoCandidate] = {}
        self._by_fingerprint: dict[str, list[str]] = {}
        self._next_id = 1
        self._candidate_level_counts: dict[tuple[str, int], int] = {}
        self._accepted_level_counts: dict[tuple[str, int], int] = {}

    def resolve_or_create(self, candidate: EgoCandidate, *, level: int) -> GraphType:
        fingerprint = candidate_fingerprint(
            candidate,
            boundary_sensitive=self.boundary_sensitive,
        )
        for type_id in self._by_fingerprint.get(fingerprint, []):
            representative = self._representatives[type_id]
            if candidates_are_isomorphic(
                candidate,
                representative,
                boundary_sensitive=self.boundary_sensitive,
            ):
                graph_type = self.types[type_id]
                graph_type.levels_seen.add(int(level))
                return graph_type

        type_id = f"GT_{self._next_id:06d}"
        self._next_id += 1
        child_types = tuple(sorted({item for item in candidate.node_types if item}))
        graph_type = GraphType(
            type_id=type_id,
            fingerprint=fingerprint,
            node_count=int(candidate.adjacency.shape[0]),
            edge_count=int(candidate.adjacency.nnz),
            relation_signature=_relation_signature(candidate),
            boundary_signature=(
                tuple(candidate.boundary_signature) if self.boundary_sensitive else ()
            ),
            child_types=child_types,
            first_level=int(level),
            levels_seen={int(level)},
        )
        self.types[type_id] = graph_type
        self._representatives[type_id] = candidate
        self._by_fingerprint.setdefault(fingerprint, []).append(type_id)
        return graph_type

    def match(self, candidate: EgoCandidate) -> GraphType | None:
        fingerprint = candidate_fingerprint(
            candidate,
            boundary_sensitive=self.boundary_sensitive,
        )
        for type_id in self._by_fingerprint.get(fingerprint, []):
            if candidates_are_isomorphic(
                candidate,
                self._representatives[type_id],
                boundary_sensitive=self.boundary_sensitive,
            ):
                return self.types[type_id]
        return None

    def representative(self, type_id: str) -> EgoCandidate:
        return self._representatives[type_id]

    def record_candidate_counts(self, *, level: int, counts: Mapping[str, int]) -> None:
        for type_id, value in counts.items():
            if type_id not in self.types:
                raise KeyError(type_id)
            value = int(value)
            if value < 0:
                raise ValueError("candidate frequency must be non-negative")
            key = (type_id, int(level))
            previous = self._candidate_level_counts.get(key, 0)
            self._candidate_level_counts[key] = value
            self.types[type_id].candidate_frequency += value - previous

    def record_accepted(
        self,
        *,
        level: int,
        type_id: str,
        count: int = 1,
        mdl_gain_bits: float = 0.0,
    ) -> None:
        if type_id not in self.types:
            raise KeyError(type_id)
        key = (type_id, int(level))
        previous = self._accepted_level_counts.get(key, 0)
        self._accepted_level_counts[key] = previous + int(count)
        graph_type = self.types[type_id]
        graph_type.accepted_frequency += int(count)
        graph_type.total_mdl_gain_bits += float(mdl_gain_bits)

    def frequency_counts(self, *, accepted: bool = False) -> dict[str, int]:
        return {
            type_id: (
                graph_type.accepted_frequency
                if accepted
                else graph_type.candidate_frequency
            )
            for type_id, graph_type in self.types.items()
            if (
                graph_type.accepted_frequency
                if accepted
                else graph_type.candidate_frequency
            )
            > 0
        }

    def write(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "graph_types.jsonl").write_text(
            "".join(
                json.dumps(self.types[type_id].to_dict(), sort_keys=True) + "\n"
                for type_id in sorted(self.types)
            ),
            encoding="utf-8",
        )
        stats = {
            "dictionary_size": len(self.types),
            "candidate_occurrences": int(
                sum(item.candidate_frequency for item in self.types.values())
            ),
            "accepted_occurrences": int(
                sum(item.accepted_frequency for item in self.types.values())
            ),
            "recursive_types": int(
                sum(bool(item.child_types) for item in self.types.values())
            ),
        }
        (target / "statistics.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
