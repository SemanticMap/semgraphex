"""Compact columnar representation of exact directed typed edge records."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grammar_binary import bits_to_float, float_to_bits
from .graphex_components import EdgeRecord


@dataclass(frozen=True)
class EdgeTable:
    edge_id: np.ndarray
    source: np.ndarray
    target: np.ndarray
    relation_id: np.ndarray
    weight_bits: np.ndarray
    relations: tuple[str, ...]

    @classmethod
    def from_records(cls, records: tuple[EdgeRecord, ...] | list[EdgeRecord]) -> "EdgeTable":
        rows = tuple(records)
        if len({int(row.edge_id) for row in rows}) != len(rows):
            raise ValueError("edge_id must be unique")
        relations = tuple(sorted({str(row.relation) for row in rows}))
        relation_to_id = {name: index for index, name in enumerate(relations)}
        edge_id = np.fromiter((int(row.edge_id) for row in rows), dtype=np.uint64, count=len(rows))
        source = np.fromiter((int(row.source) for row in rows), dtype=np.uint32, count=len(rows))
        target = np.fromiter((int(row.target) for row in rows), dtype=np.uint32, count=len(rows))
        relation_id = np.fromiter(
            (relation_to_id[str(row.relation)] for row in rows),
            dtype=np.uint32,
            count=len(rows),
        )
        weight_bits = np.fromiter(
            (float_to_bits(float(row.weight)) for row in rows),
            dtype=np.uint64,
            count=len(rows),
        )
        return cls(edge_id, source, target, relation_id, weight_bits, relations)

    def __len__(self) -> int:
        return int(self.edge_id.size)

    def validate(self, *, vertex_count: int | None = None) -> None:
        n = len(self)
        if any(array.size != n for array in (
            self.source, self.target, self.relation_id, self.weight_bits
        )):
            raise ValueError("column lengths differ")
        if np.unique(self.edge_id).size != n:
            raise ValueError("edge_id must be unique")
        if self.relation_id.size and int(self.relation_id.max()) >= len(self.relations):
            raise ValueError("relation_id outside dictionary")
        if vertex_count is not None and n:
            if int(self.source.max()) >= vertex_count or int(self.target.max()) >= vertex_count:
                raise ValueError("edge endpoint outside graph")

    def to_records(self) -> tuple[EdgeRecord, ...]:
        self.validate()
        return tuple(
            EdgeRecord(
                int(self.edge_id[index]),
                int(self.source[index]),
                int(self.target[index]),
                self.relations[int(self.relation_id[index])],
                bits_to_float(int(self.weight_bits[index])),
            )
            for index in range(len(self))
        )

    @property
    def nbytes(self) -> int:
        return int(sum(array.nbytes for array in (
            self.edge_id, self.source, self.target, self.relation_id, self.weight_bits
        )))
