"""Streaming, provenance-preserving parser for ConceptNet five-field assertions."""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TextIO


class AssertionParseError(ValueError):
    """Raised for a malformed assertion in fail-fast mode."""


@dataclass(frozen=True)
class Assertion:
    assertion_uri: str
    relation_uri: str
    start_uri: str
    end_uri: str
    weight: float
    dataset: str | None
    sources: object
    license: str | None

    @property
    def relation_name(self) -> str:
        return self.relation_uri.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class AssertionFilters:
    language: str | None = None
    relations: frozenset[str] = frozenset()
    min_weight: float = 0.0
    datasets: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()


@dataclass
class ParseReport:
    total_lines: int = 0
    accepted: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    max_rows: int | None = None
    row_limit_reached: bool = False

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "total_lines": self.total_lines,
            "accepted": self.accepted,
            "rejected": dict(sorted(self.rejected.items())),
        }
        if self.max_rows is not None:
            result["max_rows"] = self.max_rows
            result["row_limit_reached"] = self.row_limit_reached
        return result


def _language(uri: str) -> str | None:
    parts = uri.split("/")
    return parts[2] if len(parts) > 2 and parts[1] == "c" else None


def _source_texts(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    if isinstance(value, dict):
        return {str(item) for item in value.values() if isinstance(item, str)}
    return set()


def _open_text(path: Path) -> TextIO:
    """Open plain or single-stream compressed ConceptNet assertions as text.

    gzip/bzip2/xz are decompressed incrementally, so the full corpus is never
    expanded in memory. ZIP is intentionally rejected because an archive can
    contain multiple members and therefore needs a separate explicit contract.
    """
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    if suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", newline="")
    if suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", newline="")
    if suffix == ".zip":
        raise ValueError("ZIP ConceptNet input is ambiguous; extract one assertions member or use .gz/.bz2/.xz")
    return path.open("rt", encoding="utf-8", newline="")


def stream_assertions(path: str | Path, filters: AssertionFilters = AssertionFilters(), *, invalid_mode: str = "fail_fast", report: ParseReport | None = None, max_rows: int | None = None) -> Iterator[Assertion]:
    """Yield filtered assertions from a local plain/gzip/bzip2/xz TSV stream."""
    if invalid_mode not in {"fail_fast", "skip_invalid"}:
        raise ValueError("invalid_mode must be 'fail_fast' or 'skip_invalid'")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0):
        raise ValueError("max_rows must be a positive integer or None")
    destination = Path(path)
    actual_report = report if report is not None else ParseReport()
    actual_report.max_rows = max_rows
    with _open_text(destination) as stream:
        for line_number, line in enumerate(stream, start=1):
            if max_rows is not None and line_number > max_rows:
                break
            actual_report.total_lines += 1
            try:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5:
                    raise AssertionParseError("expected exactly five TSV fields")
                metadata = json.loads(fields[4])
                weight = metadata.get("weight") if isinstance(metadata, dict) else None
                if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
                    raise AssertionParseError("invalid weight: requires a finite non-boolean number")
                assertion = Assertion(fields[0], fields[1], fields[2], fields[3], float(metadata["weight"]), metadata.get("dataset"), metadata.get("sources"), metadata.get("license"))
            except (json.JSONDecodeError, AssertionParseError) as error:
                actual_report.rejected["invalid_weight" if "weight" in str(error) else "invalid"] += 1
                if invalid_mode == "fail_fast":
                    raise AssertionParseError(f"line {line_number}: {error}") from error
                continue
            if filters.language and (_language(assertion.start_uri) != filters.language or _language(assertion.end_uri) != filters.language):
                actual_report.rejected["language"] += 1
            elif filters.relations and assertion.relation_name not in filters.relations and assertion.relation_uri not in filters.relations:
                actual_report.rejected["relation"] += 1
            elif assertion.weight < filters.min_weight:
                actual_report.rejected["weight"] += 1
            elif filters.datasets and assertion.dataset not in filters.datasets:
                actual_report.rejected["dataset"] += 1
            elif filters.sources and not (_source_texts(assertion.sources) & filters.sources):
                actual_report.rejected["source"] += 1
            else:
                actual_report.accepted += 1
                yield assertion
        actual_report.row_limit_reached = max_rows is not None and actual_report.total_lines == max_rows
