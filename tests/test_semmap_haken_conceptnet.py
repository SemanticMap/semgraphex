from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from semmap_haken.conceptnet import AssertionFilters, AssertionParseError, ParseReport, stream_assertions


FIXTURE = Path(__file__).parent / "fixtures" / "conceptnet_tiny.tsv"


def test_stream_parser_preserves_sense_uri_and_filters() -> None:
    report = ParseReport()
    records = list(stream_assertions(FIXTURE, AssertionFilters(language="en", relations=frozenset({"RelatedTo"}), min_weight=1.5), report=report))
    assert records[0].start_uri == "/c/en/dog/n/animal"
    assert records[0].weight == 2.0
    assert report.to_dict() == {"total_lines": 3, "accepted": 1, "rejected": {"language": 1, "relation": 1}}


def test_plain_and_gzip_are_equivalent(tmp_path: Path) -> None:
    compressed = tmp_path / "tiny.tsv.gz"
    with gzip.open(compressed, "wb") as output:
        output.write(FIXTURE.read_bytes())
    assert list(stream_assertions(FIXTURE, invalid_mode="skip_invalid")) == list(stream_assertions(compressed, invalid_mode="skip_invalid"))


def test_invalid_records_fail_or_skip(tmp_path: Path) -> None:
    broken = tmp_path / "broken.tsv"
    broken.write_text("not\ta\tvalid\trow\n", encoding="utf-8")
    with pytest.raises(AssertionParseError, match="line 1"):
        list(stream_assertions(broken))
    report = ParseReport()
    assert list(stream_assertions(broken, invalid_mode="skip_invalid", report=report)) == []
    assert report.rejected["invalid"] == 1
