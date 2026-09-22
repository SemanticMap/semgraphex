from __future__ import annotations

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


def test_stream_parser_limits_physical_rows_before_filtering() -> None:
    report = ParseReport()

    records = list(
        stream_assertions(
            FIXTURE,
            AssertionFilters(language="en", relations=frozenset({"RelatedTo"}), min_weight=1.5),
            max_rows=2,
            report=report,
        )
    )

    assert [record.start_uri for record in records] == ["/c/en/dog/n/animal"]
    assert report.to_dict() == {
        "total_lines": 2,
        "accepted": 1,
        "max_rows": 2,
        "row_limit_reached": True,
        "rejected": {"relation": 1},
    }


def test_stream_parser_rejects_ambiguous_zip_archive(tmp_path: Path) -> None:
    """Single-stream gzip/bzip2/xz are supported; multi-member ZIP is not."""
    archive = tmp_path / "tiny.tsv.zip"
    archive.write_bytes(FIXTURE.read_bytes())

    with pytest.raises(ValueError, match="ZIP ConceptNet input is ambiguous"):
        list(stream_assertions(archive))


def test_invalid_records_fail_or_skip(tmp_path: Path) -> None:
    broken = tmp_path / "broken.tsv"
    broken.write_text("not\ta\tvalid\trow\n", encoding="utf-8")
    with pytest.raises(AssertionParseError, match="line 1"):
        list(stream_assertions(broken))
    report = ParseReport()
    assert list(stream_assertions(broken, invalid_mode="skip_invalid", report=report)) == []
    assert report.rejected["invalid"] == 1
