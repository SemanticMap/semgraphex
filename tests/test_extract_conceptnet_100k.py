from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import pytest

from scripts.extract_conceptnet_100k import main, open_source, parse_args


_REQUIRED_OUTPUT_ARGS = ["--output", "subset.tsv", "--metadata", "subset.json"]


def test_parse_args_accepts_explicit_decompressed_input() -> None:
    args = parse_args(
        [
            "--decompressed-input",
            "data/cache/conceptnet-assertions-5.7.0.csv",
            *_REQUIRED_OUTPUT_ARGS,
        ]
    )

    assert args.input is None
    assert args.decompressed_input == "data/cache/conceptnet-assertions-5.7.0.csv"


def test_parse_args_rejects_both_input_modes() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                "conceptnet.csv.gz",
                "--decompressed-input",
                "conceptnet.csv",
                *_REQUIRED_OUTPUT_ARGS,
            ]
        )


def test_open_source_reads_decompressed_text_without_gzip(tmp_path: Path) -> None:
    source = tmp_path / "conceptnet.csv"
    source.write_text("plain assertion\n", encoding="utf-8")

    with open_source(source, compressed=False) as stream:
        assert stream.read() == "plain assertion\n"


def test_open_source_preserves_compressed_input_support(tmp_path: Path) -> None:
    source = tmp_path / "conceptnet.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        stream.write("compressed assertion\n")

    with open_source(source, compressed=True) as stream:
        assert stream.read() == "compressed assertion\n"


def test_main_extracts_from_decompressed_input_and_records_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "conceptnet.csv"
    rows = [
        _assertion("ab", "/c/en/a", "/c/en/b"),
        _assertion("bc", "/c/en/b", "/c/en/c"),
        _assertion("cd", "/c/en/c", "/c/en/d"),
    ]
    source.write_text("".join(rows), encoding="utf-8")
    output = tmp_path / "subset.tsv"
    metadata = tmp_path / "subset.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_conceptnet_100k.py",
            "--decompressed-input",
            str(source),
            "--output",
            str(output),
            "--metadata",
            str(metadata),
            "--target-nodes",
            "3",
            "--candidate-multiplier",
            "2",
            "--relations",
            "RelatedTo",
        ],
    )

    assert main() == 0
    report = json.loads(metadata.read_text(encoding="utf-8"))

    assert report["input"] == str(source)
    assert report["input_compression"] == "none"
    assert report["selected_nodes_seen_in_induced_edges"] == 3
    assert output.read_text(encoding="utf-8")


def _assertion(assertion_id: str, start: str, end: str) -> str:
    metadata = json.dumps({"weight": 1.0})
    return f"/a/{assertion_id}\t/r/RelatedTo\t{start}\t{end}\t{metadata}\n"
