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
    assert args.workers == 1
    assert args.batch_size == 10_000


def test_parse_args_accepts_parallel_execution_settings() -> None:
    args = parse_args(
        [
            "--decompressed-input",
            "conceptnet.csv",
            "--workers",
            "3",
            "--batch-size",
            "17",
            *_REQUIRED_OUTPUT_ARGS,
        ]
    )

    assert args.workers == 3
    assert args.batch_size == 17


@pytest.mark.parametrize(("option", "value"), [("--workers", "0"), ("--batch-size", "-1")])
def test_parse_args_rejects_non_positive_parallel_settings(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--decompressed-input",
                "conceptnet.csv",
                option,
                value,
                *_REQUIRED_OUTPUT_ARGS,
            ]
        )


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


def test_parallel_extraction_matches_single_process_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "conceptnet.csv"
    source.write_text(
        "".join(
            [
                _assertion("ab", "/c/en/a", "/c/en/b"),
                _assertion("bc", "/c/en/b", "/c/en/c"),
                _assertion("cd", "/c/en/c", "/c/en/d"),
                _assertion("ad", "/c/en/a", "/c/en/d"),
            ]
        ),
        encoding="utf-8",
    )

    serial_output, serial_report = _run_extractor(tmp_path, monkeypatch, source, workers=1)
    parallel_output, parallel_report = _run_extractor(tmp_path, monkeypatch, source, workers=2)

    assert parallel_output == serial_output
    assert parallel_report["workers"] == 2
    assert parallel_report["batch_size"] == 2
    assert parallel_report["parallel_strategy"] == "ordered_batched_process_pool"
    ignored_metadata = {"workers", "elapsed_seconds", "parallel_strategy"}
    assert {
        key: value for key, value in parallel_report.items() if key not in ignored_metadata
    } == {
        key: value for key, value in serial_report.items() if key not in ignored_metadata
    }


def _run_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    workers: int,
) -> tuple[str, dict[str, object]]:
    output = tmp_path / f"subset-{workers}.tsv"
    metadata = tmp_path / f"subset-{workers}.json"
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
            "--workers",
            str(workers),
            "--batch-size",
            "2",
        ],
    )
    assert main() == 0
    return output.read_text(encoding="utf-8"), json.loads(metadata.read_text(encoding="utf-8"))


def _assertion(assertion_id: str, start: str, end: str) -> str:
    metadata = json.dumps({"weight": 1.0})
    return f"/a/{assertion_id}\t/r/RelatedTo\t{start}\t{end}\t{metadata}\n"
