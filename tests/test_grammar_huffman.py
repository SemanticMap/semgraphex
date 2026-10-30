"""Huffman entropy stream and grammar codec must preserve exact edge records."""
from __future__ import annotations

import zipfile
import pytest

from semmap_haken.grammar_huffman import encode_symbols, decode_symbols
from semmap_haken.grammar_codec import encode_grammar, decode_grammar
from semmap_haken.graphex_components import EdgeRecord


@pytest.mark.parametrize("symbols", [[], [3], [3] * 50, [9, 1, 9, 5, 1, 9], list(range(300))])
def test_huffman_codes_roundtrip(symbols):
    assert decode_symbols(encode_symbols(symbols)) == tuple(symbols)


def test_huffman_rejects_truncation_and_bad_header():
    raw = encode_symbols([0, 1, 1, 2, 3, 3, 3])
    with pytest.raises(ValueError):
        decode_symbols(raw[:-1])
    with pytest.raises(ValueError):
        decode_symbols(b"XXXX" + raw[4:])


def test_huffman_grammar_roundtrip_and_varint_compatibility(tmp_path):
    edges = (
        EdgeRecord(0, 0, 1, "RelatedTo", 1.25),
        EdgeRecord(1, 1, 2, "IsA", 2.0),
        EdgeRecord(2, 3, 4, "RelatedTo", 3.0),
        EdgeRecord(3, 4, 5, "IsA", 4.0),
        EdgeRecord(4, 2, 3, "UsedFor", 5.0),
    )
    for coding in ("varint", "huffman"):
        path = tmp_path / f"{coding}.zip"
        result = encode_grammar(6, edges, ((0, 1, 2), (3, 4, 5)), path,
                                symbol_coding=coding)
        assert result["roundtrip_exact"]
        assert result["symbol_coding"] == coding
        with zipfile.ZipFile(path) as archive:
            assert ("symbols.huff" in archive.namelist()) == (coding == "huffman")
        assert decode_grammar(path) == (6, edges)
