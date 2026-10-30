"""Canonical Huffman coding for exact grammar occurrence symbol IDs.

Separate bitstream deliberately: a code table + packed symbol bits. DEFLATE
still compresses the archive envelope; the experiment measures both costs.
"""
from __future__ import annotations

import io
from collections import Counter
from typing import Sequence

from .grammar_binary import read_uvarint, write_uvarint
from .graph_mdl import build_canonical_huffman_codes

MAGIC = b"SHF1"


def _canonical(lengths: dict[int, int]) -> dict[int, str]:
    out = {}
    code = 0
    previous = 0
    for symbol, length in sorted(lengths.items(), key=lambda pair: (pair[1], str(pair[0]))):
        if not 1 <= length <= 64:
            raise ValueError("invalid Huffman code length")
        code <<= length - previous
        if code >= (1 << length):
            raise ValueError("oversubscribed Huffman code lengths")
        out[symbol] = format(code, f"0{length}b")
        code += 1
        previous = length
    return out


def encode_symbols(symbols: Sequence[int]) -> bytes:
    """Store canonical code lengths and exact MSB-first packed symbol stream."""
    ids = tuple(int(s) for s in symbols)
    if any(s < 0 for s in ids):
        raise ValueError("Huffman symbol IDs must be nonnegative")
    raw_codes = build_canonical_huffman_codes({str(k): v for k, v in Counter(ids).items()})
    lengths = {int(k): len(v) for k, v in raw_codes.items()}
    codes = _canonical(lengths)
    output = io.BytesIO()
    output.write(MAGIC)
    write_uvarint(output, len(ids))
    write_uvarint(output, len(lengths))
    for symbol in sorted(lengths):
        write_uvarint(output, symbol)
        write_uvarint(output, lengths[symbol])
    bit_length = sum(lengths[s] for s in ids)
    write_uvarint(output, bit_length)
    byte = 0
    used = 0
    for symbol in ids:
        for bit in codes[symbol]:
            byte = (byte << 1) | (bit == "1")
            used += 1
            if used == 8:
                output.write(bytes((byte,)))
                byte = used = 0
    if used:
        output.write(bytes((byte << (8 - used),)))
    return output.getvalue()


def decode_symbols(data: bytes) -> tuple[int, ...]:
    stream = io.BytesIO(data)
    if stream.read(4) != MAGIC:
        raise ValueError("invalid Huffman magic")
    count = read_uvarint(stream)
    size = read_uvarint(stream)
    if (count == 0) != (size == 0) or size > count:
        raise ValueError("invalid Huffman alphabet cardinality")
    lengths = {}
    for _ in range(size):
        symbol = read_uvarint(stream)
        length = read_uvarint(stream)
        if symbol in lengths:
            raise ValueError("duplicate Huffman symbol")
        lengths[symbol] = length
    by_code = {code: symbol for symbol, code in _canonical(lengths).items()}
    nbits = read_uvarint(stream)
    if nbits < count or nbits > count * 64:
        if count or nbits:
            raise ValueError("invalid Huffman bit count")
    packed = stream.read()
    if len(packed) != (nbits + 7) // 8:
        raise ValueError("Huffman bitstream truncated or trailing bytes")
    if nbits % 8 and packed[-1] & ((1 << (8 - nbits % 8)) - 1):
        raise ValueError("nonzero Huffman padding")
    result = []
    prefix = ""
    for i in range(nbits):
        prefix += "1" if packed[i // 8] & (1 << (7 - i % 8)) else "0"
        if prefix in by_code:
            result.append(by_code[prefix])
            prefix = ""
            if len(result) > count:
                raise ValueError("too many Huffman symbols")
        elif len(prefix) > 64:
            raise ValueError("Huffman prefix exceeds limit")
    if prefix or len(result) != count:
        raise ValueError("incomplete Huffman code stream")
    return tuple(result)
