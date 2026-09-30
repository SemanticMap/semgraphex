"""Binary primitives shared by the exact grammar codec.

The functions are deliberately small and deterministic. They encode unsigned
integers as varints and preserve floating-point weights by serializing their
exact IEEE-754 binary64 bit patterns.
"""
from __future__ import annotations

import io
import struct
from typing import BinaryIO


def encode_uvarint(value: int) -> bytes:
    value = int(value)
    if value < 0:
        raise ValueError("uvarint requires a non-negative integer")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def write_uvarint(stream: BinaryIO, value: int) -> None:
    stream.write(encode_uvarint(value))


def read_uvarint(stream: BinaryIO) -> int:
    value = 0
    shift = 0
    for _ in range(10):
        raw = stream.read(1)
        if not raw:
            raise ValueError("truncated uvarint")
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value
        shift += 7
    raise ValueError("uvarint exceeds 64 bits")


def float_to_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def bits_to_float(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]


def write_u64(stream: BinaryIO, value: int) -> None:
    stream.write(struct.pack("<Q", int(value)))


def read_u64(stream: BinaryIO) -> int:
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("truncated uint64")
    return int(struct.unpack("<Q", raw)[0])


def encoded_uvarint_size(value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError("uvarint requires a non-negative integer")
    size = 1
    while value >= 0x80:
        value >>= 7
        size += 1
    return size


def bytes_reader(data: bytes) -> io.BytesIO:
    return io.BytesIO(data)
