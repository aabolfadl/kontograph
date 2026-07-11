"""The Confluent Schema Registry wire format.

Every Avro message on the topic is framed as:

    +-------------+------------------+---------------------+
    | magic byte  | schema id        | Avro binary payload |
    | 1 byte 0x00 | 4 bytes, big-end | remainder           |
    +-------------+------------------+---------------------+

Implemented here rather than pulled from ``confluent-kafka[avro]`` for two
reasons. First, the framing is the *contract* between producer and consumer:
five lines of code that decide whether historical data stays readable, and that
belongs in the repository where a reviewer can see it. Second, the serializer in
the Confluent extra couples serialisation to its own registry client and caching
strategy, and we want the schema id resolution to be explicit — it is what makes
a message decodable years later against a schema nobody remembers registering.

Redpanda's built-in registry implements the same API, so nothing here is
Redpanda-specific (ADR-0001).
"""

from __future__ import annotations

import io
import struct
from typing import Any, Final

import fastavro

MAGIC_BYTE: Final = 0
HEADER_FORMAT: Final = ">bI"  # magic byte, then unsigned 32-bit big-endian id
HEADER_SIZE: Final = struct.calcsize(HEADER_FORMAT)


class WireFormatError(ValueError):
    """Raised when a message does not carry a valid Confluent frame."""


def encode(schema: Any, schema_id: int, record: dict[str, Any]) -> bytes:
    """Frame *record* as magic byte + schema id + Avro binary."""
    if schema_id < 0 or schema_id > 0xFFFF_FFFF:
        raise WireFormatError(f"schema id {schema_id} does not fit in 4 bytes")
    buffer = io.BytesIO()
    buffer.write(struct.pack(HEADER_FORMAT, MAGIC_BYTE, schema_id))
    fastavro.schemaless_writer(buffer, schema, record)
    return buffer.getvalue()


def peek_schema_id(payload: bytes) -> int:
    """Read the schema id without decoding the body.

    Useful for routing and for diagnosing a poison message: knowing *which*
    schema a bad record claims is most of the investigation.
    """
    if len(payload) < HEADER_SIZE:
        raise WireFormatError(
            f"payload is {len(payload)} bytes, too short for a {HEADER_SIZE}-byte frame"
        )
    magic, schema_id = struct.unpack(HEADER_FORMAT, payload[:HEADER_SIZE])
    if magic != MAGIC_BYTE:
        raise WireFormatError(
            f"expected magic byte 0x00, got 0x{magic & 0xFF:02x} — this message was "
            "not produced with Schema Registry framing"
        )
    return int(schema_id)


def decode(payload: bytes, reader_schema: Any, writer_schema: Any | None = None) -> dict[str, Any]:
    """Decode a framed message.

    Passing ``writer_schema`` invokes Avro schema resolution, which is what lets
    a consumer on a newer contract read messages produced under an older one.
    That path is the entire point of the backward-compatibility gate in
    ``kg_core.contracts.registry``, so it is exercised here rather than assumed.
    """
    peek_schema_id(payload)
    body = io.BytesIO(payload[HEADER_SIZE:])
    if writer_schema is None:
        return fastavro.schemaless_reader(body, reader_schema)  # type: ignore[return-value]
    return fastavro.schemaless_reader(body, writer_schema, reader_schema)  # type: ignore[return-value]
