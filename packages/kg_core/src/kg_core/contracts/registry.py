"""Schema loading, Avro (de)serialisation, and the CI compatibility gate.

The platform scope: *"Producer-side schema ownership ... backward-compat gates in
CI."* The gate lives here rather than in a Schema Registry plugin so it runs in
CI without a broker, and so the *rules* are readable in the repository instead
of being a vendor setting somebody once clicked.

Compatibility direction: **BACKWARD** — a consumer running the new schema must
be able to read data produced under the old one. That is the right default for
a payments tap, where producers are upgraded last and historical Iceberg
partitions must stay readable forever.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Final

import fastavro
import fastavro.read

SCHEMA_DIR: Final = Path(__file__).parent / "avro"


# ---------------------------------------------------------------------------
# Keep timestamps as integer epoch milliseconds on decode.
#
# By default fastavro converts an Avro `timestamp-millis` logical type into a
# Python datetime when reading. That is helpful in general and wrong here: this
# project's core invariant is that event time is an integer millisecond value
# from the wire all the way through the feature layer (see kg_core.timeutil).
# A silent int -> datetime conversion at the Kafka boundary reintroduces exactly
# the timezone ambiguity that invariant exists to eliminate, and it does so
# invisibly -- a comparison against an int simply starts returning False.
#
# The logical type stays declared in the .avsc, because it is semantically
# correct and other engines (Flink, Spark, DuckDB) should read these columns as
# timestamps. Only *our* decode path opts out, using fastavro's documented
# extension point.
#
# Found by a round-trip check that compared the Kafka message timestamp with the
# decoded acceptance_ts and got False for what should have been identical values.
# ---------------------------------------------------------------------------
def _passthrough_millis(data: Any, writer_schema: Any, reader_schema: Any) -> Any:
    return data


fastavro.read.LOGICAL_READERS["long-timestamp-millis"] = _passthrough_millis
fastavro.read.LOGICAL_READERS["long-timestamp-micros"] = _passthrough_millis

#: Kafka subject naming: TopicNameStrategy with an explicit value suffix.
SUBJECT_SUFFIX: Final = "-value"

_PROMOTABLE: Final[dict[str, set[str]]] = {
    # Avro's documented type-promotion rules for schema resolution.
    "int": {"long", "float", "double"},
    "long": {"float", "double"},
    "float": {"double"},
    "string": {"bytes"},
    "bytes": {"string"},
}


class IncompatibleSchemaError(Exception):
    """Raised when a schema change would break existing consumers or data."""


def schema_path(name: str) -> Path:
    """Resolve a schema stem such as ``"sct_inst_event.v1"`` to its file."""
    path = SCHEMA_DIR / f"{name}.avsc"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in SCHEMA_DIR.glob("*.avsc")))
        raise FileNotFoundError(f"no Avro schema {name!r}; available: {available}")
    return path


def load_schema_dict(name: str) -> dict[str, Any]:
    """Load a schema as raw JSON (unparsed), for compatibility analysis."""
    return json.loads(schema_path(name).read_text(encoding="utf-8"))


def load_schema(name: str) -> Any:
    """Load and parse a schema for (de)serialisation."""
    return fastavro.parse_schema(load_schema_dict(name))


def list_schemas() -> list[str]:
    """Every contract stem in the repository, sorted."""
    return sorted(p.stem for p in SCHEMA_DIR.glob("*.avsc"))


def subject_for(name: str) -> str:
    """Schema Registry subject for a contract stem."""
    return f"{load_schema_dict(name)['name']}{SUBJECT_SUFFIX}"


# ------------------------------------------------------- serialisation -----


def encode(name: str, record: dict[str, Any]) -> bytes:
    """Serialise one record to Avro binary (no registry framing)."""
    buffer = io.BytesIO()
    fastavro.schemaless_writer(buffer, load_schema(name), record)
    return buffer.getvalue()


def decode(name: str, payload: bytes, *, writer_schema: str | None = None) -> dict[str, Any]:
    """Deserialise Avro binary, optionally resolving from an older writer schema.

    Passing ``writer_schema`` exercises Avro's schema-resolution path, which is
    exactly what a consumer does when reading an old Iceberg partition.
    """
    reader = load_schema(name)
    writer = load_schema(writer_schema) if writer_schema else reader
    return fastavro.schemaless_reader(io.BytesIO(payload), writer, reader)  # type: ignore[return-value]


# --------------------------------------------------- compatibility gate ----


def _fields_by_name(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["name"]: f for f in schema.get("fields", [])}


def _has_default(field: dict[str, Any]) -> bool:
    return "default" in field


def _type_name(type_spec: Any) -> str:
    """Reduce a field's type to a comparable name, unwrapping nullable unions."""
    if isinstance(type_spec, str):
        return type_spec
    if isinstance(type_spec, list):
        non_null = [t for t in type_spec if t != "null"]
        if len(non_null) == 1:
            return _type_name(non_null[0])
        return "union[" + ",".join(sorted(_type_name(t) for t in type_spec)) + "]"
    if isinstance(type_spec, dict):
        kind = type_spec.get("type")
        if kind in ("enum", "record", "fixed"):
            return f"{kind}:{type_spec.get('name')}"
        if kind == "array":
            return f"array<{_type_name(type_spec.get('items'))}>"
        if kind == "map":
            return f"map<{_type_name(type_spec.get('values'))}>"
        return str(kind)
    return str(type_spec)


def _is_nullable(type_spec: Any) -> bool:
    return isinstance(type_spec, list) and "null" in type_spec


def _enum_symbols(type_spec: Any) -> list[str] | None:
    if isinstance(type_spec, dict) and type_spec.get("type") == "enum":
        return list(type_spec.get("symbols", []))
    if isinstance(type_spec, list):
        for member in type_spec:
            symbols = _enum_symbols(member)
            if symbols is not None:
                return symbols
    return None


def check_backward_compatible(new: dict[str, Any], old: dict[str, Any]) -> list[str]:
    """Return the list of backward-incompatibility reasons; empty means compatible.

    Returning reasons rather than a bool is the point: a CI failure that says
    *which* field broke *how* is actionable, and a red build that only says
    "incompatible" trains people to bypass the gate.
    """
    problems: list[str] = []

    if new.get("name") != old.get("name"):
        problems.append(f"record renamed: {old.get('name')!r} -> {new.get('name')!r}")
    if new.get("namespace") != old.get("namespace"):
        problems.append(f"namespace changed: {old.get('namespace')!r} -> {new.get('namespace')!r}")

    new_fields, old_fields = _fields_by_name(new), _fields_by_name(old)

    # A field the new reader requires but old data does not carry is
    # unreadable unless the reader supplies a default.
    for name, field in new_fields.items():
        if name not in old_fields and not _has_default(field):
            problems.append(f"field {name!r} added without a default: old records cannot be read")

    for name, old_field in old_fields.items():
        new_field = new_fields.get(name)
        if new_field is None:
            # Dropping a field is fine for reading old data (it is ignored),
            # but it silently deletes information from every downstream mart.
            problems.append(
                f"field {name!r} removed: readable, but historical data becomes "
                "inaccessible through this contract - bump the major version instead"
            )
            continue

        old_type, new_type = _type_name(old_field["type"]), _type_name(new_field["type"])
        if old_type != new_type:
            promotable = new_type in _PROMOTABLE.get(old_type, set())
            widened_to_nullable = _is_nullable(new_field["type"]) and not _is_nullable(
                old_field["type"]
            )
            if not (promotable or widened_to_nullable):
                problems.append(
                    f"field {name!r} type changed {old_type} -> {new_type}, "
                    "which is not an Avro-promotable widening"
                )

        old_symbols, new_symbols = (
            _enum_symbols(old_field["type"]),
            _enum_symbols(new_field["type"]),
        )
        if old_symbols and new_symbols:
            dropped = [s for s in old_symbols if s not in new_symbols]
            if dropped and "default" not in (
                new_field["type"] if isinstance(new_field["type"], dict) else {}
            ):
                problems.append(
                    f"field {name!r} dropped enum symbol(s) {dropped}: old records "
                    "carrying them become undecodable"
                )

    return problems


def assert_backward_compatible(new: dict[str, Any], old: dict[str, Any]) -> None:
    """Raise :class:`IncompatibleSchemaError` listing every reason at once."""
    problems = check_backward_compatible(new, old)
    if problems:
        bullets = "\n".join(f"  - {p}" for p in problems)
        raise IncompatibleSchemaError(
            f"schema {new.get('name')!r} is not backward compatible:\n{bullets}"
        )


# ------------------------------------------------------- drift detection ---


def avro_field_names(name: str) -> set[str]:
    """Top-level field names declared in an Avro contract."""
    return set(_fields_by_name(load_schema_dict(name)))


def field_parity(name: str, model: type) -> tuple[set[str], set[str]]:
    """Compare an Avro contract with its pydantic mirror.

    Returns ``(only_in_avro, only_in_model)``. The parity test asserts both are
    empty, which is what stops the two representations drifting apart.
    """
    avro_fields = avro_field_names(name)
    model_fields = set(getattr(model, "model_fields", {}))
    return avro_fields - model_fields, model_fields - avro_fields


def schema_fingerprint(name: str) -> str:
    """Stable hash of a contract, for stamping onto artifacts and decisions."""
    from kg_core.hashing import content_hash

    return content_hash(load_schema_dict(name), prefix="avro-schema")
