"""Time primitives.

Design principle P2: *"Time is the adversary, not a column."*

Two rules are enforced here and relied upon everywhere else:

1. **Every timestamp is UTC.** Naive datetimes are rejected at the boundary
   rather than silently localised, because a naive datetime that picks up
   ``Europe/Berlin`` on one machine and UTC on another is a leakage bug that
   only shows up in production.
2. **The wire format is integer milliseconds since the Unix epoch.** Floats
   lose precision at second-level magnitudes and make ``<`` comparisons on
   event time subtly non-deterministic; point-in-time correctness depends on
   those comparisons being exact.
"""

from __future__ import annotations

import datetime as _dt
from typing import Final

UTC: Final = _dt.UTC
EPOCH: Final = _dt.datetime(1970, 1, 1, tzinfo=UTC)

MILLIS_PER_SECOND: Final = 1_000
MILLIS_PER_MINUTE: Final = 60 * MILLIS_PER_SECOND
MILLIS_PER_HOUR: Final = 60 * MILLIS_PER_MINUTE
MILLIS_PER_DAY: Final = 24 * MILLIS_PER_HOUR


class NaiveDatetimeError(ValueError):
    """Raised when a datetime without tzinfo reaches a boundary that needs UTC."""


def ensure_utc(value: _dt.datetime) -> _dt.datetime:
    """Return *value* as UTC, refusing naive datetimes.

    We refuse rather than assume. Assuming UTC for a naive datetime is how a
    two-hour offset silently becomes a two-hour lookahead in a feature window.
    """
    if value.tzinfo is None:
        raise NaiveDatetimeError(
            f"naive datetime {value!r}: attach tzinfo explicitly "
            "(kg_core.timeutil.UTC) rather than relying on the host timezone"
        )
    return value.astimezone(UTC)


def to_millis(value: _dt.datetime) -> int:
    """UTC datetime -> integer epoch milliseconds (truncating, never rounding).

    Truncation is deliberate: rounding could push an event's timestamp *forward*
    past a window boundary, which is the one direction that creates leakage.
    """
    delta = ensure_utc(value) - EPOCH
    return (
        (delta.days * MILLIS_PER_DAY)
        + (delta.seconds * MILLIS_PER_SECOND)
        + (delta.microseconds // 1000)
    )


def from_millis(millis: int) -> _dt.datetime:
    """Integer epoch milliseconds -> aware UTC datetime."""
    return EPOCH + _dt.timedelta(milliseconds=int(millis))


def parse_millis(value: str | int | float | _dt.datetime) -> int:
    """Best-effort coercion of the usual suspects into epoch milliseconds."""
    if isinstance(value, _dt.datetime):
        return to_millis(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = value.strip()
    if text.isdigit():
        return int(text)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return to_millis(_dt.datetime.fromisoformat(text))


def iso(millis: int) -> str:
    """Epoch milliseconds -> RFC3339 string with millisecond precision."""
    return from_millis(millis).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def day_start_millis(millis: int) -> int:
    """Truncate to the start of the UTC day containing *millis*."""
    return (millis // MILLIS_PER_DAY) * MILLIS_PER_DAY


def duration_millis(
    *, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0, millis: int = 0
) -> int:
    """Compose a duration in milliseconds. Keeps window definitions readable."""
    return (
        days * MILLIS_PER_DAY
        + hours * MILLIS_PER_HOUR
        + minutes * MILLIS_PER_MINUTE
        + seconds * MILLIS_PER_SECOND
        + millis
    )


_WINDOW_UNITS: Final[dict[str, int]] = {
    "ms": 1,
    "s": MILLIS_PER_SECOND,
    "m": MILLIS_PER_MINUTE,
    "h": MILLIS_PER_HOUR,
    "d": MILLIS_PER_DAY,
}


def parse_window(spec: str) -> int:
    """Parse a compact window spec such as ``"7d"``, ``"90m"``, ``"500ms"``.

    Used by the feature DSL so that a window is written once, in one syntax, and
    lowered identically into DuckDB SQL, Flink SQL and the online executor.
    """
    text = spec.strip().lower()
    for unit in ("ms", "s", "m", "h", "d"):
        if text.endswith(unit):
            number = text[: -len(unit)]
            if not number:
                break
            try:
                magnitude = float(number)
            except ValueError:
                break
            if magnitude <= 0:
                raise ValueError(f"window {spec!r} must be strictly positive")
            return int(magnitude * _WINDOW_UNITS[unit])
    raise ValueError(
        f"cannot parse window {spec!r}; expected e.g. '30s', '15m', '24h', '7d', '500ms'"
    )


def format_window(millis: int) -> str:
    """Inverse of :func:`parse_window`, choosing the largest exact unit."""
    for unit, size in (
        ("d", MILLIS_PER_DAY),
        ("h", MILLIS_PER_HOUR),
        ("m", MILLIS_PER_MINUTE),
        ("s", MILLIS_PER_SECOND),
    ):
        if millis % size == 0 and millis >= size:
            return f"{millis // size}{unit}"
    return f"{millis}ms"
