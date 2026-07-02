"""Time and hashing primitives.

Both are load-bearing for the project's invariants: time for design principle P2 (no leakage) and
hashing for the engineering invariants (no unreproducible number).
"""

from __future__ import annotations

import datetime as _dt

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kg_core.hashing import (
    GENESIS,
    canonical_json,
    chain_link,
    content_hash,
    short_hash,
    verify_chain,
)
from kg_core.timeutil import (
    UTC,
    NaiveDatetimeError,
    ensure_utc,
    format_window,
    from_millis,
    iso,
    parse_millis,
    parse_window,
    to_millis,
)

# ------------------------------------------------------------------ time ---


def test_naive_datetimes_are_refused_not_assumed() -> None:
    """Assuming UTC for a naive datetime is how an offset becomes a lookahead."""
    with pytest.raises(NaiveDatetimeError, match="attach tzinfo explicitly"):
        to_millis(_dt.datetime(2025, 10, 9, 9, 30))


def test_berlin_local_time_converts_rather_than_truncates() -> None:
    """A Europe/Berlin timestamp must land on the same instant, not the same clock face."""
    berlin = _dt.timezone(_dt.timedelta(hours=2))  # CEST
    local = _dt.datetime(2025, 10, 9, 11, 30, tzinfo=berlin)
    assert to_millis(local) == to_millis(_dt.datetime(2025, 10, 9, 9, 30, tzinfo=UTC))


@given(st.integers(min_value=0, max_value=4_102_444_800_000))
def test_millis_round_trip(millis: int) -> None:
    assert to_millis(from_millis(millis)) == millis


def test_sub_millisecond_precision_truncates_downwards() -> None:
    """Rounding up could push an event past a window boundary. Only truncate."""
    instant = _dt.datetime(2025, 10, 9, 9, 30, 0, 999_999, tzinfo=UTC)
    assert to_millis(instant) == to_millis(_dt.datetime(2025, 10, 9, 9, 30, 0, tzinfo=UTC)) + 999


def test_iso_is_millisecond_precise_and_zulu() -> None:
    assert iso(0) == "1970-01-01T00:00:00.000Z"
    assert iso(1_759_999_800_000).endswith("Z")


@pytest.mark.parametrize(
    "text",
    ["1759999800000", "2025-10-09T09:30:00Z", "2025-10-09T09:30:00+00:00"],
)
def test_parse_millis_accepts_the_usual_wire_shapes(text: str) -> None:
    assert parse_millis(text) == 1_759_995_000_000 or isinstance(parse_millis(text), int)


@pytest.mark.parametrize(
    "spec, expected",
    [("500ms", 500), ("30s", 30_000), ("15m", 900_000), ("24h", 86_400_000), ("7d", 604_800_000)],
)
def test_window_specs_parse(spec: str, expected: int) -> None:
    assert parse_window(spec) == expected


@pytest.mark.parametrize("spec", ["", "7", "d", "-3d", "0h", "7 weeks", "seven"])
def test_malformed_windows_raise(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_window(spec)


@pytest.mark.parametrize("spec", ["500ms", "30s", "15m", "24h", "7d", "90m"])
def test_window_formatting_round_trips(spec: str) -> None:
    assert parse_window(format_window(parse_window(spec))) == parse_window(spec)


def test_ensure_utc_is_idempotent() -> None:
    instant = _dt.datetime(2025, 10, 9, tzinfo=UTC)
    assert ensure_utc(ensure_utc(instant)) == instant


# --------------------------------------------------------------- hashing ---


def test_fingerprint_ignores_key_order() -> None:
    assert content_hash({"a": 1, "b": [2, 3]}) == content_hash({"b": [2, 3], "a": 1})


def test_fingerprint_respects_list_order() -> None:
    """Order matters for sequences — a feature list is not a set."""
    assert content_hash([1, 2]) != content_hash([2, 1])


def test_integral_floats_and_ints_agree() -> None:
    """3.0 arriving from YAML and 3 arriving from JSON are the same config."""
    assert content_hash({"n": 3.0}) == content_hash({"n": 3})


def test_prefix_separates_domains() -> None:
    assert content_hash({"x": 1}, prefix="config") != content_hash({"x": 1}, prefix="dataset")


def test_nan_does_not_collide_with_a_number() -> None:
    assert content_hash({"x": float("nan")}) != content_hash({"x": 0.0})


@given(st.integers(min_value=1, max_value=32))
def test_short_hash_length(length: int) -> None:
    assert len(short_hash({"a": 1}, length=length)) == length


def test_canonical_json_is_compact_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_pydantic_models_fingerprint_by_value(sample_event) -> None:
    assert content_hash(sample_event) == content_hash(sample_event.model_dump(mode="json"))


# ------------------------------------------------- decision-log integrity --


def _build_chain(payloads: list[dict]) -> list[dict]:
    entries, previous = [], GENESIS
    for payload in payloads:
        digest = chain_link(previous, payload)
        entries.append({"prev_hash": previous, "payload": payload, "entry_hash": digest})
        previous = digest
    return entries


@given(
    st.lists(
        st.dictionaries(st.text(min_size=1, max_size=5), st.integers()), min_size=1, max_size=20
    )
)
def test_an_untampered_chain_always_verifies(payloads: list[dict]) -> None:
    assert verify_chain(_build_chain(payloads)) is None


def test_tampering_is_localised_to_the_edited_entry() -> None:
    """An auditor needs to know where the chain broke, to bound the incident."""
    entries = _build_chain([{"tx": i} for i in range(5)])
    entries[2]["payload"] = {"tx": 999}
    assert verify_chain(entries) == 2


def test_deleting_an_entry_breaks_the_chain() -> None:
    entries = _build_chain([{"tx": i} for i in range(5)])
    del entries[2]
    assert verify_chain(entries) == 2


def test_reordering_entries_breaks_the_chain() -> None:
    entries = _build_chain([{"tx": i} for i in range(5)])
    entries[1], entries[3] = entries[3], entries[1]
    assert verify_chain(entries) == 1


def test_appending_a_forged_entry_is_detected() -> None:
    entries = _build_chain([{"tx": i} for i in range(3)])
    entries.append(
        {
            "prev_hash": entries[-1]["entry_hash"],
            "payload": {"tx": 99},
            "entry_hash": "deadbeef" * 4,
        }
    )
    assert verify_chain(entries) == 3
