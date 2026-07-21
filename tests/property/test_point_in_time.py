"""Point-in-time correctness and cross-backend equivalence.

This is the suite design principles P2 and P3 are about, and it is a required CI
check. Two properties, checked on Hypothesis-generated event streams:

1. **No feature can see the future.** Perturb every event at or after time ``t``
   and nothing computed for an event before ``t`` may move.
2. **The backfill and the online executor agree.** Value by value, on the same
   stream, within float tolerance.

The streams are generated adversarially rather than realistically: duplicate
timestamps, a tiny account pool so histories actually overlap, windows short
enough that boundary conditions bite, and simultaneous events on the same
entity. A realistic stream would pass these tests without exercising anything.
"""

from __future__ import annotations

import math
import os

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kg_features.compilers.online import OnlineFeatureExecutor
from kg_features.dsl import (
    count,
    distinct,
    maximum,
    mean,
    minimum,
    stddev,
    time_since_last,
    total,
    when,
)
from kg_features.ir import Entity, FeatureSet
from kg_features.registry import PAYMENTS_V1
from kg_features.verify import (
    assert_no_lookahead,
    compare_backends,
    run_duckdb,
    run_online,
)

# ---------------------------------------------------------------- streams ---

# A deliberately tiny pool: with 200 accounts every history is empty and the
# aggregations are never exercised. Four accounts guarantees collisions.
ACCOUNTS = ["A", "B", "C", "D"]
VOP = ["MATCH", "CLOSE_MATCH", "NO_MATCH", "UNAVAILABLE"]
CHANNELS = ["MOBILE_APP", "ONLINE_BANKING", "POS", "STANDING_ORDER"]


@st.composite
def event_streams(draw, min_size: int = 2, max_size: int = 40):
    """Adversarial payment streams.

    Timestamps are drawn from a small range so duplicates are common. Two events
    at the identical millisecond on the same account is the case that separates
    a right-open window from a right-closed one, and it is where the Flink
    subtraction correction breaks — so it must be frequent, not rare.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    timestamps = sorted(
        draw(
            st.lists(
                st.integers(min_value=1_000_000, max_value=1_000_000 + 20 * 3_600_000),
                min_size=size,
                max_size=size,
            )
        )
    )
    events = []
    for index, timestamp in enumerate(timestamps):
        debtor = draw(st.sampled_from(ACCOUNTS))
        creditor = draw(st.sampled_from([a for a in ACCOUNTS if a != debtor]))
        events.append(
            {
                "tx_id": f"tx{index:04d}",
                "acceptance_ts": timestamp,
                "debtor_account_id": debtor,
                "creditor_account_id": creditor,
                "amount_cents": draw(st.integers(min_value=1, max_value=5_000_000)),
                "vop_result": draw(st.sampled_from(VOP)),
                "vop_name_similarity": draw(
                    st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
                ),
                "vop_overridden": draw(st.booleans()),
                "is_new_beneficiary": draw(st.booleans()),
                "is_cross_border": draw(st.booleans()),
                "channel": draw(st.sampled_from(CHANNELS)),
            }
        )
    return events


# A compact set covering every aggregation, every entity, and both windowed and
# unbounded forms. Short windows so boundaries are hit constantly.
ALL_AGGREGATIONS = FeatureSet(
    name="coverage",
    features=(
        count("c_1h", Entity.DEBTOR, "1h"),
        count("c_filtered", Entity.DEBTOR, "2h", where=(when("vop_result", "!=", "MATCH"),)),
        count("c_unbounded", Entity.DEBTOR),
        total("s_1h", Entity.DEBTOR, "amount_cents", "1h"),
        mean("m_2h", Entity.DEBTOR, "amount_cents", "2h"),
        stddev("sd_2h", Entity.DEBTOR, "amount_cents", "2h"),
        maximum("mx_2h", Entity.DEBTOR, "amount_cents", "2h"),
        minimum("mn_2h", Entity.DEBTOR, "amount_cents", "2h"),
        distinct("d_creditors_2h", Entity.DEBTOR, "creditor_account_id", "2h"),
        distinct("d_channels_2h", Entity.DEBTOR, "channel", "2h"),
        time_since_last("tsl_debtor", Entity.DEBTOR),
        count("cr_count_1h", Entity.CREDITOR, "1h"),
        total("cr_sum_2h", Entity.CREDITOR, "amount_cents", "2h"),
        distinct("cr_distinct_2h", Entity.CREDITOR, "debtor_account_id", "2h"),
        time_since_last("cr_tsl", Entity.CREDITOR),
        count("pair_count_2h", Entity.PAIR, "2h"),
        total("pair_sum_2h", Entity.PAIR, "amount_cents", "2h"),
        time_since_last("pair_tsl", Entity.PAIR),
        mean("m_similarity", Entity.DEBTOR, "vop_name_similarity", "2h", default=1.0),
    ),
)

#: The expensive properties run a reduced budget locally and a 5x budget under
#: the CI profile, which is where the leakage gate is enforced.
SLOW = settings(
    max_examples=200 if os.environ.get("HYPOTHESIS_PROFILE") == "ci" else 40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ============================================== 1. no feature sees the future


@given(event_streams())
@SLOW
def test_backfill_cannot_see_the_future(events: list[dict]) -> None:
    """Perturb every later event; nothing earlier may move.

    This makes no assumption about *how* a leak might happen — it tests the
    property directly. A leak through a window bound, a join condition, an
    ordering mistake or a future-looking enrichment is caught identically.
    """
    assert_no_lookahead(ALL_AGGREGATIONS, events, backend="duckdb")


@given(event_streams())
@SLOW
def test_online_executor_cannot_see_the_future(events: list[dict]) -> None:
    assert_no_lookahead(ALL_AGGREGATIONS, events, backend="online")


@given(event_streams(min_size=4))
@SLOW
def test_production_feature_set_cannot_see_the_future(events: list[dict]) -> None:
    """The real registry, not just the coverage set."""
    assert_no_lookahead(PAYMENTS_V1, events, backend="duckdb")


def test_the_lookahead_detector_actually_detects_lookahead() -> None:
    """A test for the test.

    A leakage detector that always passes is worse than none, so it is pointed
    at a deliberately leaky feature set and must fail. The leak here is the
    classic one: an inclusive upper bound, so the event contributes to its own
    aggregate.
    """
    from kg_features.compilers.duckdb_sql import compile_feature_set

    leaky = FeatureSet(name="leaky", features=(count("c", Entity.DEBTOR, "2h"),))
    query = compile_feature_set(leaky).sql
    leaking_sql = query.replace(
        "h.acceptance_ts < e.acceptance_ts", "h.acceptance_ts <= e.acceptance_ts"
    )
    assert leaking_sql != query, "the strict bound should have been present to replace"

    import duckdb
    import pyarrow as pa

    events = [
        {
            "tx_id": f"tx{i}",
            "acceptance_ts": 1_000_000 + i * 60_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
        }
        for i in range(6)
    ]
    connection = duckdb.connect(":memory:")
    connection.register("events_input", pa.Table.from_pylist(events))
    connection.execute("CREATE OR REPLACE TEMP VIEW events AS SELECT * FROM events_input")
    leaked = connection.execute(leaking_sql).to_arrow_table().to_pylist()
    honest = connection.execute(query).to_arrow_table().to_pylist()
    connection.close()

    # The inclusive bound counts the event itself: every value is one higher.
    assert [r["c"] for r in leaked] == [r["c"] + 1 for r in honest]
    assert honest[0]["c"] == 0, "the first event has no history"
    assert leaked[0]["c"] == 1, "the leaky variant counts the event itself"


# =========================================== 2. the backends must agree ======


@given(event_streams())
@SLOW
def test_backfill_and_online_agree(events: list[dict]) -> None:
    """The anti-skew claim, as an executable assertion.

    A disagreement here is training-serving skew: the model would be trained on
    one number and served another. design principle P3 says these must be the same
    computation, and this is what makes that checkable instead of aspirational.
    """
    compare_backends(ALL_AGGREGATIONS, events).raise_if_diverged()


@given(event_streams(min_size=4))
@SLOW
def test_production_feature_set_agrees_across_backends(events: list[dict]) -> None:
    compare_backends(PAYMENTS_V1, events).raise_if_diverged()


# ======================================= 3. boundary and ordering semantics ==


def test_window_is_left_closed_right_open() -> None:
    """``[t - W, t)`` — stated in the IR, verified here.

    Three events placed exactly on the boundary: one just outside the window,
    one exactly at ``t - W`` (included), and one at ``t`` itself (excluded).
    Each of the three is an off-by-one somebody has shipped.
    """
    window_ms = 3_600_000
    now = 10_000_000
    events = [
        {
            "tx_id": "too_old",
            "acceptance_ts": now - window_ms - 1,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
        },
        {
            "tx_id": "exactly_at_edge",
            "acceptance_ts": now - window_ms,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
        },
        {
            "tx_id": "inside",
            "acceptance_ts": now - 1,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
        },
        {
            "tx_id": "simultaneous",
            "acceptance_ts": now,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
        },
        {
            "tx_id": "subject",
            "acceptance_ts": now,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
        },
    ]
    feature_set = FeatureSet(name="edge", features=(count("c", Entity.DEBTOR, "1h"),))

    for label, result in (
        ("duckdb", run_duckdb(feature_set, events)),
        ("online", run_online(feature_set, events)),
    ):
        # subject's window is [now - W, now). It must contain exactly:
        #   exactly_at_edge  (at now - W)      -> left-closed, INCLUDED
        #   inside           (at now - 1)                      INCLUDED
        # and must exclude:
        #   too_old          (at now - W - 1)  -> one ms too early
        #   simultaneous     (at now)          -> right-open
        #   subject itself                     -> strict
        assert result["subject"]["c"] == 2, (
            f"{label}: expected the boundary event and the inside event, got "
            f"{result['subject']['c']}"
        )

    # Now shift the boundary event one millisecond earlier. If the window were
    # left-OPEN it would already have been excluded and this would change
    # nothing; because it is left-closed, the count drops.
    shifted = [
        {**e, "acceptance_ts": e["acceptance_ts"] - 1} if e["tx_id"] == "exactly_at_edge" else e
        for e in events
    ]
    for label, result in (
        ("duckdb", run_duckdb(feature_set, shifted)),
        ("online", run_online(feature_set, shifted)),
    ):
        assert result["subject"]["c"] == 1, (
            f"{label}: moving the boundary event 1ms earlier must drop it out of "
            f"the window, got {result['subject']['c']}"
        )


def test_simultaneous_events_never_see_each_other() -> None:
    """Two payments in the same millisecond must not contribute to each other.

    A right-closed window would have each counting the other, which is a
    symmetric, self-consistent, and completely wrong answer — the kind that
    survives a smoke test.
    """
    events = [
        {
            "tx_id": f"tx{i}",
            "acceptance_ts": 5_000_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
        }
        for i in range(4)
    ]
    feature_set = FeatureSet(name="sim", features=(count("c", Entity.DEBTOR, "1h"),))
    for result in (run_duckdb(feature_set, events), run_online(feature_set, events)):
        assert {row["c"] for row in result.values()} == {0.0}


def test_online_executor_rejects_out_of_order_events() -> None:
    """Out-of-order arrival would compute features against a partly-future state."""
    executor = OnlineFeatureExecutor(ALL_AGGREGATIONS)
    events = [
        {
            "tx_id": "a",
            "acceptance_ts": 2_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 1,
        },
        {
            "tx_id": "b",
            "acceptance_ts": 1_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 1,
        },
    ]
    with pytest.raises(ValueError, match="non-decreasing event-time order"):
        executor.run(events)


def test_compute_does_not_mutate_state() -> None:
    """Re-scoring the same event must return the same answer.

    The explanation-stability metric in M5 re-scores an alert to check that the
    story does not change. If ``compute`` folded the event into state, the
    second call would differ from the first and stability would be unmeasurable.
    """
    executor = OnlineFeatureExecutor(ALL_AGGREGATIONS)
    event = {
        "tx_id": "x",
        "acceptance_ts": 1_000,
        "debtor_account_id": "A",
        "creditor_account_id": "B",
        "amount_cents": 500,
        "vop_result": "MATCH",
        "vop_name_similarity": 1.0,
        "channel": "MOBILE_APP",
    }
    first = executor.compute(event)
    second = executor.compute(event)
    assert first == second
    assert executor.tracked_entities == 0, "compute must not create state"

    executor.observe(event)
    assert executor.tracked_entities == 3, "observe tracks debtor, creditor and pair"


@given(event_streams())
@SLOW
def test_feature_values_are_independent_of_input_ordering(events: list[dict]) -> None:
    """The backfill must not depend on the order rows happen to be stored in.

    A backfill whose correctness depends on input ordering is one that will
    eventually be run on unsorted input — after a repartition, a parallel read,
    or a compaction.
    """
    import random

    from kg_features.verify import _equal

    shuffled = list(events)
    random.Random(12345).shuffle(shuffled)
    original = run_duckdb(ALL_AGGREGATIONS, events)
    reordered = run_duckdb(ALL_AGGREGATIONS, shuffled)

    assert set(original) == set(reordered)
    for tx_id, row in original.items():
        for name in ALL_AGGREGATIONS.column_names:
            # Compared with tolerance, not bit-exactly. DuckDB's parallel
            # aggregation sums in input order, so STDDEV_POP and AVG differ in
            # the last bits when rows arrive shuffled -- measured at 8.9e-16,
            # which is machine epsilon and not a semantic difference. Demanding
            # bit equality here would be asserting something SQL does not
            # promise; what matters is that no VALUE changes.
            assert _equal(row[name], reordered[tx_id][name]), (
                f"{name} @ {tx_id}: {row[name]} vs {reordered[tx_id][name]}"
            )


# ================================================ 4. aggregation semantics ===


def test_empty_history_yields_the_declared_default() -> None:
    """The first event for an entity has no history, so every feature defaults."""
    from kg_features.dsl import NEVER_SEEN_MILLIS

    events = [
        {
            "tx_id": "first",
            "acceptance_ts": 1_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 999,
            "vop_result": "MATCH",
            "vop_name_similarity": 0.5,
            "channel": "POS",
        }
    ]
    for result in (run_duckdb(ALL_AGGREGATIONS, events), run_online(ALL_AGGREGATIONS, events)):
        row = result["first"]
        assert row["c_1h"] == 0
        assert row["s_1h"] == 0
        assert row["mx_2h"] == 0
        assert row["tsl_debtor"] == NEVER_SEEN_MILLIS, (
            "never-seen must read as 'very long ago', not as 'just now'"
        )
        assert row["m_similarity"] == 1.0, "per-feature defaults must be honoured"


def test_population_stddev_of_a_single_prior_event_is_zero() -> None:
    """Sample stddev would be undefined here and each engine invents an answer."""
    events = [
        {
            "tx_id": "a",
            "acceptance_ts": 1_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 100,
        },
        {
            "tx_id": "b",
            "acceptance_ts": 2_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 500,
        },
    ]
    feature_set = FeatureSet(
        name="sd", features=(stddev("sd", Entity.DEBTOR, "amount_cents", "2h"),)
    )
    for result in (run_duckdb(feature_set, events), run_online(feature_set, events)):
        assert result["b"]["sd"] == pytest.approx(0.0)


def test_distinct_count_ignores_nulls_like_sql_does() -> None:
    events = [
        {
            "tx_id": "a",
            "acceptance_ts": 1_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 1,
            "channel": None,
        },
        {
            "tx_id": "b",
            "acceptance_ts": 2_000,
            "debtor_account_id": "A",
            "creditor_account_id": "C",
            "amount_cents": 1,
            "channel": "POS",
        },
        {
            "tx_id": "c",
            "acceptance_ts": 3_000,
            "debtor_account_id": "A",
            "creditor_account_id": "D",
            "amount_cents": 1,
            "channel": None,
        },
    ]
    feature_set = FeatureSet(name="dc", features=(distinct("d", Entity.DEBTOR, "channel", "2h"),))
    for result in (run_duckdb(feature_set, events), run_online(feature_set, events)):
        assert result["c"]["d"] == 1, "two NULLs and one POS is one distinct value"


def test_filters_treat_null_as_non_matching_like_sql() -> None:
    """SQL comparisons against NULL are false; the online path must match."""
    events = [
        {
            "tx_id": "a",
            "acceptance_ts": 1_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 1,
            "vop_result": None,
        },
        {
            "tx_id": "b",
            "acceptance_ts": 2_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 1,
            "vop_result": "NO_MATCH",
        },
        {
            "tx_id": "c",
            "acceptance_ts": 3_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 1,
            "vop_result": "MATCH",
        },
    ]
    feature_set = FeatureSet(
        name="f",
        features=(count("c", Entity.DEBTOR, "2h", where=(when("vop_result", "!=", "MATCH"),)),),
    )
    for result in (run_duckdb(feature_set, events), run_online(feature_set, events)):
        assert result["c"]["c"] == 1, "the NULL row must not match '!= MATCH'"


def test_pair_entity_distinguishes_direction() -> None:
    """A->B and B->A are different relationships, not the same one."""
    events = [
        {
            "tx_id": "ab",
            "acceptance_ts": 1_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 1,
        },
        {
            "tx_id": "ba",
            "acceptance_ts": 2_000,
            "debtor_account_id": "B",
            "creditor_account_id": "A",
            "amount_cents": 1,
        },
        {
            "tx_id": "ab2",
            "acceptance_ts": 3_000,
            "debtor_account_id": "A",
            "creditor_account_id": "B",
            "amount_cents": 1,
        },
    ]
    feature_set = FeatureSet(name="p", features=(count("c", Entity.PAIR, "2h"),))
    for result in (run_duckdb(feature_set, events), run_online(feature_set, events)):
        assert result["ba"]["c"] == 0, "B->A has no prior history of its own"
        assert result["ab2"]["c"] == 1, "A->B has exactly one prior"


# ==================================================== 5. the feature set =====


def test_max_window_defines_the_purge_width() -> None:
    """The evaluation protocol purges by this number; it must be real."""
    assert PAYMENTS_V1.max_window_millis == 30 * 86_400_000


def test_no_feature_reads_a_label_or_processing_time_column() -> None:
    """Structural anti-leakage at the feature layer.

    Ground truth lives in oracle tables, but a feature that read ``ingest_ts``
    would learn our pipeline's load rather than the payment — an artefact that
    changes the moment the pipeline is tuned.
    """
    forbidden = {
        "label",
        "typology",
        "campaign_id",
        "is_evasive",
        "role",
        "is_criminal",
        "ingest_ts",
        "bronze_ingested_ts",
        "silver_processed_ts",
        "decided_ts",
    }
    for feature in PAYMENTS_V1:
        assert feature.source_column not in forbidden, feature.name
        for filter_ in feature.filters:
            assert filter_.column not in forbidden, feature.name


def test_feature_set_fingerprint_is_stable_and_order_sensitive() -> None:
    """The fingerprint stamped onto every decision must mean something."""
    assert PAYMENTS_V1.fingerprint() == PAYMENTS_V1.fingerprint()

    reordered = FeatureSet(
        name=PAYMENTS_V1.name,
        version=PAYMENTS_V1.version,
        features=tuple(reversed(PAYMENTS_V1.features)),
    )
    assert reordered.fingerprint() != PAYMENTS_V1.fingerprint(), (
        "column order is part of the contract: the model consumes the vector "
        "positionally, so a reordering must change the fingerprint"
    )


def test_duplicate_feature_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate feature names"):
        FeatureSet(
            name="dupes",
            features=(count("c", Entity.DEBTOR, "1h"), count("c", Entity.CREDITOR, "1h")),
        )


def test_string_filter_values_are_escaped() -> None:
    """Filter values come from the repo, but a SQL builder should escape anyway."""
    from kg_features.ir import Filter

    assert Filter("vop_result", "==", "O'Brien").to_sql("h") == "h.vop_result = 'O''Brien'"


@given(event_streams(min_size=3))
@SLOW
def test_no_feature_is_ever_nan_or_infinite(events: list[dict]) -> None:
    """A NaN reaching a model is a silent failure, not a loud one."""
    for result in (run_duckdb(PAYMENTS_V1, events), run_online(PAYMENTS_V1, events)):
        for tx_id, row in result.items():
            for name, value in row.items():
                assert value is not None, f"{name} @ {tx_id} is None"
                assert math.isfinite(float(value)), f"{name} @ {tx_id} is {value}"
