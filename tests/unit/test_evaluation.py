"""The evaluation protocol, as tests.

Everything here defends a rule from ``docs/evaluation-protocol.md``. The rules
were committed before any model existed; these tests are what stop them being
quietly relaxed once a number is disappointing.
"""

from __future__ import annotations

import numpy as np
import pytest

from kg_core.timeutil import MILLIS_PER_DAY
from kg_eval.bootstrap import Comparison, bootstrap_difference, bootstrap_metric
from kg_eval.cost import CostModel, choose_threshold, sensitivity_analysis
from kg_eval.metrics import (
    average_precision,
    daily_capacity_k,
    expected_calibration_error,
    precision_at_k,
    recall_at_k,
    roc_auc,
)
from kg_eval.splits import (
    Fold,
    SplitConfigurationError,
    assign_folds,
    blocked_cv_folds,
    build_targets,
    make_boundaries,
)

DAY = MILLIS_PER_DAY


def _stamps(days: int, per_day: int = 20) -> np.ndarray:
    return np.array(
        [d * DAY + i * (DAY // per_day) for d in range(days) for i in range(per_day)],
        dtype=np.int64,
    )


# ================================================================== splits ===


def test_split_is_chronological_and_contiguous() -> None:
    stamps = _stamps(100)
    assignment = assign_folds(stamps, make_boundaries(stamps))

    for earlier, later in ((Fold.TRAIN, Fold.VALIDATION), (Fold.VALIDATION, Fold.TEST)):
        left, right = stamps[assignment.mask(earlier)], stamps[assignment.mask(later)]
        assert left.max() < right.min(), f"{earlier} must end before {later} begins"


def test_purge_and_embargo_clear_the_boundary() -> None:
    """No kept event may sit within the band on either side of a boundary."""
    stamps = _stamps(100)
    boundaries = make_boundaries(stamps, purge_millis=DAY, embargo_millis=DAY)
    assignment = assign_folds(stamps, boundaries)

    for fold in (Fold.TRAIN, Fold.VALIDATION, Fold.TEST):
        kept = stamps[assignment.mask(fold)]
        for boundary in (boundaries.t1, boundaries.t2):
            in_band = (kept >= boundary - boundaries.purge_millis) & (
                kept < boundary + boundaries.embargo_millis
            )
            assert not in_band.any(), f"{fold} retains events inside the band at {boundary}"


def test_an_over_wide_band_is_refused_not_silently_applied() -> None:
    """The failure this guard prevents surfaced as an untrainable model.

    A purge width equal to the longest feature window (30 days) discarded 79% of
    a 30-day dataset and left only the test fold. LightGBM then reported "no
    positives", several layers away from the cause.
    """
    stamps = _stamps(30)
    with pytest.raises(SplitConfigurationError, match="too wide for the shortest fold"):
        make_boundaries(stamps, purge_millis=30 * DAY)


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        make_boundaries(_stamps(50), fractions=(0.5, 0.3, 0.3))


def test_blocked_cv_never_trains_on_the_future() -> None:
    """Expanding window: every training index precedes every validation index."""
    stamps = _stamps(100)
    for train_rows, validation_rows in blocked_cv_folds(stamps, blocks=5):
        assert stamps[train_rows].max() < stamps[validation_rows].min()


# ====================================================== label availability ===


def _label(tx: str, event_day: int, decided_day: int, value: str = "CONFIRMED_FRAUD") -> dict:
    return {
        "subject_id": tx,
        "event_ts": event_day * DAY,
        "decided_ts": decided_day * DAY,
        "label": value,
    }


def test_a_label_decided_after_the_cutoff_does_not_exist() -> None:
    """The rule that actually stops time travel.

    The transaction happened well before the boundary; the *decision* did not.
    Filtering on the transaction's own timestamp — the common mistake — would
    hand the model an answer nobody had yet.
    """
    tx_ids = np.array(["a", "b"], dtype=object)
    labels = [_label("a", event_day=10, decided_day=20), _label("b", event_day=10, decided_day=40)]

    targets = build_targets(tx_ids, labels, available_as_of=30 * DAY)
    assert targets.y.tolist() == [1, 0]
    assert targets.labelled.tolist() == [True, False]

    # After the fact, both are known.
    later = build_targets(tx_ids, labels, available_as_of=50 * DAY)
    assert later.y.tolist() == [1, 1]


def test_unproven_suspicion_is_excluded_from_both_classes() -> None:
    """Protocol §1. Counting it as fraud teaches the model to imitate the
    incumbent alerting system rather than to detect crime."""
    tx_ids = np.array(["a", "b", "c"], dtype=object)
    labels = [
        _label("a", 1, 2, "CONFIRMED_LAUNDERING"),
        _label("b", 1, 2, "SUSPICIOUS_UNPROVEN"),
        _label("c", 1, 2, "BENIGN"),
    ]
    targets = build_targets(tx_ids, labels, available_as_of=10 * DAY)

    assert targets.y.tolist() == [1, 0, 0]
    assert targets.excluded.tolist() == [False, True, False]
    assert targets.labelled.tolist() == [True, False, True]


def test_the_oracle_gap_is_measured_not_assumed() -> None:
    """Unlabelled crime sits in the negative class; the gap quantifies that."""
    tx_ids = np.array(["a", "b", "c", "d"], dtype=object)
    labels = [_label("a", 1, 2)]
    targets = build_targets(tx_ids, labels, available_as_of=10 * DAY, oracle_tx_ids={"a", "b", "c"})

    gap = targets.oracle_gap()
    assert gap is not None
    assert gap["labelled_positives"] == 1
    assert gap["true_positives"] == 3
    # Reported values are rounded to 4 dp for the report; compare accordingly.
    assert gap["recall_of_truth"] == pytest.approx(1 / 3, abs=1e-4)


def test_the_earliest_decision_wins() -> None:
    """A later revision is information the model would not have had either."""
    tx_ids = np.array(["a"], dtype=object)
    labels = [
        _label("a", 1, 30, "BENIGN"),
        _label("a", 1, 10, "CONFIRMED_FRAUD"),
    ]
    assert build_targets(tx_ids, labels, available_as_of=50 * DAY).y.tolist() == [1]


# ================================================================= metrics ===


def test_average_precision_on_a_perfect_ranking() -> None:
    y = np.array([1, 1, 0, 0, 0])
    assert average_precision(y, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == pytest.approx(1.0)


def test_average_precision_on_the_worst_ranking() -> None:
    y = np.array([1, 1, 0, 0, 0])
    assert average_precision(y, np.array([0.1, 0.2, 0.7, 0.8, 0.9])) < 0.5


def test_tied_scores_are_not_ranked_by_arrival_order() -> None:
    """A model that outputs one value for everything must not be flattered.

    At a 0.1% positive rate a degenerate model can produce huge tie groups, and
    scoring them in row order would award it whatever the data happened to be
    sorted by.
    """
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    constant = np.full(10, 0.5)
    # Every ordering of the same tied scores must give the same answer.
    shuffled = np.random.default_rng(0).permutation(10)
    assert average_precision(y, constant) == pytest.approx(average_precision(y[shuffled], constant))
    # And it should land near the base rate, not near 1.0.
    assert average_precision(y, constant) == pytest.approx(0.2, abs=0.01)


def test_roc_auc_is_blind_at_this_class_balance() -> None:
    """The demonstration behind design principle P6.

    A model that ranks a handful of positives well scores an impressive ROC-AUC
    while its top alerts are mostly wrong. PR-AUC notices; ROC-AUC does not.
    """
    rng = np.random.default_rng(7)
    n = 20_000
    y = np.zeros(n, dtype=np.int8)
    y[rng.choice(n, 20, replace=False)] = 1

    # Positives land around the 90th percentile: better than most traffic, but
    # with roughly a thousand negatives still scoring above them. That is the
    # realistic shape of a decent-but-not-great model at this class balance.
    scores = rng.random(n)
    scores[y == 1] = rng.uniform(0.85, 0.95, size=int(y.sum()))

    assert roc_auc(y, scores) > 0.85, "ROC-AUC looks excellent"
    assert average_precision(y, scores) < 0.10, (
        "PR-AUC shows the truth: the top alerts are almost all false positives"
    )
    # The operational consequence, which is what design principle P6 is really about.
    assert precision_at_k(y, scores, 20) < 0.10


def test_precision_and_recall_at_k_agree_at_the_extremes() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert precision_at_k(y, scores, 3) == pytest.approx(1.0)
    assert recall_at_k(y, scores, 3) == pytest.approx(1.0)
    assert recall_at_k(y, scores, len(y)) == pytest.approx(1.0)


def test_daily_capacity_scales_with_the_evaluation_window() -> None:
    """200 alerts/day over 18 days is a budget of 3600, not 200.

    Using the daily number directly would make precision@k look far better than
    the compliance team's actual experience.
    """
    assert daily_capacity_k(_stamps(18), per_day=200) == pytest.approx(3400, rel=0.1)
    assert daily_capacity_k(np.array([0], dtype=np.int64), per_day=200) == 200


def test_calibration_error_uses_equal_mass_bins() -> None:
    """Equal-width bins would put 99% of predictions in the first bin."""
    y = np.zeros(1000, dtype=np.int8)
    y[:5] = 1
    perfect = np.full(1000, 0.005)
    assert expected_calibration_error(y, perfect) < 0.02


# ==================================================================== cost ===


def test_catching_fraud_reduces_cost_and_never_creates_income() -> None:
    """The bug the threshold optimiser exposed.

    An earlier `C_TP = investigation - recovery x amount` made a true positive
    *profitable*, mixing an absolute cost with one measured against a
    do-nothing baseline. The optimiser then preferred alerting on almost
    nothing.
    """
    model = CostModel()
    amount = np.array([100_000_00], dtype=np.int64)  # EUR 100,000

    caught = model.true_positive_cents(amount, 1)[0]
    missed = model.false_negative_cents(amount, 1)[0]

    assert caught > 0, "catching fraud must never be recorded as income"
    assert caught < missed, "catching fraud must cost less than missing it"


def test_a_perfect_model_beats_doing_nothing() -> None:
    rng = np.random.default_rng(3)
    n = 5_000
    y = (rng.random(n) < 0.01).astype(np.int8)
    amounts = np.full(n, 500_000, dtype=np.int64)  # EUR 5,000 each
    model = CostModel()

    do_nothing = model.evaluate(y, np.zeros(n), 0.5, amounts).cost_per_10k
    perfect = model.evaluate(y, y.astype(float), 0.5, amounts).cost_per_10k
    assert perfect < do_nothing


def test_the_capacity_constraint_binds() -> None:
    """Without it the optimiser proposes an alert volume nobody could staff."""
    rng = np.random.default_rng(11)
    n = 10_000
    y = (rng.random(n) < 0.01).astype(np.int8)
    scores = np.clip(rng.random(n) * 0.4 + y * 0.3, 0, 1)
    amounts = np.full(n, 1_000_000, dtype=np.int64)

    unconstrained = choose_threshold(CostModel(), y, scores, amounts_cents=amounts)
    constrained = choose_threshold(
        CostModel(), y, scores, amounts_cents=amounts, max_alert_rate=0.01
    )
    rate = constrained.outcome.alerts / constrained.outcome.rows
    assert rate <= 0.011
    assert constrained.threshold >= unconstrained.threshold


def test_sensitivity_analysis_moves_the_threshold_monotonically() -> None:
    """A higher cost of missing fraud should never make the gate stricter."""
    rng = np.random.default_rng(5)
    n = 5_000
    y = (rng.random(n) < 0.01).astype(np.int8)
    scores = np.clip(rng.random(n) * 0.5 + y * 0.4, 0, 1)

    rows = sensitivity_analysis(y, scores, max_alert_rate=0.05)
    thresholds = [row["threshold"] for row in rows]
    assert thresholds == sorted(thresholds, reverse=True), (
        f"thresholds should fall as the FN/FP ratio rises, got {thresholds}"
    )


# =============================================================== bootstrap ===


def test_intervals_bracket_the_point_estimate() -> None:
    rng = np.random.default_rng(2)
    n = 4_000
    y = (rng.random(n) < 0.02).astype(np.int8)
    scores = np.clip(rng.random(n) * 0.5 + y * 0.4, 0, 1)
    stamps = np.sort(rng.integers(0, 30 * DAY, n))

    interval = bootstrap_metric(average_precision, y, scores, stamps, resamples=200)
    assert interval.low <= interval.point <= interval.high
    assert interval.width > 0


def test_a_model_is_not_distinguishable_from_itself() -> None:
    """The paired bootstrap's sanity check.

    Comparing a model with itself must produce a difference of exactly zero and
    an interval that includes zero — otherwise the pairing is broken and every
    comparison is noise.
    """
    rng = np.random.default_rng(4)
    n = 3_000
    y = (rng.random(n) < 0.03).astype(np.int8)
    scores = np.clip(rng.random(n) * 0.5 + y * 0.3, 0, 1)
    stamps = np.sort(rng.integers(0, 20 * DAY, n))

    difference = bootstrap_difference(average_precision, y, scores, scores, stamps, resamples=200)
    assert difference.point == pytest.approx(0.0)
    assert not difference.excludes(0.0)


def test_a_clearly_better_model_is_distinguishable() -> None:
    rng = np.random.default_rng(6)
    n = 5_000
    y = (rng.random(n) < 0.03).astype(np.int8)
    good = np.clip(rng.random(n) * 0.2 + y * 0.7, 0, 1)
    noise = rng.random(n)
    stamps = np.sort(rng.integers(0, 30 * DAY, n))

    difference = bootstrap_difference(average_precision, y, good, noise, stamps, resamples=300)
    assert difference.excludes(0.0) and difference.point > 0


def test_a_null_result_reads_as_a_finding() -> None:
    """Protocol §6's honesty clause is generated, not chosen.

    The sentence exists before any result does, so publishing it is not a
    decision anyone has to make while looking at a disappointing number.
    """
    from kg_eval.bootstrap import Interval

    comparison = Comparison(
        metric="pr_auc",
        model_a="tgn",
        model_b="lightgbm",
        difference=Interval(point=0.004, low=-0.002, high=0.011),
    )
    assert not comparison.distinguishable
    assert comparison.verdict == "not distinguishable"
    assert "includes zero" in comparison.sentence()


def test_resampling_is_by_day_not_by_row() -> None:
    """Row resampling breaks within-day clustering and gives false confidence.

    A single-day dataset cannot be resampled by day, and the code says so with a
    degenerate interval rather than silently falling back to rows.
    """
    rng = np.random.default_rng(8)
    n = 500
    y = (rng.random(n) < 0.05).astype(np.int8)
    scores = rng.random(n)
    one_day = np.full(n, 12 * 3600 * 1000, dtype=np.int64)

    interval = bootstrap_metric(average_precision, y, scores, one_day, resamples=100)
    assert interval.resamples == 0
    assert interval.width == 0.0
