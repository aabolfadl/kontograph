"""Chronological splitting, purging, embargo, and label availability.

Implements §2 of ``docs/evaluation-protocol.md``, which was frozen and committed
before any model in this repository existed.

Three mechanisms, doing three different jobs. They are routinely conflated, and
conflating them is how a backtest quietly becomes a time machine.

**Chronological split.** Train on ``[t0, t1)``, validate on ``[t1, t2)``, test on
``[t2, t3)``. Never random. A random split over a payment stream puts an
account's later transactions in train and its earlier ones in test, which is
both leakage and a flattering result.

**Purge and embargo.** These do *not* prevent feature leakage — the feature layer
already guarantees that no feature reads forward (see :mod:`kg_features.ir`).
What they address is narrower: a training event at ``t1 - e`` and a validation
event at ``t1 + e`` share almost all of their feature history, so their errors
are correlated and the validation estimate is slightly optimistic.

**The width must be modest, and the reason is worth stating.** An earlier
version of the protocol set the purge width to the longest feature window (30
days), on the stated grounds that "a training transaction whose feature window
overlaps the validation start" should be dropped. That rationale is simply
wrong: a training event's window points *backwards*, so it can never reach the
future. Applying it removed 79% of a 30-day dataset and left only the test fold.

The residual correlation being addressed is second-order, and a wide band buys
very little while costing a great deal of data. More importantly, a validation
event whose features were partly computed from training-period history is not
anomalous — that is *exactly* what happens in production on the day after
deployment. Treating it as contamination would be modelling a problem that does
not exist.

So the default band is one day, and :func:`make_boundaries` refuses a
configuration that would consume most of a fold rather than silently returning
an empty training set.

**Label availability.** The one that actually stops time travel. A label carries
``decided_ts`` — when somebody confirmed the outcome — typically weeks after the
transaction. A label may enter training only if ``decided_ts`` precedes the
boundary. Filtering on the transaction's own timestamp instead is the single
most common way a fraud backtest overstates itself, because it trains on the
answer to a question that had not been asked yet.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kg_core.timeutil import MILLIS_PER_DAY, iso


class Fold(enum.StrEnum):
    """Which part of the protocol an event belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    #: Dropped from training because it sits inside the purge band before a
    #: boundary; its history overlaps the evaluation period's.
    PURGED = "purged"
    #: Dropped from evaluation because it sits inside the embargo band after a
    #: boundary, for the same reason from the other side.
    EMBARGOED = "embargoed"


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """The instants that define a chronological split, in epoch milliseconds."""

    t0: int
    t1: int
    t2: int
    t3: int
    purge_millis: int
    embargo_millis: int

    def describe(self) -> dict[str, Any]:
        return {
            "train": f"{iso(self.t0)} .. {iso(self.t1)}",
            "validation": f"{iso(self.t1)} .. {iso(self.t2)}",
            "test": f"{iso(self.t2)} .. {iso(self.t3)}",
            "purge_days": round(self.purge_millis / MILLIS_PER_DAY, 2),
            "embargo_days": round(self.embargo_millis / MILLIS_PER_DAY, 2),
        }


@dataclass(slots=True)
class SplitAssignment:
    """Per-event fold labels plus the counts, for reporting."""

    folds: np.ndarray
    boundaries: SplitBoundaries
    counts: dict[str, int] = field(default_factory=dict)

    def mask(self, fold: Fold) -> np.ndarray:
        return self.folds == fold.value

    def summary(self) -> dict[str, Any]:
        total = int(self.folds.size)
        return {
            "boundaries": self.boundaries.describe(),
            "counts": self.counts,
            "kept_fraction": round(
                sum(self.counts.get(f.value, 0) for f in (Fold.TRAIN, Fold.VALIDATION, Fold.TEST))
                / max(1, total),
                4,
            ),
        }


#: Default purge and embargo band. See the module docstring for why this is one
#: day rather than the longest feature window.
DEFAULT_BAND_MILLIS = MILLIS_PER_DAY

#: A band wider than this share of the shortest fold is refused. At a quarter,
#: the split still resembles what was asked for; beyond it the folds stop
#: meaning what their names say.
MAX_BAND_SHARE_OF_FOLD = 0.25


class SplitConfigurationError(ValueError):
    """Raised when a split would discard most of the data it was given."""


def make_boundaries(
    timestamps: np.ndarray,
    *,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
    purge_millis: int = DEFAULT_BAND_MILLIS,
    embargo_millis: int = DEFAULT_BAND_MILLIS,
) -> SplitBoundaries:
    """Split boundaries at fractions of the observed time span.

    Fractions of *time*, not of row count. Volume varies by weekday and by month
    (payroll day is enormous), so a count-based split would put a different
    number of days in each fold on every dataset and make two runs
    incomparable. Time-based boundaries are a property of the period, not of how
    busy it happened to be.

    Refuses a purge or embargo band that would consume more than
    :data:`MAX_BAND_SHARE_OF_FOLD` of the shortest fold, because the failure mode
    is silent: an over-wide band produces an empty training set, LightGBM
    reports "no positives", and the cause is several layers away from the
    symptom.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {fractions} = {sum(fractions)}")
    if len(timestamps) == 0:
        raise ValueError("cannot split an empty set of timestamps")

    start, end = int(np.min(timestamps)), int(np.max(timestamps)) + 1
    span = end - start
    t1 = start + int(span * fractions[0])
    t2 = start + int(span * (fractions[0] + fractions[1]))

    shortest_fold = min(t1 - start, t2 - t1, end - t2)
    widest_band = max(purge_millis, embargo_millis)
    if shortest_fold <= 0:
        raise SplitConfigurationError(
            f"fractions {fractions} produce an empty fold over a span of "
            f"{span / MILLIS_PER_DAY:.1f} days"
        )
    if widest_band > shortest_fold * MAX_BAND_SHARE_OF_FOLD:
        raise SplitConfigurationError(
            f"a purge/embargo band of {widest_band / MILLIS_PER_DAY:.1f} days is too wide for "
            f"the shortest fold ({shortest_fold / MILLIS_PER_DAY:.1f} days): it would discard "
            f"most of that fold and leave a training set that is empty or unrepresentative.\n"
            f"Either shorten the band (the default is "
            f"{DEFAULT_BAND_MILLIS / MILLIS_PER_DAY:.0f} day) or simulate a longer horizon — "
            f"as a rule of thumb each fold wants to be several times the longest feature window."
        )

    return SplitBoundaries(
        t0=start,
        t1=t1,
        t2=t2,
        t3=end,
        purge_millis=purge_millis,
        embargo_millis=embargo_millis,
    )


def assign_folds(timestamps: np.ndarray, boundaries: SplitBoundaries) -> SplitAssignment:
    """Label every event with its fold, purging and embargoing the boundaries."""
    stamps = np.asarray(timestamps, dtype=np.int64)
    folds = np.full(stamps.shape, Fold.TEST.value, dtype=object)

    in_train = stamps < boundaries.t1
    in_validation = (stamps >= boundaries.t1) & (stamps < boundaries.t2)
    in_test = stamps >= boundaries.t2

    folds[in_train] = Fold.TRAIN.value
    folds[in_validation] = Fold.VALIDATION.value
    folds[in_test] = Fold.TEST.value

    # Purge the training side of each boundary: these events share most of their
    # feature history with the evaluation events immediately after it.
    purged = in_train & (stamps >= boundaries.t1 - boundaries.purge_millis)
    purged |= in_validation & (stamps >= boundaries.t2 - boundaries.purge_millis)
    folds[purged] = Fold.PURGED.value

    # Embargo the evaluation side, for the same overlap from the other
    # direction. Applied after purging so a purged event is not relabelled.
    embargoed = (
        in_validation
        & (stamps < boundaries.t1 + boundaries.embargo_millis)
        & (folds != Fold.PURGED.value)
    )
    embargoed |= (
        in_test
        & (stamps < boundaries.t2 + boundaries.embargo_millis)
        & (folds != Fold.PURGED.value)
    )
    folds[embargoed] = Fold.EMBARGOED.value

    counts = {
        value: int(count)
        for value, count in zip(*np.unique(folds, return_counts=True), strict=True)
    }
    return SplitAssignment(folds=folds, boundaries=boundaries, counts=counts)


# --------------------------------------------------------- label availability


#: Labels that count as a positive target. SUSPICIOUS_UNPROVEN is deliberately
#: absent — see :func:`build_targets`.
POSITIVE_LABELS: frozenset[str] = frozenset({"CONFIRMED_FRAUD", "CONFIRMED_LAUNDERING"})

#: Excluded from training and from the primary evaluation entirely.
EXCLUDED_LABELS: frozenset[str] = frozenset({"SUSPICIOUS_UNPROVEN"})


@dataclass(slots=True)
class TargetVector:
    """Binary targets plus everything needed to report them honestly."""

    y: np.ndarray
    #: True where a confirmed label existed by the availability cutoff.
    labelled: np.ndarray
    #: True where the row must be dropped (unproven suspicion).
    excluded: np.ndarray
    #: Oracle truth, when available. Never used for training.
    y_oracle: np.ndarray | None = None

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if self.y.size else 0.0

    @property
    def label_coverage(self) -> float:
        """Share of rows that carry a confirmed label of any kind."""
        return float(self.labelled.mean()) if self.labelled.size else 0.0

    def oracle_gap(self) -> dict[str, float] | None:
        """How much of the truth the labels miss.

        The number design principle P7 is about. Reported next to every metric so a
        reader can see what the label-generating process cost, rather than
        having to take the caveat on faith.
        """
        if self.y_oracle is None:
            return None
        observed = float(self.y.sum())
        actual = float(self.y_oracle.sum())
        return {
            "labelled_positives": observed,
            "true_positives": actual,
            "recall_of_truth": round(observed / actual, 4) if actual else 0.0,
            "missed_positives": actual - observed,
        }


def build_targets(
    tx_ids: np.ndarray,
    labels: list[dict[str, Any]],
    *,
    available_as_of: int | None = None,
    oracle_tx_ids: set[str] | None = None,
) -> TargetVector:
    """Assemble the binary target, honouring label availability.

    ``available_as_of`` is the instant the model is imagined to be trained at. A
    label only counts if ``decided_ts < available_as_of``. Passing ``None``
    disables the filter, which is correct only for a final test-set evaluation
    performed after the fact — and never for building a training set.

    Rows with no label become negative. That is wrong in a known direction:
    undetected crime is counted as legitimate, so every reported precision is an
    upper bound. ``oracle_tx_ids`` supplies the simulator's ground truth so the
    size of that error can be *measured* instead of merely acknowledged.
    """
    by_tx: dict[str, dict[str, Any]] = {}
    for label in labels:
        if available_as_of is not None and int(label["decided_ts"]) >= available_as_of:
            continue  # not yet decided at training time: it does not exist
        subject = label["subject_id"]
        existing = by_tx.get(subject)
        # Keep the earliest decision for a transaction: a later revision is
        # information the model would not have had at the cutoff either.
        if existing is None or int(label["decided_ts"]) < int(existing["decided_ts"]):
            by_tx[subject] = label

    y = np.zeros(len(tx_ids), dtype=np.int8)
    labelled = np.zeros(len(tx_ids), dtype=bool)
    excluded = np.zeros(len(tx_ids), dtype=bool)

    for index, tx_id in enumerate(tx_ids):
        # Named `entry`, not `label`: `label` is already bound above as the loop
        # variable over `labels`, and rebinding it to an Optional here made the
        # two uses disagree about their type.
        entry = by_tx.get(tx_id)
        if entry is None:
            continue
        value = entry["label"]
        if value in EXCLUDED_LABELS:
            # Unproven suspicion is neither positive nor negative. Counting it
            # as fraud trains the model to reproduce the incumbent alerting
            # system rather than to detect crime.
            excluded[index] = True
            continue
        labelled[index] = True
        if value in POSITIVE_LABELS:
            y[index] = 1

    y_oracle = None
    if oracle_tx_ids is not None:
        y_oracle = np.fromiter(
            (1 if tx_id in oracle_tx_ids else 0 for tx_id in tx_ids),
            dtype=np.int8,
            count=len(tx_ids),
        )

    return TargetVector(y=y, labelled=labelled, excluded=excluded, y_oracle=y_oracle)


def label_delay_cutoff(boundary: int, delay_days: float) -> int:
    """Availability cutoff for a label-delay experiment.

    Protocol §7 retrains under 14 / 30 / 60-day verification latency. Simulating
    that is a matter of moving the cutoff *earlier*, so fewer labels have been
    decided by training time — which is exactly what a longer delay means.
    """
    return boundary - int(delay_days * MILLIS_PER_DAY)


def blocked_cv_folds(
    timestamps: np.ndarray, *, blocks: int = 5, purge_millis: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window cross-validation over contiguous time blocks.

    Protocol §2. Each block is validated once, trained only on *earlier* blocks.
    No future block ever trains a model evaluated on an earlier one, which is
    what separates this from k-fold over shuffled rows.
    """
    stamps = np.asarray(timestamps, dtype=np.int64)
    start, end = int(stamps.min()), int(stamps.max()) + 1
    edges = np.linspace(start, end, blocks + 1).astype(np.int64)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(1, blocks):
        boundary = edges[index]
        train_mask = stamps < boundary - purge_millis
        validation_mask = (stamps >= boundary) & (stamps < edges[index + 1])
        if train_mask.sum() and validation_mask.sum():
            folds.append((np.flatnonzero(train_mask), np.flatnonzero(validation_mask)))
    return folds
