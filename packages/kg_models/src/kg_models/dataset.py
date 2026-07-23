"""Assembling the training matrix.

Where the feature table, the label stream and the split protocol meet. This is
the most dangerous file in the modelling layer, because every leak that survives
the feature platform has to come through here — a label joined without its
availability cutoff, a fold assigned before purging, an oracle column left in
the matrix by accident.

So the assembly is deliberately explicit about three things:

1. **Labels are filtered by ``decided_ts``, per fold.** The training matrix uses
   only labels decided before ``t1``; validation uses only labels decided before
   ``t2``. A single global join would silently hand the model answers that had
   not been determined yet.
2. **Oracle truth is loaded separately and never enters ``X``.** It is carried
   alongside so the reported metrics can be compared against the truth, which is
   how the cost of label bias gets measured instead of assumed.
3. **The feature matrix is built from the feature-set column list**, in order.
   Anything else in the table — identifiers, timestamps, amounts — is kept as
   metadata and cannot reach the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kg_core.logging import get_logger
from kg_eval.splits import (
    DEFAULT_BAND_MILLIS,
    Fold,
    SplitAssignment,
    TargetVector,
    assign_folds,
    build_targets,
    make_boundaries,
)
from kg_features.ir import FeatureSet

log = get_logger(__name__)

#: Columns that exist in the feature table but must never be model inputs.
#: Identifiers would let a tree memorise accounts; timestamps would let it learn
#: "fraud happened in week 3 of this dataset", which generalises to nothing.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "tx_id",
        "acceptance_ts",
        "debtor_account_id",
        "creditor_account_id",
        "amount_cents",
        "ingest_ts",
        "event_date",
    }
)


@dataclass(slots=True)
class ModellingData:
    """Everything a training run needs, with the folds already applied."""

    feature_names: tuple[str, ...]
    X: np.ndarray
    tx_ids: np.ndarray
    timestamps: np.ndarray
    amounts_cents: np.ndarray
    split: SplitAssignment
    #: Targets per fold, each built with that fold's availability cutoff.
    targets: dict[str, TargetVector] = field(default_factory=dict)
    feature_set_fingerprint: str = ""
    dataset_name: str = ""

    def fold_rows(self, fold: Fold) -> np.ndarray:
        return np.flatnonzero(self.split.mask(fold))

    def matrix(self, fold: Fold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(X, y, rows)`` for one fold, with excluded rows already dropped."""
        rows = self.fold_rows(fold)
        target = self.targets[fold.value]
        keep = ~target.excluded[rows]
        rows = rows[keep]
        return self.X[rows], target.y[rows], rows

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset": self.dataset_name,
            "rows": int(self.X.shape[0]),
            "features": len(self.feature_names),
            "feature_set_fingerprint": self.feature_set_fingerprint,
            "split": self.split.summary(),
            "folds": {},
        }
        for fold in (Fold.TRAIN, Fold.VALIDATION, Fold.TEST):
            target = self.targets[fold.value]
            rows = self.fold_rows(fold)
            entry = {
                "rows": int(rows.size),
                "positives": int(target.y[rows].sum()),
                "positive_rate": round(float(target.y[rows].mean()), 6) if rows.size else 0.0,
                "label_coverage": round(float(target.labelled[rows].mean()), 6)
                if rows.size
                else 0.0,
            }
            gap: dict[str, float] | None = target.oracle_gap()
            if gap is not None:
                oracle = target.y_oracle[rows] if target.y_oracle is not None else None
                entry["oracle_positives"] = int(oracle.sum()) if oracle is not None else 0
                entry["labels_capture_of_truth"] = round(
                    entry["positives"] / max(1, entry["oracle_positives"]), 4
                )
            payload["folds"][fold.value] = entry
        return payload


def load_modelling_data(
    features_path: Path,
    dataset_dir: Path,
    feature_set: FeatureSet,
    *,
    fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
    embargo_millis: int | None = None,
    label_delay_days: float = 0.0,
    include_oracle: bool = True,
) -> ModellingData:
    """Build the modelling matrix from a feature table and a dataset directory.

    ``label_delay_days`` shifts every fold's availability cutoff *earlier*,
    simulating longer verification latency (protocol §7). Zero means "labels are
    available as soon as they were decided", which is the base case, not an
    optimistic one — the decided_ts filter still applies.
    """
    import pyarrow.parquet as pq

    from kg_core.timeutil import MILLIS_PER_DAY
    from kg_sim.dataset import read_table

    table = pq.read_table(features_path)
    columns = set(table.column_names)

    missing = [name for name in feature_set.column_names if name not in columns]
    if missing:
        raise ValueError(
            f"the feature table is missing {len(missing)} column(s) the feature set "
            f"declares: {missing[:5]}. Rebuild it with `kg features build` — the "
            "fingerprint of the set has probably changed."
        )

    # Column order follows the feature set, not the table: the model consumes
    # the vector positionally and a reordering would silently swap features.
    feature_names = feature_set.column_names
    X = np.column_stack(
        [
            np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)
            for name in feature_names
        ]
    )
    _assert_no_forbidden_columns(feature_names)

    tx_ids = np.asarray(table.column("tx_id").to_pylist(), dtype=object)
    timestamps = np.asarray(
        table.column("acceptance_ts").to_numpy(zero_copy_only=False), dtype=np.int64
    )

    amounts = (
        np.asarray(table.column("amount_cents").to_numpy(zero_copy_only=False), dtype=np.int64)
        if "amount_cents" in columns
        else _amounts_from_dataset(dataset_dir, tx_ids)
    )

    # A one-day band, not the feature window. See kg_eval.splits: a window-wide
    # purge removed 79% of a 30-day dataset, and the rationale for it was wrong
    # anyway — a training event's feature window points backwards and cannot
    # reach the validation period.
    band = embargo_millis if embargo_millis is not None else DEFAULT_BAND_MILLIS
    boundaries = make_boundaries(
        timestamps,
        fractions=fractions,
        purge_millis=band,
        embargo_millis=band,
    )
    split = assign_folds(timestamps, boundaries)

    labels = read_table(dataset_dir, "labels").to_pylist()
    oracle_ids: set[str] | None = None
    if include_oracle:
        # Deliberate, greppable oracle access: this is the evaluation path, and
        # the result never enters X.
        oracle_ids = {
            row["tx_id"]
            for row in read_table(dataset_dir, "ground_truth", allow_oracle=True).to_pylist()
        }

    delay = int(label_delay_days * MILLIS_PER_DAY)
    cutoffs = {
        Fold.TRAIN.value: boundaries.t1 - delay,
        Fold.VALIDATION.value: boundaries.t2 - delay,
        # Test is scored after the fact, so every decided label is fair game.
        # Using a cutoff here would understate the test positives and flatter
        # precision by removing true positives from the denominator.
        Fold.TEST.value: None,
    }
    targets = {
        fold: build_targets(tx_ids, labels, available_as_of=cutoff, oracle_tx_ids=oracle_ids)
        for fold, cutoff in cutoffs.items()
    }

    data = ModellingData(
        feature_names=feature_names,
        X=X,
        tx_ids=tx_ids,
        timestamps=timestamps,
        amounts_cents=amounts,
        split=split,
        targets=targets,
        feature_set_fingerprint=feature_set.fingerprint(),
        dataset_name=dataset_dir.name,
    )
    log.info(
        "modelling data assembled", **{k: v for k, v in data.summary().items() if k != "split"}
    )
    return data


def _assert_no_forbidden_columns(names: tuple[str, ...]) -> None:
    """Guard against an identifier or a timestamp reaching the model."""
    offenders = sorted(set(names) & NON_FEATURE_COLUMNS)
    if offenders:
        raise ValueError(
            f"these are not model inputs and must not be in the feature set: {offenders}. "
            "Identifiers let a tree memorise accounts; timestamps let it learn which week "
            "of this particular dataset contained fraud."
        )


def _amounts_from_dataset(dataset_dir: Path, tx_ids: np.ndarray) -> np.ndarray:
    """Fetch per-transaction amounts, which the cost model needs."""
    from kg_sim.dataset import read_table

    lookup = {
        row["tx_id"]: row["amount_cents"]
        for row in read_table(dataset_dir, "events").select(["tx_id", "amount_cents"]).to_pylist()
    }
    return np.fromiter((lookup.get(tx, 0) for tx in tx_ids), dtype=np.int64, count=len(tx_ids))
