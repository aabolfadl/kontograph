"""The training and evaluation run.

One entry point that executes the protocol end to end: assemble the matrix, fit
the ladder's rungs, calibrate and threshold on validation, evaluate on test, and
write a report every number in the README can be regenerated from.

The ordering is the protocol, not a convenience:

    train      fit the model
    validation early stopping, calibration, and the operating point
    test       touched exactly once, at the end, with nothing fitted on it

Test is read last and used for nothing but reporting. That is the whole
discipline, and it is easy to break by accident — which is why the threshold
carried into the test evaluation is the one already stored on the model rather
than one recomputed here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kg_core.config import get_settings
from kg_core.logging import get_logger
from kg_eval.cost import CostModel, sensitivity_analysis
from kg_eval.metrics import daily_capacity_k, evaluate
from kg_eval.report import (
    EvaluationReport,
    RunResult,
    add_intervals,
    compare_models,
    protocol_fingerprint,
)
from kg_eval.splits import Fold
from kg_features.registry import get_feature_set
from kg_models.baseline import (
    TrainedModel,
    calibrate_and_threshold,
    train_constant_baseline,
    train_lightgbm,
)
from kg_models.dataset import ModellingData, load_modelling_data

log = get_logger(__name__)


@dataclass(slots=True)
class RunConfig:
    """What one evaluation run does."""

    features_path: Path
    dataset_dir: Path
    feature_set_name: str = "payments"
    capacity_per_day: int = 200
    #: Cap on the share of traffic that may be alerted, so the cost optimiser
    #: cannot propose a threshold the compliance team could never staff.
    max_alert_rate: float = 0.01
    bootstrap_resamples: int = 2_000
    label_delay_days: float = 0.0
    seed: int = 20251009


def run(config: RunConfig) -> tuple[EvaluationReport, dict[str, TrainedModel], ModellingData]:
    """Execute the ladder and produce a report."""
    from kg_cli.main import git_commit

    settings = get_settings()
    feature_set = get_feature_set(config.feature_set_name)
    started = time.perf_counter()

    data = load_modelling_data(
        config.features_path,
        config.dataset_dir,
        feature_set,
        label_delay_days=config.label_delay_days,
    )

    X_train, y_train, _ = data.matrix(Fold.TRAIN)
    X_validation, y_validation, validation_rows = data.matrix(Fold.VALIDATION)
    X_test, y_test, test_rows = data.matrix(Fold.TEST)

    log.info(
        "folds ready",
        train=f"{len(y_train):,} ({int(y_train.sum())} positive)",
        validation=f"{len(y_validation):,} ({int(y_validation.sum())} positive)",
        test=f"{len(y_test):,} ({int(y_test.sum())} positive)",
    )

    cost_model = CostModel()
    models: dict[str, TrainedModel] = {}
    evidence: dict[str, Any] = {}

    # ---- rung 0: constant ------------------------------------------------
    models["constant"] = train_constant_baseline(y_train)

    # ---- rung 1: LightGBM on tabular features ----------------------------
    models["lightgbm"] = train_lightgbm(
        X_train,
        y_train,
        X_validation,
        y_validation,
        data.feature_names,
        provenance={
            "dataset": data.dataset_name,
            "feature_set_fingerprint": data.feature_set_fingerprint,
            "label_delay_days": config.label_delay_days,
        },
    )

    for name, model in models.items():
        evidence[name] = calibrate_and_threshold(
            model,
            X_validation,
            y_validation,
            amounts_cents=data.amounts_cents[validation_rows],
            cost_model=cost_model,
            max_alert_rate=config.max_alert_rate,
        )

    # ---- evaluate --------------------------------------------------------
    report = EvaluationReport(
        dataset=data.dataset_name,
        feature_set_fingerprint=data.feature_set_fingerprint,
        code_commit=git_commit(),
        protocol_hash=protocol_fingerprint(settings.repo_root),
        context={
            "capacity_per_day": config.capacity_per_day,
            "max_alert_rate": config.max_alert_rate,
            "label_delay_days": config.label_delay_days,
            "cost_model": cost_model.describe(),
            "split": data.split.summary(),
            "folds": data.summary()["folds"],
        },
    )

    test_scores: dict[str, np.ndarray] = {}
    capacity_k = daily_capacity_k(data.timestamps[test_rows], config.capacity_per_day)

    for fold, X, y, rows in (
        (Fold.VALIDATION, X_validation, y_validation, validation_rows),
        (Fold.TEST, X_test, y_test, test_rows),
    ):
        for name, model in models.items():
            scores = model.predict(X)
            bundle = evaluate(
                y,
                scores,
                data.timestamps[rows],
                capacity_per_day=config.capacity_per_day,
                # The threshold from validation, applied unchanged. Recomputing
                # it here would be choosing the operating point on test.
                threshold=model.threshold,
                cost_model=cost_model,
                amounts_cents=data.amounts_cents[rows],
            )
            result = RunResult(model=name, fold=fold.value, bundle=bundle)

            if fold is Fold.TEST:
                test_scores[name] = scores
                add_intervals(
                    result,
                    y,
                    scores,
                    data.timestamps[rows],
                    capacity_k=capacity_k,
                    resamples=config.bootstrap_resamples,
                )
                result.extras["importance"] = model.feature_importance(top=15)
                result.extras["evidence"] = evidence.get(name, {})
            report.add(result)

    # ---- comparisons -----------------------------------------------------
    report.comparisons = compare_models(
        y_test,
        test_scores,
        data.timestamps[test_rows],
        baseline="constant",
        challenger="lightgbm",
        capacity_k=capacity_k,
        resamples=config.bootstrap_resamples,
    )

    # ---- context that keeps the numbers honest ---------------------------
    report.context["label_gap"] = data.targets[Fold.TEST.value].oracle_gap()
    report.context["oracle_comparison"] = _oracle_comparison(
        data, test_rows, test_scores, config.capacity_per_day
    )
    report.context["cost_sensitivity"] = sensitivity_analysis(
        y_test,
        test_scores["lightgbm"],
        max_alert_rate=config.max_alert_rate,
    )
    report.context["runtime_seconds"] = round(time.perf_counter() - started, 2)

    return report, models, data


def _oracle_comparison(
    data: ModellingData,
    test_rows: np.ndarray,
    test_scores: dict[str, np.ndarray],
    capacity_per_day: int,
) -> dict[str, Any] | None:
    """The same metrics scored against ground truth instead of against labels.

    design principle P7 made measurable. Label-based precision is an upper bound —
    unconfirmed crime sits in the negative class — and the oracle figures show by
    how much. A real institution could not compute this; the simulator can, which
    is exactly the point of design principle P9.
    """
    target = data.targets[Fold.TEST.value]
    if target.y_oracle is None:
        return None

    y_oracle = target.y_oracle[test_rows]
    timestamps = data.timestamps[test_rows]

    comparison: dict[str, Any] = {}
    for name, scores in test_scores.items():
        against_labels = evaluate(
            data.targets[Fold.TEST.value].y[test_rows],
            scores,
            timestamps,
            capacity_per_day=capacity_per_day,
        )
        against_truth = evaluate(y_oracle, scores, timestamps, capacity_per_day=capacity_per_day)
        comparison[name] = {
            "pr_auc_vs_labels": round(against_labels.value("pr_auc"), 6),
            "pr_auc_vs_oracle": round(against_truth.value("pr_auc"), 6),
            f"precision_at_{capacity_per_day}_vs_labels": round(
                against_labels.value(f"precision_at_{capacity_per_day}_per_day"), 6
            ),
            f"precision_at_{capacity_per_day}_vs_oracle": round(
                against_truth.value(f"precision_at_{capacity_per_day}_per_day"), 6
            ),
        }
    return comparison


def label_delay_curve(
    config: RunConfig, delays: tuple[float, ...] = (0.0, 14.0, 30.0, 60.0)
) -> list[dict[str, Any]]:
    """Retrain under increasing verification latency — protocol §7.

    Longer delay means fewer labels have been decided by training time, so the
    model is trained on less. The decay curve is the observable consequence of
    design principle P7, and publishing it is what turns "labels arrive late" from a caveat
    into a measurement.
    """
    curve: list[dict[str, Any]] = []
    for delay in delays:
        variant = RunConfig(**{**config.__dict__, "label_delay_days": delay})
        report, _, data = run(variant)
        test = next(r for r in report.for_fold("test") if r.model == "lightgbm")
        train_target = data.targets[Fold.TRAIN.value]
        train_rows = data.fold_rows(Fold.TRAIN)
        curve.append(
            {
                "delay_days": delay,
                "training_positives": int(train_target.y[train_rows].sum()),
                "pr_auc": round(test.bundle.value("pr_auc"), 6),
                f"precision_at_{config.capacity_per_day}": round(
                    test.bundle.value(f"precision_at_{config.capacity_per_day}_per_day"), 6
                ),
            }
        )
        log.info("label delay run complete", delay_days=delay, pr_auc=curve[-1]["pr_auc"])
    return curve
