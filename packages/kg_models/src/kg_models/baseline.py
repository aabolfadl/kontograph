"""The LightGBM baseline — rung 1 of the ablation ladder.

The platform scope: *"Not everything is a neural network."* This model exists to be
beaten, and to be **hard to beat**. A deliberately weak baseline would make the
graph model look good and prove nothing, so this one gets proper class-imbalance
handling, early stopping on the validation fold, and calibrated probabilities.

Rung 0 (a constant predictor) is included too. It answers the question that
should be asked before any other: is the headline metric even meaningful at this
class balance? At a 0.1 % positive rate a constant predictor scores 0.999
accuracy and 0.5 ROC-AUC, which is exactly why the protocol leads with PR-AUC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kg_core.hashing import content_hash
from kg_core.logging import get_logger

log = get_logger(__name__)

#: Default hyperparameters.
#:
#: Chosen for a small, extremely imbalanced tabular problem and NOT tuned
#: against test results — the engineering invariants forbid choosing anything after seeing
#: them. `is_unbalance` rather than a hand-set `scale_pos_weight` so the
#: weighting follows the fold's own class balance instead of a number that would
#: silently become wrong when the dataset changes.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "verbosity": -1,
    "num_threads": 0,
    "seed": 20251009,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass(slots=True)
class TrainedModel:
    """A fitted model plus the provenance needed to reproduce and audit it."""

    name: str
    booster: Any = None
    calibrator: Any = None
    feature_names: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    best_iteration: int = 0
    threshold: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Constant score, for the rung-0 baseline which has no booster.
    constant_score: float | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probabilities."""
        raw = self.predict_raw(X)
        return self.calibrator.transform(raw) if self.calibrator is not None else raw

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated model output."""
        if self.constant_score is not None:
            return np.full(len(X), self.constant_score, dtype=np.float64)
        if self.booster is None:
            raise RuntimeError(f"{self.name} has no booster and no constant score")
        return np.asarray(
            self.booster.predict(X, num_iteration=self.best_iteration or None), dtype=np.float64
        )

    def feature_importance(self, top: int = 20) -> list[dict[str, Any]]:
        """Gain-based importance.

        Gain rather than split count: split count rewards high-cardinality
        features for being splittable, which flatters exactly the features that
        carry least information per split.
        """
        if self.booster is None:
            return []
        gains = self.booster.feature_importance(importance_type="gain")
        ranked = sorted(zip(self.feature_names, gains, strict=True), key=lambda pair: -pair[1])
        total = float(sum(gains)) or 1.0
        return [
            {"feature": name, "gain": float(gain), "share": round(float(gain) / total, 4)}
            for name, gain in ranked[:top]
        ]

    def save(self, directory: Path) -> Path:
        """Write the model, calibrator and provenance to *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        if self.booster is not None:
            self.booster.save_model(
                str(directory / "model.txt"), num_iteration=self.best_iteration or None
            )
        payload = {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "params": self.params,
            "best_iteration": self.best_iteration,
            "threshold": self.threshold,
            "constant_score": self.constant_score,
            "calibrator": self.calibrator.to_dict() if self.calibrator is not None else None,
            "provenance": self.provenance,
        }
        (directory / "model.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        log.info("model saved", name=self.name, path=str(directory))
        return directory

    @classmethod
    def load(cls, directory: Path) -> TrainedModel:
        import lightgbm as lgb

        from kg_models.calibration import from_dict as calibrator_from_dict

        payload = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        booster = None
        if (directory / "model.txt").exists():
            booster = lgb.Booster(model_file=str(directory / "model.txt"))
        return cls(
            name=payload["name"],
            booster=booster,
            calibrator=(
                calibrator_from_dict(payload["calibrator"]) if payload.get("calibrator") else None
            ),
            feature_names=tuple(payload["feature_names"]),
            params=payload["params"],
            best_iteration=payload["best_iteration"],
            threshold=payload["threshold"],
            provenance=payload.get("provenance", {}),
            constant_score=payload.get("constant_score"),
        )


def train_constant_baseline(y_train: np.ndarray, name: str = "constant") -> TrainedModel:
    """Rung 0: predict the training base rate for everything.

    Not a joke entry. It establishes what "no model" scores on each metric, which
    is the only way to read the others. Its ROC-AUC of 0.5 against an accuracy
    above 0.99 is the clearest possible demonstration of why design principle P6 rejects
    accuracy and ROC-AUC as headlines here.
    """
    rate = float(np.asarray(y_train).mean()) if len(y_train) else 0.0
    return TrainedModel(
        name=name,
        constant_score=rate,
        provenance={"kind": "constant", "base_rate": rate},
    )


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    name: str = "lightgbm",
    params: dict[str, Any] | None = None,
    num_boost_round: int = 2_000,
    early_stopping_rounds: int = 100,
    provenance: dict[str, Any] | None = None,
) -> TrainedModel:
    """Train the gradient-boosted baseline with early stopping on validation.

    Early stopping uses the validation fold, which is also where the calibrator
    and the threshold are fitted. That is three uses of one fold, and it is worth
    naming: validation is being spent, and the *test* fold stays untouched until
    the final evaluation. Reusing test for any of these would be the error the
    protocol is built to prevent.
    """
    import lightgbm as lgb

    settings = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=list(feature_names))
    validation_set = lgb.Dataset(
        X_validation, label=y_validation, feature_name=list(feature_names), reference=train_set
    )

    if int(np.asarray(y_train).sum()) == 0:
        raise ValueError(
            "the training fold contains no positives. With label availability enforced this "
            "usually means every label was decided after the split boundary — try a longer "
            "horizon or a smaller label delay."
        )

    callbacks: list[Any] = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(
        settings,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=callbacks,
    )

    log.info(
        "lightgbm trained",
        name=name,
        best_iteration=booster.best_iteration,
        train_rows=len(y_train),
        train_positives=int(np.asarray(y_train).sum()),
    )

    return TrainedModel(
        name=name,
        booster=booster,
        feature_names=feature_names,
        params=settings,
        best_iteration=int(booster.best_iteration or 0),
        provenance={
            "kind": "lightgbm",
            "param_hash": content_hash(settings, prefix="lgbm-params"),
            **(provenance or {}),
        },
    )


def calibrate_and_threshold(
    model: TrainedModel,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    amounts_cents: np.ndarray | None = None,
    cost_model: Any = None,
    max_alert_rate: float | None = None,
) -> dict[str, Any]:
    """Fit the calibrator and choose the operating point, both on validation.

    Mutates *model* with the fitted calibrator and threshold, and returns the
    evidence for both so the model card can show how they were chosen rather
    than merely asserting them.
    """
    from kg_eval.cost import CostModel, choose_threshold
    from kg_models.calibration import choose_calibrator

    raw = model.predict_raw(X_validation)
    calibration = choose_calibrator(raw, y_validation)
    model.calibrator = calibration.calibrator

    calibrated = model.predict(X_validation)
    cost = cost_model or CostModel()
    choice = choose_threshold(
        cost,
        y_validation,
        calibrated,
        amounts_cents=amounts_cents,
        max_alert_rate=max_alert_rate,
        chosen_on="validation",
    )
    model.threshold = choice.threshold

    return {"calibration": calibration.as_dict(), "threshold": choice.as_dict()}
