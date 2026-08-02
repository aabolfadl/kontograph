"""The ONNX export, and the parity check that decides whether it may serve.

The point of these tests is not that a conversion succeeds. It is that a
conversion which *changes decisions* is reported as a failure rather than
accepted because the mean error looked small — which is exactly what happened
when this was first measured (ADR-0008, amendment).
"""

from __future__ import annotations

import numpy as np
import pytest

from kg_serve.onnx_export import (
    PARITY_TOLERANCE,
    OnnxExportError,
    ParityReport,
    export_booster,
    verify_parity,
)


def _report(**overrides) -> ParityReport:
    base = {
        "rows": 1000,
        "max_abs_diff": 1e-9,
        "mean_abs_diff": 1e-12,
        "decisions_changed": 0,
        "threshold": 4e-4,
        "alerts_booster": 50,
        "alerts_onnx": 50,
        "alerts_agreed": 50,
    }
    base.update(overrides)
    return ParityReport(**base)


def test_parity_holds_when_nothing_moves():
    report = _report()
    assert report.ok is True
    assert report.alert_jaccard == 1.0


def test_a_tiny_score_difference_that_flips_a_decision_fails_parity():
    """The decision count governs, not the score difference.

    A mean absolute error of 7e-08 is the number that makes an export look fine.
    It is the wrong number when the operating threshold is 4e-04.
    """
    report = _report(
        max_abs_diff=1.665e-4,
        mean_abs_diff=7.4e-8,
        decisions_changed=774,
        alerts_booster=6458,
        alerts_onnx=7232,
        alerts_agreed=6458,
    )
    assert report.ok is False
    # The headline an operator needs: 12% more alerts than the evaluated model.
    assert report.alerts_onnx > report.alerts_booster
    assert report.alert_jaccard == pytest.approx(0.8930, abs=1e-4)


def test_a_large_score_difference_alone_also_fails_parity():
    """Even with no decision flipped in this sample, drift this big is a warning."""
    report = _report(max_abs_diff=PARITY_TOLERANCE * 10)
    assert report.ok is False


def test_jaccard_is_one_when_neither_path_alerts():
    """No alerts either side is perfect agreement, not a division by zero."""
    report = _report(alerts_booster=0, alerts_onnx=0, alerts_agreed=0)
    assert report.alert_jaccard == 1.0


def test_report_serialises_every_number_a_reader_needs():
    payload = _report(decisions_changed=774).as_dict()
    for key in (
        "rows",
        "max_abs_diff",
        "decisions_changed",
        "decisions_changed_share",
        "alerts_booster",
        "alerts_onnx",
        "alert_jaccard",
        "ok",
    ):
        assert key in payload
    assert payload["decisions_changed_share"] == pytest.approx(0.774)


def test_a_model_without_a_booster_refuses_to_export(tmp_path):
    """The constant baseline is a number, not a tree ensemble."""

    class Constant:
        name = "constant"
        booster = None
        feature_names = ("a", "b")

    with pytest.raises(OnnxExportError):
        export_booster(Constant(), tmp_path / "model.onnx")


# ------------------------------------------------------ the real conversion ---

lightgbm = pytest.importorskip("lightgbm")
pytest.importorskip("onnxmltools")
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def trained():
    """A small real booster, so the conversion path is genuinely exercised."""
    from kg_models.baseline import TrainedModel

    rng = np.random.default_rng(11)
    X = rng.normal(size=(600, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0.4).astype(int)
    booster = lightgbm.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 7, "seed": 11},
        lightgbm.Dataset(X, label=y),
        num_boost_round=25,
    )
    model = TrainedModel(
        name="lightgbm",
        booster=booster,
        feature_names=tuple(f"f{i}" for i in range(6)),
        threshold=0.5,
    )
    return model, X


def test_export_produces_a_loadable_graph(trained, tmp_path):
    model, X = trained
    path = export_booster(model, tmp_path / "model.onnx")
    assert path.exists() and path.stat().st_size > 0

    report = verify_parity(model, path, X)
    assert report.rows == len(X)
    # Scores must be probabilities, not the graph's `label` output. Reading
    # outputs[0] by position returns int64 class labels and produced a mean
    # error small enough to pass for rounding - the bug this asserts against.
    assert report.max_abs_diff < 0.01


def test_scorer_returns_probabilities_not_labels(trained, tmp_path):
    from kg_serve.onnx_export import OnnxScorer

    model, X = trained
    path = export_booster(model, tmp_path / "model.onnx")
    scores = OnnxScorer(path, model.feature_names).predict_raw(X)

    assert scores.shape == (len(X),)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    # Class labels would be exactly 0 or 1 everywhere; probabilities are not.
    assert not np.all(np.isin(scores, (0.0, 1.0)))
