"""ONNX export for the serving path, and the parity check that keeps it honest.

[ADR-0008](../../../../docs/adr/0008-onnx-runtime-not-triton.md) chose ONNX
Runtime in-process over a Triton sidecar. This module is the half of that
decision which actually has to work: turning a trained LightGBM booster into an
ONNX graph, and proving the graph agrees with the booster before anything is
allowed to serve from it.

**What goes into the graph, and what does not.** The tree ensemble is exported.
The calibrator is not, and that is deliberate rather than a shortcut: it is
already stored as isotonic knots evaluated by ``np.interp`` — a handful of
floating-point operations with no sklearn dependency at serving time. Putting it
in the graph would add a converter dependency and a second thing to keep in sync,
to accelerate an operation that does not appear in the latency budget. The
expensive part is the ensemble, and the ensemble is what ONNX runs.

So the honest description is: **the tree ensemble is ONNX; the monotone
calibration map is thirty lines of numpy.** That is what the model card says too.

**Parity is checked, not assumed.** Converters change, opsets change, and a tree
model that silently disagrees with its trained form in the fifth decimal place is
a model that will cross a 4e-4 threshold differently in production than it did in
evaluation. :func:`verify_parity` re-scores real feature vectors through both
paths and reports the largest disagreement — including, separately, whether any
row lands on a different side of the operating threshold, which is the only
disagreement that changes a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kg_core.logging import get_logger

log = get_logger(__name__)

#: Opset pinned rather than left to the converter default: the default moves
#: between releases, and a serving artifact should not change shape because a
#: dependency was upgraded. 15 is the ceiling onnxmltools 1.16 supports.
ML_OPSET = 3
ONNX_OPSET = 15

#: Largest absolute score difference tolerated between booster and ONNX graph.
#: Tree ensembles are sums of float32 leaf values, so exact equality is not
#: available; 1e-6 is far below the smallest threshold the cost model produces.
PARITY_TOLERANCE = 1e-6


class OnnxExportError(RuntimeError):
    """Raised when a model cannot be exported, or the export disagrees."""


@dataclass(slots=True)
class ParityReport:
    """How closely the ONNX graph reproduces the booster it came from."""

    rows: int
    max_abs_diff: float
    mean_abs_diff: float
    decisions_changed: int
    threshold: float
    alerts_booster: int = 0
    alerts_onnx: int = 0
    alerts_agreed: int = 0

    @property
    def alert_jaccard(self) -> float:
        """Overlap of the two alert sets.

        The headline number for an operator. "0.26% of decisions differ" sounds
        negligible until you notice that decisions are overwhelmingly ALLOW, and
        that the disagreements concentrate exactly where the score sits near the
        threshold — which is where the alerts are.
        """
        union = self.alerts_booster + self.alerts_onnx - self.alerts_agreed
        return self.alerts_agreed / union if union else 1.0

    @property
    def ok(self) -> bool:
        """Parity holds only if no decision flips, whatever the raw diff.

        The score tolerance is the early warning; the decision count is the
        thing that matters. A disagreement of 1e-9 that happens to straddle the
        operating threshold changes an outcome, and a disagreement of 1e-5 that
        does not, does not.
        """
        return self.decisions_changed == 0 and self.max_abs_diff <= PARITY_TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "decisions_changed": self.decisions_changed,
            "decisions_changed_share": self.decisions_changed / self.rows if self.rows else 0.0,
            "threshold": self.threshold,
            "alerts_booster": self.alerts_booster,
            "alerts_onnx": self.alerts_onnx,
            "alerts_agreed": self.alerts_agreed,
            "alert_jaccard": self.alert_jaccard,
            "tolerance": PARITY_TOLERANCE,
            "ok": self.ok,
        }


def export_booster(model: Any, path: Path) -> Path:
    """Convert a trained LightGBM booster to ONNX and write it to ``path``.

    Raises rather than falling back silently. A gateway that quietly served the
    Python booster while the ADR claimed ONNX would be the exact failure mode
    this project exists to avoid.
    """
    if model.booster is None:
        raise OnnxExportError(
            f"{model.name} has no booster to export "
            "(the constant baseline is a number, not a tree ensemble)"
        )

    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise OnnxExportError("onnxmltools is not installed; it is declared by kg_serve") from error

    n_features = len(model.feature_names)
    graph = convert_lightgbm(
        model.booster,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=ONNX_OPSET,
        zipmap=False,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(graph.SerializeToString())
    log.info(
        "exported to onnx",
        model=model.name,
        features=n_features,
        bytes=path.stat().st_size,
        path=str(path),
    )
    return path


def verify_parity(model: Any, path: Path, X: np.ndarray) -> ParityReport:
    """Score ``X`` through both paths and report the disagreement.

    ``X`` should be real feature vectors, not random noise: a tree ensemble's
    disagreements live at split boundaries, and random inputs mostly land in the
    interior of leaves where any two implementations trivially agree.
    """
    session = OnnxScorer(path, tuple(model.feature_names))
    onnx_raw = session.predict_raw(X)
    booster_raw = model.predict_raw(X)

    diff = np.abs(onnx_raw - booster_raw)
    calibrate = model.calibrator.transform if model.calibrator is not None else (lambda s: s)
    onnx_alerts = calibrate(onnx_raw) >= model.threshold
    booster_alerts = calibrate(booster_raw) >= model.threshold

    report = ParityReport(
        rows=int(X.shape[0]),
        max_abs_diff=float(diff.max()) if diff.size else 0.0,
        mean_abs_diff=float(diff.mean()) if diff.size else 0.0,
        decisions_changed=int(np.sum(onnx_alerts != booster_alerts)),
        threshold=float(model.threshold),
        alerts_booster=int(booster_alerts.sum()),
        alerts_onnx=int(onnx_alerts.sum()),
        alerts_agreed=int(np.sum(onnx_alerts & booster_alerts)),
    )
    log.info("onnx parity", **report.as_dict())
    return report


@dataclass(slots=True)
class OnnxScorer:
    """Runs the exported ensemble under ONNX Runtime, in this process.

    Single-threaded on purpose. The gateway scores one payment per request on a
    vector of a few dozen features; letting ONNX Runtime spin up an intra-op
    thread pool for that adds synchronisation cost to a matrix that fits in
    cache, and it competes with the event loop for the same four cores.
    """

    path: Path
    feature_names: tuple[str, ...]
    _session: Any = None
    _input: str = ""
    _output: str = ""
    _probability_column: int = -1

    def __post_init__(self) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

        # A binary-objective graph has TWO outputs, and the first is `label` —
        # the argmax class, not the score. Taking outputs[0] because it is first
        # gives 0/1 integers where probabilities belong, and the mistake is easy
        # to miss: against a model whose scores are mostly near 0 or 1, the mean
        # absolute error stays small enough to look like rounding. The parity
        # check is what caught it. Select by name, and fail loudly if absent.
        names = [o.name for o in self._session.get_outputs()]
        if "probabilities" in names:
            self._output = "probabilities"
            self._probability_column = 1
        elif len(names) == 1:
            # A regression-objective graph emits the raw score directly.
            self._output = names[0]
            self._probability_column = -1
        else:
            raise OnnxExportError(
                f"cannot identify the score output among {names}; refusing to guess"
            )

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated ensemble output, matching ``TrainedModel.predict_raw``."""
        outputs = self._session.run([self._output], {self._input: np.asarray(X, dtype=np.float32)})
        scores = np.asarray(outputs[0], dtype=np.float64)
        if self._probability_column >= 0:
            return scores[:, self._probability_column]
        return scores.reshape(-1)
