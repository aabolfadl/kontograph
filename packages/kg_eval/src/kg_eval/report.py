"""Assembling results into the tables the README publishes.

The engineering invariants: *"No unreproducible number. Every figure in the README
regenerates from a command."* Everything here is generated from run artifacts;
nothing is typed by hand.

The ablation renderer implements the commitment in protocol §6 literally: if the
graph model does not beat LightGBM by more than its bootstrap confidence
interval, the generated table says so in those words. That sentence is produced
by code precisely so that nobody has to choose to write it later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kg_core.hashing import content_hash
from kg_eval.bootstrap import Comparison, Interval, bootstrap_difference, bootstrap_metric
from kg_eval.metrics import MetricBundle, average_precision, precision_at_k


@dataclass(slots=True)
class RunResult:
    """One model's evaluation on one fold, with provenance."""

    model: str
    fold: str
    bundle: MetricBundle
    intervals: dict[str, Interval] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "fold": self.fold,
            **self.bundle.as_dict(),
            "intervals": {name: ci.as_dict() for name, ci in self.intervals.items()},
            **self.extras,
        }


@dataclass(slots=True)
class EvaluationReport:
    """Everything one evaluation run produced."""

    dataset: str
    feature_set_fingerprint: str
    code_commit: str
    protocol_hash: str
    results: list[RunResult] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add(self, result: RunResult) -> None:
        self.results.append(result)

    def for_fold(self, fold: str) -> list[RunResult]:
        return [r for r in self.results if r.fold == fold]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "feature_set_fingerprint": self.feature_set_fingerprint,
            "code_commit": self.code_commit,
            "protocol_hash": self.protocol_hash,
            "context": self.context,
            "results": [r.as_dict() for r in self.results],
            "comparisons": [c.as_dict() for c in self.comparisons],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8")
        return path


def protocol_fingerprint(repo_root: Path) -> str:
    """Content hash of the evaluation protocol document.

    Stamped onto every report. A result computed under a different protocol
    version is not comparable to one computed under this version, and the hash
    is what makes that checkable rather than a matter of memory.
    """
    protocol = repo_root / "docs" / "evaluation-protocol.md"
    if not protocol.exists():
        return "missing"
    return content_hash(protocol.read_text(encoding="utf-8"), prefix="eval-protocol")


def add_intervals(
    result: RunResult,
    y_true: np.ndarray,
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    capacity_k: int,
    resamples: int = 2_000,
) -> None:
    """Attach bootstrap intervals to the two headline metrics."""
    result.intervals["pr_auc"] = bootstrap_metric(
        average_precision, y_true, scores, timestamps, resamples=resamples
    )
    result.intervals["precision_at_k"] = bootstrap_metric(
        lambda y, s: precision_at_k(y, s, capacity_k),
        y_true,
        scores,
        timestamps,
        resamples=resamples,
    )


def compare_models(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    timestamps: np.ndarray,
    *,
    baseline: str,
    challenger: str,
    capacity_k: int,
    resamples: int = 2_000,
) -> list[Comparison]:
    """Paired bootstrap comparisons on both headline metrics."""
    if baseline not in scores or challenger not in scores:
        return []
    comparisons = []
    for name, metric in (
        ("pr_auc", average_precision),
        ("precision_at_k", lambda y, s: precision_at_k(y, s, capacity_k)),
    ):
        comparisons.append(
            Comparison(
                metric=name,
                model_a=challenger,
                model_b=baseline,
                difference=bootstrap_difference(
                    metric,
                    y_true,
                    scores[challenger],
                    scores[baseline],
                    timestamps,
                    resamples=resamples,
                ),
            )
        )
    return comparisons


# ------------------------------------------------------------- rendering ----


def render_ablation_table(report: EvaluationReport, fold: str = "test") -> str:
    """The results table, as Markdown, generated from artifacts."""
    results = report.for_fold(fold)
    if not results:
        return "_No results for this fold._"

    capacity = report.context.get("capacity_per_day", 200)
    header = (
        f"| # | Model | PR-AUC (95% CI) | Precision@{capacity}/day | "
        f"Recall@{capacity}/day | Expected cost / 10k |\n"
        "|---|---|---|---|---|---|"
    )
    rows = []
    for index, result in enumerate(results):
        metrics = result.bundle.metrics
        pr = result.intervals.get("pr_auc")
        precision = metrics.get(f"precision_at_{capacity}_per_day")
        recall = metrics.get(f"recall_at_{capacity}_per_day")
        cost = metrics.get("expected_cost_per_10k")
        rows.append(
            "| {i} | {name} | {pr} | {p} | {r} | {c} |".format(
                i=index,
                name=result.model,
                pr=pr.format() if pr else f"{metrics['pr_auc'].value:.4f}",
                p=f"{precision.value:.4f}" if precision else "—",
                r=f"{recall.value:.4f}" if recall else "—",
                c=f"€{cost.value:,.0f}" if cost else "—",
            )
        )
    return "\n".join([header, *rows])


def render_verdict(report: EvaluationReport) -> str:
    """The honesty clause from protocol §6, written by code rather than by choice.

    If the challenger is not distinguishable from the baseline, this returns a
    paragraph saying so plainly. That is the whole point: the sentence exists
    before the result does, so publishing it is not a decision anyone has to
    make while looking at a disappointing number.
    """
    if not report.comparisons:
        return "_No model comparison available yet — the ablation ladder needs at least two rungs._"

    lines: list[str] = []
    undistinguishable = [c for c in report.comparisons if not c.distinguishable]

    for comparison in report.comparisons:
        lines.append(f"- {comparison.sentence()}")

    if undistinguishable:
        challenger = report.comparisons[0].model_a
        baseline = report.comparisons[0].model_b
        lines.append("")
        lines.append(
            f"**{challenger} did not beat {baseline} convincingly.** On "
            f"{len(undistinguishable)} of {len(report.comparisons)} headline metrics the 95% "
            "confidence interval on the difference includes zero, which means the two models "
            "are not distinguishable on this data. Reporting the higher point estimate as a "
            "win would be reading noise. This paragraph is generated by "
            "`kg_eval.report.render_verdict`, per §6 of the evaluation protocol."
        )
    return "\n".join(lines)


def render_summary(report: EvaluationReport) -> str:
    """A complete, self-contained results section."""
    parts = [
        "## Results",
        "",
        f"Dataset `{report.dataset}` · feature set `{report.feature_set_fingerprint[:12]}` · "
        f"commit `{report.code_commit}` · protocol `{report.protocol_hash[:12]}`",
        "",
        render_ablation_table(report, "test"),
        "",
        "### Does the added complexity earn its place?",
        "",
        render_verdict(report),
    ]

    label_gap = report.context.get("label_gap")
    if label_gap:
        parts += [
            "",
            "### What label bias costs",
            "",
            f"The label stream captures **{label_gap.get('recall_of_truth', 0):.1%}** of the "
            f"crime the simulator actually injected "
            f"({int(label_gap.get('labelled_positives', 0)):,} of "
            f"{int(label_gap.get('true_positives', 0)):,} transactions). Every precision figure "
            "above is therefore an upper bound: transactions that are genuinely criminal but "
            "were never confirmed are counted as legitimate in the denominator. This gap is "
            "measured rather than assumed because the data is simulated — a real institution "
            "would not be able to quote it.",
        ]
    return "\n".join(parts)
