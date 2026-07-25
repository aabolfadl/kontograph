"""``kg eval ingest-gpu`` — fold Kaggle GPU results into the local report.

The GPU half of this project returns a narrow bundle: per-event scores, the
transaction ids they belong to, and a ``metrics.json``. This command turns that
into rungs of the ablation ladder.

**The notebook's own metrics are not trusted, they are checked.** The scores are
re-scored here with the same ``kg_eval`` code that produced every other row in
the table, against labels resolved from the local dataset — and the notebook's
reported figure is compared against the recomputed one. A disagreement means the
two implementations have drifted, which is precisely the failure the
the design's "one definition, verified" claim is about, so it is reported loudly
rather than silently preferring one number.

Labels are joined by ``tx_id``, never by position. The notebook orders events by
time within its own split; the local test fold is built independently. Two
orderings that happen to agree today would silently disagree the moment either
side changed, and the resulting metrics would look plausible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from kg_core.config import get_settings

console = Console()

#: How far a recomputed metric may sit from the notebook's own figure before
#: this is treated as drift rather than floating-point noise. The notebook
#: rounds when it serialises, so an exact match is not available.
METRIC_TOLERANCE = 5e-4

#: Ladder position for each score file the notebooks emit. Keeps the published
#: ordering in one place rather than inferred from a filename at render time.
RUNG_NAMES: dict[str, str] = {
    "tgn_no_memory": "3_tgn_no_memory",
    "tgn_memory": "4_tgn_memory",
    "student": "5_amortised_explainer",
}


def _rung(name: str) -> str:
    """Ladder label for a score file. 03 already names its files by rung."""
    return RUNG_NAMES.get(name, name)


def _load_bundle(
    directory: Path, ids_from: Path | None = None
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    """Read metrics.json, the test tx_ids, and every scores_*.npy beside them.

    ``ids_from`` borrows the id vector from a sibling notebook's output, for a
    run that did not emit its own. That is only sound because the recomputed
    metrics are then checked against the notebook's reported figures: a wrong
    ordering collapses PR-AUC to roughly the base rate, so agreement to four
    decimals is strong evidence the join is right. It is still second best —
    the notebook should emit its own ids — so it has to be asked for explicitly.
    """
    metrics_path = directory / "metrics.json"
    ids_path = directory / "test_tx_ids.npy"
    if not metrics_path.exists():
        raise typer.BadParameter(f"no metrics.json in {directory}")
    if not ids_path.exists() and ids_from is not None:
        borrowed = ids_from / "test_tx_ids.npy"
        if not borrowed.exists():
            raise typer.BadParameter(f"no test_tx_ids.npy in {ids_from} either")
        console.print(
            f"[yellow]borrowing test_tx_ids.npy from {ids_from.name}[/] — "
            "the recomputed-vs-reported check below is what validates the join."
        )
        ids_path = borrowed
    if not ids_path.exists():
        raise typer.BadParameter(
            f"no test_tx_ids.npy in {directory}; without it the scores cannot be "
            "joined to labels by id, and joining by position would be a guess. "
            "Pass --tx-ids-from <notebook> to borrow them from a run on the same "
            "dataset."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    tx_ids = np.load(ids_path, allow_pickle=True).astype(str)
    scores = {
        path.stem.removeprefix("scores_"): np.load(path)
        for path in sorted(directory.glob("scores_*.npy"))
    }
    if not scores:
        raise typer.BadParameter(f"no scores_*.npy in {directory}")
    return metrics, tx_ids, scores


def ingest_gpu(
    notebook: str = typer.Argument("01_train_tgn", help="Notebook whose artifacts to fold in."),
    dataset: Path = typer.Option(None, "--dataset"),
    resamples: int = typer.Option(2000, "--resamples", help="Bootstrap resamples."),
    tx_ids_from: str = typer.Option(
        None,
        "--tx-ids-from",
        help="Borrow test_tx_ids.npy from another notebook's output (same dataset).",
    ),
) -> None:
    """Fold pulled GPU scores into the evaluation report, recomputing every metric."""
    from kg_eval.bootstrap import bootstrap_difference, bootstrap_metric
    from kg_eval.cost import CostModel
    from kg_eval.metrics import average_precision, daily_capacity_k, evaluate
    from kg_eval.splits import Fold
    from kg_features.registry import get_feature_set
    from kg_models.dataset import load_modelling_data
    from kg_sim.dataset import find_latest

    settings = get_settings()
    dataset_dir = dataset or find_latest(settings.data_dir)
    if dataset_dir is None:
        console.print("[red]no dataset found.[/] Run `kg sim generate` first.")
        raise typer.Exit(1)

    directory = settings.artifacts_dir / "kaggle" / "out" / notebook
    if not directory.exists():
        console.print(
            f"[red]nothing pulled for {notebook}.[/] Run `kg kaggle pull-artifacts {notebook}`."
        )
        raise typer.Exit(1)

    borrowed_from = settings.artifacts_dir / "kaggle" / "out" / tx_ids_from if tx_ids_from else None
    metrics, tx_ids, score_sets = _load_bundle(directory, borrowed_from)

    # Guard the join before doing anything with it. A fingerprint mismatch means
    # the GPU run used different data than this report describes, and the
    # numbers would be meaningless rather than merely wrong.
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    local_fingerprint = manifest.get("config_fingerprint", "")
    remote_fingerprint = metrics.get("dataset_fingerprint", "")
    if remote_fingerprint and local_fingerprint != remote_fingerprint:
        console.print(
            f"[red]dataset mismatch.[/] The GPU run used {remote_fingerprint}, "
            f"this dataset is {local_fingerprint}. Re-push the data, or pass "
            "--dataset for the one that was trained on."
        )
        raise typer.Exit(1)

    features_path = settings.artifacts_dir / "features" / f"{dataset_dir.name}.parquet"
    data = load_modelling_data(features_path, dataset_dir, get_feature_set())
    _, y_test, rows = data.matrix(Fold.TEST)
    local_ids = data.tx_ids[rows].astype(str)
    local_ts = data.timestamps[rows]
    local_amounts = data.amounts_cents[rows]

    # Join by id. Positional alignment is not assumed even when the lengths
    # match, because two independently-built orderings agreeing is a
    # coincidence, not an invariant.
    order = {tx: i for i, tx in enumerate(local_ids)}
    mapped = np.array([order.get(tx, -1) for tx in tx_ids])
    known = mapped >= 0
    if not known.any():
        console.print("[red]no transaction ids in common.[/] Wrong dataset or wrong fold.")
        raise typer.Exit(1)
    if not known.all():
        console.print(
            f"[yellow]{(~known).sum():,} of {len(tx_ids):,} scored ids are not in the "
            "local test fold; they are dropped from the comparison.[/]"
        )

    index = mapped[known]
    y_true = y_test[index]
    timestamps = local_ts[index]
    amounts = local_amounts[index]
    cost_model = CostModel()
    capacity = settings.analyst_capacity_per_day

    report_path = settings.artifacts_dir / "eval" / f"{dataset_dir.name}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = next(
        (r for r in report["results"] if r["model"] == "lightgbm" and r["fold"] == Fold.TEST.value),
        None,
    )

    table = Table("rung", "PR-AUC (recomputed)", "notebook said", "agrees", "recall@k")
    added: list[dict[str, Any]] = []

    for name, values in sorted(score_sets.items()):
        scores = np.asarray(values, dtype=np.float64)[known]

        # Operating point = the analyst capacity constraint, NOT a fitted
        # threshold. The protocol chooses thresholds on validation, and the GPU
        # notebooks exported test scores only; fitting one on test would be
        # leakage dressed as a cost figure. Taking the k-th highest score uses
        # the capacity budget and the score distribution, never the test labels,
        # which is the same basis precision@k already reports on.
        k_capacity = daily_capacity_k(timestamps, capacity)
        threshold = (
            float(np.partition(scores, -k_capacity)[-k_capacity])
            if 0 < k_capacity <= scores.size
            else None
        )

        bundle = evaluate(
            y_true,
            scores,
            timestamps,
            capacity_per_day=capacity,
            threshold=threshold,
            cost_model=cost_model,
            amounts_cents=amounts,
        )
        recomputed = bundle.value("pr_auc")

        # 01 reports under "results", 03 under "ladder". Both are checked.
        reported = metrics.get("results") or metrics.get("ladder") or {}
        claimed = (reported.get(name, {}) or {}).get("pr_auc")
        agrees = claimed is None or abs(recomputed - float(claimed)) <= METRIC_TOLERANCE

        # as_dict() already rounds and carries the extras (cost breakdown,
        # reliability curve) alongside "metrics"; spread it so an ingested rung
        # is shaped exactly like a locally-trained one.
        entry: dict[str, Any] = {
            "model": _rung(name),
            "fold": Fold.TEST.value,
            **bundle.as_dict(),
            "positives": int(y_true.sum()),
            "rows": len(y_true),
            "k": daily_capacity_k(timestamps, capacity),
            "source": f"kaggle:{notebook}",
            "threshold_basis": "capacity",
            "notebook_pr_auc": claimed,
            "recompute_agrees": bool(agrees),
        }

        # Bootstrap CIs, on the same day-blocked resampling as every other row.
        interval = bootstrap_metric(
            average_precision, y_true, scores, timestamps, resamples=resamples
        )
        entry["intervals"] = {"pr_auc": interval.as_dict()}

        added.append(entry)
        table.add_row(
            _rung(name),
            f"{recomputed:.4f}",
            "—" if claimed is None else f"{float(claimed):.4f}",
            "[green]yes[/]" if agrees else "[red]NO[/]",
            f"{bundle.value(f'recall_at_{capacity}_per_day'):.4f}",
        )

    console.print(table)

    disagreed = [e["model"] for e in added if not e["recompute_agrees"]]
    if disagreed:
        console.print(
            f"\n[red]recomputed metrics disagree with the notebook for: "
            f"{', '.join(disagreed)}.[/]\n"
            "The GPU and local metric implementations have drifted. "
            "tests/unit/test_kaggle_parity.py exists to prevent exactly this."
        )

    # Paired comparisons against the tabular baseline. The ladder's published
    # commitment is about THIS interval, not about which point estimate is
    # larger.
    comparisons = report.setdefault("comparisons", [])
    if baseline is not None:
        base_scores = _baseline_scores(settings, dataset_dir, index, known, tx_ids)
        if base_scores is not None:
            for entry, (_name, values) in zip(added, sorted(score_sets.items()), strict=True):
                scores = np.asarray(values, dtype=np.float64)[known]
                diff = bootstrap_difference(
                    average_precision,
                    y_true,
                    scores,
                    base_scores,
                    timestamps,
                    resamples=resamples,
                )
                comparisons.append(
                    {
                        "metric": "pr_auc",
                        "model": entry["model"],
                        "against": "lightgbm",
                        "difference": diff.as_dict(),
                        "conclusive": diff.excludes(0.0),
                    }
                )
                verdict = (
                    "[green]beats[/]"
                    if diff.excludes(0.0) and diff.point > 0
                    else "[yellow]does not separate from[/]"
                )
                console.print(f"{entry['model']} {verdict} lightgbm on pr_auc: {diff.format()}")
        else:
            console.print(
                "[yellow]no stored baseline scores;[/] skipping the paired comparison. "
                "Re-run `kg train baseline` to write them."
            )

    # Replace rather than append, so re-ingesting is idempotent.
    names = {e["model"] for e in added}
    report["results"] = [r for r in report["results"] if r["model"] not in names] + added
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"\n[green]written[/] {report_path}")
    console.print("Regenerate the published tables with `kg eval report --markdown`.")


def _baseline_scores(
    settings: Any, dataset_dir: Path, index: np.ndarray, known: np.ndarray, tx_ids: np.ndarray
) -> np.ndarray | None:
    """Load the stored LightGBM test scores, aligned to the same rows."""
    path = settings.artifacts_dir / "models" / dataset_dir.name / "lightgbm" / "test_scores.npz"
    if not path.exists():
        return None
    stored = np.load(path, allow_pickle=True)
    lookup = dict(zip(stored["tx_ids"].astype(str), stored["scores"], strict=True))
    aligned = np.array([lookup.get(tx, np.nan) for tx in tx_ids[known]], dtype=np.float64)
    return None if np.isnan(aligned).any() else aligned
