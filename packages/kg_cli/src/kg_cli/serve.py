"""``kg serve`` — run the gateway, load-test it, and publish the latency budget."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kg_core.config import get_settings

serve_app = typer.Typer(
    name="serve", help="The decision gateway and its latency budget.", no_args_is_help=True
)
console = Console()


@serve_app.command("run")
def run(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
    warm: bool = typer.Option(True, "--warm/--cold", help="Warm the feature state on boot."),
) -> None:
    """Run the FastAPI decision gateway.

    Warms by default. A cold gateway is *silently* wrong — every feature reads
    its default and the model scores against a world where nobody has ever
    transacted — so serving cold has to be an explicit choice.
    """
    import uvicorn

    settings = get_settings()
    console.print(
        f"[bold]KONTOGRAPH gateway[/] on http://{host}:{port}\n"
        f"  SLA p99 <= {settings.sla_p99_millis} ms\n"
        f"  /score /health /latency /metrics /decisions/verify\n"
        + ("  warming on boot\n" if warm else "  [yellow]starting COLD[/]\n")
    )
    if warm:
        import os

        os.environ["KG_WARM_ON_BOOT"] = "1"

    uvicorn.run(
        "kg_serve.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
        # One worker: the online feature state lives in this process. Multiple
        # workers would each hold a different partial view of every account,
        # and score the same payment differently depending on which one got it.
        # Sharing state properly means Redis, which is the next step, not a
        # flag.
        workers=1,
    )


@serve_app.command("smoke")
def smoke(
    events: int = typer.Option(500, "--events", "-n"),
    warm: int = typer.Option(20_000, "--warm", help="Events to fold in before scoring."),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Score real payments in-process and emit the latency budget.

    In-process rather than over HTTP: this measures the *decision path*, which
    is what the architecture claims a budget for. The k6 run adds the network
    and the server, and both numbers are published — a p99 that excludes the
    transport is a partial answer, and one that hides the stage breakdown is
    unactionable.
    """
    import pyarrow.parquet as pq

    from kg_features.registry import get_feature_set
    from kg_models.baseline import TrainedModel
    from kg_serve.latency import LatencyBudget
    from kg_serve.scoring import FeatureAttributionExplainer, ScoringEngine
    from kg_sim.dataset import find_latest

    settings = get_settings()
    settings.ensure_dirs()
    dataset = find_latest(settings.data_dir)
    if dataset is None:
        console.print("[red]no dataset.[/] Run `kg sim generate`.")
        raise typer.Exit(1)

    directory = settings.artifacts_dir / "models" / dataset.name / "lightgbm"
    if not directory.exists():
        console.print(f"[red]no model at {directory}.[/] Run `kg train baseline`.")
        raise typer.Exit(1)

    model = TrainedModel.load(directory)
    feature_set = get_feature_set()
    engine = ScoringEngine(
        feature_set,
        model,
        threshold=model.threshold,
        explainer=FeatureAttributionExplainer(model, feature_set.column_names),
    )

    parquet = pq.ParquetFile(dataset / "events.parquet")
    stream: list[dict] = []
    for batch in parquet.iter_batches(batch_size=10_000):
        stream.extend(batch.to_pylist())
        if len(stream) >= warm + events:
            break

    console.print(f"warming with {min(warm, len(stream)):,} events…")
    engine.warm(stream[:warm])

    budget = LatencyBudget(sla_p99_ms=float(settings.sla_p99_millis))
    decisions = {"ALLOW": 0, "ALERT": 0, "HOLD": 0, "BLOCK": 0}
    explained = 0

    console.print(f"scoring {events:,} payments…")
    started = time.perf_counter()
    for event in stream[warm : warm + events]:
        decision = engine.score(event)
        budget.record(decision.latency)
        decisions[decision.decision.value] += 1
        if decision.explanation:
            explained += 1
    elapsed = time.perf_counter() - started

    console.print(
        f"\n[bold]{budget.count:,} decisions in {elapsed:.2f}s[/] "
        f"({budget.count / elapsed:,.0f}/s single-threaded)\n"
    )

    table = Table("stage", "p50", "p95", "p99", "max", "share of p99")
    for row in budget.table():
        style = "bold" if row["stage"] == "total" else ""
        table.add_row(
            f"[{style}]{row['stage']}[/]" if style else row["stage"],
            f"{row['p50_ms']:.3f} ms",
            f"{row['p95_ms']:.3f} ms",
            f"[{style}]{row['p99_ms']:.3f} ms[/]" if style else f"{row['p99_ms']:.3f} ms",
            f"{row['max_ms']:.3f} ms",
            "" if row["stage"] == "total" else f"{row['share_of_p99']:.0%}",
        )
    console.print(table)

    total = budget.quantiles("total")
    if budget.meets_sla:
        console.print(
            f"\n[green]PASS[/] — p99 of {total['p99']:.2f} ms leaves "
            f"{budget.sla_p99_ms - total['p99']:.1f} ms of headroom against the "
            f"{budget.sla_p99_ms:.0f} ms SLA."
        )
    else:
        console.print(
            f"\n[red]FAIL[/] — p99 of {total['p99']:.2f} ms exceeds the "
            f"{budget.sla_p99_ms:.0f} ms SLA."
        )

    console.print(f"\ndecisions: {decisions}   explained: {explained:,}")

    destination = out or (settings.artifacts_dir / "loadtest" / "latency_budget.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(budget.as_dict(), indent=2), encoding="utf-8")
    (destination.parent / "latency_budget.md").write_text(
        budget.render_markdown(), encoding="utf-8"
    )
    console.print(f"\n[green]written[/] {destination} and latency_budget.md")

    if not budget.meets_sla:
        raise typer.Exit(1)


@serve_app.command("export-onnx")
def export_onnx(
    dataset: Path = typer.Option(None, "--dataset"),
    model_name: str = typer.Option("lightgbm", "--model"),
    rows: int = typer.Option(0, "--rows", help="Limit parity rows; 0 uses the whole test fold."),
) -> None:
    """Export the trained booster to ONNX and measure its parity with it.

    The export is not the deliverable — the *parity report* is. An ONNX graph
    that disagrees with the model that was evaluated is a different model, and
    shipping it on the strength of "the conversion succeeded" is how a serving
    stack quietly stops matching its own evaluation.
    """
    from kg_eval.splits import Fold
    from kg_features.registry import get_feature_set
    from kg_models.baseline import TrainedModel
    from kg_models.dataset import load_modelling_data
    from kg_serve.onnx_export import export_booster, verify_parity
    from kg_sim.dataset import find_latest

    settings = get_settings()
    dataset_dir = dataset or find_latest(settings.data_dir)
    if dataset_dir is None:
        console.print("[red]no dataset found.[/] Run `kg sim generate` first.")
        raise typer.Exit(1)

    directory = settings.artifacts_dir / "models" / dataset_dir.name / model_name
    if not directory.exists():
        console.print(f"[red]no trained model at {directory}.[/] Run `kg train baseline` first.")
        raise typer.Exit(1)

    model = TrainedModel.load(directory)
    path = export_booster(model, directory / "model.onnx")

    features_path = settings.artifacts_dir / "features" / f"{dataset_dir.name}.parquet"
    data = load_modelling_data(features_path, dataset_dir, get_feature_set())
    X, _, _ = data.matrix(Fold.TEST)
    if rows:
        X = X[:rows]

    report = verify_parity(model, path, X)
    payload = report.as_dict()

    table = Table("measure", "value")
    table.add_row("rows compared", f"{payload['rows']:,}")
    table.add_row("max score difference", f"{payload['max_abs_diff']:.3e}")
    table.add_row("mean score difference", f"{payload['mean_abs_diff']:.3e}")
    table.add_row("decisions changed", f"{payload['decisions_changed']:,}")
    table.add_row("  as a share", f"{payload['decisions_changed_share']:.4%}")
    table.add_row("alerts (booster)", f"{payload['alerts_booster']:,}")
    table.add_row("alerts (ONNX)", f"{payload['alerts_onnx']:,}")
    table.add_row("alert-set Jaccard", f"{payload['alert_jaccard']:.4f}")
    console.print(table)

    out = settings.artifacts_dir / "models" / dataset_dir.name / model_name / "onnx_parity.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]written[/] {out}")

    if report.ok:
        console.print("\n[green]PARITY HOLDS[/] — the ONNX graph reproduces the booster exactly.")
        return

    # Reported, not raised. The export is a legitimate portability artifact even
    # when it cannot be the serving path; what must not happen is it being
    # swapped in silently. See ADR-0008.
    console.print(
        f"\n[yellow]PARITY GAP[/] — {payload['decisions_changed']:,} of {payload['rows']:,} "
        f"decisions differ ({payload['decisions_changed_share']:.4%}).\n"
        "The ONNX operator set scores tree ensembles in float32; LightGBM uses float64 "
        "internally, and the converter rejects a double-precision input for classifiers.\n"
        "At an operating threshold of "
        f"{payload['threshold']:.2e} that rounding is enough to move scores across it.\n"
        "[bold]The gateway therefore keeps serving the booster[/], and this graph ships as a "
        "portability artifact with the gap recorded. ADR-0008 has the argument."
    )


@serve_app.command("loadtest")
def loadtest(
    rate: int = typer.Option(200, "--rate", help="Arrival rate per second."),
    duration: str = typer.Option("30s", "--duration"),
    base: str = typer.Option("http://localhost:8000", "--base"),
) -> None:
    """Run the k6 load test against a running gateway. Fails if p99 exceeds the SLA."""
    settings = get_settings()
    script = settings.repo_root / "loadtest" / "score.js"

    args = [
        "k6",
        "run",
        "-e",
        f"RATE={rate}",
        "-e",
        f"DURATION={duration}",
        "-e",
        f"BASE={base}",
        "-e",
        f"SLA_MS={settings.sla_p99_millis}",
        str(script),
    ]
    console.print(f"[dim]$ {' '.join(args)}[/]")
    try:
        result = subprocess.run(args, cwd=settings.repo_root, check=False)
    except FileNotFoundError:
        console.print(
            "[red]k6 is not installed.[/]\n"
            "  winget install k6 --source winget\n"
            "  or: docker run --rm -i --network host grafana/k6 run - < loadtest/score.js\n\n"
            "[dim]`kg serve smoke` measures the decision path in-process without k6.[/]"
        )
        raise typer.Exit(1) from None
    if result.returncode:
        console.print("[red]the load test failed its thresholds.[/]")
        raise typer.Exit(result.returncode)


@serve_app.command("verify-log")
def verify_log() -> None:
    """Audit the decision log's hash chain."""
    from kg_serve.decision_log import DecisionLog

    settings = get_settings()
    log = DecisionLog(settings.artifacts_dir / "decisions" / "decisions.jsonl")
    result = log.verify()

    if result["intact"]:
        console.print(
            f"[green]chain intact[/] — {result['entries']:,} decisions, "
            "each committing to the one before it."
        )
    else:
        console.print(
            f"[red]CHAIN BROKEN at entry {result['broken_at']}[/]\n"
            f"  entries 0..{result['trustworthy_prefix'] - 1} are still verifiable; "
            "everything after that instant is not."
        )
        raise typer.Exit(1)
