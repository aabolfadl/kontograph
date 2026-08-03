"""``kg gov`` — regenerate the governance bundle from artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kg_core.config import get_settings

gov_app = typer.Typer(
    name="gov",
    help="Model cards, lineage, decision-log audit — generated, never typed.",
    no_args_is_help=True,
)
console = Console()


def _report_path() -> Path:
    from kg_sim.dataset import find_latest

    settings = get_settings()
    dataset = find_latest(settings.data_dir)
    if dataset is None:
        console.print("[red]no dataset.[/] Run `kg sim generate`.")
        raise typer.Exit(1)
    path = settings.artifacts_dir / "eval" / f"{dataset.name}.json"
    if not path.exists():
        console.print(f"[red]no evaluation report at {path}.[/] Run `kg train baseline`.")
        raise typer.Exit(1)
    return path


@gov_app.command("model-card")
def model_card(
    model: str = typer.Option("lightgbm", "--model"),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Generate the model card from the evaluation report."""
    from kg_gov.modelcard import generate

    settings = get_settings()
    destination = out or (settings.repo_root / "docs" / "model-cards" / f"{model}.md")
    generate(_report_path(), destination, model=model)
    console.print(f"[green]written[/] {destination}")


@gov_app.command("lineage")
def lineage(out: Path = typer.Option(None, "--out")) -> None:
    """Render the data-lineage document from the Dagster asset graph."""
    from kg_gov.lineage import collect, generate
    from kg_sim.dataset import find_latest

    settings = get_settings()
    dataset = find_latest(settings.data_dir)
    destination = out or (settings.repo_root / "docs" / "lineage.md")
    generate(destination, dataset=dataset.name if dataset else None)

    graph = collect()
    console.print(
        f"[green]written[/] {destination}\n"
        f"  {len(graph.assets)} assets, {len(graph.edges)} edges — "
        + ("derived from the Dagster graph" if graph.derived else "[yellow]static fallback[/]")
    )


@gov_app.command("decision-log")
def decision_log() -> None:
    """Audit the hash-chained decision log."""
    from kg_serve.decision_log import DecisionLog

    settings = get_settings()
    log = DecisionLog(settings.artifacts_dir / "decisions" / "decisions.jsonl")
    result = log.verify()

    table = Table("check", "result")
    table.add_row("entries", f"{result['entries']:,}")
    table.add_row("chain intact", "[green]yes[/]" if result["intact"] else "[red]NO[/]")
    if not result["intact"]:
        table.add_row("broken at", str(result["broken_at"]))
        table.add_row("trustworthy prefix", str(result["trustworthy_prefix"]))
    console.print(table)

    if result["entries"]:
        entries = log.entries()
        sample = entries[-1]["payload"]
        console.print("\n[bold]Most recent decision record[/]")
        console.print(json.dumps(sample, indent=2)[:1200])
        console.print(
            "\n[dim]Every field here is written AT decision time. Reconstructing it later "
            "from a model file and a guess about which version was live is not the same "
            "thing, and an auditor will say so.[/]"
        )
    if not result["intact"]:
        raise typer.Exit(1)


@gov_app.command("sar")
def sar(
    alert: str = typer.Option(None, "--alert", help="Alert id; defaults to the latest."),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Draft a bilingual SAR narrative for a logged decision."""
    from kg_gov.sar import draft, render_markdown
    from kg_serve.decision_log import DecisionLog

    settings = get_settings()
    log = DecisionLog(settings.artifacts_dir / "decisions" / "decisions.jsonl")
    entries = [e for e in log.entries() if e["payload"].get("decision") != "ALLOW"]
    if not entries:
        console.print(
            "[yellow]no alerts in the decision log.[/] Run `kg serve smoke` or score some "
            "payments through the gateway first."
        )
        raise typer.Exit(1)

    if alert:
        # An explicit --alert that silently fell back to the latest would hand
        # someone a narrative about a different payment than the one they asked
        # for, under the id they typed. Fail instead.
        chosen = next((e for e in entries if e["payload"]["alert_id"] == alert), None)
        if chosen is None:
            console.print(f"[red]no alert {alert!r} in the decision log.[/]")
            raise typer.Exit(1)
    else:
        chosen = entries[-1]
    payload = chosen["payload"]
    explanation = payload.get("explanation") or {}

    document = draft(
        alert_id=payload["alert_id"],
        tx_id=payload["tx_id"],
        debtor=payload.get("debtor_account_id", "—"),
        creditor=payload.get("creditor_account_id", "—"),
        amount_cents=payload.get("amount_cents", 0),
        # When the payment was accepted, not when the model scored it. The two
        # differ by the decision latency, and a SAR narrative dates the payment.
        acceptance_ts=payload.get("acceptance_ts") or payload.get("scored_ts", 0),
        score=payload["score"],
        threshold=payload["threshold"],
        counterfactual=explanation.get("counterfactual"),
        vop_result=payload.get("vop_result") or "MATCH",
        vop_overridden=bool(payload.get("vop_overridden", False)),
        is_new_beneficiary=bool(payload.get("is_new_beneficiary", False)),
    )

    text = render_markdown(document)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]written[/] {out}")
    else:
        console.print(text)


@gov_app.command("all")
def generate_all() -> None:
    """Regenerate the whole governance bundle.

    the engineering invariants: no unreproducible number, and no claimed capability without
    an artifact. Everything this produces is derived from run artifacts, so a
    change to the model produces a changed bundle rather than a stale one.
    """
    from kg_gov.lineage import generate as generate_lineage
    from kg_gov.modelcard import generate as generate_card
    from kg_sim.dataset import find_latest

    settings = get_settings()
    dataset = find_latest(settings.data_dir)
    produced: list[Path] = []

    report = _report_path()
    for model in ("lightgbm", "constant"):
        try:
            produced.append(
                generate_card(
                    report, settings.repo_root / "docs" / "model-cards" / f"{model}.md", model=model
                )
            )
        except Exception as error:
            console.print(f"[yellow]skipped {model}: {error}[/]")

    produced.append(
        generate_lineage(
            settings.repo_root / "docs" / "lineage.md",
            dataset=dataset.name if dataset else None,
        )
    )

    latency = settings.artifacts_dir / "loadtest" / "latency_budget.md"
    if latency.exists():
        produced.append(latency)

    table = Table("artifact", "size")
    for path in produced:
        table.add_row(str(path.relative_to(settings.repo_root)), f"{path.stat().st_size:,} bytes")
    console.print(table)

    console.print(
        "\n[dim]The AI Act memo (docs/ai-act-memo.md) is written rather than generated: "
        "it is a legal analysis, and a generated legal analysis would be worth less than "
        "the paper it is not printed on. It is reviewed and dated.[/]"
    )
