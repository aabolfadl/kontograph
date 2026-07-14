"""``kg lake`` and ``kg stream`` — the streaming spine and the lakehouse."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kg_core.config import get_settings
from kg_lake.medallion import LayerStats

lake_app = typer.Typer(
    name="lake", help="Iceberg lakehouse: bronze, silver, gold.", no_args_is_help=True
)
stream_app = typer.Typer(
    name="stream", help="Kafka ingest, schema registry and replay.", no_args_is_help=True
)
console = Console()


# ================================================================== stream ===


@stream_app.command("register")
def register_contracts() -> None:
    """Register every Avro contract with the Schema Registry."""
    from kg_stream.registry import register_all_contracts

    settings = get_settings()
    registered = register_all_contracts(settings.schema_registry_url)

    table = Table("contract", "subject", "schema id", "version")
    for name, entry in sorted(registered.items()):
        table.add_row(name, entry.subject, str(entry.schema_id), f"v{entry.version}")
    console.print(table)
    console.print(
        "\n[dim]Compatibility is pinned to BACKWARD server-side. The same rule is "
        "enforced without a broker by `kg contracts check` in CI.[/]"
    )


@stream_app.command("replay")
def replay(
    dataset: Path = typer.Option(None, "--dataset", help="Dataset dir (default: latest)."),
    mode: str = typer.Option(
        "as_fast_as_possible", "--mode", help="as_fast_as_possible | wall_clock | constant_rate"
    ),
    speed: float = typer.Option(10_000.0, "--speed", help="wall_clock multiplier."),
    rate: float = typer.Option(500.0, "--rate", help="constant_rate events/second."),
    limit: int = typer.Option(None, "--limit", help="Stop after N events."),
) -> None:
    """Replay a generated dataset onto the payments topic.

    ``wall_clock`` preserves the relative spacing between events, so the diurnal
    rhythm and the month-end payroll spike survive into the stream. That is what
    the live demo needs; a uniform firehose would show a flat graph.
    """
    from kg_sim.dataset import find_latest
    from kg_stream.producer import stream_dataset_events
    from kg_stream.replay import ReplayMode, replay_to_kafka

    settings = get_settings()
    target = dataset or find_latest(settings.data_dir)
    if target is None:
        console.print("[red]no dataset found.[/] Run `kg sim generate` first.")
        raise typer.Exit(1)

    console.print(f"replaying [bold]{target.name}[/] -> {settings.topic_payments} ({mode})")
    progress = replay_to_kafka(
        stream_dataset_events(target),
        mode=ReplayMode(mode),
        speed=speed,
        rate_per_second=rate,
        limit=limit,
    )
    console.print(
        f"[green]sent[/] {progress.sent:,} events in {progress.elapsed_seconds:.1f}s "
        f"({progress.rate:,.0f}/s)"
    )


@stream_app.command("consume")
def consume(
    limit: int = typer.Option(10, "--limit", "-n", help="Messages to show."),
    group: str = typer.Option("kg-cli-inspect", "--group"),
) -> None:
    """Consume and print messages, to eyeball what is actually on the topic."""
    from kg_stream.consumer import AvroConsumer

    settings = get_settings()
    with AvroConsumer("sct_inst_event.v1", [settings.topic_payments], group_id=group) as consumer:
        table = Table("offset", "key", "amount", "vop", "channel", "tx_id")
        for message in consumer.stream(max_messages=limit, idle_timeout=8.0):
            value = message.value
            table.add_row(
                f"p{message.partition}@{message.offset}",
                message.key or "",
                f"{value['amount_cents'] / 100:,.2f}",
                value["vop_result"],
                value["channel"],
                value["tx_id"][:13],
            )
        console.print(table)
        if consumer.stats.poison:
            console.print(f"[yellow]{consumer.stats.poison} undecodable message(s)[/]")


# ==================================================================== lake ===


@lake_app.command("init")
def init_tables() -> None:
    """Create the Iceberg namespaces and tables."""
    from kg_lake.medallion import ensure_tables

    tables = ensure_tables()
    console.print(f"[green]{len(tables)} table(s) ready:[/] {', '.join(sorted(tables))}")


@lake_app.command("reset")
def reset_tables(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Drop every managed Iceberg table and recreate it empty.

    Needed when switching to a different generated dataset: each one is a
    separate world that reuses account identifiers, so appending a second on top
    of the first merges two unrelated graphs.
    """
    from kg_lake.catalog import get_catalog
    from kg_lake.medallion import ensure_tables
    from kg_lake.schemas import TABLES

    if not yes:
        typer.confirm("This deletes every row in bronze and silver. Continue?", abort=True)

    catalog = get_catalog()
    dropped = []
    for identifier in TABLES:
        try:
            catalog.drop_table(identifier)
            dropped.append(identifier)
        except Exception:
            pass
    ensure_tables(catalog)
    console.print(f"[green]reset[/] {len(dropped)} table(s): {', '.join(dropped)}")


@lake_app.command("ingest")
def ingest(
    dataset: Path = typer.Option(None, "--dataset", help="Dataset dir (default: latest)."),
    from_kafka: bool = typer.Option(
        False, "--from-kafka", help="Consume from the topic instead of the Parquet backfill."
    ),
    limit: int = typer.Option(None, "--limit", help="Max messages when consuming."),
) -> None:
    """Load payments into bronze — from Parquet (backfill) or Kafka (stream)."""
    from kg_lake.medallion import (
        bronze_rows_from_messages,
        ingest_from_dataset,
        load_dimension,
        write_bronze,
    )
    from kg_sim.dataset import find_latest

    settings = get_settings()

    if from_kafka:
        from kg_stream.consumer import AvroConsumer

        with AvroConsumer(
            "sct_inst_event.v1", [settings.topic_payments], group_id="kg-lake-bronze"
        ) as consumer:
            messages = list(consumer.stream(max_messages=limit, idle_timeout=10.0))
            stats = write_bronze(list(bronze_rows_from_messages(messages)))
            # Offsets are committed only after bronze is durably written.
            consumer.commit()
        console.print(f"[green]bronze[/] +{stats.rows_written:,} rows from Kafka")
        return

    target = dataset or find_latest(settings.data_dir)
    if target is None:
        console.print("[red]no dataset found.[/] Run `kg sim generate` first.")
        raise typer.Exit(1)

    for dimension in ("accounts", "labels"):
        loaded = load_dimension(target, dimension)
        console.print(f"  silver.{dimension}: {loaded.rows_written:,} rows")

    started = time.perf_counter()
    stats = ingest_from_dataset(target)
    console.print(
        f"[green]bronze[/] +{stats.rows_written:,} rows "
        f"[dim]({time.perf_counter() - started:.1f}s)[/]"
    )


@lake_app.command("promote")
def promote(
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Block the promotion on a failed expectation."
    ),
) -> None:
    """Promote bronze to silver through the data-quality gate."""
    from kg_lake.medallion import promote_to_silver
    from kg_lake.quality import DataQualityError

    try:
        stats = promote_to_silver(strict=strict)
    except DataQualityError as error:
        console.print(f"[red]PROMOTION BLOCKED[/]\n{error}")
        console.print(
            "\n[dim]The batch stays in bronze. Fix the producer or the expectation, "
            "then re-run — bronze is append-only and nothing was lost.[/]"
        )
        raise typer.Exit(1) from None

    _print_quality(stats)


def _print_quality(stats: LayerStats) -> None:
    report = stats.report
    console.print(
        f"[green]silver[/] +{stats.rows_written:,} rows, {stats.rows_quarantined:,} quarantined"
    )
    if report is None:
        return

    table = Table("severity", "expectation", "failed", "rate", "status")
    for result in report.results:
        style = "green" if result.passed else ("red" if result.severity == "BLOCKING" else "yellow")
        table.add_row(
            result.severity.value,
            result.name,
            f"{result.failed_rows:,}",
            f"{result.failure_rate:.2%}",
            f"[{style}]{'pass' if result.passed else 'FAIL'}[/]",
        )
    console.print(table)


@lake_app.command("quality")
def quality(
    layer: str = typer.Option("bronze", "--layer", help="bronze | silver"),
    out: Path = typer.Option(None, "--out", help="Write the report as JSON."),
) -> None:
    """Run the quality suite and print (or write) the report."""
    from kg_lake.catalog import get_catalog
    from kg_lake.quality import validate

    identifier = "bronze.payments" if layer == "bronze" else "silver.payments"
    table = get_catalog().load_table(identifier).scan().to_arrow()
    report = validate(table, "payments")

    console.print(f"[bold]{identifier}[/] — {table.num_rows:,} rows")
    grouped = Table("severity", "expectation", "failed", "rate", "status")
    for result in report.results:
        style = "green" if result.passed else ("red" if result.severity == "BLOCKING" else "yellow")
        grouped.add_row(
            result.severity.value,
            result.name,
            f"{result.failed_rows:,}",
            f"{result.failure_rate:.2%}",
            f"[{style}]{'pass' if result.passed else 'FAIL'}[/]",
        )
    console.print(grouped)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
        console.print(f"[green]report written[/] {out}")

    if not report.ok:
        raise typer.Exit(1)


@lake_app.command("export")
def export() -> None:
    """Materialise silver into DuckDB for the analytics marts."""
    from kg_lake.duck import export_to_duckdb, warehouse_db_path

    provenance = export_to_duckdb()
    table = Table("iceberg table", "duckdb table", "rows", "snapshot")
    for identifier, meta in provenance.items():
        table.add_row(
            identifier, meta["duckdb_table"], f"{meta['rows']:,}", str(meta["snapshot_id"])
        )
    console.print(table)
    console.print(f"[green]exported[/] {warehouse_db_path()}")


@lake_app.command("dbt")
def run_dbt(
    command: str = typer.Argument("build", help="dbt subcommand: build | run | test | docs"),
) -> None:
    """Build the gold marts with dbt-duckdb.

    Requires the ``orchestration`` extra: ``uv sync --extra orchestration``.
    """
    from kg_lake.duck import warehouse_db_path

    settings = get_settings()
    project = settings.repo_root / "dbt" / "kontograph"
    if not warehouse_db_path().exists():
        console.print("[red]no analytics database.[/] Run `kg lake export` first.")
        raise typer.Exit(1)

    args = ["dbt", command, "--project-dir", str(project), "--profiles-dir", str(project)]
    console.print(f"[dim]$ {' '.join(args)}[/]")
    try:
        result = subprocess.run(
            args,
            cwd=project,
            env={
                **__import__("os").environ,
                "KG_DUCKDB_PATH": str(warehouse_db_path()),
            },
            check=False,
        )
    except FileNotFoundError:
        console.print("[red]dbt is not installed.[/] Run: uv sync --extra orchestration")
        raise typer.Exit(1) from None
    if result.returncode:
        raise typer.Exit(result.returncode)


@lake_app.command("status")
def status() -> None:
    """Row counts and Iceberg snapshot history for every managed table."""
    from kg_lake.medallion import datasets_present, layer_counts, snapshot_history

    present = datasets_present()
    if len(present) > 1:
        console.print(
            f"[yellow]silver holds {len(present)} source datasets:[/] "
            + ", ".join(f"{name} ({rows:,})" for name, rows in present.items())
            + "\n[dim]Each is a separate simulated world reusing account ids. "
            "Run `kg lake reset` before loading a different one.[/]\n"
        )
    counts = layer_counts()
    table = Table("table", "rows", "snapshots")
    for identifier, rows in counts.items():
        try:
            snapshots = len(snapshot_history(identifier))
        except Exception:
            snapshots = 0
        table.add_row(identifier, f"{rows:,}", str(snapshots))
    console.print(table)


@lake_app.command("history")
def history(identifier: str = typer.Argument("bronze.payments")) -> None:
    """Iceberg snapshot history — the time-travel handle for lineage."""
    from kg_core.timeutil import iso
    from kg_lake.medallion import snapshot_history

    table = Table("snapshot id", "parent", "when", "operation", "added records")
    for snapshot in snapshot_history(identifier):
        table.add_row(
            str(snapshot["snapshot_id"]),
            str(snapshot["parent_id"] or "-"),
            iso(snapshot["timestamp_ms"]),
            str(snapshot["operation"] or "-"),
            str(snapshot["added_records"] or "-"),
        )
    console.print(table)
