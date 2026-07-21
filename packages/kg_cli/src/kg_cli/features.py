"""``kg features`` — inspect, compile and verify the feature platform."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from kg_core.config import get_settings

features_app = typer.Typer(
    name="features",
    help="One feature definition, three compiled backends, one verified answer.",
    no_args_is_help=True,
)
console = Console()


@features_app.command("list")
def list_features(
    feature_set: str = typer.Option("payments", "--set", "-s"),
) -> None:
    """Show every feature, and which backends can compute it."""
    from kg_features.registry import get_feature_set

    fs = get_feature_set(feature_set)
    table = Table("feature", "entity", "aggregation", "window", "streams", "description")
    for row in fs.summary():
        table.add_row(
            row["name"],
            row["entity"],
            row["aggregation"],
            row["window"],
            "[green]yes[/]" if row["streams"] else "[yellow]online only[/]",
            (row["description"] or "")[:60],
        )
    console.print(table)

    streamable = len(fs.flink_compatible())
    console.print(
        f"\n[bold]{len(fs)}[/] features · fingerprint [bold]{fs.fingerprint()[:12]}[/] · "
        f"max window {fs.max_window_millis / 86_400_000:.0f}d (the evaluation purge width)"
    )
    console.print(
        f"[dim]{streamable}/{len(fs)} compile to Flink SQL. The rest use non-subtractable "
        "aggregations (MAX/MIN/DISTINCT/TIME_SINCE_LAST); a Flink OVER frame must end at "
        "CURRENT ROW, so their current-row contribution cannot be removed. They are computed "
        "by the online executor instead. See ADR-0005 and kg_features.compilers.flink.[/]"
    )


@features_app.command("compile")
def compile_features(
    backend: str = typer.Option("duckdb", "--backend", "-b", help="duckdb | flink"),
    feature_set: str = typer.Option("payments", "--set", "-s"),
    out: Path = typer.Option(None, "--out", help="Write the SQL to a file."),
) -> None:
    """Emit the generated SQL for a backend."""
    from kg_features.registry import get_feature_set

    fs = get_feature_set(feature_set)

    if backend == "duckdb":
        from kg_features.compilers.duckdb_sql import compile_feature_set as compile_backfill

        compiled = compile_backfill(fs)
        sql, note = compiled.sql, f"{compiled.groups} scan groups"
    elif backend == "flink":
        from kg_features.compilers.flink import compile_feature_set as compile_stream

        settings = get_settings()
        stream = compile_stream(
            fs,
            topic=settings.topic_payments,
            bootstrap=settings.kafka_bootstrap.replace("localhost:19092", "redpanda:9092"),
            registry=settings.schema_registry_url.replace("localhost:18081", "redpanda:8081"),
        )
        sql = f"{stream.ddl}\n\n{stream.sql};"
        note = f"{len(stream.columns) - 4} features; {len(stream.skipped)} not streamable"
        if stream.skipped:
            console.print(
                f"[yellow]not compiled to Flink:[/] {', '.join(stream.skipped)}\n"
                "[dim]non-subtractable aggregations; computed by the online executor.[/]\n"
            )
    else:
        console.print(f"[red]unknown backend {backend!r}[/]; use duckdb or flink")
        raise typer.Exit(1)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(sql, encoding="utf-8")
        console.print(f"[green]written[/] {out}  [dim]({note})[/]")
    else:
        console.print(Syntax(sql, "sql", theme="ansi_dark", line_numbers=False))
        console.print(f"[dim]{note}[/]")


@features_app.command("verify")
def verify(
    events: int = typer.Option(400, "--events", "-n", help="How many events to test on."),
    feature_set: str = typer.Option("payments", "--set", "-s"),
    dataset: Path = typer.Option(None, "--dataset", help="Use real data instead of synthetic."),
) -> None:
    """Prove the backends agree and that nothing sees the future.

    This is the anti-skew claim, executed. It runs the same event stream through
    the DuckDB backfill and the incremental online executor, diffs every value,
    then perturbs the future and checks nothing in the past moved.
    """
    from kg_features.registry import get_feature_set
    from kg_features.verify import compare_backends, find_lookahead

    fs = get_feature_set(feature_set)
    stream = _load_events(dataset, events)
    console.print(f"verifying [bold]{len(fs)}[/] features over [bold]{len(stream):,}[/] events…\n")

    started = time.perf_counter()
    report = compare_backends(fs, stream)
    elapsed = time.perf_counter() - started

    summary = report.summary()
    table = Table("check", "result")
    table.add_row("values compared", f"{summary['values_compared']:,}")
    table.add_row(
        "backfill == online",
        "[green]agree[/]" if report.agrees else f"[red]{summary['divergences']} divergences[/]",
    )
    console.print(table)

    if not report.agrees:
        for divergence in report.divergences[:10]:
            console.print(f"  [red]{divergence}[/]")
        console.print(
            "\n[red]This is training-serving skew.[/] The model would be trained on one "
            "number and served another."
        )
        raise typer.Exit(1)

    console.print("\nchecking for lookahead (perturbing the future)…")
    findings = find_lookahead(fs, stream[: min(len(stream), 200)], backend="duckdb")
    if findings:
        for finding in findings[:10]:
            console.print(f"  [red]{finding}[/]")
        console.print(f"\n[red]LEAKAGE: {len(findings)} value(s) depend on the future.[/]")
        raise typer.Exit(1)

    console.print(
        f"[green]no lookahead[/] — no feature changed when later events were altered\n"
        f"\n[dim]{elapsed:.1f}s. Design principles P2 and P3, as an executable assertion.[/]"
    )


def _load_events(dataset: Path | None, limit: int) -> list[dict]:
    """Load events to compute features over, preferring the silver layer.

    Features are defined against **silver**, not against raw simulator output.
    Some of them read columns that only exist after the bronze->silver
    promotion — ``is_cross_border`` and ``is_cross_institution`` are derived
    there by joining the account dimension. Reading the raw Parquet instead
    produces a DuckDB binder error about a missing column, which is a confusing
    way to be told "run the pipeline first".

    Order of preference:
      1. an explicit dataset directory, if given
      2. silver.payments from the Iceberg lakehouse
      3. the raw dataset, with the feature set checked against it first
      4. a freshly simulated stream
    """
    from kg_features.ir import validate_against_schema
    from kg_features.registry import get_feature_set

    settings = get_settings()

    if dataset is None:
        rows = _load_from_silver(limit)
        if rows is not None:
            return rows

    from kg_sim.dataset import find_latest

    target = dataset or find_latest(settings.data_dir)
    if target is not None and (target / "events.parquet").exists():
        import pyarrow.parquet as pq

        table = pq.read_table(target / "events.parquet")
        problems = validate_against_schema(get_feature_set(), set(table.column_names))
        if problems:
            console.print(
                f"[red]{target.name} cannot supply every feature:[/]\n"
                + "\n".join(f"  - {problem}" for problem in problems)
                + "\n\n[dim]Features are defined against the SILVER layer, which derives "
                "columns like is_cross_border by joining the account dimension. Run:\n"
                "  kg lake ingest && kg lake promote && kg lake export[/]"
            )
            raise typer.Exit(1)
        console.print(f"[dim]using raw dataset {target.name}[/]")
        return table.slice(0, limit).to_pylist()

    console.print("[dim]no dataset found; generating a synthetic stream[/]")
    from kg_lake.medallion import silver_rows_from_events
    from kg_sim.config import get_profile
    from kg_sim.generator import simulate

    result = simulate(get_profile("smoke"))
    return silver_rows_from_events(result.events[:limit], result.accounts)


def _load_from_silver(limit: int) -> list[dict] | None:
    """Read silver.payments in event-time order, or None if it is not built."""
    try:
        from kg_lake.duck import query, warehouse_db_path
    except ImportError:
        return None

    if not warehouse_db_path().exists():
        return None
    try:
        table = query(
            f"SELECT * FROM silver.payments ORDER BY acceptance_ts, tx_id LIMIT {int(limit)}"
        )
    except Exception:
        return None
    if table.num_rows == 0:
        return None
    console.print(f"[dim]using silver.payments ({table.num_rows:,} rows)[/]")
    return table.to_pylist()


@features_app.command("build")
def build(
    dataset: Path = typer.Option(None, "--dataset"),
    feature_set: str = typer.Option("payments", "--set", "-s"),
    out: Path = typer.Option(None, "--out", help="Parquet output path."),
) -> None:
    """Materialise the training feature table from silver."""
    import pyarrow.parquet as pq

    from kg_features.compilers.duckdb_sql import execute_chunked
    from kg_features.ir import validate_against_schema
    from kg_features.registry import get_feature_set
    from kg_sim.dataset import find_latest

    settings = get_settings()
    settings.ensure_dirs()
    fs = get_feature_set(feature_set)
    target = dataset or find_latest(settings.data_dir)
    if target is None:
        console.print("[red]no dataset found.[/] Run `kg sim generate` first.")
        raise typer.Exit(1)

    import pyarrow as pa

    silver = _load_from_silver(limit=10_000_000)
    if silver is not None:
        events = pa.Table.from_pylist(silver)
    else:
        events = pq.read_table(target / "events.parquet")
        problems = validate_against_schema(fs, set(events.column_names))
        if problems:
            console.print(
                "[red]the raw dataset cannot supply every feature.[/] "
                "Features are defined against silver; run "
                "`kg lake ingest && kg lake promote && kg lake export` first.\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )
            raise typer.Exit(1)

    console.print(f"building {len(fs)} features over {events.num_rows:,} events…")

    started = time.perf_counter()

    def _progress(done: int, total: int, rows: int) -> None:
        console.print(f"  [dim]chunk {done}/{total}: {rows:,} rows[/]")

    # Chunked: the backfill is a self-join whose intermediate size scales with
    # events-in-window, and a high-in-degree account (a utility provider with
    # thousands of payers) makes a single pass exhaust DuckDB's spill budget.
    table = execute_chunked(fs, events, on_progress=_progress)
    elapsed = time.perf_counter() - started

    destination = out or (settings.artifacts_dir / "features" / f"{target.name}.parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")

    console.print(
        f"[green]written[/] {destination}\n"
        f"  {table.num_rows:,} rows x {table.num_columns} columns in {elapsed:.1f}s\n"
        f"  feature-set fingerprint [bold]{fs.fingerprint()}[/]"
    )
    console.print(
        "\n[dim]The fingerprint is stamped onto every model and every decision. "
        "A score is only comparable to another computed under the same fingerprint.[/]"
    )


@features_app.command("fingerprint")
def fingerprint(feature_set: str = typer.Option("payments", "--set", "-s")) -> None:
    """Print the feature-set fingerprint and each feature's own hash."""
    from kg_features.registry import get_feature_set

    fs = get_feature_set(feature_set)
    console.print(
        json.dumps(
            {
                "feature_set": fs.name,
                "version": fs.version,
                "fingerprint": fs.fingerprint(),
                "max_window_ms": fs.max_window_millis,
                "features": {f.name: f.fingerprint() for f in fs},
            },
            indent=2,
        )
    )
