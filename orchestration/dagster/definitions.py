"""Dagster software-defined assets for the KONTOGRAPH pipeline.

The platform scope asks for "assets, not cron jobs". The distinction is not cosmetic:
a cron job says *when to run something*, an asset says *what should exist and
what it is derived from*. The second is what produces a lineage graph you can
hand an auditor, and this project needs one.

Two Dagster features are doing real work here:

**Asset checks with ``blocking=True``.** The quality suite is expressed as
Dagster asset checks rather than as assertions buried in the transform. A
blocking check that fails stops downstream assets from materialising, which is
exactly the "pipeline-blocking assertion" the platform scope asks for — expressed in the
orchestrator, where it is visible in the UI and in the run history, instead of
as an exception nobody sees until they read a log.

**Metadata on every materialisation.** Row counts, quality outcomes and Iceberg
snapshot ids are attached to the materialisation event. That turns the asset
history into the lineage record the governance layer reads in M7, rather than
something reconstructed later from logs.
"""

# NOTE: deliberately no `from __future__ import annotations` in this module.
# Dagster resolves the `context` parameter's annotation at runtime to decide
# what to inject; PEP 563 turns annotations into strings and Dagster then fails
# with the memorably circular "Cannot annotate `context` parameter with type
# AssetExecutionContext. `context` must be annotated with
# AssetExecutionContext...". Every other module in this repo uses the future
# import; this one cannot.

from typing import Any

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetKey,
    AssetSelection,
    DefaultScheduleStatus,
    Definitions,
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
)

GROUP_SOURCE = "source"
GROUP_BRONZE = "bronze"
GROUP_SILVER = "silver"
GROUP_GOLD = "gold"


# ============================================================ source ========


@asset(
    group_name=GROUP_SOURCE,
    description=(
        "The synthetic payments dataset. A pure function of (config, seed, code "
        "commit), so it is identified by a content hash rather than versioned "
        "(ADR-0009)."
    ),
    compute_kind="python",
)
def synthetic_dataset(context: AssetExecutionContext) -> MaterializeResult:
    from kg_core.config import get_settings
    from kg_sim.config import get_profile
    from kg_sim.dataset import write_dataset
    from kg_sim.generator import simulate

    profile = context.run.tags.get("profile", "dev")
    config = get_profile(profile)
    result = simulate(config)
    target = write_dataset(result, get_settings().data_dir, profile=profile)

    manifest = result.manifest
    return MaterializeResult(
        metadata={
            "profile": profile,
            "path": MetadataValue.path(str(target)),
            "config_fingerprint": manifest["config_fingerprint"],
            "events": MetadataValue.int(manifest["counts"]["events"]),
            "accounts": MetadataValue.int(manifest["counts"]["accounts"]),
            "campaigns": MetadataValue.int(manifest["counts"]["campaigns"]),
            "positive_rate": MetadataValue.float(manifest["rates"]["positive_rate"]),
            # Surfaced deliberately: it is the design principle P7 headline and it belongs
            # where somebody will see it, not only in a docstring.
            "criminal_label_coverage": MetadataValue.float(
                manifest["rates"]["criminal_label_coverage"]
            ),
            "digest": (target / "DIGEST").read_text().strip(),
        }
    )


@asset_check(
    asset=synthetic_dataset,
    blocking=True,
    description=(
        "The dataset regenerates byte-identically. ADR-0009 replaces DVC with "
        "content hashing on exactly this premise; if it stops holding, the "
        "decision to drop dataset versioning has become wrong."
    ),
)
def dataset_is_reproducible() -> AssetCheckResult:
    from kg_sim.config import get_profile
    from kg_sim.generator import simulate

    # The smoke profile, not the materialised one: this check runs on every
    # pipeline and needs to cost seconds, not minutes. Determinism is a property
    # of the generator, not of the profile size.
    config = get_profile("smoke")
    first, second = simulate(config), simulate(config)
    identical = first.manifest["content_hashes"] == second.manifest["content_hashes"]
    return AssetCheckResult(
        passed=identical,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "events_hash": first.manifest["content_hashes"]["events"],
            "matched": identical,
        },
    )


# ============================================================ bronze ========


@asset(
    deps=[synthetic_dataset],
    group_name=GROUP_BRONZE,
    description=(
        "Raw payments in Iceberg, append-only. The audit substrate: if it can "
        "be edited, the lineage report is decoration."
    ),
    compute_kind="iceberg",
)
def bronze_payments(context: AssetExecutionContext) -> MaterializeResult:
    from kg_core.config import get_settings
    from kg_lake.medallion import ingest_from_dataset, snapshot_history
    from kg_sim.dataset import find_latest

    dataset = find_latest(get_settings().data_dir)
    if dataset is None:
        raise RuntimeError("no dataset found; materialise synthetic_dataset first")

    stats = ingest_from_dataset(dataset)
    snapshots = snapshot_history("bronze.payments")
    return MaterializeResult(
        metadata={
            "rows_written": MetadataValue.int(stats.rows_written),
            "source_dataset": dataset.name,
            "snapshot_id": str(snapshots[-1]["snapshot_id"]) if snapshots else "none",
            "snapshot_count": MetadataValue.int(len(snapshots)),
        }
    )


@asset_check(
    asset=bronze_payments,
    blocking=True,
    description=(
        "The blocking half of the data-quality suite. A failure here stops the "
        "promotion to silver; the batch stays in bronze, where it can be "
        "inspected and replayed, because bronze is append-only and loses nothing."
    ),
)
def bronze_quality_gate() -> AssetCheckResult:
    from kg_lake.catalog import get_catalog
    from kg_lake.quality import Severity, validate

    table = get_catalog().load_table("bronze.payments").scan().to_arrow()
    report = validate(table, "payments")
    blocking = [r for r in report.results if r.severity is Severity.BLOCKING]

    return AssetCheckResult(
        passed=report.ok,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "rows": MetadataValue.int(report.total_rows),
            "blocking_expectations": MetadataValue.int(len(blocking)),
            "blocking_failures": MetadataValue.int(len(report.blocking_failures)),
            "failed": MetadataValue.json([r.name for r in report.blocking_failures]),
            "report": MetadataValue.json(report.summary()),
        },
    )


@asset_check(
    asset=bronze_payments,
    blocking=False,
    description=(
        "Distributional warnings: VoP match rate, ingest lag, large amounts. "
        "Non-blocking on purpose — these are information, not error, and a "
        "pipeline that halts on a 0.01% anomaly trains everyone to bypass it."
    ),
)
def bronze_distribution_warnings() -> AssetCheckResult:
    from kg_lake.catalog import get_catalog
    from kg_lake.quality import Severity, validate

    table = get_catalog().load_table("bronze.payments").scan().to_arrow()
    report = validate(table, "payments")
    warnings = [r for r in report.results if r.severity is Severity.WARN]
    failed = [r for r in warnings if not r.passed]

    return AssetCheckResult(
        passed=not failed,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "checked": MetadataValue.int(len(warnings)),
            "exceeded_tolerance": MetadataValue.json([r.name for r in failed]),
            "rates": MetadataValue.json({r.name: round(r.failure_rate, 5) for r in warnings}),
        },
    )


# ============================================================ silver ========


@asset(
    deps=[synthetic_dataset],
    group_name=GROUP_SILVER,
    description="Account dimension. Attributes are fixed at account opening, so joining them is point-in-time safe.",
    compute_kind="iceberg",
)
def silver_accounts() -> MaterializeResult:
    from kg_core.config import get_settings
    from kg_lake.medallion import load_dimension
    from kg_sim.dataset import find_latest

    stats = load_dimension(find_latest(get_settings().data_dir), "accounts")
    return MaterializeResult(metadata={"rows": MetadataValue.int(stats.rows_written)})


@asset(
    deps=[synthetic_dataset],
    group_name=GROUP_SILVER,
    description=(
        "Confirmed outcomes, partitioned by decision date. Partitioning by when "
        "a label was DECIDED rather than when the event happened is what makes "
        "'which labels existed at time T?' a pruned scan — the question every "
        "chronological split asks."
    ),
    compute_kind="iceberg",
)
def silver_labels() -> MaterializeResult:
    from kg_core.config import get_settings
    from kg_lake.medallion import load_dimension
    from kg_sim.dataset import find_latest

    stats = load_dimension(find_latest(get_settings().data_dir), "labels")
    return MaterializeResult(metadata={"rows": MetadataValue.int(stats.rows_written)})


@asset(
    deps=[bronze_payments, silver_accounts],
    group_name=GROUP_SILVER,
    description=(
        "Validated, conformed payments. Rows failing a row-level expectation are "
        "diverted to bronze.quarantine with the name of the expectation they "
        "failed — never silently dropped."
    ),
    compute_kind="iceberg",
)
def silver_payments() -> MaterializeResult:
    from kg_lake.medallion import promote_to_silver

    stats = promote_to_silver(strict=True)
    metadata: dict[str, Any] = {
        "rows_in": MetadataValue.int(stats.rows_in),
        "rows_written": MetadataValue.int(stats.rows_written),
        "rows_quarantined": MetadataValue.int(stats.rows_quarantined),
        "quarantine_reasons": MetadataValue.json(stats.quarantine_reasons),
    }
    if stats.report is not None:
        metadata["quality"] = MetadataValue.json(stats.report.summary())
    return MaterializeResult(metadata=metadata)


@asset_check(
    asset=silver_payments,
    blocking=True,
    description=(
        "Referential integrity: every silver payment references accounts that "
        "exist in the dimension. A dangling reference means a mart will quietly "
        "produce nulls instead of failing."
    ),
)
def silver_referential_integrity() -> AssetCheckResult:
    from kg_lake.catalog import get_catalog

    catalog = get_catalog()
    payments = catalog.load_table("silver.payments").scan().to_arrow()
    accounts = catalog.load_table("silver.accounts").scan().to_arrow()

    known = set(accounts.column("account_id").to_pylist())
    debtors = set(payments.column("debtor_account_id").to_pylist())
    creditors = set(payments.column("creditor_account_id").to_pylist())
    missing = (debtors | creditors) - known

    return AssetCheckResult(
        passed=not missing,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "distinct_accounts_referenced": MetadataValue.int(len(debtors | creditors)),
            "missing": MetadataValue.int(len(missing)),
            "sample_missing": MetadataValue.json(sorted(missing)[:10]),
        },
    )


@asset_check(
    asset=silver_payments,
    blocking=True,
    description=(
        "No label may be decided before its transaction happened. A negative "
        "verification latency is a time-travel bug, and this is the cheapest "
        "place to catch one (design principle P2)."
    ),
)
def labels_never_precede_their_events() -> AssetCheckResult:
    from kg_lake.catalog import get_catalog

    labels = get_catalog().load_table("silver.labels").scan().to_arrow()
    if labels.num_rows == 0:
        return AssetCheckResult(passed=True, metadata={"rows": 0})

    rows = labels.select(["label_id", "event_ts", "decided_ts"]).to_pylist()
    offenders = [r for r in rows if r["decided_ts"] < r["event_ts"]]
    latencies = sorted((r["decided_ts"] - r["event_ts"]) / 86_400_000 for r in rows)

    return AssetCheckResult(
        passed=not offenders,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "labels": MetadataValue.int(len(rows)),
            "time_travelling_labels": MetadataValue.int(len(offenders)),
            "median_latency_days": MetadataValue.float(round(latencies[len(latencies) // 2], 2)),
        },
    )


# ============================================================== gold ========


@asset(
    deps=[silver_payments, silver_labels],
    group_name=GROUP_GOLD,
    description=(
        "Silver materialised into DuckDB for the analytics marts, with the "
        "Iceberg snapshot id of each source recorded so a mart is traceable to "
        "the exact table state it was built from."
    ),
    compute_kind="duckdb",
)
def analytics_database() -> MaterializeResult:
    from kg_lake.duck import export_to_duckdb, warehouse_db_path

    provenance = export_to_duckdb()
    return MaterializeResult(
        metadata={
            "path": MetadataValue.path(str(warehouse_db_path())),
            "tables": MetadataValue.json(
                {
                    k: {"rows": v["rows"], "snapshot_id": str(v["snapshot_id"])}
                    for k, v in provenance.items()
                }
            ),
            "total_rows": MetadataValue.int(sum(v["rows"] for v in provenance.values())),
        }
    )


@asset(
    deps=[analytics_database],
    group_name=GROUP_GOLD,
    description=(
        "Gold marts built by dbt: account activity, payment corridors, VoP "
        "effectiveness, label availability, and the AnaCredit-shaped regulatory "
        "reporting mart."
    ),
    compute_kind="dbt",
)
def gold_marts(context: AssetExecutionContext) -> MaterializeResult:
    import os
    import subprocess

    from kg_core.config import get_settings
    from kg_lake.duck import query, warehouse_db_path

    project = get_settings().repo_root / "dbt" / "kontograph"
    result = subprocess.run(
        ["dbt", "build", "--project-dir", str(project), "--profiles-dir", str(project)],
        cwd=project,
        env={**os.environ, "KG_DUCKDB_PATH": str(warehouse_db_path())},
        capture_output=True,
        text=True,
        check=False,
    )
    context.log.info(result.stdout[-4000:])
    if result.returncode != 0:
        raise RuntimeError(f"dbt build failed:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")

    marts = (
        query(
            "select table_name from information_schema.tables "
            "where table_schema = 'main_gold' order by table_name"
        )
        .column("table_name")
        .to_pylist()
    )

    return MaterializeResult(
        metadata={
            "marts": MetadataValue.json(marts),
            "mart_count": MetadataValue.int(len(marts)),
        }
    )


# ============================================================== jobs ========

# The medallion job stops at silver: gold depends on dbt, which lives behind an
# extra. Splitting them means a contributor without the orchestration extra can
# still run the core pipeline.
medallion_job = define_asset_job(
    name="medallion",
    selection=AssetSelection.assets(
        synthetic_dataset, bronze_payments, silver_accounts, silver_labels, silver_payments
    ),
    description="Simulator through to validated silver, gated by the quality suite.",
)

full_pipeline_job = define_asset_job(
    name="full_pipeline",
    selection=AssetSelection.all(),
    description="Everything, including the dbt gold marts.",
)

# Daily, matching the cadence at which a bank would rebuild these marts. Stopped
# by default: a schedule that starts running the moment somebody opens the UI is
# a surprise, and surprises in a pipeline are expensive.
daily_schedule = ScheduleDefinition(
    name="daily_medallion",
    job=medallion_job,
    cron_schedule="0 3 * * *",
    default_status=DefaultScheduleStatus.STOPPED,
    description="Rebuild the medallion layers overnight.",
)

defs = Definitions(
    assets=[
        synthetic_dataset,
        bronze_payments,
        silver_accounts,
        silver_labels,
        silver_payments,
        analytics_database,
        gold_marts,
    ],
    asset_checks=[
        dataset_is_reproducible,
        bronze_quality_gate,
        bronze_distribution_warnings,
        silver_referential_integrity,
        labels_never_precede_their_events,
    ],
    jobs=[medallion_job, full_pipeline_job],
    schedules=[daily_schedule],
)


def lineage_edges() -> list[tuple[str, str]]:
    """The asset graph as (upstream, downstream) pairs.

    Read by the governance layer in M7 to render the lineage diagram. Derived
    from the Dagster graph rather than hand-drawn, so the published diagram
    cannot drift from what actually runs.
    """
    edges: list[tuple[str, str]] = []
    for spec in defs.get_all_asset_specs():
        downstream = spec.key.to_user_string()
        for dep in spec.deps:
            edges.append((AssetKey(dep.asset_key.path).to_user_string(), downstream))
    return sorted(edges)
