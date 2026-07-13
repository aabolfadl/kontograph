"""The medallion pipeline: bronze → silver.

**Bronze is append-only and never rewritten.** It records what arrived,
including the Kafka coordinates it arrived on. If bronze can be edited then the
lineage report is decoration, so nothing in this module updates or deletes a
bronze row — bad records are *quarantined on promotion*, not removed at source.

**The gate lives on the promotion.** Blocking expectations run before silver is
written; a failure raises and the batch stays in bronze, where it can be
inspected and replayed. Quarantined rows are diverted to ``bronze.quarantine``
with the name of the expectation they failed, never silently dropped.

**Silver enrichment is point-in-time safe by construction.** The only joins are
against account attributes that are fixed at account opening (bank, country,
opening date), and derived values like account age are computed against *this*
transaction's timestamp. Nothing reads a later snapshot, so no enrichment can
see the future.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.table import Table

from kg_core.logging import get_logger
from kg_core.timeutil import MILLIS_PER_DAY, from_millis
from kg_lake.catalog import get_catalog
from kg_lake.quality import (
    ValidationReport,
    partition_valid_rows,
    validate,
)
from kg_lake.schemas import TABLES

log = get_logger(__name__)


@dataclass(slots=True)
class LayerStats:
    """What a pipeline step did. Returned rather than logged so assets can assert."""

    rows_in: int = 0
    rows_written: int = 0
    rows_quarantined: int = 0
    quarantine_reasons: list[str] = field(default_factory=list)
    report: ValidationReport | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "rows_in": self.rows_in,
            "rows_written": self.rows_written,
            "rows_quarantined": self.rows_quarantined,
            "quarantine_reasons": self.quarantine_reasons,
        }
        if self.report is not None:
            payload["quality"] = self.report.summary()
        return payload


# ------------------------------------------------------------ table admin ---


def ensure_tables(catalog: Catalog | None = None) -> dict[str, Table]:
    """Create every managed Iceberg table that does not yet exist."""
    catalog = catalog or get_catalog()
    tables: dict[str, Table] = {}
    for identifier, (schema, spec, sort_order) in TABLES.items():
        try:
            tables[identifier] = catalog.load_table(identifier)
        except NoSuchTableError:
            kwargs: dict[str, Any] = {"schema": schema, "partition_spec": spec}
            if sort_order is not None:
                kwargs["sort_order"] = sort_order
            tables[identifier] = catalog.create_table(identifier, **kwargs)
            log.info("created iceberg table", table=identifier)
    return tables


def _event_date(millis: int) -> _dt.date:
    """UTC date of an epoch-millisecond timestamp — the partition key.

    Routed through kg_core.timeutil rather than date.fromtimestamp(), which
    takes no tz argument and would silently have used the host timezone. On a
    machine east of UTC that shifts late-evening payments into the next day's
    partition, which is a leakage bug wearing a partitioning costume.
    """
    return from_millis(millis).date()


def _arrow_schema_for(identifier: str) -> pa.Schema:
    """Arrow schema matching an Iceberg table, so writes never rely on inference."""
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    return schema_to_pyarrow(TABLES[identifier][0])


def _to_arrow(rows: list[dict[str, Any]], identifier: str) -> pa.Table:
    schema = _arrow_schema_for(identifier)
    columns = {f.name: [row.get(f.name) for row in rows] for f in schema}
    return pa.Table.from_pydict(columns, schema=schema)


# ----------------------------------------------------------------- bronze ---


def bronze_rows_from_messages(messages: Iterable[Any]) -> Iterator[dict[str, Any]]:
    """Turn consumed Kafka messages into bronze rows.

    The Kafka coordinates are carried through deliberately: they are what let an
    auditor answer "where exactly did this row come from?" with an offset rather
    than a shrug.
    """
    now = int(_dt.datetime.now(tz=_dt.UTC).timestamp() * 1000)
    for message in messages:
        row = dict(message.value)
        acceptance = int(row["acceptance_ts"])
        row.update(
            event_date=_event_date(acceptance),
            kafka_topic=message.topic,
            kafka_partition=message.partition,
            kafka_offset=message.offset,
            kafka_timestamp_ms=message.timestamp_ms,
            writer_schema_id=message.schema_id,
            bronze_ingested_ts=now,
        )
        yield row


def write_bronze(rows: list[dict[str, Any]], catalog: Catalog | None = None) -> LayerStats:
    """Append rows to ``bronze.payments``. Append-only, always."""
    catalog = catalog or get_catalog()
    tables = ensure_tables(catalog)
    if not rows:
        return LayerStats()

    table = _to_arrow(rows, "bronze.payments")
    tables["bronze.payments"].append(table)
    log.info("bronze append", rows=len(rows))
    return LayerStats(rows_in=len(rows), rows_written=len(rows))


def ingest_from_dataset(
    dataset_dir: Any,
    catalog: Catalog | None = None,
    *,
    batch_size: int = 50_000,
    replace: bool = False,
) -> LayerStats:
    """Load a generated dataset straight into bronze, bypassing Kafka.

    The backfill path. Bronze rows produced this way carry null Kafka
    coordinates, which is honest: they did not arrive on a topic. Anything that
    needs to distinguish streamed from backfilled data can test that column
    rather than guess.
    """
    import pyarrow.parquet as pq

    catalog = catalog or get_catalog()
    tables = ensure_tables(catalog)
    now = int(_dt.datetime.now(tz=_dt.UTC).timestamp() * 1000)
    stats = LayerStats()

    source_dataset = str(getattr(dataset_dir, "name", dataset_dir))
    parquet = pq.ParquetFile(dataset_dir / "events.parquet")

    # Idempotence. An orchestrated asset gets re-materialised routinely, and
    # bronze is append-only by design -- so without this guard a second run
    # silently doubles the layer. Probing one known tx_id is enough and costs a
    # single partition-pruned scan.
    if not replace and _already_ingested(tables["bronze.payments"], parquet):
        log.warning(
            "dataset already present in bronze; skipping",
            dataset=str(getattr(dataset_dir, "name", dataset_dir)),
        )
        return stats
    for batch in parquet.iter_batches(batch_size=batch_size):
        rows = batch.to_pylist()
        for row in rows:
            row["event_date"] = _event_date(int(row["acceptance_ts"]))
            row["bronze_ingested_ts"] = now
            row["source_dataset"] = source_dataset
        tables["bronze.payments"].append(_to_arrow(rows, "bronze.payments"))
        stats.rows_in += len(rows)
        stats.rows_written += len(rows)
        log.info("bronze backfill batch", rows=len(rows), total=stats.rows_written)
    return stats


def _already_ingested(bronze: Table, parquet: Any) -> bool:
    """True if bronze already contains this dataset's first transaction."""
    first_batch = next(parquet.iter_batches(batch_size=1), None)
    if first_batch is None or first_batch.num_rows == 0:
        return False
    probe = first_batch.to_pylist()[0]
    existing = (
        bronze.scan(
            row_filter=f"tx_id == '{probe['tx_id']}'",
            selected_fields=("tx_id",),
        )
        .to_arrow()
        .num_rows
    )
    return existing > 0


def load_dimension(dataset_dir: Any, name: str, catalog: Catalog | None = None) -> LayerStats:
    """Load ``accounts`` or ``labels`` into silver.

    Labels gain a ``decided_date`` partition column. Partitioning labels by when
    they were *decided* rather than when the event happened is what makes
    "which labels existed at time T?" a partition-pruned scan instead of a full
    table read — and that question is asked on every chronological split.
    """
    import pyarrow.parquet as pq

    catalog = catalog or get_catalog()
    tables = ensure_tables(catalog)
    identifier = f"silver.{name}"

    rows = pq.read_table(dataset_dir / f"{name}.parquet").to_pylist()
    if name == "labels":
        for row in rows:
            row["decided_date"] = _event_date(int(row["decided_ts"]))

    # Idempotent, for the same reason bronze ingest is: a dimension is a
    # full-refresh table, and appending it twice doubles every account. Replace
    # rather than append -- these are snapshots of a current state, not an
    # event log, so overwriting is the semantically correct operation.
    if rows:
        tables[identifier].overwrite(_to_arrow(rows, identifier))
    log.info("dimension loaded", table=identifier, rows=len(rows))
    return LayerStats(rows_in=len(rows), rows_written=len(rows))


# ----------------------------------------------------------------- silver ---

#: Countries whose payments count as domestic for the cross-border flag.
_DOMESTIC = "DE"


def promote_to_silver(
    catalog: Catalog | None = None,
    *,
    since_date: _dt.date | None = None,
    strict: bool = True,
) -> LayerStats:
    """Promote bronze to silver, gated by the quality suite.

    ``strict=True`` raises on a blocking failure and writes nothing — the batch
    stays in bronze for inspection. ``strict=False`` records the failures and
    proceeds, which exists only for the quality *report* path and should never
    be used by the pipeline.
    """
    catalog = catalog or get_catalog()
    tables = ensure_tables(catalog)

    scan = tables["bronze.payments"].scan()
    if since_date is not None:
        scan = tables["bronze.payments"].scan(
            row_filter=f"event_date >= '{since_date.isoformat()}'"
        )
    bronze = scan.to_arrow()
    stats = LayerStats(rows_in=bronze.num_rows)
    if bronze.num_rows == 0:
        log.warning("nothing to promote: bronze is empty")
        return stats

    # ---- the gate ---------------------------------------------------------
    report = validate(bronze, "payments")
    stats.report = report
    if strict:
        report.raise_if_blocked()

    # Anti-join against silver so promotion is idempotent. Re-running an asset
    # is normal; re-running it must not duplicate rows. Done on tx_id rather
    # than a timestamp watermark because late-arriving bronze rows for an
    # already-promoted day are a real case, and a watermark would skip them.
    already = _promoted_tx_ids(tables["silver.payments"])
    if already:
        fresh_mask = pa.array(
            [tx not in already for tx in bronze.column("tx_id").to_pylist()],
            type=pa.bool_(),
        )
        bronze = bronze.filter(fresh_mask)
        stats.rows_in = bronze.num_rows
        if bronze.num_rows == 0:
            log.info("nothing new to promote; silver is up to date")
            return stats

    clean, quarantined, reasons = partition_valid_rows(bronze, "payments")
    stats.rows_quarantined = quarantined.num_rows
    stats.quarantine_reasons = reasons

    if quarantined.num_rows:
        _write_quarantine(tables["bronze.quarantine"], quarantined, reasons)

    accounts = _account_lookup(tables["silver.accounts"])
    silver_rows = list(_enrich(clean, accounts))
    if silver_rows:
        tables["silver.payments"].append(_to_arrow(silver_rows, "silver.payments"))
    stats.rows_written = len(silver_rows)

    log.info(
        "promoted to silver",
        rows_in=stats.rows_in,
        written=stats.rows_written,
        quarantined=stats.rows_quarantined,
    )
    return stats


def _promoted_tx_ids(silver: Table) -> set[str]:
    """Transaction references already present in silver."""
    try:
        arrow = silver.scan(selected_fields=("tx_id",)).to_arrow()
    except Exception:
        return set()
    return set(arrow.column("tx_id").to_pylist())


def _write_quarantine(table: Table, rows: pa.Table, reasons: list[str]) -> None:
    now = int(_dt.datetime.now(tz=_dt.UTC).timestamp() * 1000)
    reason = ", ".join(reasons) or "unspecified"
    payload = [
        {
            "tx_id": record.get("tx_id"),
            "event_date": record.get("event_date"),
            "quarantined_ts": now,
            "failed_expectation": reason,
            "severity": "QUARANTINE",
            "detail": "diverted on promotion to silver",
            # The full record is kept as JSON so a quarantined row can be
            # replayed after the defect is fixed, rather than being a dead end.
            "raw_record": json.dumps(record, default=str)[:8000],
        }
        for record in rows.to_pylist()
    ]
    table.append(_to_arrow(payload, "bronze.quarantine"))
    log.warning("rows quarantined", count=len(payload), reasons=reasons)


def _account_lookup(accounts_table: Table) -> dict[str, dict[str, Any]]:
    """Account attributes fixed at opening — the only PIT-safe things to join."""
    try:
        arrow = accounts_table.scan(
            selected_fields=("account_id", "bank_blz", "country", "opened_ts", "kind")
        ).to_arrow()
    except Exception:
        return {}
    return {row["account_id"]: row for row in arrow.to_pylist()}


def _enrich(payments: pa.Table, accounts: dict[str, dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Derive silver columns.

    Every derived value here depends only on the transaction itself and on
    account attributes fixed before it. Account *age* is computed against this
    transaction's acceptance time, so replaying the same row a year later
    produces the same number — which is the property a feature store needs and
    the one a naive "join the current account table" would destroy.
    """
    now = int(_dt.datetime.now(tz=_dt.UTC).timestamp() * 1000)

    for row in payments.to_pylist():
        acceptance = int(row["acceptance_ts"])
        debtor = accounts.get(row["debtor_account_id"], {})
        creditor = accounts.get(row["creditor_account_id"], {})

        debtor_country = row.get("debtor_country") or debtor.get("country")
        creditor_country = row.get("creditor_country") or creditor.get("country")
        debtor_blz = debtor.get("bank_blz")
        creditor_blz = creditor.get("bank_blz")

        yield {
            "tx_id": row["tx_id"],
            "acceptance_ts": acceptance,
            "event_date": row["event_date"],
            "debtor_account_id": row["debtor_account_id"],
            "creditor_account_id": row["creditor_account_id"],
            "amount_cents": row["amount_cents"],
            "currency": row.get("currency"),
            "channel": row.get("channel"),
            "initiation_source": row.get("initiation_source"),
            "purpose_code": row.get("purpose_code"),
            "vop_result": row.get("vop_result"),
            "vop_name_similarity": row.get("vop_name_similarity"),
            "vop_overridden": row.get("vop_overridden"),
            "is_new_beneficiary": row.get("is_new_beneficiary"),
            "settlement_status": row.get("settlement_status"),
            "debtor_bank_blz": debtor_blz,
            "creditor_bank_blz": creditor_blz,
            "debtor_country": debtor_country,
            "creditor_country": creditor_country,
            "is_cross_border": (
                None
                if not (debtor_country and creditor_country)
                else debtor_country != creditor_country
                or debtor_country != _DOMESTIC
                or creditor_country != _DOMESTIC
            ),
            # Germany's banking system is federated by design (see
            # kg_sim.institutions), so "did this payment cross an institutional
            # boundary?" is a meaningful question here in a way it is not
            # everywhere.
            "is_cross_institution": (
                None if not (debtor_blz and creditor_blz) else debtor_blz != creditor_blz
            ),
            "debtor_account_age_days": _age_days(debtor.get("opened_ts"), acceptance),
            "creditor_account_age_days": _age_days(creditor.get("opened_ts"), acceptance),
            "ingest_lag_ms": int(row["ingest_ts"]) - acceptance,
            "silver_processed_ts": now,
            "source_dataset": row.get("source_dataset"),
        }


def silver_rows_from_events(
    events: list[dict[str, Any]], accounts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply the bronze->silver derivation to in-memory events, without a catalog.

    Features are defined against silver. Anything that computes them over a
    freshly simulated stream must go through the *same* enrichment as the
    lakehouse, or it is testing a different input from the one production sees.
    """
    if not events:
        return []
    rows = [{**event, "event_date": _event_date(int(event["acceptance_ts"]))} for event in events]
    lookup = {account["account_id"]: account for account in accounts}
    return list(_enrich(pa.Table.from_pylist(rows), lookup))


def _age_days(opened_ts: int | None, at_ts: int) -> float | None:
    """Account age in days at *at_ts*, or None if the opening date is unknown."""
    if opened_ts is None:
        return None
    return max(0.0, (at_ts - int(opened_ts)) / MILLIS_PER_DAY)


# ------------------------------------------------------------------ stats ---


def datasets_present(catalog: Catalog | None = None) -> dict[str, int]:
    """Row counts per source dataset in silver.

    Reported by `kg lake status`, because "the lake has 574,893 rows" is
    reassuring and wrong when two of those datasets are different worlds.
    """
    import collections

    catalog = catalog or get_catalog()
    try:
        arrow = (
            catalog.load_table("silver.payments")
            .scan(selected_fields=("source_dataset",))
            .to_arrow()
        )
    except Exception:
        return {}
    return dict(collections.Counter(arrow.column("source_dataset").to_pylist()))


def layer_counts(catalog: Catalog | None = None) -> dict[str, int]:
    """Row counts per managed table, for the CLI and the freshness checks."""
    catalog = catalog or get_catalog()
    counts: dict[str, int] = {}
    for identifier in TABLES:
        try:
            counts[identifier] = catalog.load_table(identifier).scan().to_arrow().num_rows
        except NoSuchTableError:
            counts[identifier] = 0
    return counts


def snapshot_history(identifier: str, catalog: Catalog | None = None) -> list[dict[str, Any]]:
    """Iceberg snapshot history — the time-travel handle the lineage report uses."""
    catalog = catalog or get_catalog()
    table = catalog.load_table(identifier)
    return [
        {
            "snapshot_id": snapshot.snapshot_id,
            "parent_id": snapshot.parent_snapshot_id,
            "timestamp_ms": snapshot.timestamp_ms,
            "operation": snapshot.summary.operation.value if snapshot.summary else None,
            # Summary is a mapping-like object, not a dict; go through
            # additional_properties so this keeps working across pyiceberg
            # versions rather than relying on dict.get.
            "added_records": (
                snapshot.summary.additional_properties.get("added-records")
                if snapshot.summary is not None
                else None
            ),
        }
        for snapshot in table.metadata.snapshots
    ]
