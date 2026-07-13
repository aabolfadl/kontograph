"""Iceberg table definitions for the medallion layers.

**Why event time is a `long`, not a `timestamptz`.** Iceberg's hidden
partitioning (`days(ts)`) is a genuinely nice feature and the obvious thing to
reach for. It is not used for the primary event-time column, because this
project's invariant is that event time is an exact integer millisecond value
everywhere (see :mod:`kg_core.timeutil`), and a timestamp column invites engines
to apply a session timezone on read. We already found one instance of exactly
that class of bug at the Kafka boundary, where a logical-type conversion turned
an int into a datetime and silently broke a comparison.

So: ``acceptance_ts`` is the exact wire value as a ``long``, and ``event_date``
is a derived ``date`` used as an explicit identity partition. Time-ranged scans
still prune partitions; the trade is that partitioning is explicit rather than
hidden. That is the honest exchange, and it is written down rather than
discovered later.

Bronze additionally carries the Kafka coordinates each record arrived on. Those
are what make the bronze layer an audit substrate rather than just a copy: a row
can be traced back to a specific offset on a specific partition.
"""

from __future__ import annotations

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table.sorting import SortField, SortOrder
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
)

# Field ids must be stable forever: Iceberg tracks columns by id, not by name,
# which is what makes a rename a metadata-only operation. Ids are assigned in
# blocks per table so a new column can be added without disturbing others.
BRONZE_PAYMENTS = Schema(
    NestedField(1, "tx_id", StringType(), required=True),
    NestedField(2, "end_to_end_id", StringType(), required=False),
    NestedField(3, "acceptance_ts", LongType(), required=True),
    NestedField(4, "ingest_ts", LongType(), required=True),
    NestedField(5, "debtor_account_id", StringType(), required=True),
    NestedField(6, "debtor_iban", StringType(), required=False),
    NestedField(7, "debtor_bic", StringType(), required=False),
    NestedField(8, "debtor_name", StringType(), required=False),
    NestedField(9, "debtor_country", StringType(), required=False),
    NestedField(10, "creditor_account_id", StringType(), required=True),
    NestedField(11, "creditor_iban", StringType(), required=False),
    NestedField(12, "creditor_bic", StringType(), required=False),
    NestedField(13, "creditor_name", StringType(), required=False),
    NestedField(14, "creditor_country", StringType(), required=False),
    NestedField(15, "creditor_name_claimed", StringType(), required=False),
    NestedField(16, "vop_result", StringType(), required=False),
    NestedField(17, "vop_name_similarity", DoubleType(), required=False),
    NestedField(18, "vop_overridden", BooleanType(), required=False),
    NestedField(19, "amount_cents", LongType(), required=True),
    NestedField(20, "currency", StringType(), required=False),
    NestedField(21, "purpose_code", StringType(), required=False),
    NestedField(22, "remittance_info", StringType(), required=False),
    NestedField(23, "channel", StringType(), required=False),
    NestedField(24, "initiation_source", StringType(), required=False),
    NestedField(25, "device_id", StringType(), required=False),
    NestedField(26, "session_ip_prefix", StringType(), required=False),
    NestedField(27, "is_new_beneficiary", BooleanType(), required=False),
    NestedField(28, "settlement_status", StringType(), required=False),
    NestedField(29, "schema_version", IntegerType(), required=False),
    # ---- ingestion provenance: what makes bronze auditable ----
    NestedField(30, "event_date", DateType(), required=True),
    NestedField(31, "kafka_topic", StringType(), required=False),
    NestedField(32, "kafka_partition", IntegerType(), required=False),
    NestedField(33, "kafka_offset", LongType(), required=False),
    NestedField(34, "kafka_timestamp_ms", LongType(), required=False),
    NestedField(35, "writer_schema_id", IntegerType(), required=False),
    NestedField(36, "bronze_ingested_ts", LongType(), required=False),
    #: Which generated dataset this row came from.
    #:
    #: In production there is one stream and this would be constant. Here,
    #: regenerating the simulator produces a NEW WORLD that happens to reuse
    #: account identifiers, so appending a second dataset silently merges two
    #: unrelated graphs into one. Recording the source makes that visible and
    #: lets the feature build scope itself to a single world.
    NestedField(37, "source_dataset", StringType(), required=False),
)

# Silver keeps the same payload plus derived, *point-in-time safe* enrichments.
# Nothing here may depend on a future event: every field is computable from the
# transaction itself and from account attributes known at acceptance time.
SILVER_PAYMENTS = Schema(
    NestedField(1, "tx_id", StringType(), required=True),
    NestedField(2, "acceptance_ts", LongType(), required=True),
    NestedField(3, "event_date", DateType(), required=True),
    NestedField(4, "debtor_account_id", StringType(), required=True),
    NestedField(5, "creditor_account_id", StringType(), required=True),
    NestedField(6, "amount_cents", LongType(), required=True),
    NestedField(7, "currency", StringType(), required=False),
    NestedField(8, "channel", StringType(), required=False),
    NestedField(9, "initiation_source", StringType(), required=False),
    NestedField(10, "purpose_code", StringType(), required=False),
    NestedField(11, "vop_result", StringType(), required=False),
    NestedField(12, "vop_name_similarity", DoubleType(), required=False),
    NestedField(13, "vop_overridden", BooleanType(), required=False),
    NestedField(14, "is_new_beneficiary", BooleanType(), required=False),
    NestedField(15, "settlement_status", StringType(), required=False),
    NestedField(16, "debtor_bank_blz", StringType(), required=False),
    NestedField(17, "creditor_bank_blz", StringType(), required=False),
    NestedField(18, "debtor_country", StringType(), required=False),
    NestedField(19, "creditor_country", StringType(), required=False),
    NestedField(20, "is_cross_border", BooleanType(), required=False),
    NestedField(21, "is_cross_institution", BooleanType(), required=False),
    # Account age at acceptance time. Point-in-time safe: derived from the
    # account's opening date and *this* transaction's timestamp, never from a
    # later snapshot of the account.
    NestedField(22, "debtor_account_age_days", DoubleType(), required=False),
    NestedField(23, "creditor_account_age_days", DoubleType(), required=False),
    NestedField(24, "ingest_lag_ms", LongType(), required=False),
    NestedField(25, "silver_processed_ts", LongType(), required=False),
    NestedField(26, "source_dataset", StringType(), required=False),
)

# Rows that failed a quality gate. Quarantined rather than dropped: a silently
# discarded row is an unexplained gap in a reconciliation months later.
QUARANTINE = Schema(
    NestedField(1, "tx_id", StringType(), required=False),
    NestedField(2, "event_date", DateType(), required=True),
    NestedField(3, "quarantined_ts", LongType(), required=True),
    NestedField(4, "failed_expectation", StringType(), required=True),
    NestedField(5, "severity", StringType(), required=True),
    NestedField(6, "detail", StringType(), required=False),
    NestedField(7, "raw_record", StringType(), required=False),
)

ACCOUNTS = Schema(
    NestedField(1, "account_id", StringType(), required=True),
    NestedField(2, "iban", StringType(), required=False),
    NestedField(3, "bic", StringType(), required=False),
    NestedField(4, "holder_name", StringType(), required=False),
    NestedField(5, "kind", StringType(), required=False),
    NestedField(6, "bank_blz", StringType(), required=False),
    NestedField(7, "country", StringType(), required=False),
    NestedField(8, "opened_ts", LongType(), required=False),
)

LABELS = Schema(
    NestedField(1, "label_id", StringType(), required=True),
    NestedField(2, "subject_type", StringType(), required=False),
    NestedField(3, "subject_id", StringType(), required=True),
    NestedField(4, "event_ts", LongType(), required=True),
    NestedField(5, "decided_ts", LongType(), required=True),
    NestedField(6, "decided_date", DateType(), required=True),
    NestedField(7, "label", StringType(), required=True),
    NestedField(8, "source", StringType(), required=False),
    NestedField(9, "sar_filed", BooleanType(), required=False),
    NestedField(10, "confidence", DoubleType(), required=False),
)


def _partition_by(schema: Schema, column: str, field_id: int = 1000) -> PartitionSpec:
    """Identity partitioning on *column*. See the module docstring for why."""
    source = schema.find_field(column)
    return PartitionSpec(
        PartitionField(
            source_id=source.field_id,
            field_id=field_id,
            transform=IdentityTransform(),
            name=column,
        )
    )


def _sort_by(schema: Schema, column: str) -> SortOrder:
    """Sort data files by *column* so time-ranged scans skip whole files.

    Iceberg keeps per-file min/max statistics; writing sorted data is what makes
    those statistics selective. Unsorted files each span the whole range and
    prune nothing.
    """
    return SortOrder(SortField(source_id=schema.find_field(column).field_id))


#: Every managed table: identifier -> (schema, partition spec, sort order).
TABLES: dict[str, tuple[Schema, PartitionSpec, SortOrder | None]] = {
    "bronze.payments": (
        BRONZE_PAYMENTS,
        _partition_by(BRONZE_PAYMENTS, "event_date"),
        _sort_by(BRONZE_PAYMENTS, "acceptance_ts"),
    ),
    "bronze.quarantine": (QUARANTINE, _partition_by(QUARANTINE, "event_date"), None),
    "silver.payments": (
        SILVER_PAYMENTS,
        _partition_by(SILVER_PAYMENTS, "event_date"),
        _sort_by(SILVER_PAYMENTS, "acceptance_ts"),
    ),
    "silver.accounts": (ACCOUNTS, PartitionSpec(), None),
    # Labels are partitioned by *decision* date, not event date. That is the
    # whole point of section I.7: a label becomes available when it is decided,
    # and a chronological split has to be able to ask "which labels existed at
    # time T?" -- which is a scan bounded by decided_date.
    "silver.labels": (LABELS, _partition_by(LABELS, "decided_date"), None),
}
