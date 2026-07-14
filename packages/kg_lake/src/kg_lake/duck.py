"""DuckDB bridge between Iceberg and the analytics marts.

Iceberg is the storage format; DuckDB is what actually answers questions about
it locally. pyiceberg reads a table into Arrow, DuckDB queries Arrow with zero
copy, and dbt builds the gold marts on top.

**Why materialise into a DuckDB file rather than querying Iceberg in place.**
DuckDB's Iceberg extension can read a table directly, but the path through a
local `file://` warehouse and a SQLite catalog is fragile enough that a broken
dbt run would be indistinguishable from a broken model. Materialising silver
into `warehouse/kontograph.duckdb` makes the boundary explicit: one step exports,
dbt then owns everything downstream, and a failure has one obvious owner.

The export is a *projection of a specific Iceberg snapshot*, and the snapshot id
is recorded alongside it. That is what keeps a mart traceable back to the exact
table state it was built from — the lineage claim in the platform scope is only worth making if
that link exists.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
from pyiceberg.catalog import Catalog

from kg_core.config import get_settings
from kg_core.logging import get_logger
from kg_lake.catalog import get_catalog

log = get_logger(__name__)

#: Tables exported for analytics. Bronze is deliberately absent: it is the audit
#: substrate, not an analytics surface, and anything reading it should be doing
#: so explicitly through the Iceberg API.
EXPORTED = ("silver.payments", "silver.accounts", "silver.labels")


def warehouse_db_path() -> Path:
    """Location of the DuckDB file the marts live in."""
    settings = get_settings()
    settings.ensure_dirs()
    return settings.warehouse_dir / "kontograph.duckdb"


def connect(path: Path | None = None, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the analytics database."""
    target = path or warehouse_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(target), read_only=read_only)


def export_to_duckdb(
    catalog: Catalog | None = None,
    path: Path | None = None,
    *,
    tables: tuple[str, ...] = EXPORTED,
) -> dict[str, Any]:
    """Materialise Iceberg tables into DuckDB and record their snapshot ids.

    Returns a provenance record: table -> (rows, snapshot_id). Written into a
    ``_lineage`` table in the same database so a mart can always be traced back
    to the snapshot it was built from.
    """
    catalog = catalog or get_catalog()
    connection = connect(path)
    provenance: dict[str, Any] = {}

    try:
        for identifier in tables:
            iceberg_table = catalog.load_table(identifier)
            arrow: pa.Table = iceberg_table.scan().to_arrow()
            # Mirror the medallion naming as real DuckDB schemas, so
            # "silver.payments" in Iceberg is "silver.payments" in DuckDB too.
            # dbt resolves a source's name to a schema, so flattening these to
            # silver_payments in the default schema makes every dbt source
            # reference fail to resolve.
            namespace, name = identifier.split(".", 1)
            staging = f"_arrow_{namespace}_{name}"

            connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{namespace}"')
            connection.register(staging, arrow)
            connection.execute(
                f'CREATE OR REPLACE TABLE "{namespace}"."{name}" AS SELECT * FROM {staging}'
            )
            connection.unregister(staging)

            snapshot = iceberg_table.current_snapshot()
            provenance[identifier] = {
                "rows": arrow.num_rows,
                "snapshot_id": snapshot.snapshot_id if snapshot else None,
                "duckdb_table": identifier,
            }
            log.info("exported to duckdb", table=identifier, rows=arrow.num_rows)

        _write_lineage(connection, provenance)
    finally:
        connection.close()

    return provenance


def _write_lineage(connection: duckdb.DuckDBPyConnection, provenance: dict[str, Any]) -> None:
    """Record which Iceberg snapshot each exported table came from."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS _lineage (
            exported_at   TIMESTAMP,
            iceberg_table VARCHAR,
            duckdb_table  VARCHAR,
            snapshot_id   BIGINT,
            row_count     BIGINT
        )
        """
    )
    now = _dt.datetime.now(tz=_dt.UTC)
    connection.executemany(
        "INSERT INTO _lineage VALUES (?, ?, ?, ?, ?)",
        [
            (now, identifier, meta["duckdb_table"], meta["snapshot_id"], meta["rows"])
            for identifier, meta in provenance.items()
        ],
    )


def query(sql: str, path: Path | None = None) -> pa.Table:
    """Run a read-only query against the analytics database."""
    connection = connect(path, read_only=True)
    try:
        # fetch_arrow_table(), not .arrow(): in current DuckDB the latter
        # returns a RecordBatchReader, which is a streaming cursor tied to the
        # connection. Returning one from here would hand the caller a reader
        # that is already dead by the time the finally-block has closed the
        # connection.
        result = connection.execute(sql)
        return (
            result.to_arrow_table()
            if hasattr(result, "to_arrow_table")
            else result.fetch_arrow_table()
        )
    finally:
        connection.close()


def table_names(path: Path | None = None) -> list[str]:
    """Every table and view in the analytics database."""
    connection = connect(path, read_only=True)
    try:
        rows = connection.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]
