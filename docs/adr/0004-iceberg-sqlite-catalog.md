# ADR-0004 — Iceberg with a SQLite catalog, not a REST catalog service

**Status:** accepted · **Date:** 2026-07-13

## Context

The lakehouse needs a table format with schema evolution, hidden partitioning, snapshot isolation and
time travel. Iceberg is specified in the platform scope and is the right call — time travel in particular
is not a nicety here, it is how "what did the feature store believe at time T?" gets answered during
an audit.

Iceberg needs a catalog. The options are a REST catalog service (another container), a Hive
metastore (another container plus a database), a JDBC/SQL catalog, or filesystem-only "HadoopCatalog"
(deprecated in practice and unsafe for concurrent writes).

## Decision

Use pyiceberg's **SQL catalog backed by SQLite**, stored under `data/warehouse/catalog.db`, with
table data in S3-compatible object storage.

The local object store is **SeaweedFS** (Apache-2.0), run as a single `weed server` with its S3
gateway. MinIO was the obvious first choice, but its community edition no longer publishes container
images, so a pinned `minio/minio` tag cannot be pulled, and a stack pinned to an image nobody can
fetch is not reproducible. Nothing in the code is specific to either: pyiceberg, MLflow and Flink's
S3A connector all speak plain S3 with path-style addressing.

## Consequences

**Good.** Zero extra containers. The catalog is a single file, so a broken local state is fixed by
deleting it and re-registering — which matters when someone is 20 minutes into their first run.

**Good.** Genuinely Iceberg: snapshots, schema evolution and time travel all work. Nothing about the
table format is compromised, only the catalog backend.

**Bad.** SQLite is single-writer. Concurrent commits from Flink and a Dagster backfill will contend.
Acceptable because our writes are serialised by design; it would not survive a real multi-tenant
platform.

**Bad.** Engine interop is narrower. DuckDB and pyiceberg read it; Trino/Spark would want a REST or
Hive catalog. Since the whole point of the SQL catalog is that it is swappable, the catalog
construction is isolated in one function (`kg_lake.catalog.get_catalog`) and configured by URI, so
moving to a REST catalog is a config change plus one container.

## What would change our mind

- More than one concurrent writer.
- Wanting Trino or Spark to query the same tables.
- Deploying anywhere that a durable, backed-up catalog matters — SQLite on a laptop disk is not that.
