# ADR-0003 — Not Spark Structured Streaming

**Status:** accepted · **Date:** 2026-07-16

## Context

Spark is the default answer for most data platforms, and Structured Streaming would let one
engine cover both the backfill and the streaming path — which is superficially the exact
training-serving-skew solution this project is built around.

## Decision

Do not use Spark for the streaming path.

## Rationale

**Micro-batch is the wrong shape for a 10-second settlement window.** Structured Streaming is
fundamentally micro-batch; continuous processing has remained experimental and does not support the
stateful aggregations we need. A trigger interval large enough to be efficient is a meaningful
fraction of the SLA in design principle P4, and one small enough to be timely spends most of its time on scheduling
overhead. Flink's per-event model matches the problem; Spark's does not.

**Event-time state with per-key ordering is Flink's home ground.** Our features are
"aggregate over the last N of this account's transactions, strictly before this event". Flink's
keyed state and watermark semantics express this directly. Doing the same in Spark means
`flatMapGroupsWithState` and hand-rolled timers, which is more code for a weaker guarantee.

**Resource footprint.** A Spark cluster on a laptop is worse than Flink, not better.

**The unification argument does not survive contact.** Even with Spark on both sides, the offline
job reads a table and the online job reads a stream, so the *definitions* still diverge unless
something forces them to agree. That "something" is the feature DSL (ADR-0005) — which works
regardless of engine. Spark buys the appearance of unification without the mechanism.

## Consequences

We give up Spark's ecosystem: MLlib, its Iceberg maintenance procedures, and the familiarity of a
tool most teams already run. Iceberg maintenance is handled through pyiceberg and DuckDB instead,
which is adequate at our data volumes and would need revisiting at genuine scale.

## What would change our mind

- Data volume beyond what a single Flink TaskManager and DuckDB can hold — Spark's batch engine is
  genuinely better above a few hundred GB.
- An employer whose platform is Spark-shaped. Fitting the organisation beats being right in a vacuum.
