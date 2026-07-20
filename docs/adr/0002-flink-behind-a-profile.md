# ADR-0002 — Flink is real, but behind an optional profile

**Status:** accepted · **Date:** 2026-07-20

## Context

The platform scope names Flink as the streaming compute engine, and the load-bearing claim of the whole
project is that **one feature definition compiles to both a streaming and a backfill backend**. That
claim is only meaningful if the streaming target is a genuine distributed stream processor rather
than a Python loop wearing a costume.

But a Flink JobManager + TaskManager on a 4-core laptop costs ~2 GB of RAM and 60+ seconds of
startup, and the live demo — a stream of payments animating across a graph — becomes visibly
sluggish. That directly damages design principle P10 ("under ten seconds to comprehension").

## Decision

The feature DSL emits **three** targets from one IR:

1. `to_duckdb_sql()` — the backfill/training path
2. `to_flink_sql()` — a real Flink SQL job, runnable via `make flink-up`
3. `to_online()` — an incremental Python executor over Redis state, used by the serving gateway and
   as the default local stream engine

Flink lives in `docker/compose.flink.yml` behind the `flink` profile. The default `make up` does not
start it.

The equivalence of the three is not asserted, it is **tested**: the point-in-time property suite
runs the same hypothesis-generated event stream through all available backends and diffs the
outputs. The Flink leg is skipped locally when the profile is down, and runs in CI.

## Consequences

**Good.** Fast default loop. The anti-skew claim keeps its teeth, because the Flink compiler is
exercised by tests rather than being a slide.

**Good.** The online executor is not a toy we tolerate — it is the code that *actually runs in the
serving path*, so it gets the most scrutiny of the three, which is the correct allocation.

**Bad.** Three backends is three times the compiler surface, and a bug can hide in the one that is
skipped locally. Mitigated by making the CI Flink job non-optional: a red CI on the Flink leg blocks
the merge even though the developer's laptop never ran it.

**Bad.** Someone skimming the repo may conclude Flink is decorative. This ADR and the CI badge are
the answer.

## What would change our mind

- If the Flink SQL compiler's output diverges from DuckDB in a way that cannot be tested cheaply,
  the honest move is to drop one backend rather than ship an untested equivalence claim.
- A target deployment that is Flink-only would flip the default.
