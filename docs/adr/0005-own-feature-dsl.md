# ADR-0005 — A small in-house feature DSL, not Feast or Tecton

**Status:** accepted · **Date:** 2026-07-18

## Context

Design principle P3 makes eliminating training-serving skew a structural requirement, and the
dual-compiled feature layer is the one non-obvious decision the rest of the architecture rests on.
Feature stores exist for exactly this problem; not using one demands justification.

## Decision

Build a small declarative DSL (`Feature(entity, agg, window, filter, source)`) lowered to a typed IR,
then compiled to DuckDB SQL, Flink SQL, and an incremental online executor.

## Rationale

**Feast does not solve the skew problem; it relocates it.** Feast materialises features computed by
*your* pipeline into an online store. The offline definition and the streaming definition are still
two pieces of code you wrote separately — Feast just gives them a shared serving API. The skew lives
in the transformation, and that is precisely the part Feast leaves to you.

**Tecton does solve it, and is a commercial platform.** It cannot be part of a reproducible,
self-hosted, `make up` artifact.

**Point-in-time correctness has to be a property we can *test*, not trust.** Owning the IR is what
makes the property suite in design principle P2 possible: we can assert that every compiled query orders on event
time with a strict `<`, generate random event streams, and diff all three backends against each
other. With a vendor store, the strongest available claim is "we configured it correctly".

**The scope is genuinely small.** Windowed aggregations over two entity types. That is a few hundred
lines of compiler, not a platform. The engineering judgment being demonstrated is knowing which
small things are worth owning.

## Consequences

**Good.** The anti-skew claim has an artifact behind it: a failing test if it is ever untrue.
**Good.** Feature definitions are reviewable in a pull request as code.

**Bad.** No feature discovery UI, no lineage-across-teams, no serving infrastructure for free. All
irrelevant at one-team scale, all decisive at fifty-team scale.
**Bad.** We own the bugs. Mitigated by the property suite being the most heavily tested code here.

## What would change our mind

- More than a handful of teams sharing features — discovery and governance then dominate.
- Needing aggregation types (sessionisation, sequence features) whose compiler cost outgrows the
  benefit of owning it.
