# ADR-0011 — A native quality gate for the blocking path, Great Expectations for the report

**Status:** accepted · **Date:** 2026-07-13
**Supersedes:** the assumption in the M2 plan that Great Expectations would carry the whole
data-quality pillar.

## Context

The platform scope asks for "Great Expectations / Soda checks as **pipeline-blocking assertions**". The
operative word is *blocking*: the check must run before every promotion from bronze to silver, in
every process, in CI, and on every Dagster asset materialisation.

Great Expectations was installed and measured rather than assumed:

| | measured |
|---|---|
| install footprint (with Dagster + dbt + confluent-kafka) | 538 MB, 81 packages |
| `import great_expectations` — cold | **62 s** |
| `import great_expectations` — warm | **~10 s** |
| `import dagster` — warm (for comparison) | 2.1 s |
| `import pyiceberg, confluent_kafka, duckdb` — warm | 0.1 s |

Ten seconds of import, paid per process, for an assertion whose actual evaluation takes
milliseconds. Dagster runs asset checks in subprocesses; pytest pays it on every run; CI pays it on
every job. A gate that expensive to *load* gets disabled, and a disabled gate is worse than no gate
because it looks like coverage.

## Decision

Split by where the cost lands.

**The blocking path is native.** `kg_lake.quality` implements the expectation suite directly over
Arrow and DuckDB compute: zero import cost, exact, and testable without a `DataContext`. This is
what runs on every promotion and in every Dagster asset check.

**Great Expectations stays for the published quality report**, behind the `quality` extra, where a
slow import is irrelevant and GE's validation rendering earns its place.

**Both read the same suite.** The expectations are defined once, as data, in
`kg_lake.quality.PAYMENTS_SUITE`. The two engines cannot disagree about what the rules are — only
about how they present the outcome.

## Rationale

**The severity model is the part that matters, and it is ours either way.** Expectations are tagged
`BLOCKING`, `QUARANTINE` or `WARN`, and the pipeline behaves differently for each: block the
promotion, divert the row, or record it. Making everything blocking produces a pipeline that halts
on a 0.01 % anomaly and trains everyone to bypass it; making nothing blocking produces a report
nobody reads. That design is independent of the engine.

**Row-level results are needed, not just pass/fail.** Quarantine requires knowing *which* rows
failed so they can be diverted with the name of the expectation they broke. Predicates that return a
validity mask give that directly.

**It earned its keep immediately.** On its first run against real data the gate blocked the
promotion on `debtor_and_creditor_differ`: 3 of 34,769 rows were payments from an account to itself.
The cause was a type confusion in the simulator — `social_circle` stores household *positions* but
the self-exclusion filter compared against an *account index*, so a household could appear in its
own social circle. That bug had survived code review and the M1 test suite, and the gate caught it
in the first minute of M2.

## Consequences

**Good.** No import tax on the hot path. The rules are readable Python in the repository rather than
JSON in a store somebody configured once.

**Good.** The gate runs in CI with no `DataContext`, no store, and no filesystem state.

**Bad.** We own this code. Mitigated by its small scope — predicates over Arrow — and by its test
coverage.

**Bad.** No Data Docs by default, and none of GE's expectation library. If a suite ever needs
something GE already has, it should be written against GE in the report path rather than
reimplemented here.

**Honest limitation:** the GE report path is wired but thin. It is not a substitute for GE's full
feature set, and this ADR does not claim it is.

## What would change our mind

- GE import cost dropping to the low single digits of seconds, which would remove the entire basis
  for the split.
- A team that already operates GE, where matching their tooling beats saving ten seconds.
- Needing expectation types whose implementation cost here exceeds the import cost there.
