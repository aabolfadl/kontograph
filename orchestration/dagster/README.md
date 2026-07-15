# Dagster orchestration

```bash
uv sync --extra orchestration
uv run dagster dev -f orchestration/dagster/definitions.py
# then open http://localhost:3000
```

## Why assets rather than a DAG of tasks

A task DAG says *when to run something*. An asset says *what should exist and
what it is derived from*. Only the second produces a lineage graph, and this
project needs one for the governance layer (M7), which renders its diagram from
`lineage_edges()` rather than from a hand-drawn picture that can drift.

## Blocking asset checks are the quality gate

The data-quality suite is expressed as Dagster **asset checks with
`blocking=True`**, not as assertions buried inside a transform. A failing
blocking check stops downstream assets materialising, and it does so visibly —
in the UI, in the run history, and in the asset's metadata.

That is what the platform scope means by "pipeline-blocking assertions". An
`assert` inside a function is invisible until someone reads a log.

| Check | Asset | Blocking | Guards |
|---|---|---|---|
| `dataset_is_reproducible` | `synthetic_dataset` | yes | ADR-0009's premise: the recipe regenerates the bytes |
| `bronze_quality_gate` | `bronze_payments` | yes | structural validity before promotion |
| `bronze_distribution_warnings` | `bronze_payments` | no | VoP rate, ingest lag, large amounts |
| `silver_referential_integrity` | `silver_payments` | yes | no dangling account references |
| `labels_never_precede_their_events` | `silver_payments` | yes | no time-travelling labels (design principle P2) |

## Jobs

- `medallion` — simulator through to validated silver. No extra dependencies
  beyond the default install.
- `full_pipeline` — adds the dbt gold marts.

The daily schedule is **stopped by default**. A schedule that begins firing the
moment someone opens the UI is a surprise, and surprises in a pipeline are
expensive.
