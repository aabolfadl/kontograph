# ADR-0006 — Continuous-time TGN, not a snapshot GNN or static graph features

**Status:** accepted · **Date:** 2026-07-26

## Context

Design principle P1: a transaction is an edge, and mules look anomalous *in formation*. Three ways to
model that:

1. **Static graph features** — build one graph over a window, compute degree / PageRank / component
   size, feed to LightGBM.
2. **Snapshot GNN** — discretise time into buckets, run a GNN per snapshot, stitch with an RNN
   (EvolveGCN, DySAT).
3. **Continuous-time dynamic graph** — model the event stream directly with per-node memory and
   temporal attention (TGN family).

## Decision

Continuous-time TGN, with (1) kept as an ablation rung and (2) rejected.

## Rationale

**Discretisation destroys the signal we are hunting.** Structuring, smurfing and rapid layering are
defined by *inter-event timing* — twelve transfers in nine minutes is the typology; the same twelve
over three weeks is a small business. Any snapshot granularity coarse enough to be affordable erases
exactly that, and one fine enough to preserve it produces mostly-empty snapshots.

**The 10-second settlement window forbids waiting for a bucket boundary.** A snapshot model must
either wait for the current bucket to close — unaffordable under design principle P4 — or score against a stale
snapshot. Continuous-time models update node memory on each event, so state is current at scoring
time by construction.

**Memory is the mechanism that makes an account's history cheap at serving time.** A per-node memory
vector summarises everything the account has done without re-traversing its history for every score.
That is what makes a graph model feasible inside 200 ms at all — the architecture choice is *driven
by* the latency constraint, which is precisely the design principle P4 argument.

**Static features stay as the honest control.** Rung 2 of the ablation ladder exists specifically so
we can answer "is it the graph, or is it the neural network?" If static graph features on LightGBM
capture most of the lift, the README will say so.

## Consequences

**Good.** The architecture matches the physical process. Memory doubles as the serving cache.

**Bad.** Training is inherently sequential over the event stream — batching is awkward and this is a
real throughput cost. It is the main reason training needs a GPU and goes to Kaggle.

**Bad.** Memory state must be checkpointed and restored consistently, or the model silently scores
against cold memory after a restart. Handled explicitly in the serving design.

**Bad.** Reproducibility is harder: memory makes results order-dependent, so event order is part of
the dataset fingerprint rather than an incidental detail.

## What would change our mind

- If the memory-disabled ablation (rung 3) matches rung 4, the memory module is complexity without
  benefit and should be cut — and that result would be published either way.
- Typologies defined by slow structure rather than fast timing would favour snapshots.
