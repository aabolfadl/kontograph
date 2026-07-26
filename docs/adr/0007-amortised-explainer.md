# ADR-0007 — Amortised explainer distilled from a slow teacher, not GNNExplainer at serving time

**Status:** accepted · **Date:** 2026-07-26

This is the vertical spike, so this is the most important record in the folder.

## Context

Regulation (DSGVO Art. 22, and AI Act Art. 13/14 for the credit module) requires meaningful
information about the logic of an automated decision. For a graph model that means *the subgraph* —
not a SHAP bar chart over flattened features.

The standard tools — GNNExplainer, PGExplainer, SubgraphX — are perturbation-based: they optimise an
edge mask by running many forward passes per instance. That costs **seconds**. The SLA is 200 ms
total, of which explanation gets a slice. This is not a tuning problem; it is two orders of
magnitude.

## Decision

Train an **amortised explainer head** jointly with the detector. It consumes the TGN's temporal
neighbourhood embeddings and predicts edge importances in a **single forward pass**, distilled from
an expensive offline perturbation teacher.

## Rationale

**The cost structure is wrong, so move the cost.** Perturbation search is per-instance
inference-time work. Distillation moves it to training time, where seconds per instance are
affordable, and leaves a forward pass at serving time.

**Distillation does not automatically preserve faithfulness — so we measure it.** The student could
learn to produce plausible-looking masks that are not causally load-bearing. That is the central risk
of this design, and it is why the evaluation protocol commits *in advance* to reporting
comprehensiveness, sufficiency, deletion/insertion curves, and top-k agreement against the teacher.
The teacher is the fidelity ceiling; the gap gets published.

**Counterfactuals must be searched, not predicted.** A predicted counterfactual is unfalsifiable. We
run a small constrained beam search over the salient subgraph and **re-score** each candidate, so a
counterfactual is only emitted if it demonstrably moves the score across the threshold. Validity is
verified per instance, not modelled.

**Stability is a first-class requirement.** An explainer that tells a different story one second
later is worse than none, because an investigator builds a case on it. We define a stability metric
across event ticks and report the share of unstable explanations.

## Consequences

**Good.** Explanation inside the latency budget — the thing most graph deployments cannot do.

**Bad.** Two models to train, and the student is coupled to the detector's internals: a detector
change invalidates the student.

**Bad.** The honest outcome might be a large fidelity gap. If so, that is the write-up's finding, and
it is a more useful contribution than a flattering number.

## What would change our mind

- If teacher agreement is poor enough that the explanation misleads, the correct product decision is
  to serve explanations **asynchronously** — score in 200 ms, explain in 2 s, show a spinner — and
  say plainly that real-time faithful explanation did not work at this budget.
- A perturbation method fast enough to run inline would remove the need for distillation entirely.
