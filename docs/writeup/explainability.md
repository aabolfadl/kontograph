# Explaining a payment decision in under 200 milliseconds

*What the explainer does, what it costs, and where it is weaker than it looks.*

---

## The problem, stated precisely

SEPA Instant gives the bank about ten seconds end to end. KONTOGRAPH's internal
budget is **200 ms p99** for scoring plus explanation, and the explanation is not
optional: an alert an investigator cannot act on is an alert that costs money and
produces nothing.

That rules out the standard toolkit by construction, not by preference:

| Method | Cost per instance | Why it does not fit |
|---|---|---|
| Exact Shapley | exponential | Not a latency problem, a computability one |
| KernelSHAP | 10²–10⁴ model calls | Seconds, and variance depends on sample count |
| GNNExplainer | 100–300 gradient steps | Seconds per instance, by design |
| Integrated gradients | 20–200 forward passes | Borderline; attributes to *features*, not payments |

All four are per-instance optimisation. The budget allows one forward pass.

## The move: pay the cost once, offline

The explanation problem is *amortised* — turned from a per-instance optimisation
into a learned function.

```
                    OFFLINE  (Kaggle GPU, hours)                ONLINE (< 3 ms)
   ┌────────────────────────────────────────────┐        ┌──────────────────────┐
   │  teacher: per-instance edge-mask descent   │  ───▶  │  student: one forward│
   │  ~300 steps/instance, seconds each         │ distil │  pass over the same  │
   │  produces (event, salient subgraph) pairs  │        │  neighbourhood embeds│
   └────────────────────────────────────────────┘        └──────────────────────┘
```

The teacher is the slow, well-understood perturbation explainer: learn a soft
mask over the temporal neighbourhood's edges by gradient descent, regularised
toward sparsity. Seconds per instance, run offline over a training slice.

The student reads the TGN's temporal-neighbourhood embeddings — already computed
by the detector, so the marginal cost is one small head — and predicts the mask
directly. It is trained on the teacher's masks with a distillation loss plus a
comprehensiveness regulariser, so it is optimised toward *being faithful*, not
only toward *imitating the teacher*.

**The claim is not that the student equals the teacher.** It is that the student
is fast enough to run inline, and that the gap is measured and published.

## Faithfulness, and why the gap is published

An explanation that is fast and wrong is worse than no explanation: it launders a
model's decision into a sentence an investigator will believe.

Four measurements, all in `kg_models/explain/faithfulness.py`:

- **Comprehensiveness** — remove the cited edges; the score should drop. If it
  does not, the explanation did not identify what drove the decision.
- **Sufficiency** — keep *only* the cited edges; the score should hold up.
- **Deletion / insertion curves** — the ranking, not just the top-k set.
- **Stability** — top-k Jaccard and rank correlation across consecutive event
  ticks. An explanation that reshuffles on every tick cannot be reasoned about,
  even if each individual version is faithful.

The teacher is the **fidelity ceiling**, and the student's shortfall against it
is reported as a number in the ablation table rather than described in prose.
Per the evaluation protocol, that number is generated, never typed.

## Measured: the amortisation works, the fidelity measurement does not

Run on a Kaggle T4 over 57 explained instances (`02_distill_explainer`):

| measure | value |
|---|---|
| teacher, median per explanation | **575.6 ms** |
| student, median per explanation | **2.38 ms** |
| speedup | **241x** |
| top-5 Jaccard vs teacher | 0.904 |
| Spearman rank correlation vs teacher | -0.038 |
| **median candidates per instance** | **1.0** |

**The speed claim holds, and by more than the budget needs.** The teacher at
575 ms does not merely strain a 200 ms budget, it exceeds the entire SLA nearly
threefold — the notebook's own line is that the budget allows `0.32`
explanations. The student fits in 2.38 ms, leaving the rest of the budget for
the work the explanation is about. This is the amortisation claim, and it is
the one the spike was built to make.

**The fidelity numbers, however, measure nothing.** Read the last row first: the
median explained instance has **one candidate edge**.

With a single candidate, top-5 Jaccard is 1.0 by construction — both "top 5"
sets are the same singleton — and the stability helper returns 0.0 by its own
`if a.size < 2` guard. So the 0.904 Jaccard is very largely an artifact of
set size, and the -0.038 Spearman is very largely that guard firing. Neither
number is evidence about the student.

This is worth stating precisely, because the two readings differ and only one is
supportable:

- *"The student is 241x faster but does not reproduce the teacher's ranking."*
  **Not supported.** With one candidate there is no ranking to reproduce.
- *"The fidelity comparison is vacuous at this candidate-set size."*
  **This is what the data says.**

Which is a null result about the experiment, not about the method — and the
distinction matters, because the first version would have been a much more
interesting claim to publish and it would have been wrong.

### Why the neighbourhoods are singletons, and what would fix it

The teacher explains the top-scoring test events, and its candidate set is the
temporal neighbourhood the detector actually attended to. For most of those
events the two accounts have almost no prior history inside the neighbour
loader's window — which is consistent with what the detector finds suspicious
(a first payment to a fresh beneficiary) but leaves nothing to attribute among.

An explanation of a one-edge neighbourhood is trivially correct and useless, and
the fidelity metrics faithfully report that it is trivial.

To make the measurement mean something:

- **Explain instances selected for neighbourhood size**, not only for score.
  Stratify by candidate count and report fidelity per stratum; an aggregate over
  mostly-singletons will always be dominated by them.
- **Widen the candidate set** — a larger `LastNeighborLoader` size, or
  candidates drawn from both endpoints' full histories rather than the attended
  subgraph.
- **Report the candidate-count distribution alongside every fidelity figure.**
  Its absence is what let a 0.86 Jaccard look like a result for one whole
  iteration of this analysis.

Only then is the ranking question — whether pointwise distillation loss teaches
order — actually testable. It remains the most likely weakness (the student is
trained with per-edge `binary_cross_entropy_with_logits`, which never asks the
order to match), but on this evidence it is a hypothesis, not a finding.

A bug found while reading these numbers is worth recording: the first run
reported `speedup` as `NaN`, because the fidelity cell re-initialised
`teacher_times` and wiped the timings the teacher loop had collected. Fixed —
and `median_candidates` was added in the same pass, which is the field that
turned the analysis around.

## Counterfactuals: the part an investigator can actually use

Attribution says *feature 17 mattered*. That sentence cannot go in a
Verdachtsmeldung. What can is:

> Remove these two payments through this intermediary, and the score falls below
> threshold.

That is a claim an investigator can check against the account. So the console's
narrative is built from the **counterfactual**, not from the attribution — beam
search over minimal edits to the temporal subgraph, constrained to edits that are
semantically possible in a payment network (you can remove a payment; you cannot
give an account a negative balance), and budgeted in milliseconds.

### Two kinds of counterfactual, and they are not interchangeable

This distinction caused a real bug, and it is worth stating plainly because the
failure mode is the kind that reaches a regulator.

- **Edge-space** (the TGN path): names specific transactions. Checkable.
- **Feature-space** (the LightGBM gateway path): names an *aggregate indicator* —
  "hold `creditor_txn_24h` at its baseline and the score drops". Weaker, because
  it identifies no transaction at all.

The SAR generator originally rendered both through the same template. Given a
feature-space counterfactual, which carries no transaction ids, it produced:

> The alert rests on 0 related payment(s): . Excluding them, the score falls to…

A sentence asserting payment evidence that does not exist, in a document destined
for a suspicious-activity filing. It now branches on the counterfactual's kind and
says what the feature-space version actually is, including the sentence *"this is
a statement about an aggregate indicator, not about any identified transaction,
and does not on its own identify conduct to report."*

Regression test:
`tests/unit/test_serving_governance.py::test_feature_space_counterfactual_is_not_dressed_up_as_evidence`.

## What it costs, measured

From `artifacts/loadtest/latency_budget.md`, regenerated by `kg serve smoke`:

| stage | p50 | p95 | p99 | share of p99 |
|---|---|---|---|---|
| feature_fetch | 1.84 ms | 3.13 ms | 4.67 ms | 59% |
| graph_assemble | 0.00 ms | 0.01 ms | 0.01 ms | 0% |
| inference | 0.42 ms | 0.74 ms | 1.78 ms | 22% |
| **explain** | 0.00 ms | 0.00 ms | **1.87 ms** | 23% |
| **counterfactual** | 0.00 ms | 0.00 ms | **3.59 ms** | 45% |
| unaccounted | 0.17 ms | 0.26 ms | 0.46 ms | 6% |
| **total** | 2.71 ms | 5.59 ms | **7.97 ms** | |

**192.0 ms of headroom against the 200 ms SLA.**

Four things about this table are worth reading carefully:

1. **The number moves.** Repeat runs on the same machine gave p99 between 3.6 ms
   and 8.0 ms depending on background load — a 4-core laptop, not a serving host.
   The claim is not the specific figure; it is a margin of more than an order of
   magnitude, which survives that variance.
2. **The p50 for explain and counterfactual is 0.000 ms** because only alerts are
   explained — 98 of 2,000 payments in that run. The p99 is the honest number for
   the explained path, and it is the one the budget is checked against.
3. **The shares do not sum to 100%.** Each stage's p99 is measured independently,
   and the slowest feature fetch is not in the same request as the slowest
   counterfactual. This is deliberate: a budget table that forced the shares to
   sum would have to pretend one request was worst at everything.
4. **The percentiles are nearest-rank, not interpolated.** numpy's default linear
   interpolation reports a latency no request experienced, on the flattering
   side — with 99 fast samples and one at 900 ms it puts p99 at ~14 ms. For a
   number that gates an SLA claim, the reported figure rounds against us.

## Honest limitations

- **The gateway measured above runs LightGBM, not the TGN.** That figure is
  the tabular path with a feature-space counterfactual. The TGN's amortised
  explainer is trained and evaluated on Kaggle; its serving latency is a separate
  measurement and is not folded into this table.
- **The student's fidelity gap against the teacher is real** and will be
  published in the ablation table whichever way it falls.
- **`explain` at p99 1.87 ms is feature attribution over a gradient-boosted
  tree** — cheap because the model is cheap. A TGN's explanation head is a
  neural forward pass and will cost more. The headroom is large enough that this
  is not expected to threaten the SLA, but it is not yet measured, and an
  unmeasured claim is not a claim.
- **Faithfulness metrics are computed against the model, not against the truth.**
  A perfectly faithful explanation of a wrong model is a perfect explanation of a
  wrong answer.

## Where the code is

| Piece | Path |
|---|---|
| Teacher (perturbation) | [teacher.py](../../packages/kg_models/src/kg_models/explain/teacher.py) |
| Student (amortised) | [student.py](../../packages/kg_models/src/kg_models/explain/student.py) |
| Faithfulness metrics | [faithfulness.py](../../packages/kg_models/src/kg_models/explain/faithfulness.py) |
| Counterfactual search | [counterfactual.py](../../packages/kg_models/src/kg_models/explain/counterfactual.py) |
| Serving-path explainer | [scoring.py](../../packages/kg_serve/src/kg_serve/scoring.py) |
| SAR narrative (DE/EN) | [sar.py](../../packages/kg_gov/src/kg_gov/sar.py) |
| Latency budget | [latency.py](../../packages/kg_serve/src/kg_serve/latency.py) |
