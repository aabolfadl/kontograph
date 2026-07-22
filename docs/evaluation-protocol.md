# Evaluation protocol

**Status:** frozen.
**Committed:** before any model in this repository was trained. The git history is the evidence.

> The engineering invariants: *"No metric chosen after seeing the results. The evaluation protocol is
> written first, and committed before the model is trained."*

This document is the contract against which every number in the README is produced. Changing it
after results exist is permitted, but only as a **new commit that states what changed and why**,
leaving the original in history. A protocol that can be silently edited is not a protocol.

---

## 1. What we are predicting

**Target.** For a transaction `t`, the binary label `y(t) = 1` iff a `LabelOutcome` exists with
`subject_id = t.tx_id` and `label ∈ {CONFIRMED_FRAUD, CONFIRMED_LAUNDERING}`.

**`SUSPICIOUS_UNPROVEN` is not a positive.** It is excluded from the positive class and from the
negative class — those rows are dropped from training and from the primary evaluation. Counting
unproven suspicion as fraud trains the model to reproduce the *incumbent alerting system's*
behaviour, which is the failure mode design principle P7 warns about, not to detect crime.

**Unlabelled is negative, with a caveat we state rather than hide.** Transactions with no
`LabelOutcome` are treated as negative. This is wrong in a known direction: undetected crime is
silently counted as legitimate, so all reported precision is an *upper* bound on true precision and
all recall is an *upper* bound on true recall. Because the data is simulated we can also report the
oracle metrics against `typology_membership` ground truth — and we report **both**, with the gap
between them labelled as "what label bias costs you".

---

## 2. Splitting

**Chronological only.** Never random, never stratified-random, never k-fold over shuffled rows.

```
|<-- train [t0, t1) -->|  purge  |<-- val [t1, t2) -->|  purge  |<-- test [t2, t3) -->|
                       |<-embargo->|                  |<-embargo->|
```

- **Purge and embargo, 1 day each.** Transactions within one day on either side of a boundary are
  dropped. What this addresses is *residual correlation*, not leakage: a training event just before
  `t1` and a validation event just after it share almost all of their feature history, so their
  errors are correlated and the validation estimate is slightly optimistic.

  It does **not** prevent feature leakage — the feature layer already guarantees no feature reads
  forward, and that guarantee is enforced by the point-in-time property suite, not by this band.

  The band is deliberately narrow. A validation event whose features were partly computed from
  training-period history is not contaminated; that is exactly what happens in production on the
  day after deployment, and treating it as a defect would be modelling a problem that does not
  exist. `kg_eval.splits.make_boundaries` refuses a band wider than a quarter of the shortest
  fold, because an over-wide band empties the training set and the resulting error surfaces
  several layers away from its cause.
- **Label availability.** A label may enter training only if `decided_ts < t1`. A label whose
  `event_ts` is in the training window but whose `decided_ts` is after the boundary **does not
  exist** as far as the training run is concerned. This is the difference between an honest backtest
  and a time-travelling one.

Split boundaries are defined as fractions of the simulated period (`0.60 / 0.20 / 0.20`) and stored
in the dataset manifest, so they cannot be nudged per model.

**Blocked cross-validation** for variance estimates: 5 contiguous, non-overlapping time blocks, each
used once as validation with only *earlier* blocks for training (expanding-window). No future block
ever trains a model evaluated on an earlier one.

---

## 3. Headline metrics

Reported in this order. ROC-AUC is **not** a headline metric and never appears above the fold — at a
~0.1 % positive rate it is dominated by the negative class and flatters everything.

| Metric | Definition | Why |
|---|---|---|
| **PR-AUC** | Area under precision–recall | The only threshold-free metric that respects class imbalance |
| **Precision@k** | Precision among the top `k` scored transactions per day | `k` is the compliance team's actual capacity, not a tuning knob |
| **Recall@k** | Share of true positives caught inside that same budget | What the bank loses by having a finite team |
| **Expected cost / 10k tx** | `Σ cost(decision, truth) / n × 10 000` | The quantity the bank actually pays |
| **Alert-to-SAR conversion** | Share of alerts an analyst escalates | The operational metric compliance is measured on |

**`k = 200 alerts/day.`** Stated here, before results. It is configurable (`KG_ANALYST_CAPACITY_PER_DAY`)
but the headline number is always the `k = 200` one. Sensitivity to `k` is reported as a curve over
`k ∈ {50, 100, 200, 500, 1000}`, not by choosing the flattering `k`.

---

## 4. Cost model

All four cells are **absolute costs**: the money the institution is out by, given what happened.
Mixing an absolute cost with one measured relative to a do-nothing baseline is an easy and
consequential mistake — see the changelog entry for 2026-07-22.

| | Predicted negative | Predicted positive |
|---|---|---|
| **Truly negative** | 0 | `C_FP` = €35 investigation + €12 friction = **€47** |
| **Truly positive** | `C_FN` = the transaction amount, capped at €100 000 | `C_TP` = €35 investigation + (1 − *p*) × amount |

where *p* = **0.85** is the share of the loss actually prevented by alerting in time. It is high
because that is the premise of the project: a decision inside the ten-second window can hold the
payment before it settles irrevocably. It is not 1.0 — some alerts land after settlement, some are
overridden by the payer, and some funds are already gone.

`C_TP` is therefore always **positive and below `C_FN`**: catching fraud reduces cost, it never
creates income. The per-transaction `C_FN` matters too — missing a €40 mule hop and missing a
€90,000 APP fraud are not the same mistake, and a threshold tuned against an averaged constant
systematically under-alerts on the large ones.

`C_FP`, the investigation cost and *p* are order-of-magnitude estimates for a mid-sized German
institution, stated as assumptions and varied in a sensitivity analysis over
`C_FN/C_FP ∈ {10, 50, 100, 500}`. They are not measured facts and are not presented as such.

**The operating threshold is chosen on the validation split by minimising expected cost**, then
applied unchanged to test. Choosing it on test is the single most common way published evaluations
overstate results.

---

## 5. Calibration

A score that feeds a cost-based threshold must be a probability, not a ranking.

- Isotonic regression and Platt scaling, both fitted **on the validation split only**.
- Reported: Brier score, Expected Calibration Error (15 equal-mass bins), and a reliability diagram.
- The calibrator is part of the model artifact. An uncalibrated score never reaches the threshold.

---

## 6. Ablations — does the graph earn its complexity?

Run as a ladder, each rung adding exactly one thing:

| # | Model | Question it answers |
|---|---|---|
| 0 | Majority / random baseline | Is the metric even meaningful here? |
| 1 | LightGBM on tabular features | What does a competent non-graph model get? |
| 2 | + static graph features (degree, PageRank, component size, computed point-in-time) | Is it the *graph* or the *neural net*? |
| 3 | + temporal GNN (TGN), memory module **disabled** | Does temporal attention alone help? |
| 4 | + memory module | Does per-node state help? |
| 5 | + amortised explainer head (joint training) | Does explanation supervision cost accuracy? |

**Commitment made in advance:** if rung 4 does not beat rung 1 by more than the width of its
bootstrap confidence interval on PR-AUC *and* precision@200, the README will say the graph model did
not win, in those words, above the fold. Rung 5's accuracy delta relative to rung 4 is reported as
the *price of explainability* whether it is positive or negative.

---

## 7. Robustness evaluations

Each of these is a headline-adjacent table, not an appendix.

- **Adversarial / evasive typologies.** The simulator emits evasive variants (sub-threshold
  structuring, deliberate dwell time, fan-out through fresh accounts). Report metrics separately on
  standard vs. evasive, and the degradation between them.
- **Label-delay simulation.** Retrain under 14 / 30 / 60-day verification latency; publish the decay
  curve. Metric: precision@200 at each latency, relative to the zero-delay oracle.
- **Concept drift.** Evaluate on the final time block with a model trained only on the first,
  quantifying decay without retraining.
- **Cold start.** Metrics restricted to accounts with fewer than 5 prior transactions, where the
  graph has least to say.

---

## 8. Explanation evaluation

The spike needs its own protocol, because "the explanation looked reasonable" is not a measurement.

| Metric | Definition |
|---|---|
| **Comprehensiveness** | `score(full) − score(full \ salient subgraph)`. Higher = the subgraph was load-bearing |
| **Sufficiency** | `score(salient subgraph alone)` relative to `score(full)`. Higher = the subgraph suffices |
| **Deletion AUC** | Score as edges are removed in descending importance. Lower = better |
| **Insertion AUC** | Score as edges are added in descending importance. Higher = better |
| **Teacher agreement** | Top-`k` rank correlation (Spearman) and Jaccard between amortised student and perturbation teacher |
| **Explanation stability** | Top-`k` Jaccard between explanations of the *same* alert at consecutive event ticks |
| **Counterfactual validity** | Share of counterfactuals that genuinely push the score below threshold when re-scored |
| **Counterfactual sparsity** | Median number of edges removed |

**The fidelity gap is published.** The perturbation teacher is the ceiling; the student will be
worse. The size of that gap is a headline result of the write-up, reported whether it is flattering
or not. An explainer that is fast and unfaithful is worse than no explainer, and the only way to
know which one we built is to measure it.

**Stability threshold committed in advance:** an explanation whose top-5 Jaccard across one event
tick falls below **0.6** is reported as unstable, and the share of unstable explanations is a
reported metric.

---

## 9. Latency

- Per-stage p50 / p95 / p99: feature fetch, graph assembly, inference, explanation, counterfactual.
- Measured under load with k6, not with a single-request timer in a notebook.
- **SLA: p99 ≤ 200 ms end-to-end.** The load test fails the build if it is exceeded.
- Any accuracy sacrificed to meet the SLA is reported explicitly as an
  accuracy-vs-latency table, per design principle P4.

---

## 10. Uncertainty

**Every reported point estimate carries a 95 % confidence interval.** Bootstrap over transactions,
2000 resamples, stratified by day to respect temporal structure. A point estimate without an
interval is an opinion, and two models whose intervals overlap are not distinguishable — the README
says so rather than reporting the higher number as a win.

---

## 11. Reproducibility

Every number regenerates with:

```
make sim && make eval
```

Each result artifact records: dataset fingerprint, feature-set hash, model version, code commit,
protocol version (this file's content hash), and the random seed. A number whose provenance stamp
does not match the current tree is regenerated, never quoted.

---

## Changelog

| Date | Change | Reason |
|---|---|---|
| 2026-07-04 | Initial protocol, frozen before any model was trained | Engineering invariant: no metric chosen after seeing the results |
| 2026-07-22 | **Corrected the purge rule** from "purge width = the maximum feature window (30 d)" to a 1-day band on both sides, with a guard | The original rule was justified by "any training transaction whose feature window overlaps the validation start", which is not possible — a training event's window points *backwards*. Applied literally on a 30-day dataset it purged 27,405 of 34,769 rows (79 %) and left only the test fold. The mechanism that actually prevents leakage is the point-in-time feature layer plus the `decided_ts` label filter, both of which are unchanged. Found before any model was trained. |
| 2026-07-22 | **Corrected `C_TP`** from `€35 − recovered amount` to `€35 + (1 − p) × amount`, and named the prevention fraction *p* = 0.85 | The original formula priced a true positive *relative to a do-nothing baseline* while pricing a false negative *absolutely*, which made catching fraud profitable rather than merely cheaper. The threshold optimiser exploited it exactly as written: on a synthetic check it preferred catching 2.7 % of fraud over catching 97 %, because alerting less scored a lower "cost". Found before any model was trained, so no reported result was ever computed under the old formula. |
