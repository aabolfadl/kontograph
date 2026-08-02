# ADR-0008 — ONNX Runtime in-process, not NVIDIA Triton

**Status:** accepted · **Date:** 2026-07-30

## Context

The platform scope names "ONNX export → Triton" under Serving. Triton is the right answer for a fleet of
GPU models with dynamic batching, multi-framework support and model-repository hot-reload.

## Decision

Export to ONNX and run it **in-process inside the FastAPI gateway** via ONNX Runtime. No Triton.

## Rationale

**The image does not fit.** The Triton server image is roughly 15 GB. The target machine has about
32 GB free. That alone ends it, and pretending otherwise would break the engineering invariants — an unrunnable
stack is not an artifact.

**A network hop is a large fraction of a small budget.** The SLA is 200 ms end-to-end including
feature fetch, graph assembly, explanation and counterfactual search. Triton adds serialisation plus
a network round trip per call. In-process inference removes that entirely, which is the right call at
one model on CPU.

**Triton's strengths do not apply here.** Dynamic batching helps throughput under GPU saturation; we
serve single events on CPU under a latency SLA, where batching *adds* latency. Multi-framework
serving matters with a zoo of models; we have one detector and one explainer head.

**We keep the part that carries the value.** The ONNX export itself is retained, so the model stays
framework-portable and the graph gets optimised. Moving to Triton later becomes a deployment change,
not a model change.

## Consequences

**Good.** About 15 GB and one network hop saved; the gateway is a single process to reason about.

**Bad.** No hot model reload — a new version means a rolling restart. Acceptable, and the canary /
shadow deployment design accounts for it.

**Bad.** No GPU serving path. We do not need one: CPU inference on a single event is well inside
budget, which the latency table will show.

**Known risk, stated up front:** exporting a *memory-carrying* TGN to ONNX is awkward. The plan is to
externalise memory to Redis and export the message / embedding / decoder submodules separately. If a
submodule resists export, the fallback is TorchScript, and this ADR gets amended to say so rather
than the README quietly claiming a clean ONNX path.

## Amendment — 2026-08-02: the export works, and it is not the serving path

This is the amendment the paragraph above promised, arriving from an unexpected direction. The
problem was not the TGN. It was **LightGBM**.

`kg serve export-onnx` converts the trained booster and then re-scores the entire test fold through
both paths. On 303,129 real feature vectors:

| measure | value |
|---|---|
| max score difference | 1.665e-04 |
| mean score difference | 7.405e-08 |
| decisions changed | 774 (0.2553%) |
| alerts — booster | 6,458 |
| alerts — ONNX | 7,232 |
| **alert-set Jaccard** | **0.8930** |

The mean difference is 7e-08, which is the number one would quote to claim the export is fine.
**It is the wrong number.** The ONNX operator set scores tree ensembles in float32 while LightGBM
accumulates in float64, and `onnxmltools` rejects a double-precision input for classifier graphs. The
cost model puts the operating threshold at 3.98e-04, and float32 rounding is comfortably enough to
move a score across a threshold that small.

The consequence is not academic: the ONNX graph raises **7,232 alerts where the evaluated model
raises 6,458 — 12% more**. Against a stated review capacity of 200 alerts/day, silently swapping in
the ONNX graph would have inflated alert volume by a tenth while every published metric continued to
describe the booster. That is training-serving skew, arriving through the serving artifact instead of
the feature layer, and this project's whole claim is that it does not tolerate that.

**Revised decision.** The gateway serves the **LightGBM booster in-process**. The ONNX export ships
as a portability artifact alongside it, with `onnx_parity.json` recording the gap. Inference is
0.42 ms p50 / 1.78 ms p99 against a 200 ms budget, so the export was never buying latency — it was
buying portability, and portability is worth exactly nothing if the portable copy makes different
decisions.

Three ways this could be revisited, in order of preference:

1. **Re-derive the operating threshold on ONNX scores** and re-run the full evaluation against the
   graph. Legitimate, and it makes the ONNX graph the evaluated model rather than an approximation
   of one. It costs a full re-evaluation, which is why it is not done casually.
2. **A double-precision converter path**, if `onnxmltools` gains one for classifiers.
3. **A model whose scores are not concentrated against a 4e-04 threshold** — a different operating
   point, or a different objective, would make float32 rounding irrelevant.

The general lesson is worth keeping: *a serving-format conversion is a model change until measured
otherwise.* "The conversion succeeded" is not evidence, and neither is a small mean error.

## What would change our mind

- Multiple models, multiple frameworks, or GPU serving under throughput pressure.
- A production environment that already operates Triton.
