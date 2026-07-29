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

## What would change our mind

- Multiple models, multiple frameworks, or GPU serving under throughput pressure.
- A production environment that already operates Triton.
