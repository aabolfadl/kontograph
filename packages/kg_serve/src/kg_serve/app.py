"""The decision gateway.

A single async FastAPI process holding the model, the online feature state and
the explainer — no network hop to an inference server (ADR-0008), because a
round trip is a large fraction of a 200 ms budget.

**Scoring is synchronous inside an async handler, deliberately.** The work is
CPU-bound and takes single-digit milliseconds; handing it to a thread pool would
add scheduling latency to save nothing. What *is* async is everything around it —
the decision log write and the alert publish — so a slow disk cannot block the
next decision.

Endpoints:

    POST /score      score one payment, the hot path
    GET  /health     liveness plus whether state is warm
    GET  /metrics    Prometheus, with the SLA histogram
    GET  /latency    the per-stage budget table, measured
    GET  /decisions/verify   audit the decision log's hash chain
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from kg_core.config import get_settings
from kg_core.logging import get_logger
from kg_serve.decision_log import DecisionLog, build_record
from kg_serve.latency import LatencyBudget

log = get_logger(__name__)

#: Populated at startup. A module-level singleton rather than dependency
#: injection because there is exactly one model per process and the hot path
#: should not pay for a lookup.
STATE: dict[str, Any] = {}

#: In-flight decision-log writes. Holding a strong reference is what keeps the
#: event loop from collecting a pending write; see the note in `score`.
PENDING_WRITES: set[asyncio.Task[None]] = set()


class PaymentRequest(BaseModel):
    """One SCT Inst payment to score.

    Mirrors the wire contract's scoring-relevant fields. Extra keys are allowed
    so a producer that adds a field does not break the gateway — the
    backward-compatibility direction that matters here.
    """

    tx_id: str
    acceptance_ts: int
    debtor_account_id: str
    creditor_account_id: str
    amount_cents: int
    currency: str = "EUR"
    vop_result: str = "MATCH"
    vop_name_similarity: float = 1.0
    vop_overridden: bool = False
    is_new_beneficiary: bool = False
    is_cross_border: bool = False
    channel: str = "ONLINE_BANKING"
    settlement_status: str = "ACSC"
    model_config = {"extra": "allow"}


class ScoreResponse(BaseModel):
    alert_id: str
    tx_id: str
    score: float
    threshold: float
    decision: str
    latency_ms: dict[str, float]
    explanation: dict[str, Any] | None = None
    within_sla: bool = Field(description="Whether this decision met the p99 budget")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and warm the feature state before accepting traffic.

    Warming matters more than it looks. A cold gateway is not merely less
    accurate — it is *silently* less accurate: every feature reads its default,
    the model scores against a world in which nobody has ever transacted, and
    nothing in the response says so.
    """
    settings = get_settings()
    STATE["settings"] = settings
    STATE["budget"] = LatencyBudget(sla_p99_ms=float(settings.sla_p99_millis))
    STATE["log"] = DecisionLog(settings.artifacts_dir / "decisions" / "decisions.jsonl")
    STATE["started"] = time.time()

    try:
        _load_engine(settings)
    except Exception as error:
        # Start anyway and report unhealthy, rather than crash-looping. An
        # operator needs /health to tell them WHY it is down.
        log.error("failed to load the model", error=str(error))
        STATE["load_error"] = str(error)

    yield

    # Drain the decision-log writes before the loop closes. A record that was
    # promised to an auditor and lost to a shutdown race is worse than a slow
    # shutdown.
    if PENDING_WRITES:
        log.info("draining decision-log writes", pending=len(PENDING_WRITES))
        await asyncio.gather(*PENDING_WRITES, return_exceptions=True)
    log.info("gateway stopping", decisions=STATE["log"].count)


def _load_engine(settings: Any) -> None:
    from kg_cli.main import git_commit
    from kg_features.registry import get_feature_set
    from kg_models.baseline import TrainedModel
    from kg_serve.scoring import FeatureAttributionExplainer, ScoringEngine
    from kg_sim.dataset import find_latest

    dataset = find_latest(settings.data_dir)
    if dataset is None:
        raise RuntimeError("no dataset found; run `kg sim generate`")

    directory = settings.artifacts_dir / "models" / dataset.name / "lightgbm"
    if not directory.exists():
        raise RuntimeError(f"no trained model at {directory}; run `kg train baseline`")

    model = TrainedModel.load(directory)
    feature_set = get_feature_set()
    engine = ScoringEngine(
        feature_set,
        model,
        threshold=model.threshold,
        explainer=FeatureAttributionExplainer(model, feature_set.column_names),
        model_name=model.name,
        model_version=model.provenance.get("param_hash", "unknown")[:12],
        code_commit=git_commit(),
    )

    STATE["engine"] = engine
    STATE["dataset"] = dataset
    STATE["warm"] = False
    log.info("model loaded", model=model.name, threshold=model.threshold)


app = FastAPI(
    title="KONTOGRAPH decision gateway",
    description=(
        "Scores SEPA Instant payments under a p99 <= 200 ms SLA, and records every "
        "decision in a hash-chained log for GDPR Art. 22."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/score", response_model=ScoreResponse)
async def score(payment: PaymentRequest) -> ScoreResponse:
    """Score one payment. The hot path."""
    engine = STATE.get("engine")
    if engine is None:
        raise HTTPException(503, STATE.get("load_error", "the model is not loaded"))

    event = payment.model_dump()
    decision = engine.score(event)

    budget: LatencyBudget = STATE["budget"]
    budget.record(decision.latency)
    _observe_prometheus(decision)

    # The decision log write is off the hot path: an auditable record must not
    # be able to add disk latency to a payment decision. It is NOT fire and
    # forget. A task with no live reference can be garbage-collected before it
    # runs, which in this path means a decision that was made but never logged —
    # and a hash chain that verifies clean while missing an entry. So the task
    # is held until it completes, and drained on shutdown.
    task = asyncio.create_task(_record(decision, engine, event))
    PENDING_WRITES.add(task)
    task.add_done_callback(PENDING_WRITES.discard)

    return ScoreResponse(
        alert_id=decision.alert_id,
        tx_id=decision.tx_id,
        score=decision.score,
        threshold=decision.threshold,
        decision=decision.decision.value,
        latency_ms={k: round(v, 3) for k, v in decision.latency.items()},
        explanation=decision.explanation,
        within_sla=decision.latency["total"] <= budget.sla_p99_ms,
    )


async def _record(decision: Any, engine: Any, event: dict[str, Any]) -> None:
    record = build_record(
        alert_id=decision.alert_id,
        tx_id=decision.tx_id,
        scored_ts=int(time.time() * 1000),
        acceptance_ts=int(event.get("acceptance_ts", 0)),
        debtor_account_id=str(event.get("debtor_account_id", "")),
        creditor_account_id=str(event.get("creditor_account_id", "")),
        amount_cents=int(event.get("amount_cents", 0)),
        vop_result=str(event.get("vop_result", "")),
        vop_overridden=bool(event.get("vop_overridden", False)),
        is_new_beneficiary=bool(event.get("is_new_beneficiary", False)),
        score=decision.score,
        threshold=decision.threshold,
        decision=decision.decision.value,
        model_name=engine.model_name,
        model_version=engine.model_version,
        feature_set_hash=engine.feature_set_hash,
        code_commit=engine.code_commit,
        latency=decision.latency,
        explanation=decision.explanation,
    )
    STATE["log"].append(record)


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness, readiness, and — importantly — whether state is warm."""
    engine = STATE.get("engine")
    budget: LatencyBudget | None = STATE.get("budget")
    healthy = engine is not None

    payload = {
        "status": "ok" if healthy else "degraded",
        "model_loaded": healthy,
        "error": STATE.get("load_error"),
        "warm": STATE.get("warm", False),
        "tracked_entities": engine.executor.tracked_entities if engine else 0,
        "decisions": STATE["log"].count if "log" in STATE else 0,
        "uptime_seconds": round(time.time() - STATE.get("started", time.time()), 1),
    }
    if budget and budget.count:
        payload["p99_ms"] = round(budget.quantiles("total")["p99"], 2)
        payload["meets_sla"] = budget.meets_sla
    return JSONResponse(payload, status_code=200 if healthy else 503)


@app.get("/latency")
async def latency() -> dict[str, Any]:
    """The per-stage budget table, measured from real decisions."""
    budget: LatencyBudget = STATE["budget"]
    if budget.count == 0:
        return {"decisions": 0, "note": "no decisions scored yet"}
    return budget.as_dict()


@app.get("/decisions/verify")
async def verify_decisions() -> dict[str, Any]:
    """Audit the decision log's hash chain.

    Returns where the chain broke, not merely whether — an auditor needs to
    know which prefix is still trustworthy.
    """
    return STATE["log"].verify()


@app.post("/warm")
async def warm(limit: int = 50_000) -> dict[str, Any]:
    """Fold recent history into the feature state.

    Exposed as an endpoint so a restarted pod can be warmed before it takes
    traffic, rather than serving defaults and looking healthy.
    """
    engine = STATE.get("engine")
    if engine is None:
        raise HTTPException(503, "the model is not loaded")

    import pyarrow.parquet as pq

    dataset = STATE["dataset"]
    table = pq.ParquetFile(dataset / "events.parquet")
    events: list[dict[str, Any]] = []
    for batch in table.iter_batches(batch_size=10_000):
        events.extend(batch.to_pylist())
        if len(events) >= limit:
            break

    entities = engine.warm(events[:limit])
    STATE["warm"] = True
    return {"warmed_events": min(limit, len(events)), "tracked_entities": entities}


# ------------------------------------------------------------- prometheus ---

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    DECISIONS = Counter("kontograph_decisions_total", "Decisions made", ["decision"])
    # Buckets chosen around the 200 ms SLA rather than the library default, so
    # the histogram has resolution exactly where the question is.
    LATENCY = Histogram(
        "kontograph_decision_latency_ms",
        "End-to-end decision latency",
        buckets=(1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500, 1000),
    )
    STAGE_LATENCY = Histogram(
        "kontograph_stage_latency_ms",
        "Per-stage decision latency",
        ["stage"],
        buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200),
    )
    _PROMETHEUS = True
except ImportError:  # pragma: no cover
    _PROMETHEUS = False


def _observe_prometheus(decision: Any) -> None:
    if not _PROMETHEUS:
        return
    DECISIONS.labels(decision=decision.decision.value).inc()
    LATENCY.observe(decision.latency["total"])
    for stage, value in decision.latency.items():
        if stage not in ("total", "unaccounted"):
            STAGE_LATENCY.labels(stage=stage).observe(value)


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus scrape endpoint."""
    if not _PROMETHEUS:
        return PlainTextResponse("prometheus_client is not installed\n", status_code=501)
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)
