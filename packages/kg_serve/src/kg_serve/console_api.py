"""The investigator console and the endpoints it needs.

Design principle P10: *"A live demo, no signup ... under ten seconds to
comprehension."*

The console is a single self-contained HTML file — no CDN, no build step, no
`npm install`. A demo that needs the network is a demo nobody will run,
and `make up` has to work on a laptop with nothing installed.

Kept in its own module rather than in :mod:`kg_serve.app` because the demo
surface and the decision path have different reasons to change, and the hot path
should not grow a dependency on the SAR narrative generator.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from kg_core.logging import get_logger

log = get_logger(__name__)

router = APIRouter()
CONSOLE = Path(__file__).parent / "console" / "index.html"

#: Set by kg_serve.app at startup. A module-level handle rather than an import,
#: to keep the dependency pointing one way.
STATE: dict[str, Any] = {}


def bind(state: dict[str, Any]) -> None:
    """Give the console access to the gateway's loaded engine."""
    STATE.clear()
    STATE.update(state)


@router.get("/", response_class=HTMLResponse)
async def console() -> HTMLResponse:
    """The investigator console."""
    if not CONSOLE.exists():
        return HTMLResponse("<h1>console not found</h1>", status_code=404)
    return HTMLResponse(CONSOLE.read_text(encoding="utf-8"))


class FeedbackRequest(BaseModel):
    alert_id: str
    tx_id: str
    verdict: str
    explanation_helpful: bool | None = None
    review_seconds: float | None = None
    rationale: str | None = None


@router.post("/api/feedback")
async def feedback(request: FeedbackRequest) -> dict[str, Any]:
    """Capture an analyst's verdict and publish it back onto Kafka.

    Closes the loop drawn in the architecture diagram. Kafka is best-effort and
    deliberately so: an investigator's decision must be recorded even when the
    broker is down, so the local record is written first and the publish is
    allowed to fail.
    """
    settings = STATE["settings"]
    record = {
        "schema_version": 1,
        "feedback_id": str(uuid.uuid4()),
        "alert_id": request.alert_id,
        "analyst_id": "console-demo",
        "submitted_ts": int(time.time() * 1000),
        "verdict": request.verdict,
        "explanation_helpful": request.explanation_helpful,
        "explanation_rating": None,
        "review_seconds": request.review_seconds,
        "rationale": request.rationale,
    }

    path = settings.artifacts_dir / "decisions" / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    published = False
    try:
        from kg_stream.producer import AvroProducer

        with AvroProducer(
            "analyst_feedback.v1",
            settings.topic_feedback,
            key_field="alert_id",
            timestamp_field="submitted_ts",
        ) as producer:
            producer.produce(record)
        published = True
    except Exception as error:
        log.warning("feedback not published to kafka", error=str(error))

    return {
        "recorded": True,
        "published_to_kafka": published,
        "feedback_id": record["feedback_id"],
    }


@router.get("/api/replay")
async def replay(limit: int = 400, warm: int = 30_000) -> dict[str, Any]:
    """Score a slice of the real stream and return the alerts, with subgraphs.

    Bounded and synchronous rather than a live SSE feed: the demo has to reach
    comprehension in seconds, and a stream a user must wait for defeats
    that. The latency figures returned are measured on this very run, not quoted
    from a previous one.
    """
    import pyarrow.parquet as pq

    from kg_serve.latency import LatencyBudget

    engine = STATE.get("engine")
    if engine is None:
        raise HTTPException(503, STATE.get("load_error", "the model is not loaded"))

    dataset = STATE["dataset"]
    parquet = pq.ParquetFile(dataset / "events.parquet")
    stream: list[dict[str, Any]] = []
    for batch in parquet.iter_batches(batch_size=10_000):
        stream.extend(batch.to_pylist())
        if len(stream) >= warm + limit:
            break

    if not STATE.get("warm"):
        engine.warm(stream[:warm])
        STATE["warm"] = True

    # An account -> recent-events index over EVERYTHING loaded, not a rolling
    # window of the last few hundred. With ~6,500 accounts a 400-event window
    # almost never contains a repeat, so the console drew a two-node graph and
    # the subgraph panel said nothing. The index is bounded per account rather
    # than globally, which is what keeps it cheap.
    index = _build_account_index(stream[: warm + limit], per_account=25)

    budget = LatencyBudget(sla_p99_ms=float(STATE["settings"].sla_p99_millis))
    alerts: list[dict[str, Any]] = []
    scored = 0

    for event in stream[warm : warm + limit]:
        decision = engine.score(event)
        budget.record(decision.latency)
        scored += 1
        if decision.alerted:
            alerts.append(_alert_payload(decision, event, index))

    alerts.sort(key=lambda a: -a["score"])
    return {
        "scored": scored,
        "alerts": alerts[:50],
        "p99_ms": budget.quantiles("total")["p99"],
        "sla_ms": budget.sla_p99_ms,
        "meets_sla": budget.meets_sla,
    }


def _build_account_index(
    events: list[dict[str, Any]], *, per_account: int = 25
) -> dict[str, list[dict[str, Any]]]:
    """Map each account to its most recent events, bounded per account.

    Bounded per account rather than globally: a utility provider receives from
    thousands of households, and an unbounded index would hold the whole stream
    under one key while a normal household held three.
    """
    index: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        for account in (event["debtor_account_id"], event["creditor_account_id"]):
            bucket = index.setdefault(account, [])
            bucket.append(event)
            if len(bucket) > per_account:
                bucket.pop(0)
    return index


def _alert_payload(
    decision: Any, event: dict[str, Any], index: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Assemble one alert for the console: score, subgraph, narrative."""
    from kg_gov.sar import draft

    explanation = decision.explanation or {}
    counterfactual = explanation.get("counterfactual")
    debtor, creditor = event["debtor_account_id"], event["creditor_account_id"]

    # The neighbourhood the console draws. STRICTLY earlier: an explanation
    # citing a later payment would describe information the model did not have,
    # which is the same leak the feature layer forbids in a different costume.
    seen: set[str] = set()
    neighbourhood: list[dict[str, Any]] = []
    for account in (debtor, creditor):
        for other in reversed(index.get(account, [])):
            if other["tx_id"] == event["tx_id"] or other["tx_id"] in seen:
                continue
            if int(other["acceptance_ts"]) >= int(event["acceptance_ts"]):
                continue
            seen.add(other["tx_id"])
            neighbourhood.append(
                {
                    "tx_id": other["tx_id"],
                    "debtor_account_id": other["debtor_account_id"],
                    "creditor_account_id": other["creditor_account_id"],
                    "amount_cents": other["amount_cents"],
                    "salient": False,
                }
            )
            if len(neighbourhood) >= 12:
                break
        if len(neighbourhood) >= 12:
            break

    narrative = draft(
        alert_id=decision.alert_id,
        tx_id=decision.tx_id,
        debtor=debtor,
        creditor=creditor,
        amount_cents=event["amount_cents"],
        acceptance_ts=int(event["acceptance_ts"]),
        score=decision.score,
        threshold=decision.threshold,
        counterfactual=counterfactual,
        vop_result=event.get("vop_result", "MATCH"),
        vop_overridden=bool(event.get("vop_overridden", False)),
        is_new_beneficiary=bool(event.get("is_new_beneficiary", False)),
    )

    return {
        "alert_id": decision.alert_id,
        "tx_id": decision.tx_id,
        "score": decision.score,
        "threshold": decision.threshold,
        "decision": decision.decision.value,
        "amount_cents": event["amount_cents"],
        "acceptance_ts": int(event["acceptance_ts"]),
        "debtor_account_id": debtor,
        "creditor_account_id": creditor,
        "vop_result": event.get("vop_result", "MATCH"),
        "vop_overridden": bool(event.get("vop_overridden", False)),
        "latency_ms": {k: round(v, 3) for k, v in decision.latency.items()},
        "top_features": explanation.get("top_features", []),
        "counterfactual": counterfactual,
        "neighbourhood": neighbourhood,
        "narrative_en": narrative.english,
        "narrative_de": narrative.german,
    }
