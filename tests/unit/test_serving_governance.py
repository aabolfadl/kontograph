"""M6-M8: the decision log, the latency budget, and the SAR narrative.

The tests that matter here are not "does it return a string". They are:

- the hash chain **detects** tampering and localises it
- the latency budget fails loudly rather than quietly when the SLA is missed
- the SAR draft never asserts evidence it does not have

That last one is a real bug this suite was written for, not a hypothetical.
"""

from __future__ import annotations

import json

import pytest

from kg_core.fmt import format_money, format_score
from kg_gov.sar import HEADER_DE, HEADER_EN, draft, render_markdown
from kg_serve.decision_log import DecisionLog, build_record
from kg_serve.latency import STAGES, LatencyBudget, StageTimer, UnknownStageError


def _record(index: int, **overrides: object) -> dict[str, object]:
    base = {
        "alert_id": f"alert-{index}",
        "tx_id": f"tx-{index}",
        "scored_ts": 1_760_000_000_000 + index,
        "score": 0.5,
        "threshold": 0.4,
        "decision": "ALERT",
        "model_name": "lightgbm",
        "model_version": "abc123",
        "feature_set_hash": "fingerprint",
        "code_commit": "deadbeef",
        "latency": {"total": 4.2},
        "explanation": {"top_features": []},
    }
    base.update(overrides)
    return build_record(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------ decision log ---


def test_chain_verifies_and_localises_tampering(tmp_path):
    """An auditor needs to know WHICH prefix is still trustworthy."""
    path = tmp_path / "decisions.jsonl"
    log = DecisionLog(path)
    for i in range(5):
        log.append(_record(i))

    assert log.verify()["intact"] is True

    # Edit entry 2 the way someone covering a bad decision would: change the
    # outcome inside the payload, leave the envelope alone.
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[2])
    tampered["payload"]["decision"] = "ALLOW"
    lines[2] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = DecisionLog(path).verify()
    assert result["intact"] is False
    assert result["broken_at"] == 2
    # Entries 0 and 1 were written before the edit and still chain correctly.
    assert result["trustworthy_prefix"] == 2


def test_appending_after_reopen_continues_the_same_chain(tmp_path):
    """A restarted gateway must not silently start a second chain."""
    path = tmp_path / "decisions.jsonl"
    first = DecisionLog(path)
    for i in range(3):
        first.append(_record(i))

    second = DecisionLog(path)
    second.append(_record(99))

    assert DecisionLog(path).verify()["intact"] is True
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 4


def test_renumbered_sequence_is_rejected(tmp_path):
    """The chain commits to the payload, so the envelope needs its own check."""
    path = tmp_path / "decisions.jsonl"
    log = DecisionLog(path)
    for i in range(4):
        log.append(_record(i))

    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[2])
    entry["sequence"] = 99
    lines[2] = json.dumps(entry)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")

    result = DecisionLog(path).verify()
    assert result["intact"] is False
    assert result["broken_at"] == 2


def test_stray_envelope_field_is_rejected(tmp_path):
    """An unknown top-level field must fail verification, not be ignored."""
    path = tmp_path / "decisions.jsonl"
    log = DecisionLog(path)
    for i in range(3):
        log.append(_record(i))

    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["reviewed_by"] = "somebody"
    lines[1] = json.dumps(entry)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")

    assert DecisionLog(path).verify()["broken_at"] == 1


def test_record_carries_the_art22_fields():
    """Model version, feature fingerprint and commit are not optional."""
    record = _record(0)
    for name in ("model_name", "model_version", "feature_set_hash", "code_commit"):
        assert record[name], f"{name} must be populated at decision time"


def test_record_identifies_the_payment_it_decided():
    """A record that cannot say whose payment it was is an unusable log.

    Not a minimisation question: these are the payment the decision was *about*,
    not extra data gathered on the subject. Without them neither an Art. 22
    explanation nor a SAR draft can name anyone - the draft rendered
    "account - sent EUR 0.00" before they were logged.
    """
    record = _record(
        0,
        debtor_account_id="ACC-1",
        creditor_account_id="ACC-2",
        amount_cents=947_500,
        acceptance_ts=1_760_271_234_330,
        vop_result="NO_MATCH",
        vop_overridden=True,
    )
    assert record["debtor_account_id"] == "ACC-1"
    assert record["creditor_account_id"] == "ACC-2"
    assert record["amount_cents"] == 947_500
    # Acceptance and scoring times are distinct: they differ by the decision
    # latency, and a SAR narrative dates the payment, not the model run.
    assert record["acceptance_ts"] != record["scored_ts"]
    assert record["vop_result"] == "NO_MATCH"
    assert record["vop_overridden"] is True


def test_feature_values_stay_out_of_the_record_by_default():
    """Derived behavioural aggregates are a minimisation decision, and default off."""
    assert "features" not in _record(0)
    with_features = _record(0, features={"debtor_txn_24h": 3.0})
    assert with_features["features"] == {"debtor_txn_24h": 3.0}


# --------------------------------------------------------- latency budget ---


def test_budget_fails_when_p99_exceeds_the_sla():
    """The budget's job is to fail. A budget that always passes is decoration."""
    budget = LatencyBudget(sla_p99_ms=200.0)
    for _ in range(95):
        budget.record({"total": 5.0, "inference": 2.0})
    for _ in range(5):
        budget.record({"total": 900.0, "inference": 800.0})

    assert budget.meets_sla is False
    # Nearest-rank: the reported p99 is a latency a request actually saw.
    assert budget.quantiles("total")["p99"] == 900.0


def test_p99_is_nearest_rank_not_interpolated():
    """Interpolation would report a latency nobody experienced, optimistically.

    With 99 fast samples and one 900 ms sample, numpy's default linear
    interpolation puts p99 at ~14 ms - a figure no request saw, on the flattering
    side. Rounding up reports 900 ms: a real observation, and one that fails the
    SLA rather than passing it on an interpolation artefact.
    """
    budget = LatencyBudget(sla_p99_ms=200.0)
    for _ in range(99):
        budget.record({"total": 5.0})
    budget.record({"total": 900.0})
    assert budget.quantiles("total")["p99"] == 900.0
    assert budget.meets_sla is False


def test_budget_meets_sla_on_fast_decisions():
    budget = LatencyBudget(sla_p99_ms=200.0)
    for _ in range(100):
        budget.record({"total": 5.0, "inference": 2.0})
    assert budget.meets_sla is True
    assert "p99" in budget.quantiles("total")


def test_stage_timer_total_covers_the_stages():
    timer = StageTimer()
    with timer.stage("feature_fetch"):
        pass
    with timer.stage("inference"):
        pass
    timings = timer.as_dict()
    # as_dict reports every declared stage, so an untimed stage shows as 0.0
    # rather than going missing - a budget table with a hole in it is worse than
    # one with an explicit zero.
    assert set(timings) == {*STAGES, "total", "unaccounted"}
    assert timings["explain"] == 0.0
    # Total is measured end to end, so it cannot be less than the parts.
    assert timings["total"] >= timings["feature_fetch"] + timings["inference"] - 1e-9


def test_timing_an_undeclared_stage_raises():
    """A typo must fail loudly, not quietly drop a stage from the budget."""
    timer = StageTimer()
    with pytest.raises(UnknownStageError), timer.stage("featurefetch"):
        pass


# ------------------------------------------------------------- formatting ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.000398, "3.98e-04"), (0.5, "0.500"), (0.01, "0.010"), (0.0, "0.00e+00")],
)
def test_small_scores_keep_their_information(value, expected):
    """A cost-optimal threshold near 4e-4 must not render as 0.00."""
    assert format_score(value) == expected


def test_money_is_formatted_from_minor_units():
    assert format_money(6942) == "EUR 69.42"


# ------------------------------------------------------- the SAR narrative ---


def _draft(counterfactual):
    return draft(
        alert_id="alert-1",
        tx_id="tx-1",
        debtor="ACC-1",
        creditor="ACC-2",
        amount_cents=6942,
        acceptance_ts=1_760_271_234_330,
        score=0.000398,
        threshold=0.000398,
        counterfactual=counterfactual,
    )


def test_feature_space_counterfactual_is_not_dressed_up_as_evidence():
    """Regression: the narrative once read "0 related payment(s): ".

    A feature-space counterfactual carries no transaction ids. Rendering it
    through the edge-space branch produced a sentence claiming payment evidence
    that did not exist - in a document destined for a Verdachtsmeldung.
    """
    result = _draft(
        {
            "found": True,
            "kind": "feature_space",
            "feature": "creditor_txn_24h",
            "score_after": 0.000169,
            "crosses_threshold": True,
        }
    )
    assert "0 related payment" not in result.english
    assert "0 zusammenhängenden Zahlung" not in result.german
    assert "No individual payment explains this alert" in result.english
    assert "Keine einzelne Zahlung" in result.german
    assert "creditor_txn_24h" in result.english
    # It names no transaction, so it must contribute no evidence rows.
    assert result.evidence == []


def test_edge_space_counterfactual_names_checkable_payments():
    result = _draft(
        {
            "found": True,
            "kind": "edge",
            "removed_tx_ids": ["tx-7", "tx-8"],
            "score_after": 0.0001,
            "crosses_threshold": True,
        }
    )
    assert "tx-7" in result.english and "tx-8" in result.english
    assert {e["tx_id"] for e in result.evidence} == {"tx-7", "tx-8"}
    assert all(e["role"] == "counterfactual" for e in result.evidence)


def test_no_counterfactual_says_so_rather_than_inventing_one():
    result = _draft({"found": False, "kind": "feature_space", "reason": "none"})
    assert "No small set of related payments" in result.english
    assert result.evidence == []


def test_every_draft_is_marked_as_not_a_filing():
    """The legal distinction is enforced by the output, not by convention."""
    result = _draft(None)
    payload = result.as_dict()
    assert payload["disclaimer_en"] == HEADER_EN
    assert payload["disclaimer_de"] == HEADER_DE

    markdown = render_markdown(result)
    assert "NOT A FILING" in markdown
    assert "KEINE MELDUNG" in markdown
    # A machine may state what the model concluded, not what the institution
    # suspects. "Es besteht der Verdacht" is the institution speaking.
    assert "Es besteht der Verdacht" not in result.german


def test_vop_mismatch_is_reported_without_being_called_fraud():
    result = draft(
        alert_id="a",
        tx_id="t",
        debtor="ACC-1",
        creditor="ACC-2",
        amount_cents=100_000,
        acceptance_ts=1_760_271_234_330,
        score=0.9,
        threshold=0.5,
        vop_result="NO_MATCH",
        vop_overridden=True,
    )
    assert "NO_MATCH" in result.english
    assert "not by itself evidence of fraud" in result.english
    assert "kein Nachweis eines Betrugs" in result.german
