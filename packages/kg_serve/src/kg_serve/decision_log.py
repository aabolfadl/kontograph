"""The GDPR Art. 22 decision log.

An automated decision with legal effect requires *meaningful information about
the logic involved*, and that information has to be persisted **at decision
time**. Reconstructing it later from a model file and a guess about which
version was live is not the same thing, and an auditor will say so.

So every decision writes a record carrying:

- what was decided, and against which threshold
- which model version and which feature-set fingerprint produced it
- the code commit the serving image was built from
- the explanation that was shown to the investigator
- the per-stage latency

**The log is hash-chained.** Each entry commits to the previous one, so altering
entry *n* invalidates every entry after it. That is what turns "we logged the
decision" from an assertion into something an auditor can verify — and it
localises tampering, which bounds the blast radius of an incident.

It is append-only JSONL on disk. In production this would be an Iceberg table or
a WORM bucket; the chain is what makes the guarantee, not the storage.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kg_core.hashing import GENESIS, chain_link, verify_chain
from kg_core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class DecisionLog:
    """Append-only, hash-chained decision records."""

    path: Path
    _previous: str = GENESIS
    _count: int = 0
    _lock: Any = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._resume()

    def _resume(self) -> None:
        """Pick up the chain where it left off.

        Re-reading rather than starting a fresh chain: a restart that began a new
        chain would leave an unverifiable seam, which is indistinguishable from
        somebody having deleted the entries either side of it.
        """
        last = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = json.loads(line)
                    self._count += 1
        if last:
            self._previous = last["entry_hash"]
        log.info("decision log resumed", entries=self._count, path=str(self.path))

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write one decision. Returns the stored entry including its hash."""
        with self._lock:
            entry_hash = chain_link(self._previous, payload)
            entry = {
                "sequence": self._count,
                "prev_hash": self._previous,
                "entry_hash": entry_hash,
                "payload": payload,
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
            self._previous = entry_hash
            self._count += 1
            return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def verify(self) -> dict[str, Any]:
        """Verify the chain end to end.

        Reports *where* it broke, not merely that it did: an auditor needs to
        know which entries are still trustworthy.
        """
        entries = self.entries()
        broken_at = _first_malformed(entries)
        if broken_at is None:
            broken_at = verify_chain(entries)
        return {
            "entries": len(entries),
            "intact": broken_at is None,
            "broken_at": broken_at,
            "trustworthy_prefix": len(entries) if broken_at is None else broken_at,
            "path": str(self.path),
        }

    @property
    def count(self) -> int:
        return self._count


#: Exactly the keys an entry may carry. Checked, not assumed.
ENVELOPE_KEYS = frozenset({"sequence", "prev_hash", "entry_hash", "payload"})


def _first_malformed(entries: list[dict[str, Any]]) -> int | None:
    """Index of the first structurally invalid entry, or None.

    The hash chain commits to ``payload`` and to the previous link — so it does
    NOT cover the rest of the envelope. Without this check an entry could carry
    a renumbered ``sequence``, or an extra top-level field that a reader might
    reasonably trust, and the chain would still verify clean. An auditor reading
    "intact: true" would be reading a narrower claim than they think.

    Rejecting unknown keys outright, rather than ignoring them, is the
    conservative direction: a log that grows a field should fail verification
    and be dealt with, not pass quietly.
    """
    for index, entry in enumerate(entries):
        if set(entry) != ENVELOPE_KEYS:
            return index
        if entry["sequence"] != index:
            return index
        if not isinstance(entry["payload"], dict):
            return index
    return None


def build_record(
    *,
    alert_id: str,
    tx_id: str,
    scored_ts: int,
    score: float,
    threshold: float,
    decision: str,
    model_name: str,
    model_version: str,
    feature_set_hash: str,
    code_commit: str,
    latency: dict[str, float],
    debtor_account_id: str = "",
    creditor_account_id: str = "",
    amount_cents: int = 0,
    acceptance_ts: int = 0,
    vop_result: str = "",
    vop_overridden: bool = False,
    is_new_beneficiary: bool = False,
    explanation: dict[str, Any] | None = None,
    features: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble one decision record.

    Two data-protection judgements, in opposite directions, and the difference
    between them is the point:

    ``features`` is optional and off by default. Those values are *derived*
    behavioural facts about a named account — how often they transact, with whom
    — and logging them is a data-protection decision rather than a debugging
    convenience (Art. 5(1)(c) data minimisation). The feature set's
    *fingerprint* is always logged, and that is what actually supports the
    Art. 22 explanation of the logic involved.

    The payment's own identifying facts — the two accounts, the amount, the
    acceptance time — are always logged. They are not extra data collected about
    the subject; they are the payment the decision was *about*. A record that
    cannot say whose payment was declined cannot support a subject's Art. 22
    right to an explanation, and cannot produce a SAR draft that names anyone.
    Omitting them is not minimisation, it is an unusable log.

    The Verification of Payee outcome travels with them for the same reason: a
    payer who proceeded past a payee-name mismatch is a material fact in a
    Verdachtsmeldung, and it cannot be reconstructed from the score.
    """
    record: dict[str, Any] = {
        "alert_id": alert_id,
        "tx_id": tx_id,
        "scored_ts": scored_ts,
        "acceptance_ts": acceptance_ts,
        "debtor_account_id": debtor_account_id,
        "creditor_account_id": creditor_account_id,
        "amount_cents": int(amount_cents),
        "vop_result": vop_result,
        "vop_overridden": bool(vop_overridden),
        "is_new_beneficiary": bool(is_new_beneficiary),
        "score": round(float(score), 6),
        "threshold": round(float(threshold), 6),
        "decision": decision,
        "model_name": model_name,
        "model_version": model_version,
        "feature_set_hash": feature_set_hash,
        "code_commit": code_commit,
        "latency_ms": {k: round(float(v), 4) for k, v in latency.items()},
    }
    if explanation is not None:
        record["explanation"] = explanation
    if features is not None:
        record["features"] = {k: round(float(v), 6) for k, v in features.items()}
    return record
