"""Suspicious activity report narrative drafts, DE and EN.

Design principle P5.5 asks for *"an auto-drafted narrative fragment for the suspicious
activity report"*, and the regulatory context makes it bilingual: a Verdachtsmeldung is filed with
the German FIU in German, while the investigator's console and the model
documentation are English.

**This drafts a fragment. It does not file anything.** The distinction matters
legally and is enforced by the output itself: every draft carries a header
marking it as machine-generated and unreviewed, and the German text uses
``Der Alarm beruht auf`` — what the *model* asserts — rather than
``Es besteht der Verdacht``, which is a statement the institution makes and a
machine may not make on its behalf.

The narrative is built from the counterfactual, not from feature attributions.
"Remove these two payments and the score falls below threshold" is a claim an
investigator can check against the account; "feature 17 had high attribution" is
not something that can go in a report at all.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

from kg_core.fmt import format_money as _money
from kg_core.fmt import format_score as _score
from kg_core.timeutil import iso

#: Typology names as an investigator would write them, in both languages.
TYPOLOGY_TERMS: dict[str, tuple[str, str]] = {
    "MULE_NETWORK": ("mule network", "Finanzagenten-Netzwerk"),
    "SMURFING": ("structuring", "Smurfing / Strukturierung"),
    "APP_FRAUD": ("authorised push payment fraud", "Betrug durch autorisierte Zahlung"),
    "CIRCULAR_FLOW": ("circular flow", "Kreislaufgeschäft"),
    "LAYERING_CHAIN": ("layering chain", "Verschleierungskette"),
}

HEADER_EN = (
    "MACHINE-GENERATED DRAFT — NOT A FILING. Produced by an automated monitoring "
    "system and not reviewed by a person. It states what the model concluded; it "
    "does not state that the institution suspects money laundering. Any "
    "Verdachtsmeldung must be assessed and filed by a responsible officer."
)
HEADER_DE = (
    "MASCHINELL ERSTELLTER ENTWURF — KEINE MELDUNG. Erstellt durch ein "
    "automatisiertes Überwachungssystem, nicht durch eine Person geprüft. Der Text "
    "gibt wieder, was das Modell festgestellt hat; er stellt keine "
    "Verdachtsäußerung des Instituts dar. Eine Verdachtsmeldung ist durch den "
    "Geldwäschebeauftragten zu bewerten und abzugeben."
)


@dataclass(slots=True)
class SarDraft:
    """A bilingual narrative fragment for one alert."""

    alert_id: str
    tx_id: str
    english: str
    german: str
    evidence: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "tx_id": self.tx_id,
            "narrative_en": self.english,
            "narrative_de": self.german,
            "evidence": self.evidence,
            "disclaimer_en": HEADER_EN,
            "disclaimer_de": HEADER_DE,
        }


def draft(
    *,
    alert_id: str,
    tx_id: str,
    debtor: str,
    creditor: str,
    amount_cents: int,
    acceptance_ts: int,
    score: float,
    threshold: float,
    counterfactual: dict[str, Any] | None = None,
    vop_result: str = "MATCH",
    vop_overridden: bool = False,
    is_new_beneficiary: bool = False,
    supporting: list[dict[str, Any]] | None = None,
) -> SarDraft:
    """Draft the narrative for one alert."""
    when = iso(acceptance_ts)
    supporting = supporting or []

    english = [
        f"On {when} account {debtor} sent {_money(amount_cents)} to account {creditor} "
        f"(reference {tx_id}). The automated monitoring system scored this payment at "
        f"{_score(score)} against an operating threshold of {_score(threshold)}."
    ]
    german = [
        f"Am {when} überwies das Konto {debtor} einen Betrag von {_money(amount_cents)} "
        f"an das Konto {creditor} (Referenz {tx_id}). Das automatisierte "
        f"Überwachungssystem bewertete diese Zahlung mit {_score(score)} bei einem "
        f"Schwellenwert von {_score(threshold)}."
    ]

    # Verification of Payee. Reported factually, and NOT as evidence of fraud on
    # its own: roughly 40% of honest payers override a warning, and married
    # names, umlaut transliterations and trading names all produce genuine
    # mismatches on entirely legitimate payments.
    if vop_result != "MATCH":
        english.append(
            f"Verification of Payee returned {vop_result} for this payment"
            + (", and the payer proceeded past the warning." if vop_overridden else ".")
            + " A mismatch is not by itself evidence of fraud: name transliteration, "
            "married names and trading names produce mismatches on legitimate payments."
        )
        german.append(
            f"Die Empfängerüberprüfung (Verification of Payee) ergab {vop_result}"
            + ("; der Zahler setzte die Zahlung trotz Warnung fort." if vop_overridden else ".")
            + " Eine Abweichung ist für sich genommen kein Nachweis eines Betrugs: "
            "Namensumschreibungen, Ehenamen und Geschäftsbezeichnungen führen auch bei "
            "rechtmäßigen Zahlungen zu Abweichungen."
        )

    if is_new_beneficiary:
        english.append(
            f"The payee {creditor} had not previously received a payment from this account."
        )
        german.append(
            f"Der Zahlungsempfänger {creditor} hatte zuvor keine Zahlung von diesem Konto erhalten."
        )

    evidence: list[dict[str, Any]] = []
    cf: dict[str, Any] = counterfactual or {}
    removed = list(cf.get("removed_tx_ids") or [])
    crosses = bool(cf.get("crosses_threshold"))
    after = _score(cf.get("score_after", 0.0))

    if crosses and removed:
        # The edge-space counterfactual: an investigator can pull these payments
        # up on the account and check the claim themselves.
        english.append(
            f"The alert rests on {len(removed)} related payment(s): "
            f"{', '.join(removed)}. Excluding them, the score falls to "
            f"{after}, below the threshold. This was "
            "verified by re-scoring the model without those payments, not inferred."
        )
        german.append(
            f"Der Alarm beruht auf {len(removed)} zusammenhängenden Zahlung(en): "
            f"{', '.join(removed)}. Ohne diese sinkt der Score auf "
            f"{after} und damit unter den Schwellenwert. "
            "Dies wurde durch erneute Bewertung des Modells überprüft, nicht abgeleitet."
        )
        evidence = [{"tx_id": t, "role": "counterfactual"} for t in removed]
    elif crosses and cf.get("kind") == "feature_space":
        # The WEAKER kind, and it must not be dressed up as the stronger one.
        # Rendering it through the edge-space branch produced the sentence "the
        # alert rests on 0 related payment(s): ." — a claim of payment evidence
        # with no payments behind it, which is exactly the kind of sentence that
        # must never reach a Verdachtsmeldung. Named honestly instead, including
        # the fact that it identifies no specific transaction.
        feature = cf.get("feature", "an indicator")
        english.append(
            f"No individual payment explains this alert. It rests on the indicator "
            f"'{feature}': holding that indicator at its baseline value, and changing "
            f"nothing else, the score falls to {after}, below the threshold. This is a "
            "statement about an aggregate indicator, not about any identified "
            "transaction, and does not on its own identify conduct to report."
        )
        german.append(
            f"Keine einzelne Zahlung erklärt diesen Alarm. Er beruht auf dem Indikator "
            f"„{feature}“: Wird dieser Indikator auf seinen Basiswert gesetzt und sonst "
            f"nichts verändert, sinkt der Score auf {after} und damit unter den "
            "Schwellenwert. Dies ist eine Aussage über eine aggregierte Kennzahl, nicht "
            "über eine bestimmte Transaktion, und benennt für sich genommen keinen "
            "meldepflichtigen Sachverhalt."
        )
    elif counterfactual is not None:
        english.append(
            "No small set of related payments explains this alert on its own; the score "
            "reflects the account's broader pattern of activity."
        )
        german.append(
            "Keine kleine Gruppe zusammenhängender Zahlungen erklärt diesen Alarm für sich "
            "genommen; der Score spiegelt das allgemeine Aktivitätsmuster des Kontos wider."
        )

    for item in supporting[:5]:
        evidence.append({**item, "role": item.get("role", "supporting")})

    english.append(
        "This draft is generated from the model's own decision record and requires review "
        "before any report is filed."
    )
    german.append(
        "Dieser Entwurf wurde aus dem Entscheidungsprotokoll des Modells erstellt und ist "
        "vor einer Meldung zu prüfen."
    )

    return SarDraft(
        alert_id=alert_id,
        tx_id=tx_id,
        english=" ".join(english),
        german=" ".join(german),
        evidence=evidence,
    )


def render_markdown(draft_: SarDraft) -> str:
    """The draft as a document an investigator can paste from."""
    generated = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            "# Verdachtsmeldung — Entwurf / SAR narrative draft",
            "",
            f"> **{HEADER_EN}**",
            "",
            f"> **{HEADER_DE}**",
            "",
            f"_Alert `{draft_.alert_id}` · transaction `{draft_.tx_id}` · generated {generated}_",
            "",
            "## Deutsch",
            "",
            draft_.german,
            "",
            "## English",
            "",
            draft_.english,
            "",
            "## Evidence referenced",
            "",
            "| Transaction | Role |",
            "|---|---|",
            *[f"| `{e['tx_id']}` | {e['role']} |" for e in draft_.evidence],
        ]
    )
