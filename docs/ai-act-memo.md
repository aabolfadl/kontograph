# Regulatory classification memo

**Subject:** KONTOGRAPH real-time financial-crime detection
**Status:** analysis of a synthetic-data research system; not legal advice
**Last reviewed:** 2026-08-04

---

## The point of this memo

The regulatory context: *"The credit module is high-risk (Annex III); the AML module is not, but is
regulated elsewhere. Getting this distinction* right *is a stronger signal than reciting the
Act."*

Most "AI Act compliance" claims attached to a fraud model are wrong in a specific way: they
assume that because the system is consequential, it must be Annex III high-risk. It is not, and
the instrument that *does* govern it imposes obligations the AI Act does not. Applying the wrong
framework produces a compliance programme that is simultaneously over-engineered in one direction
and absent in another.

---

## 1. What this system does

Scores individual SEPA Instant Credit Transfers for financial-crime risk inside the ten-second
settlement window, and raises an alert for human review.

It is a **triage** system. It determines what a compliance analyst looks at. It does not
determine what happens to a customer — except in the `BLOCK` band, which is discussed in §4.

---

## 2. EU AI Act — Regulation (EU) 2024/1689

### 2.1 Is it prohibited under Art. 5?

No. It performs no social scoring by public authorities, no real-time remote biometric
identification, no emotion inference in workplaces or education, and no untargeted facial-image
scraping.

One Art. 5 practice deserves an explicit "no" rather than silence: **predictive policing based on
profiling of natural persons** to assess the risk of criminal offence. This system assesses the
risk that *a transaction* is part of a criminal scheme, on transactional evidence, in discharge of
a statutory AML obligation. It does not predict whether a person will commit an offence from
personality traits or characteristics. The distinction is real but not decorative, and a system
that drifted toward scoring people rather than payments would need this re-examined.

### 2.2 Is it Annex III high-risk?

**No.** The relevant Annex III heading is point 5(b): *"AI systems intended to be used to evaluate
the creditworthiness of natural persons or establish their credit score."*

AML transaction monitoring is not creditworthiness assessment. Annex III contains **no heading for
financial-crime detection**, and its omission is not an oversight — the AML package legislates
that field directly and in far more operational detail than the AI Act would.

Two further Annex III headings are sometimes reached for and do not apply:

- *Access to essential private services* (point 5(a)) covers eligibility for benefits. Alerting on
  a payment for review is not an eligibility determination.
- *Law enforcement* (point 6) applies to systems used **by or on behalf of law-enforcement
  authorities**. A bank meeting its own AML obligation is not acting on behalf of one, even though
  the output may eventually reach the FIU as a Verdachtsmeldung.

**A credit module would be different.** Were this platform extended to score creditworthiness, that
module would be Annex III high-risk and would attract the full Chapter III regime: risk-management
system (Art. 9), data governance (Art. 10), technical documentation (Art. 11), logging (Art. 12),
transparency (Art. 13), human oversight (Art. 14), and accuracy/robustness/cybersecurity (Art. 15).
That is a substantially heavier programme, and it applies to the module, not the platform.

### 2.3 So does the AI Act impose nothing?

Art. 4 (**AI literacy**) applies to any deployer: staff operating the system must have adequate
understanding of it. For a compliance team acting on model-generated alerts that is a real, if
modest, obligation — and it is one this project's explanation surface directly serves.

---

## 3. What actually governs it

| Instrument | What it requires here |
|---|---|
| **AMLR / AMLD** (and the AMLA in Frankfurt) | Transaction monitoring proportionate to risk; suspicious activity reporting; record-keeping |
| **BaFin AuA / interpretive guidance** | Documented, justified monitoring rules and model choices |
| **MaRisk AT 4.3.2 / BAIT** | Model risk management: validation, monitoring, documented change control |
| **DORA** (Reg. 2022/2554) | ICT operational resilience, incident reporting, third-party risk |
| **DSGVO / GDPR** | Lawful basis, data minimisation, and Art. 22 — see §4 |
| **Reg. (EU) 2024/886** | SEPA Instant: ten-second settlement and mandatory Verification of Payee — the constraint that shapes the architecture |

The **model risk** obligations under MaRisk are, in practice, the demanding ones. They are why this
repository carries a frozen evaluation protocol, chronological splits, bootstrap intervals, and
generated model cards — not because the AI Act asked, but because a supervisor will.

---

## 4. GDPR Art. 22 — the obligation that does bite

Art. 22 grants a right not to be subject to a decision based solely on automated processing which
produces legal effects or similarly significantly affects the data subject.

**Where it applies.** The `BLOCK` band holds or refuses a payment automatically. Under SCT Inst
that is an irrevocable, immediate, customer-facing consequence — comfortably "similarly
significantly affects". `ALERT` and `HOLD` route to a human and do not, on their own, engage
Art. 22.

**How this system addresses it.**

1. **Human oversight is real, not nominal.** `ALERT` and `HOLD` require analyst review. Automated
   `BLOCK` is reserved for scores far above the operating threshold, and the console captures the
   analyst's verdict. Rubber-stamp oversight would not satisfy Art. 22, which is why the feedback
   contract records *time to decision* — a reviewer spending two seconds per alert is not
   exercising meaningful oversight, and the data would show it.

2. **Meaningful information about the logic involved.** Every decision persists, *at decision
   time*: the score, the threshold in force, the model version, the feature-set fingerprint, the
   code commit, and the explanation shown. Reconstructing this later from a model file and a guess
   about which version was live is not the same thing.

3. **The explanation is actionable, not decorative.** The counterfactual states which payments the
   alert rests on and is **verified by re-scoring** — it is emitted only when removing them
   demonstrably moves the model below threshold. This is the vertical spike
   ([ADR-0007](adr/0007-amortised-explainer.md)), and it exists because Art. 22 asks for something
   a SHAP bar chart does not provide.

4. **The log is tamper-evident.** Decisions are hash-chained: altering entry *n* invalidates every
   entry after it. `kg serve verify-log` audits the chain and reports *where* it broke, which
   bounds the blast radius of an incident.

**Data minimisation (Art. 5(1)(c)).** IP addresses are truncated to a /24 at the producer, so the
full address is never collected and cannot leak later. Feature *values* are excluded from the
decision log by default — they are behavioural facts about a named account — while the feature-set
*fingerprint* is always recorded, which is what actually supports the Art. 22 explanation.

---

## 5. Honest limitations

- **This is a research system on synthetic data.** No production deployment, no DPIA, no
  supervisory engagement. Nothing here has been reviewed by a lawyer.
- **No fairness assessment has been performed.** The simulator generates names reflecting
  Germany's largest immigrant communities, which makes a disparate-impact analysis across name
  origin *possible* — and it has not been done. Verification of Payee compares names, so a name
  similarity feature is a plausible route to disparate impact. This is a known gap, stated
  because an unstated gap is worse than an admitted one.
- **The Art. 22 analysis assumes the `BLOCK` band is used.** A deployment that only ever alerts
  would engage Art. 22 far more weakly, and the analysis should be redone against actual usage.
- **AMLA's supervisory practice is still forming.** The authority is standing up in Frankfurt and
  its expectations for model-based monitoring will develop. This memo reflects the position as of
  its review date and is not a durable conclusion.

---

## 6. What would change this classification

- Adding a **credit or creditworthiness** module — immediately Annex III high-risk for that module.
- Using the system **on behalf of a law-enforcement authority** rather than in discharge of the
  institution's own AML obligation — potentially Annex III point 6.
- Scoring **persons** rather than payments, on characteristics rather than transactional evidence —
  which would require the Art. 5 predictive-policing analysis to be revisited seriously.
- Removing human review from the `ALERT` path, making the whole pipeline solely automated.

---

_Generated as part of the governance bundle: `kg gov all`. Reviewed against the AI Act, the AML
package, DORA and the GDPR as at the date above._
