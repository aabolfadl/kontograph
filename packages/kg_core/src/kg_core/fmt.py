"""Formatting for numbers that end up in front of a person.

Small, but it exists for a reason. At a 0.02% positive rate the cost-optimal
operating threshold sits near 4e-4, and the obvious ``f"{score:.2f}"`` renders
both the score and the threshold as ``0.00`` — producing narratives that read
"scored at 0.00 against a threshold of 0.00", which tells an investigator
nothing and reads as a bug even when the model is right.

Lives in ``kg_core`` rather than in either caller because the serving hot path
and the SAR generator both need it and must not depend on each other.
"""

from __future__ import annotations

#: Below this, fixed-point notation loses the information. Chosen so a score
#: reported to three decimals still distinguishes adjacent operating points.
_SCIENTIFIC_BELOW = 0.01


def format_score(value: float) -> str:
    """A probability rendered with enough precision to be meaningful."""
    if abs(value) >= _SCIENTIFIC_BELOW:
        return f"{value:.3f}"
    return f"{value:.2e}"


def format_money(cents: int | float, *, currency: str = "EUR") -> str:
    """Minor units as an amount a person reads. Never used for arithmetic."""
    return f"{currency} {cents / 100:,.2f}"
