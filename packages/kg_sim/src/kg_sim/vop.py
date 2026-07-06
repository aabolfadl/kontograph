"""Verification of Payee.

Since 9 October 2025, Regulation (EU) 2024/886 obliges every euro-area PSP to
check the payee name a payer supplies against the name held on the creditor
account, and to warn the payer before the payment is confirmed.

This module implements that check. It matters to the project for a reason
beyond realism: **VoP is the highest-signal real-time feature available on the
German rail, and it is also a false-positive machine.** Married names,
transliterated umlauts, and trading-vs-registered names all produce genuine
mismatches on entirely legitimate payments. Any model that treats
``vop_result != MATCH`` as a rule will drown the compliance team.

So the simulator generates all three outcome bands with realistic *causes*, and
the model has to learn when a mismatch means something.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from kg_core.contracts import VopResult
from kg_sim.names import transliterate

# Words that carry no identifying information and should not dominate a
# similarity score. "GmbH" matching "GmbH" is not evidence of anything.
_NOISE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "gmbh",
        "ag",
        "kg",
        "ohg",
        "gbr",
        "eg",
        "ug",
        "haftungsbeschraenkt",
        "co",
        "e",
        "k",
        "mbh",
        "und",
        "der",
        "die",
        "das",
        "van",
        "von",
        "de",
    }
)


def normalise_name(name: str) -> str:
    """Canonical form for comparison: transliterated, lowercased, depunctuated.

    Doing this *before* scoring is what stops the umlaut problem swamping
    everything else — ``Müller`` and ``Mueller`` are the same person, and a
    comparison that says otherwise is measuring the encoding, not the identity.
    """
    folded = transliterate(name).lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in folded).split())


def _tokens(name: str) -> list[str]:
    return [t for t in normalise_name(name).split() if t not in _NOISE_TOKENS]


def levenshtein(left: str, right: str) -> int:
    """Edit distance, two-row dynamic programme.

    Hand-written to avoid a dependency: this runs on the serving path, where
    every millisecond is inside a 200 ms budget, and the two-row form keeps it
    O(min(m,n)) in memory.
    """
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        current = [i]
        for j, right_ch in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (left_ch != right_ch),  # substitution
                )
            )
        previous = current
    return previous[-1]


#: Score awarded when one token is an initial matching the other's first letter
#: ("A. Schmidt" vs "Anna Schmidt"). High, but not 1.0: an initial genuinely
#: carries less verification value than a full name, and the band it lands in
#: (CLOSE_MATCH rather than MATCH) is the honest answer.
_INITIAL_SCORE: Final = 0.85


def _ratio(left: str, right: str) -> float:
    """Normalised similarity in [0, 1] between two name tokens."""
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    # Initials are ubiquitous on German invoices and in payment references.
    if len(left) == 1 and right.startswith(left):
        return _INITIAL_SCORE
    if len(right) == 1 and left.startswith(right):
        return _INITIAL_SCORE
    longest = max(len(left), len(right))
    return 1.0 - (levenshtein(left, right) / longest)


def name_similarity(claimed: str, actual: str) -> float:
    """Similarity between the name a payer typed and the name on the account.

    Deliberately more forgiving than raw edit distance, because the failure mode
    that matters operationally is a *false* mismatch on a legitimate payment:

    - token-set comparison, so word order does not matter
      (``Anna Schmidt`` vs ``Schmidt Anna``)
    - a subset of tokens scores high, so ``Nord Handel`` matches
      ``Nord Handel GmbH``
    - per-token best-match averaging, so one mistyped word does not sink the
      whole comparison

    Returns 1.0 for an exact normalised match.
    """
    claimed_norm, actual_norm = normalise_name(claimed), normalise_name(actual)
    if claimed_norm == actual_norm:
        return 1.0

    claimed_tokens, actual_tokens = _tokens(claimed), _tokens(actual)
    if not claimed_tokens or not actual_tokens:
        return _ratio(claimed_norm, actual_norm)

    claimed_set, actual_set = set(claimed_tokens), set(actual_tokens)
    if claimed_set == actual_set:
        return 1.0

    # One name being a clean subset of the other is the trading-name case.
    # Penalise only mildly, and in proportion to how much is missing.
    if claimed_set < actual_set or actual_set < claimed_set:
        smaller, larger = sorted((claimed_set, actual_set), key=len)
        return 0.90 + 0.10 * (len(smaller) / len(larger))

    # Otherwise: symmetric soft token matching. Each token on each side scores
    # against its best counterpart on the other, and the two directions are
    # combined by harmonic mean.
    #
    # An earlier version added an exact-set-overlap "coverage" term. That
    # double-penalised near-matches: "Anna Schmidt" vs "Anna Schmitt" — a
    # one-letter surname typo, the single most common legitimate mismatch in
    # German retail payments — scored 0.78 and landed in NO_MATCH. Since the
    # whole point of modelling VoP here is that it produces false positives on
    # honest payments, a scorer that rejects ordinary typos would have made the
    # feature a fraud oracle and destroyed its value as a test.
    forward = sum(max(_ratio(c, a) for a in actual_tokens) for c in claimed_tokens)
    forward /= len(claimed_tokens)
    backward = sum(max(_ratio(a, c) for c in claimed_tokens) for a in actual_tokens)
    backward /= len(actual_tokens)
    if forward + backward == 0:
        return 0.0
    return float(2 * forward * backward / (forward + backward))


def classify(similarity: float, *, match_threshold: float, close_threshold: float) -> VopResult:
    """Map a similarity score onto the scheme's three-band outcome."""
    if similarity >= match_threshold:
        return VopResult.MATCH
    if similarity >= close_threshold:
        return VopResult.CLOSE_MATCH
    return VopResult.NO_MATCH


class VopEngine:
    """Applies the VoP check, including the payer's decision to override.

    Instantiated once per simulation and driven by the calibration block, so
    thresholds and override rates are configuration rather than magic numbers
    scattered through the generator.
    """

    def __init__(self, calibration: dict) -> None:
        vop = calibration["vop"]
        self.match_threshold: float = vop["match_threshold"]
        self.close_threshold: float = vop["close_match_threshold"]
        self.typo_probability: float = vop["legitimate_typo_probability"]
        self.alias_probability: float = vop["legitimate_alias_probability"]
        self.unavailable_probability: float = vop["unavailable_probability"]
        self.override_legit: float = vop["override_probability_legitimate"]
        self.override_fraud: float = vop["override_probability_app_fraud"]

    def check(
        self,
        claimed_name: str,
        actual_name: str,
        rng: np.random.Generator,
        *,
        is_app_fraud: bool = False,
    ) -> tuple[VopResult, float, bool]:
        """Run the check.

        Returns ``(result, similarity, overridden)``.

        ``overridden`` is separate from the result and legally significant: under
        the 2025 regime a payer who proceeds past a warning shifts liability, and
        it is a strong — but not conclusive — fraud signal. Honest payers
        override too, which is why the rate for legitimate payments is set well
        above zero.
        """
        # The creditor PSP can simply fail to answer inside the SCT Inst window.
        # Modelled because "no answer" is operationally different from "no
        # match", and a model that conflates them will misread an outage.
        if rng.random() < self.unavailable_probability:
            return VopResult.UNAVAILABLE, 0.0, False

        similarity = name_similarity(claimed_name, actual_name)
        result = classify(
            similarity,
            match_threshold=self.match_threshold,
            close_threshold=self.close_threshold,
        )

        if result is VopResult.MATCH:
            return result, similarity, False

        override_rate = self.override_fraud if is_app_fraud else self.override_legit
        return result, similarity, bool(rng.random() < override_rate)

    def perturb_claimed_name(self, actual_name: str, rng: np.random.Generator) -> str:
        """What the payer *typed*, for a legitimate payment.

        Usually the right name. Sometimes a typo, sometimes the trading name.
        These are the two mechanisms that generate legitimate CLOSE_MATCH
        traffic, and without them VoP would be a perfect fraud oracle in this
        dataset — which would make the whole feature worthless as a test of the
        model.
        """
        from kg_sim.names import introduce_typo, trading_name_variant

        draw = rng.random()
        if draw < self.typo_probability:
            return introduce_typo(actual_name, rng)
        if draw < self.typo_probability + self.alias_probability:
            return trading_name_variant(actual_name, rng)
        return actual_name
