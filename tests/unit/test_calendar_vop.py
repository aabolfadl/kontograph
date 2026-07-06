"""The German calendar and Verification of Payee.

Both encode domain rules rather than arithmetic, so they are pinned to known
values instead of to whatever the implementation currently returns.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from kg_core.contracts import VopResult
from kg_sim.calendar_de import (
    business_days_between,
    easter_sunday,
    federal_holidays,
    is_business_day,
    last_business_day_of_month,
    next_business_day,
    previous_business_day,
    seasonal_spending_multiplier,
)
from kg_sim.config import load_calibration
from kg_sim.names import introduce_typo, transliterate
from kg_sim.vop import VopEngine, classify, levenshtein, name_similarity, normalise_name

# --------------------------------------------------------------- calendar ---


@pytest.mark.parametrize(
    "year, expected",
    [
        (2024, "2024-03-31"),
        (2025, "2025-04-20"),
        (2026, "2026-04-05"),
        (2027, "2027-03-28"),
        (2030, "2030-04-21"),
        (2038, "2038-04-25"),  # latest possible Easter date
        (2285, "2285-03-22"),  # earliest possible Easter date
    ],
)
def test_computus_matches_known_easter_dates(year: int, expected: str) -> None:
    assert easter_sunday(year).isoformat() == expected


def test_easter_is_always_a_sunday() -> None:
    assert all(easter_sunday(year).weekday() == 6 for year in range(1900, 2100))


def test_movable_holidays_track_easter() -> None:
    """Karfreitag, Ostermontag, Himmelfahrt and Pfingstmontag are Easter-relative."""
    easter = easter_sunday(2025)
    holidays = federal_holidays(2025)
    assert holidays[easter - _dt.timedelta(days=2)] == "Karfreitag"
    assert holidays[easter + _dt.timedelta(days=1)] == "Ostermontag"
    assert holidays[easter + _dt.timedelta(days=39)] == "Christi Himmelfahrt"
    assert holidays[easter + _dt.timedelta(days=50)] == "Pfingstmontag"


def test_german_unity_day_is_a_holiday() -> None:
    assert not is_business_day(_dt.date(2025, 10, 3))


@pytest.mark.parametrize(
    "year, month, expected",
    [
        (2025, 8, "2025-08-29"),  # 31 Aug is a Sunday -> payroll moves to Friday
        (2025, 11, "2025-11-28"),  # 30 Nov is a Sunday -> Friday
        (2025, 10, "2025-10-31"),  # 31 Oct is a Friday -> unchanged
        (2025, 12, "2025-12-31"),  # New Year's Eve is a working day in Germany
        (2026, 5, "2026-05-29"),  # 31 May is a Sunday
    ],
)
def test_payroll_lands_on_the_last_banking_day(year: int, month: int, expected: str) -> None:
    """Not simply "the 30th".

    When month-end falls on a weekend the whole population's income arrives a
    day or two early. A model that assumes a fixed day-of-month mishandles this
    every few months.
    """
    assert last_business_day_of_month(year, month).isoformat() == expected


def test_business_day_navigation_is_consistent() -> None:
    saturday = _dt.date(2025, 10, 11)
    assert previous_business_day(saturday) == _dt.date(2025, 10, 10)
    assert next_business_day(saturday) == _dt.date(2025, 10, 13)
    assert is_business_day(previous_business_day(saturday))
    assert is_business_day(next_business_day(saturday))


def test_business_day_navigation_is_idempotent_on_business_days() -> None:
    monday = _dt.date(2025, 10, 13)
    assert previous_business_day(monday) == monday == next_business_day(monday)


def test_easter_week_loses_exactly_two_business_days() -> None:
    """Karfreitag and Ostermontag bracket the Easter weekend.

    Asserted as a difference against a weekday-only count rather than a
    hand-computed total: the arithmetic of "how many weekdays are in this
    window" is exactly the kind of thing to get wrong in a test and then
    "fix" in the implementation.
    """
    easter = easter_sunday(2025)
    start = easter - _dt.timedelta(days=5)
    end = start + _dt.timedelta(days=10)

    weekdays_only = sum(
        1
        for offset in range((end - start).days)
        if (start + _dt.timedelta(days=offset)).weekday() < 5
    )
    assert business_days_between(start, end) == weekdays_only - 2


def test_december_spending_peaks_and_then_collapses() -> None:
    assert seasonal_spending_multiplier(_dt.date(2025, 12, 20)) > 1.3
    assert seasonal_spending_multiplier(_dt.date(2025, 12, 28)) < 0.7
    assert seasonal_spending_multiplier(_dt.date(2025, 1, 10)) < 1.0
    assert seasonal_spending_multiplier(_dt.date(2025, 3, 15)) == 1.0


# -------------------------------------------------------------------- VoP ---


@pytest.fixture(scope="module")
def engine() -> VopEngine:
    return VopEngine(load_calibration())


def band(engine: VopEngine, claimed: str, actual: str) -> VopResult:
    return classify(
        name_similarity(claimed, actual),
        match_threshold=engine.match_threshold,
        close_threshold=engine.close_threshold,
    )


@pytest.mark.parametrize(
    "claimed, actual, expected, why",
    [
        ("Anna Mueller", "Anna Müller", VopResult.MATCH, "umlaut transliteration"),
        ("Anna Schmidt", "Schmidt Anna", VopResult.MATCH, "word order"),
        ("Nord Handel", "Nord Handel GmbH", VopResult.MATCH, "trading vs registered name"),
        ("Anna Schmidt", "Anna Schmitt", VopResult.CLOSE_MATCH, "one-letter typo"),
        ("A. Schmidt", "Anna Schmidt", VopResult.CLOSE_MATCH, "initial"),
        ("Anna Schmidt", "Anna Schmidt-Weber", VopResult.CLOSE_MATCH, "married name"),
        ("Anna Schmidt", "Boris Petrov", VopResult.NO_MATCH, "impersonation"),
        ("Nord Handel GmbH", "Nord Bau GmbH", VopResult.NO_MATCH, "different company"),
    ],
)
def test_vop_bands(
    engine: VopEngine, claimed: str, actual: str, expected: VopResult, why: str
) -> None:
    """The three bands, with the causes that produce each.

    The legitimate near-misses matter most: if ordinary typos and umlaut
    transliterations landed in NO_MATCH, VoP would separate fraud perfectly in
    this dataset and would be worthless as a test of the model.
    """
    assert band(engine, claimed, actual) is expected, why


def test_similarity_is_symmetric() -> None:
    pairs = [("Anna Schmidt", "Anna Schmitt"), ("Nord Handel GmbH", "Nord Bau GmbH")]
    for left, right in pairs:
        assert name_similarity(left, right) == pytest.approx(name_similarity(right, left))


def test_identical_names_score_exactly_one() -> None:
    assert name_similarity("Stadtwerke München GmbH", "Stadtwerke München GmbH") == 1.0


def test_legal_form_alone_is_not_evidence_of_identity() -> None:
    """Two unrelated GmbHs must not match just because both are GmbHs."""
    assert name_similarity("Nord Handel GmbH", "Süd Technik GmbH") < 0.5


def test_transliteration_covers_the_german_special_characters() -> None:
    assert transliterate("Müller Weiß Öl Ärger") == "Mueller Weiss Oel Aerger"


def test_normalisation_strips_punctuation_and_case() -> None:
    assert normalise_name("Nord-Handel GmbH.") == normalise_name("NORD HANDEL gmbh")


@pytest.mark.parametrize(
    "left, right, distance",
    [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "abc", 0), ("flaw", "lawn", 2)],
)
def test_levenshtein_reference_values(left: str, right: str, distance: int) -> None:
    assert levenshtein(left, right) == distance


def test_unavailable_is_distinct_from_no_match(engine: VopEngine) -> None:
    """A creditor PSP that does not answer is not a mismatch.

    Conflating them would make a VoP outage look like a fraud wave.
    """
    rng = np.random.default_rng(7)
    seen = {engine.check("Anna Schmidt", "Anna Schmidt", rng)[0] for _ in range(2000)}
    assert VopResult.UNAVAILABLE in seen
    assert VopResult.MATCH in seen


def test_matching_names_are_never_overridden(engine: VopEngine) -> None:
    """There is no warning to override when the name matches."""
    rng = np.random.default_rng(11)
    for _ in range(500):
        result, _, overridden = engine.check("Anna Schmidt", "Anna Schmidt", rng)
        if result is VopResult.MATCH:
            assert not overridden


def test_honest_payers_do_override_warnings(engine: VopEngine) -> None:
    """Override alone cannot be treated as proof of fraud.

    A material share of legitimate payers proceed past a CLOSE_MATCH warning —
    which is exactly why the model has to weigh it rather than rule on it.
    """
    rng = np.random.default_rng(13)
    overrides = sum(engine.check("Anna Schmidt", "Anna Schmitt", rng)[2] for _ in range(1000))
    assert 250 < overrides < 600


def test_app_fraud_victims_override_more_often_than_honest_payers(engine: VopEngine) -> None:
    """Social engineering defeats the warning — that is the whole typology."""
    rng = np.random.default_rng(17)
    honest = sum(engine.check("Anna Schmidt", "Boris Petrov", rng)[2] for _ in range(1000))
    rng = np.random.default_rng(17)
    coerced = sum(
        engine.check("Anna Schmidt", "Boris Petrov", rng, is_app_fraud=True)[2] for _ in range(1000)
    )
    assert coerced > honest


def test_typos_stay_recognisable(engine: VopEngine) -> None:
    """A generated typo must degrade the name, not destroy it.

    If ``introduce_typo`` produced unrecognisable strings the legitimate
    CLOSE_MATCH population would collapse into NO_MATCH and the whole
    false-positive story would disappear.
    """
    rng = np.random.default_rng(3)
    names = ["Stadtwerke Muenchen GmbH", "Anna Schmidt", "Nord Logistik GmbH"]
    scores = [name_similarity(introduce_typo(n, rng), n) for n in names for _ in range(40)]
    assert sum(s > engine.close_threshold for s in scores) / len(scores) > 0.5
