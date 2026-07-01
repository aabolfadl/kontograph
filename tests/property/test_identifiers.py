"""Property-based tests for payment identifiers.

Example-based tests confirm the cases you thought of. For a checksum algorithm
the interesting failures are the cases you did not — leading zeros in the
Kontonummer, a check digit of 00 or 97, boundary lengths. Hypothesis looks for
those, and shrinks any failure to a minimal reproducer.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kg_core.ids import (
    InvalidIbanError,
    build_iban,
    compute_check_digits,
    format_iban,
    german_iban,
    iban_account_number,
    iban_bank_code,
    iban_checksum_valid,
    normalise_iban,
    validate_bic,
    validate_iban,
)

blz = st.text(alphabet="0123456789", min_size=8, max_size=8)
kontonummer = st.integers(min_value=0, max_value=9_999_999_999)


@given(blz, kontonummer)
def test_generated_ibans_always_pass_the_checksum(bank_code: str, account: int) -> None:
    iban = german_iban(bank_code, account)
    assert iban_checksum_valid(iban)
    assert validate_iban(iban, expect_country="DE") == iban
    assert len(iban) == 22


@given(blz, kontonummer)
def test_parts_survive_a_round_trip(bank_code: str, account: int) -> None:
    """Leading zeros in the Kontonummer must not be lost — a classic bug."""
    iban = german_iban(bank_code, account)
    assert iban_bank_code(iban) == bank_code
    assert iban_account_number(iban) == str(account).zfill(10)


@given(blz, kontonummer)
def test_check_digits_stay_in_the_iso_range(bank_code: str, account: int) -> None:
    """Check digits are 98 - (N mod 97) with N mod 97 in [0, 96], so the only
    reachable values are 02..98. Producing 00, 01 or 99 means the mod-97 step
    is wrong. (Hypothesis caught an earlier version of this test that wrongly
    excluded 98, which is reachable whenever N mod 97 == 0.)"""
    digits = german_iban(bank_code, account)[2:4]
    assert digits not in ("00", "01", "99")
    assert 2 <= int(digits) <= 98


@given(blz, kontonummer, st.integers(min_value=0, max_value=99))
def test_wrong_check_digits_are_rejected(bank_code: str, account: int, wrong: int) -> None:
    iban = german_iban(bank_code, account)
    forged = f"DE{wrong:02d}{iban[4:]}"
    if forged != iban:
        assert not iban_checksum_valid(forged), f"forged {forged} must not validate"


@given(blz, kontonummer)
def test_single_digit_corruption_is_always_caught(bank_code: str, account: int) -> None:
    """MOD-97-10 must detect every single-character substitution.

    This is the guarantee that makes IBAN checksums worth having, so we assert
    it rather than assume the implementation delivers it.
    """
    iban = german_iban(bank_code, account)
    for position in range(4, len(iban)):
        original = iban[position]
        replacement = "0" if original != "0" else "1"
        corrupted = iban[:position] + replacement + iban[position + 1 :]
        assert not iban_checksum_valid(corrupted), f"undetected corruption at {position}"


@given(blz, kontonummer)
def test_adjacent_transposition_is_caught(bank_code: str, account: int) -> None:
    iban = german_iban(bank_code, account)
    for position in range(4, len(iban) - 1):
        if iban[position] == iban[position + 1]:
            continue
        swapped = iban[:position] + iban[position + 1] + iban[position] + iban[position + 2 :]
        assert not iban_checksum_valid(swapped), f"undetected transposition at {position}"


@given(blz, kontonummer)
def test_formatting_is_cosmetic_only(bank_code: str, account: int) -> None:
    iban = german_iban(bank_code, account)
    assert normalise_iban(format_iban(iban)) == iban
    assert validate_iban(format_iban(iban).lower()) == iban


@given(blz, kontonummer)
def test_check_digits_agree_with_the_builder(bank_code: str, account: int) -> None:
    bban = f"{bank_code}{str(account).zfill(10)}"
    assert german_iban(bank_code, account)[2:4] == compute_check_digits("DE", bban)


def test_reference_iban_from_the_published_example() -> None:
    """The canonical German worked example, rebuilt from its components."""
    assert german_iban("37040044", 532013000) == "DE89370400440532013000"


@pytest.mark.parametrize(
    "bad, reason",
    [
        ("DE8937040044053201300", "length"),
        ("DE88370400440532013000", "MOD-97"),
        ("XX89370400440532013000", "MOD-97"),
        ("DE89-3704-0044-0532-0130-00", "non-alphanumeric"),
    ],
)
def test_malformed_ibans_are_rejected_with_a_specific_reason(bad: str, reason: str) -> None:
    with pytest.raises(InvalidIbanError, match=reason):
        validate_iban(bad)


def test_wrong_country_is_rejected() -> None:
    with pytest.raises(InvalidIbanError, match="expected AT"):
        validate_iban("DE89370400440532013000", expect_country="AT")


def test_short_bics_are_padded_to_eleven() -> None:
    assert validate_bic("COBADEFF") == "COBADEFFXXX"
    assert validate_bic("byladem1001") == "BYLADEM1001"


@pytest.mark.parametrize("bad", ["COBA", "COBADEF", "1OBADEFFXXX", "COBADEFFXXXX"])
def test_malformed_bics_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="ISO 9362"):
        validate_bic(bad)


def test_non_german_ibans_build_and_validate() -> None:
    austrian = build_iban("AT", "19043", "00234573201")
    assert iban_checksum_valid(austrian) and len(austrian) == 20
