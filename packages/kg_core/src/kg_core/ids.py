"""Payment identifiers: IBAN, BIC, and deterministic synthetic account numbers.

These are *algorithms only*. The list of real German Bankleitzahlen and their
BICs lives in :mod:`kg_sim.institutions`, because which banks exist is domain
data about the simulated ecosystem, not a core primitive.

Why bother with a real mod-97 checksum for synthetic data? Because the moment a
downstream component validates IBANs — a data-quality gate, a schema check, a
demo audience who pastes one into a validator — fake-looking identifiers make
the whole artifact look careless. Correctness here is cheap.
"""

from __future__ import annotations

import re
import string
from typing import Final

# ISO 13616 BBAN lengths for the countries we model. Germany is the only one
# generated; the others appear as cross-border counterparties.
IBAN_LENGTHS: Final[dict[str, int]] = {
    "DE": 22,
    "AT": 20,
    "NL": 18,
    "FR": 27,
    "IT": 27,
    "ES": 24,
    "BE": 16,
    "PL": 28,
    "CZ": 24,
    "CH": 21,
    "LU": 20,
    "DK": 18,
}

_ALPHANUM = set(string.ascii_uppercase + string.digits)
_BIC_RE: Final = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")

# A -> 10, B -> 11, ... Z -> 35 (ISO 7064 alphabet substitution).
_LETTER_VALUES: Final[dict[str, str]] = {
    ch: str(10 + i) for i, ch in enumerate(string.ascii_uppercase)
}


class InvalidIbanError(ValueError):
    """Raised when an IBAN fails structural or checksum validation."""


def normalise_iban(iban: str) -> str:
    """Strip the human-readable spacing and upper-case an IBAN."""
    return iban.replace(" ", "").replace("\t", "").upper()


def _mod97(rearranged: str) -> int:
    """ISO 7064 MOD-97-10 over a possibly very long numeric string.

    Computed in chunks rather than via one huge ``int`` so the cost stays linear
    and the implementation matches how it is done on the wire.
    """
    remainder = 0
    for ch in rearranged:
        if ch in string.digits:
            remainder = (remainder * 10 + int(ch)) % 97
        else:
            for digit in _LETTER_VALUES[ch]:
                remainder = (remainder * 10 + int(digit)) % 97
    return remainder


#: ISO 13616 check digits are ``98 - (N mod 97)`` with ``N mod 97`` in [0, 96],
#: so only 02..98 are reachable. 00, 01 and 99 are excluded by *rule*, not by
#: the checksum -- see :func:`iban_checksum_valid`.
MIN_CHECK_DIGITS: Final = 2
MAX_CHECK_DIGITS: Final = 98


def iban_checksum_valid(iban: str) -> bool:
    """True iff *iban* satisfies ISO 7064 MOD-97-10 *and* has legal check digits.

    The range test is not redundant, which is easy to get wrong. MOD-97-10 alone
    accepts some forgeries: if an IBAN's true check digits are 97, then the same
    BBAN with check digits ``00`` also satisfies ``mod97 == 1``. The equivalent
    holds for ``01`` (when the true digits are 98) and ``99`` (when they are 66).
    ISO 13616 excludes 00/01/99 by rule, so validation has to as well.

    Found by the property suite, not by inspection -- see
    ``tests/property/test_identifiers.py::test_wrong_check_digits_are_rejected``.
    """
    value = normalise_iban(iban)
    if len(value) < 5 or not set(value) <= _ALPHANUM:
        return False
    check = value[2:4]
    if not check.isdigit() or not (MIN_CHECK_DIGITS <= int(check) <= MAX_CHECK_DIGITS):
        return False
    return _mod97(value[4:] + value[:4]) == 1


def compute_check_digits(country: str, bban: str) -> str:
    """Return the two check digits that make ``country + dd + bban`` valid."""
    rearranged = f"{bban.upper()}{country.upper()}00"
    return f"{98 - _mod97(rearranged):02d}"


def validate_iban(iban: str, *, expect_country: str | None = None) -> str:
    """Validate structure, length and checksum; return the normalised IBAN.

    Raises :class:`InvalidIbanError` with a message naming the specific failure,
    because "invalid IBAN" alone is useless in a data-quality report.
    """
    value = normalise_iban(iban)
    if not set(value) <= _ALPHANUM:
        raise InvalidIbanError(f"{iban!r} contains non-alphanumeric characters")
    if len(value) < 5:
        raise InvalidIbanError(f"{iban!r} is too short to be an IBAN")
    country = value[:2]
    if not country.isalpha():
        raise InvalidIbanError(f"{iban!r} does not start with a country code")
    if expect_country and country != expect_country.upper():
        raise InvalidIbanError(f"{iban!r} is {country}, expected {expect_country.upper()}")
    expected_length = IBAN_LENGTHS.get(country)
    if expected_length is not None and len(value) != expected_length:
        raise InvalidIbanError(
            f"{iban!r} has length {len(value)}, but {country} IBANs are {expected_length}"
        )
    check = value[2:4]
    if not check.isdigit() or not (MIN_CHECK_DIGITS <= int(check) <= MAX_CHECK_DIGITS):
        raise InvalidIbanError(
            f"{iban!r} has check digits {check!r}: ISO 13616 only permits "
            f"{MIN_CHECK_DIGITS:02d}..{MAX_CHECK_DIGITS}"
        )
    if not iban_checksum_valid(value):
        raise InvalidIbanError(f"{iban!r} fails the MOD-97-10 checksum")
    return value


def build_iban(country: str, bank_code: str, account_number: str) -> str:
    """Assemble a checksum-correct IBAN from its parts.

    For Germany the BBAN is the 8-digit Bankleitzahl followed by the 10-digit
    Kontonummer, zero-padded on the left as the Bundesbank specifies.
    """
    country = country.upper()
    bank_code = bank_code.strip()
    account_number = account_number.strip()
    if country == "DE":
        if not (bank_code.isdigit() and len(bank_code) == 8):
            raise ValueError(f"German BLZ must be 8 digits, got {bank_code!r}")
        if not account_number.isdigit() or len(account_number) > 10:
            raise ValueError(f"German Kontonummer must be <=10 digits, got {account_number!r}")
        bban = f"{bank_code}{account_number.zfill(10)}"
    else:
        bban = f"{bank_code}{account_number}"
    return f"{country}{compute_check_digits(country, bban)}{bban}"


def german_iban(blz: str, account_number: int | str) -> str:
    """Convenience wrapper for the German case, the only one we generate."""
    return build_iban("DE", blz, str(account_number))


#: Position of the bank identifier *within the BBAN* (i.e. after the first four
#: IBAN characters), as (offset, length), per the ISO 13616 national formats.
#: Only the countries we actually model are listed; anything else raises rather
#: than silently returning a plausible-looking wrong slice.
BANK_CODE_SPANS: Final[dict[str, tuple[int, int]]] = {
    "DE": (0, 8),  # Bankleitzahl
    "AT": (0, 5),
    "NL": (0, 4),  # alphabetic bank code
    "FR": (0, 5),  # code banque
    "ES": (0, 4),
    "BE": (0, 3),
    "PL": (0, 8),
    "CZ": (0, 4),
    "CH": (0, 5),
    "LU": (0, 3),
    "DK": (0, 4),
    "IT": (1, 5),  # BBAN opens with the CIN check character, then ABI
}


def iban_bank_code(iban: str) -> str:
    """Extract the bank identifier (the Bankleitzahl for DE) from an IBAN.

    Raises for a country whose national format we have not encoded. Guessing
    the German offset for an Italian IBAN would produce a wrong-but-plausible
    bank code, which is exactly the kind of quiet error a data-quality report
    never catches.
    """
    value = normalise_iban(iban)
    country = value[:2]
    span = BANK_CODE_SPANS.get(country)
    if span is None:
        raise InvalidIbanError(
            f"bank-code position for country {country!r} is not encoded; "
            f"add it to BANK_CODE_SPANS (known: {sorted(BANK_CODE_SPANS)})"
        )
    offset, length = span
    start = 4 + offset
    return value[start : start + length]


def iban_account_number(iban: str) -> str:
    """Extract the account number portion (the Kontonummer for DE).

    Defined as everything after the bank identifier, so it stays correct for
    every country in :data:`BANK_CODE_SPANS` rather than only for Germany.
    """
    value = normalise_iban(iban)
    offset, length = BANK_CODE_SPANS[value[:2]]
    return value[4 + offset + length :]


def format_iban(iban: str) -> str:
    """Group into blocks of four, the way a bank statement prints it."""
    value = normalise_iban(iban)
    return " ".join(value[i : i + 4] for i in range(0, len(value), 4))


def bic_valid(bic: str) -> bool:
    """True iff *bic* matches the ISO 9362 8- or 11-character shape."""
    return bool(_BIC_RE.match(bic.strip().upper()))


def validate_bic(bic: str) -> str:
    """Validate and normalise a BIC, padding an 8-character BIC to 11 with XXX."""
    value = bic.strip().upper()
    if not bic_valid(value):
        raise ValueError(f"{bic!r} is not a valid ISO 9362 BIC")
    return value if len(value) == 11 else value + "XXX"
