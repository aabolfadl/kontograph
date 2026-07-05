"""German banking institutions.

The BLZ/BIC pairs below are drawn from the Bundesbank's publicly published
Bankleitzahlen directory. They are used so that generated IBANs are
*structurally* realistic — anyone who pastes one into an IBAN validator
gets a sensible bank name back instead of nonsense.

Nothing else about these institutions is modelled, and no simulated account
represents a real customer of any of them.

Why the three-way archetype split matters (the regulatory context): the German market is
genuinely federated. Sparkassen and Genossenschaftsbanken are regionally bounded
networks, not national banks, so a payment between two Sparkassen in different
Länder crosses an institutional boundary that a UK or French model would not
have. That structure is what makes the privacy-preserving federated extension in
the roadmap a real problem rather than an artificial one.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final


class BankArchetype(enum.StrEnum):
    """The three pillars of the German banking system, plus direct banks.

    This is the *Drei-Säulen-Modell*: public-law savings banks, cooperative
    banks, and private commercial banks. Direct banks are split out because
    their payment behaviour differs sharply — no branch channel, younger
    customer base, higher mobile share, and materially higher exposure to
    account-opening fraud, which matters for the mule typologies.
    """

    SPARKASSE = "SPARKASSE"
    GENOSSENSCHAFTSBANK = "GENOSSENSCHAFTSBANK"
    GROSSBANK = "GROSSBANK"
    DIREKTBANK = "DIREKTBANK"


@dataclass(frozen=True, slots=True)
class Bank:
    """One institution."""

    blz: str
    bic: str
    name: str
    archetype: BankArchetype
    city: str

    #: Relative share of retail accounts. Sparkassen and cooperatives together
    #: hold the majority of German retail relationships, which is why the
    #: weights are not uniform.
    retail_weight: float = 1.0

    #: Probability that an account at this bank was opened recently. Direct
    #: banks onboard remotely and fast, which is precisely why mule networks
    #: prefer them — the fresh-account fan-out typology depends on this.
    fresh_account_propensity: float = 0.05

    def __post_init__(self) -> None:
        if not (self.blz.isdigit() and len(self.blz) == 8):
            raise ValueError(f"{self.name}: BLZ must be 8 digits, got {self.blz!r}")
        if len(self.bic) not in (8, 11):
            raise ValueError(f"{self.name}: BIC must be 8 or 11 characters, got {self.bic!r}")


BANKS: Final[tuple[Bank, ...]] = (
    # ---------------------------------------------------- Sparkassen (public-law)
    Bank(
        "10050000",
        "BELADEBEXXX",
        "Landesbank Berlin / Berliner Sparkasse",
        BankArchetype.SPARKASSE,
        "Berlin",
        retail_weight=3.0,
    ),
    Bank(
        "20050550",
        "HASPDEHHXXX",
        "Hamburger Sparkasse",
        BankArchetype.SPARKASSE,
        "Hamburg",
        retail_weight=2.6,
    ),
    Bank(
        "37050198",
        "COLSDE33XXX",
        "Sparkasse KoelnBonn",
        BankArchetype.SPARKASSE,
        "Koeln",
        retail_weight=2.4,
    ),
    Bank(
        "70150000",
        "SSKMDEMMXXX",
        "Stadtsparkasse Muenchen",
        BankArchetype.SPARKASSE,
        "Muenchen",
        retail_weight=2.2,
    ),
    Bank(
        "50050201",
        "HELADEF1822",
        "Frankfurter Sparkasse",
        BankArchetype.SPARKASSE,
        "Frankfurt am Main",
        retail_weight=1.9,
    ),
    Bank(
        "30050110",
        "DUSSDEDDXXX",
        "Stadtsparkasse Duesseldorf",
        BankArchetype.SPARKASSE,
        "Duesseldorf",
        retail_weight=1.5,
    ),
    Bank(
        "25050180",
        "SPKHDE2HXXX",
        "Sparkasse Hannover",
        BankArchetype.SPARKASSE,
        "Hannover",
        retail_weight=1.4,
    ),
    Bank(
        "60050101",
        "SOLADEST600",
        "Landesbank Baden-Wuerttemberg",
        BankArchetype.SPARKASSE,
        "Stuttgart",
        retail_weight=1.8,
    ),
    # ------------------------------------------ Genossenschaftsbanken (cooperative)
    Bank(
        "50060400",
        "GENODEFFXXX",
        "DZ BANK",
        BankArchetype.GENOSSENSCHAFTSBANK,
        "Frankfurt am Main",
        retail_weight=1.2,
    ),
    Bank(
        "43060967",
        "GENODEM1GLS",
        "GLS Gemeinschaftsbank",
        BankArchetype.GENOSSENSCHAFTSBANK,
        "Bochum",
        retail_weight=0.9,
    ),
    Bank(
        "70090100",
        "GENODEF1S04",
        "Sparda-Bank Muenchen",
        BankArchetype.GENOSSENSCHAFTSBANK,
        "Muenchen",
        retail_weight=1.3,
    ),
    Bank(
        "37060193",
        "GENODED1PAX",
        "PAX-Bank",
        BankArchetype.GENOSSENSCHAFTSBANK,
        "Koeln",
        retail_weight=0.7,
    ),
    Bank(
        "30060601",
        "DAAEDEDDXXX",
        "Deutsche Apotheker- und Aerztebank",
        BankArchetype.GENOSSENSCHAFTSBANK,
        "Duesseldorf",
        retail_weight=0.8,
    ),
    # ------------------------------------------------- Grossbanken (private commercial)
    Bank(
        "50070010",
        "DEUTDEFFXXX",
        "Deutsche Bank",
        BankArchetype.GROSSBANK,
        "Frankfurt am Main",
        retail_weight=2.4,
    ),
    Bank(
        "10070000",
        "DEUTDEBBXXX",
        "Deutsche Bank",
        BankArchetype.GROSSBANK,
        "Berlin",
        retail_weight=1.6,
    ),
    Bank(
        "30040000",
        "COBADEFFXXX",
        "Commerzbank",
        BankArchetype.GROSSBANK,
        "Duesseldorf",
        retail_weight=2.1,
    ),
    Bank(
        "70020270",
        "HYVEDEMMXXX",
        "UniCredit Bank - HypoVereinsbank",
        BankArchetype.GROSSBANK,
        "Muenchen",
        retail_weight=1.7,
    ),
    # ------------------------------------------------------------------ Direktbanken
    # Higher fresh-account propensity: remote onboarding is fast, which is a
    # genuine AML pressure point and the mechanism the mule typologies exploit.
    Bank(
        "50010517",
        "INGDDEFFXXX",
        "ING-DiBa",
        BankArchetype.DIREKTBANK,
        "Frankfurt am Main",
        retail_weight=2.3,
        fresh_account_propensity=0.16,
    ),
    Bank(
        "12030000",
        "BYLADEM1001",
        "Deutsche Kreditbank (DKB)",
        BankArchetype.DIREKTBANK,
        "Berlin",
        retail_weight=1.9,
        fresh_account_propensity=0.18,
    ),
    Bank(
        "10011001",
        "NTSBDEB1XXX",
        "N26",
        BankArchetype.DIREKTBANK,
        "Berlin",
        retail_weight=1.4,
        fresh_account_propensity=0.31,
    ),
    Bank(
        "76026000",
        "NORSDE71XXX",
        "norisbank",
        BankArchetype.DIREKTBANK,
        "Berlin",
        retail_weight=0.8,
        fresh_account_propensity=0.14,
    ),
)

#: Fast lookup by Bankleitzahl.
BANKS_BY_BLZ: Final[dict[str, Bank]] = {bank.blz: bank for bank in BANKS}


def banks_of(archetype: BankArchetype) -> tuple[Bank, ...]:
    """Every institution of a given archetype."""
    return tuple(bank for bank in BANKS if bank.archetype == archetype)


def retail_weights() -> tuple[tuple[Bank, ...], tuple[float, ...]]:
    """Banks and their normalised retail-account shares, for sampling."""
    total = sum(bank.retail_weight for bank in BANKS)
    return BANKS, tuple(bank.retail_weight / total for bank in BANKS)


#: Non-German counterparty BICs, for the cross-border leg of layering chains.
#: Laundering typologies routinely move value out of the domestic rail, and a
#: simulator with no foreign edges makes cross-border a trivially clean signal.
FOREIGN_BANKS: Final[tuple[tuple[str, str, str, str], ...]] = (
    # (country, bank code, BIC, name)
    ("AT", "19043", "GIBAATWWXXX", "Erste Bank"),
    ("NL", "ABNA", "ABNANL2AXXX", "ABN AMRO"),
    ("FR", "30004", "BNPAFRPPXXX", "BNP Paribas"),
    ("IT", "05428", "BPMOIT22XXX", "Banco BPM"),
    ("ES", "21000", "CAIXESBBXXX", "CaixaBank"),
    ("BE", "001", "GEBABEBBXXX", "BNP Paribas Fortis"),
    ("PL", "10201026", "BPKOPLPWXXX", "PKO Bank Polski"),
    ("CZ", "0800", "GIBACZPXXXX", "Ceska sporitelna"),
    ("LU", "001", "BILLLULLXXX", "Banque Internationale a Luxembourg"),
    ("CH", "00700", "POFICHBEXXX", "PostFinance"),
)
