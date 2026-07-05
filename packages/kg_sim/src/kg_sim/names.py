"""German person and company names.

Names are not decoration here — they are the *input to Verification of Payee*.
VoP compares the payee name a payer typed against the name held on the creditor
account, and since October 2025 every euro-area credit transfer carries that
comparison. So the simulator needs names that produce a realistic distribution
of near-misses: married names, umlaut transliterations, trading names that
differ from registered names, and ordinary typos.

Generated without an external faker dependency: the lists are small, the logic
is deterministic, and a name generator is not worth a transitive dependency
tree in a project whose reproducibility story is a content hash.
"""

from __future__ import annotations

import unicodedata
from typing import Final

import numpy as np

# Common German given names, weighted towards the cohorts that hold bank
# accounts today rather than towards this year's birth registry.
FIRST_NAMES_F: Final[tuple[str, ...]] = (
    "Anna",
    "Maria",
    "Sabine",
    "Julia",
    "Katharina",
    "Petra",
    "Sandra",
    "Nicole",
    "Claudia",
    "Andrea",
    "Susanne",
    "Monika",
    "Christina",
    "Stefanie",
    "Melanie",
    "Birgit",
    "Ursula",
    "Heike",
    "Martina",
    "Angelika",
    "Lena",
    "Laura",
    "Sophie",
    "Marie",
    "Hannah",
    "Emma",
    "Leonie",
    "Lisa",
    "Jana",
    "Franziska",
    "Elke",
    "Gabriele",
    "Renate",
    "Ingrid",
    "Beate",
    "Silke",
    "Kerstin",
    "Daniela",
)

FIRST_NAMES_M: Final[tuple[str, ...]] = (
    "Thomas",
    "Michael",
    "Andreas",
    "Stefan",
    "Peter",
    "Christian",
    "Martin",
    "Markus",
    "Frank",
    "Alexander",
    "Matthias",
    "Jan",
    "Sebastian",
    "Daniel",
    "Klaus",
    "Wolfgang",
    "Jürgen",
    "Dieter",
    "Uwe",
    "Ralf",
    "Lukas",
    "Felix",
    "Maximilian",
    "Tobias",
    "Florian",
    "Philipp",
    "Nils",
    "Jonas",
    "Leon",
    "Hans",
    "Werner",
    "Bernd",
    "Norbert",
    "Holger",
    "Kai",
    "Sven",
    "Timo",
)

SURNAMES: Final[tuple[str, ...]] = (
    "Müller",
    "Schmidt",
    "Schneider",
    "Fischer",
    "Weber",
    "Meyer",
    "Wagner",
    "Becker",
    "Schulz",
    "Hoffmann",
    "Schäfer",
    "Koch",
    "Bauer",
    "Richter",
    "Klein",
    "Wolf",
    "Schröder",
    "Neumann",
    "Schwarz",
    "Zimmermann",
    "Braun",
    "Krüger",
    "Hofmann",
    "Hartmann",
    "Lange",
    "Schmitt",
    "Werner",
    "Schmitz",
    "Krause",
    "Meier",
    "Lehmann",
    "Schmid",
    "Schulze",
    "Maier",
    "Köhler",
    "Herrmann",
    "König",
    "Walter",
    "Mayer",
    "Huber",
    "Kaiser",
    "Fuchs",
    "Peters",
    "Lang",
    "Scholz",
    "Möller",
    "Weiß",
    "Jung",
    "Hahn",
    "Vogel",
    # Names reflecting Germany's largest immigrant communities. Omitting them
    # would produce a name distribution no German bank would recognise, and
    # would make transliteration-driven VoP near-misses artificially rare.
    "Yilmaz",
    "Kaya",
    "Demir",
    "Sahin",
    "Öztürk",
    "Nguyen",
    "Rossi",
    "Kowalski",
    "Nowak",
    "Popescu",
    "Ivanov",
    "Hussein",
    "Ali",
    "Petrov",
    "Novak",
)

#: Legal forms, with their share of the German company register. GmbH dominates;
#: UG is the small-capital variant that is over-represented among shell
#: companies, which is why the layering typologies favour it.
LEGAL_FORMS: Final[tuple[tuple[str, float], ...]] = (
    ("GmbH", 0.52),
    ("GmbH & Co. KG", 0.11),
    ("e.K.", 0.08),
    ("UG (haftungsbeschränkt)", 0.09),
    ("AG", 0.03),
    ("GbR", 0.10),
    ("OHG", 0.04),
    ("eG", 0.03),
)

COMPANY_ROOTS: Final[tuple[str, ...]] = (
    "Nord",
    "Süd",
    "West",
    "Ost",
    "Rhein",
    "Main",
    "Elbe",
    "Donau",
    "Alpen",
    "Hansa",
    "Berlin",
    "Bayern",
    "Sachsen",
    "Hessen",
    "Westfalen",
    "Baden",
    "Adler",
    "Falke",
    "Linde",
    "Eiche",
    "Buche",
    "Stern",
    "Kranz",
    "Berg",
    "Feld",
    "Tal",
    "Hafen",
    "Brücke",
    "Turm",
    "Anker",
    "Kompass",
    "Horizont",
)

COMPANY_SECTORS: Final[tuple[str, ...]] = (
    "Handel",
    "Logistik",
    "Bau",
    "Technik",
    "Systeme",
    "Consulting",
    "Service",
    "Immobilien",
    "Energie",
    "Metall",
    "Textil",
    "Pharma",
    "Medien",
    "Digital",
    "Elektro",
    "Maschinenbau",
    "Vertrieb",
    "Beteiligungen",
    "Industrie",
)

MERCHANT_KINDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("Supermarkt", ("groceries",)),
    ("Bäckerei", ("groceries",)),
    ("Metzgerei", ("groceries",)),
    ("Drogerie", ("groceries", "retail")),
    ("Apotheke", ("retail",)),
    ("Buchhandlung", ("retail",)),
    ("Modehaus", ("retail",)),
    ("Elektronikmarkt", ("retail", "ecommerce")),
    ("Möbelhaus", ("retail",)),
    ("Baumarkt", ("retail",)),
    ("Restaurant", ("dining",)),
    ("Café", ("dining",)),
    ("Pizzeria", ("dining",)),
    ("Tankstelle", ("retail",)),
    ("Onlineshop", ("ecommerce",)),
    ("Versandhandel", ("ecommerce",)),
    ("Fitnessstudio", ("subscription",)),
    ("Friseursalon", ("retail",)),
)

UTILITY_NAMES: Final[tuple[str, ...]] = (
    "Stadtwerke",
    "Energieversorgung",
    "Wasserwerke",
    "Telekommunikation",
    "Netzbetrieb",
    "Fernwärme",
)


def person_name(rng: np.random.Generator) -> str:
    """A plausible German personal name."""
    pool = FIRST_NAMES_F if rng.random() < 0.51 else FIRST_NAMES_M
    first = str(rng.choice(pool))
    last = str(rng.choice(SURNAMES))
    # A minority carry a double-barrelled name, which is a real and frequent
    # source of VoP near-misses: banks and payers disagree about whether to
    # include both halves.
    if rng.random() < 0.06:
        last = f"{last}-{rng.choice(SURNAMES)}"
    return f"{first} {last}"


def company_name(rng: np.random.Generator) -> str:
    """A plausible German company name including its legal form."""
    forms, weights = zip(*LEGAL_FORMS, strict=True)
    form = str(rng.choice(forms, p=np.asarray(weights) / np.sum(weights)))
    root = str(rng.choice(COMPANY_ROOTS))
    sector = str(rng.choice(COMPANY_SECTORS))
    if rng.random() < 0.25:
        return f"{root}{sector} {form}"
    return f"{root} {sector} {form}"


def merchant_name(rng: np.random.Generator) -> tuple[str, tuple[str, ...]]:
    """A merchant's trading name plus the spend categories it serves."""
    kind, categories = MERCHANT_KINDS[int(rng.integers(len(MERCHANT_KINDS)))]
    if rng.random() < 0.55:
        trading = f"{kind} {rng.choice(SURNAMES)}"
    else:
        trading = f"{rng.choice(COMPANY_ROOTS)} {kind}"
    return trading, categories


def utility_name(rng: np.random.Generator, city: str) -> str:
    return f"{rng.choice(UTILITY_NAMES)} {city} GmbH"


# ----------------------------------------------------- name transformations


def transliterate(name: str) -> str:
    """Umlaut and eszett transliteration, as an older core banking system does.

    ``Müller`` becomes ``Mueller``, ``Weiß`` becomes ``Weiss``. This is a real
    and extremely common cause of VoP mismatches in Germany: the payer types the
    name with umlauts, the creditor's bank holds the transliterated form, and a
    naive string comparison scores them as different people.
    """
    replacements = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}
    for source, target in replacements.items():
        name = name.replace(source, target)
    # Strip any remaining diacritics (e.g. from non-German surnames).
    return "".join(
        ch for ch in unicodedata.normalize("NFD", name) if unicodedata.category(ch) != "Mn"
    )


def introduce_typo(name: str, rng: np.random.Generator) -> str:
    """Perturb a name the way a human mistypes one.

    Four mechanisms, because they produce different similarity scores and a VoP
    threshold has to cope with all of them: transposition, omission, doubling,
    and dropping everything after the first name.
    """
    if len(name) < 4:
        return name
    mechanism = int(rng.integers(4))
    position = int(rng.integers(1, len(name) - 1))

    if mechanism == 0:  # transpose adjacent characters
        return name[:position] + name[position + 1] + name[position] + name[position + 2 :]
    if mechanism == 1:  # omit a character
        return name[:position] + name[position + 1 :]
    if mechanism == 2:  # double a character
        return name[:position] + name[position] + name[position:]
    # Give only the first name — very common in P2P transfers.
    parts = name.split()
    return parts[0] if len(parts) > 1 else name


def trading_name_variant(name: str, rng: np.random.Generator) -> str:
    """A business's everyday name versus its registered name.

    ``Nord Handel GmbH`` is invoiced as ``Nord Handel``, and the payer types
    what is on the invoice. Legitimate, ubiquitous, and indistinguishable from
    impersonation by string distance alone — which is exactly why VoP produces
    so many false positives and why ``vop_result`` cannot be used as a rule.
    """
    for form, _ in LEGAL_FORMS:
        if name.endswith(form):
            stripped = name[: -len(form)].strip()
            if rng.random() < 0.5:
                return stripped
            return f"{stripped} {form.split()[0]}"
    return name
