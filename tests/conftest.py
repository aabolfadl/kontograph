"""Shared fixtures.

Deliberately small: fixtures that fabricate rich domain objects tend to hide
the very assumptions a test is supposed to pin down. Anything beyond a plain
sample event is built inside the test that needs it.
"""

from __future__ import annotations

import datetime as _dt
import os

import pytest
from hypothesis import HealthCheck, settings

from kg_core.contracts import SctInstEvent
from kg_core.timeutil import UTC, to_millis

#: Hypothesis budgets. ``dev`` keeps the local loop fast; ``ci`` is selected by the
#: point-in-time job (HYPOTHESIS_PROFILE=ci), where the failures that matter are
#: rare orderings a small example count will not reach.
settings.register_profile("dev", max_examples=100)
settings.register_profile(
    "ci",
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

#: A fixed instant, so every test that prints a timestamp prints the same one.
#: 9 October 2025 is the day SEPA Instant became mandatory in the euro area.
REFERENCE_TS = to_millis(_dt.datetime(2025, 10, 9, 9, 30, 0, tzinfo=UTC))


@pytest.fixture()
def sample_event_dict() -> dict:
    """A well-formed SCT Inst payment as a plain dict."""
    return {
        "tx_id": "0f1d3c4e-5a6b-4c7d-8e9f-a0b1c2d3e4f5",
        "end_to_end_id": "E2E-2025-10-09-0001",
        "acceptance_ts": REFERENCE_TS,
        "ingest_ts": REFERENCE_TS + 40,
        "debtor_account_id": "ACC-000001",
        "debtor_iban": "DE89370400440532013000",
        "debtor_bic": "COBADEFFXXX",
        "debtor_name": "Anna Schmidt",
        "debtor_country": "DE",
        "creditor_account_id": "ACC-000002",
        "creditor_iban": "DE02120300000000202051",
        "creditor_bic": "BYLADEM1001",
        "creditor_name": "Stadtwerke Muenchen GmbH",
        "creditor_country": "DE",
        "creditor_name_claimed": "Stadtwerke Muenchen GmbH",
        "vop_result": "MATCH",
        "vop_name_similarity": 1.0,
        "vop_overridden": False,
        "amount_cents": 8450,
        "currency": "EUR",
        "purpose_code": "UBIL",
        "remittance_info": "Abschlag Strom 10/2025 KdNr 4711",
        "channel": "ONLINE_BANKING",
        "initiation_source": "ASPSP_DIRECT",
        "device_id": "dev-7c21",
        "session_ip_prefix": "84.150.12.0/24",
        "is_new_beneficiary": False,
        "settlement_status": "ACSC",
    }


@pytest.fixture()
def sample_event(sample_event_dict: dict) -> SctInstEvent:
    return SctInstEvent(**sample_event_dict)
