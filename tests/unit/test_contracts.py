"""Contract tests.

These are the gate described in the platform scope under "Data contracts & quality":
producer-side schema ownership with backward-compatibility enforced in CI.
"""

from __future__ import annotations

import copy

import pytest

from kg_core.contracts import (
    CONTRACT_MODELS,
    IncompatibleSchemaError,
    SctInstEvent,
    assert_backward_compatible,
    check_backward_compatible,
    decode,
    encode,
    field_parity,
    list_schemas,
    load_schema_dict,
    schema_fingerprint,
    subject_for,
)


def test_every_schema_has_a_pydantic_mirror() -> None:
    """A contract without an in-process model is a contract nobody validates."""
    assert set(list_schemas()) == set(CONTRACT_MODELS), (
        "add the new contract to CONTRACT_MODELS in kg_core.contracts.models"
    )


@pytest.mark.parametrize("name", sorted(CONTRACT_MODELS))
def test_avro_and_pydantic_do_not_drift(name: str) -> None:
    """The cost of keeping two representations: this test, run on every push."""
    only_avro, only_model = field_parity(name, CONTRACT_MODELS[name])
    assert not only_avro, f"{name}: fields in Avro but not in the model: {sorted(only_avro)}"
    assert not only_model, f"{name}: fields in the model but not in Avro: {sorted(only_model)}"


@pytest.mark.parametrize("name", sorted(CONTRACT_MODELS))
def test_schema_parses_and_fingerprints_stably(name: str) -> None:
    first, second = schema_fingerprint(name), schema_fingerprint(name)
    assert first == second and len(first) == 32
    assert subject_for(name).endswith("-value")


def test_payment_event_round_trips_through_avro(sample_event: SctInstEvent) -> None:
    payload = encode("sct_inst_event.v1", sample_event.model_dump(mode="json"))
    restored = decode("sct_inst_event.v1", payload)
    assert restored["tx_id"] == sample_event.tx_id
    assert restored["amount_cents"] == sample_event.amount_cents
    assert restored["vop_result"] == sample_event.vop_result.value


def test_event_contract_carries_no_ground_truth() -> None:
    """The single most important structural anti-leakage property.

    If a typology or label field ever appears on the wire contract, a feature
    can read it, and the entire evaluation becomes fiction. This test is why
    ground truth lives in a separate table.
    """
    forbidden = {"label", "is_fraud", "typology", "is_mule", "ground_truth", "y", "target"}
    fields = {f["name"].lower() for f in load_schema_dict("sct_inst_event.v1")["fields"]}
    assert not (fields & forbidden), (
        f"ground truth leaked into the wire contract: {fields & forbidden}"
    )


def test_ingest_cannot_precede_acceptance(sample_event_dict: dict) -> None:
    sample_event_dict["ingest_ts"] = sample_event_dict["acceptance_ts"] - 1
    with pytest.raises(ValueError, match="precedes acceptance_ts"):
        SctInstEvent(**sample_event_dict)


def test_suspicious_unproven_is_not_a_positive_label() -> None:
    from kg_core.contracts import LabelValue

    assert LabelValue.CONFIRMED_FRAUD.is_positive
    assert LabelValue.CONFIRMED_LAUNDERING.is_positive
    assert not LabelValue.SUSPICIOUS_UNPROVEN.is_positive, (
        "training on unproven suspicion teaches the model to imitate the "
        "incumbent alerting system rather than to detect crime"
    )


# ----------------------------------------------- the compatibility gate ----


@pytest.fixture()
def base_schema() -> dict:
    return load_schema_dict("sct_inst_event.v1")


def test_schema_is_compatible_with_itself(base_schema: dict) -> None:
    assert check_backward_compatible(base_schema, base_schema) == []


def test_adding_a_field_without_default_is_rejected(base_schema: dict) -> None:
    new = copy.deepcopy(base_schema)
    new["fields"].append({"name": "risk_band", "type": "string"})
    problems = check_backward_compatible(new, base_schema)
    assert any("risk_band" in p and "default" in p for p in problems)


def test_adding_a_field_with_default_is_accepted(base_schema: dict) -> None:
    new = copy.deepcopy(base_schema)
    new["fields"].append({"name": "risk_band", "type": "string", "default": "UNKNOWN"})
    assert check_backward_compatible(new, base_schema) == []


def test_widening_int_to_long_is_accepted(base_schema: dict) -> None:
    new = copy.deepcopy(base_schema)
    for field in new["fields"]:
        if field["name"] == "schema_version":
            field["type"] = "long"
    assert check_backward_compatible(new, base_schema) == []


def test_narrowing_long_to_int_is_rejected(base_schema: dict) -> None:
    new = copy.deepcopy(base_schema)
    for field in new["fields"]:
        if field["name"] == "amount_cents":
            field["type"] = "int"
    problems = check_backward_compatible(new, base_schema)
    assert any("amount_cents" in p for p in problems)


def test_dropping_an_enum_symbol_is_rejected(base_schema: dict) -> None:
    """Removing UNAVAILABLE would make every historical record carrying it undecodable."""
    new = copy.deepcopy(base_schema)
    for field in new["fields"]:
        if field["name"] == "vop_result":
            field["type"]["symbols"] = [s for s in field["type"]["symbols"] if s != "UNAVAILABLE"]
    problems = check_backward_compatible(new, base_schema)
    assert any("UNAVAILABLE" in p for p in problems)


def test_removing_a_field_is_rejected(base_schema: dict) -> None:
    new = copy.deepcopy(base_schema)
    new["fields"] = [f for f in new["fields"] if f["name"] != "vop_overridden"]
    problems = check_backward_compatible(new, base_schema)
    assert any("vop_overridden" in p for p in problems)


def test_assert_raises_with_every_reason_at_once(base_schema: dict) -> None:
    new = copy.deepcopy(base_schema)
    new["fields"].append({"name": "a", "type": "string"})
    new["fields"].append({"name": "b", "type": "string"})
    with pytest.raises(IncompatibleSchemaError) as excinfo:
        assert_backward_compatible(new, base_schema)
    message = str(excinfo.value)
    assert "'a'" in message and "'b'" in message, "report all problems, not just the first"


def test_old_records_remain_readable_under_the_new_schema(sample_event: SctInstEvent) -> None:
    """End-to-end proof of the gate: encode with v1, decode with an evolved v1.

    This is the behaviour the structural rules are a proxy for, so we check the
    behaviour too rather than trusting the proxy alone.
    """
    import fastavro

    old = load_schema_dict("sct_inst_event.v1")
    new = copy.deepcopy(old)
    new["fields"].append({"name": "risk_band", "type": "string", "default": "UNKNOWN"})
    assert_backward_compatible(new, old)

    import io

    buffer = io.BytesIO()
    fastavro.schemaless_writer(
        buffer, fastavro.parse_schema(old), sample_event.model_dump(mode="json")
    )
    buffer.seek(0)
    restored = fastavro.schemaless_reader(
        buffer, fastavro.parse_schema(old), fastavro.parse_schema(new)
    )
    assert restored["risk_band"] == "UNKNOWN"
    assert restored["tx_id"] == sample_event.tx_id
