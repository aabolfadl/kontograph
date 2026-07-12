"""Schema Registry client.

Producer-side schema ownership (the platform scope): the producer registers the
contract it writes, the registry assigns an id, and that id travels in every
message. A consumer years later resolves the id back to the exact schema the
data was written with — which is what makes an Iceberg partition from 2025
readable by code from 2027.

The registry is the *runtime* half of the story. The compatibility **rules**
live in ``kg_core.contracts.registry`` and are enforced in CI without a broker,
so a breaking change is caught in a pull request rather than at 3 a.m. when a
consumer starts failing. This client additionally sets the subject's
compatibility level on registration, so the same rule is enforced server-side
by anyone who bypasses CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

import httpx

from kg_core.contracts import load_schema_dict, subject_for
from kg_core.logging import get_logger

log = get_logger(__name__)

#: Confluent's registry content type. Redpanda accepts it too.
CONTENT_TYPE: Final = "application/vnd.schemaregistry.v1+json"

#: BACKWARD: a consumer on the new schema must read data written under the old.
#: The right default for a payments tap, where producers upgrade last and
#: historical partitions must stay readable forever.
COMPATIBILITY_LEVEL: Final = "BACKWARD"


class SchemaRegistryError(RuntimeError):
    """Raised when the registry rejects a request."""


@dataclass(frozen=True, slots=True)
class RegisteredSchema:
    """A schema as the registry knows it."""

    subject: str
    schema_id: int
    version: int


class SchemaRegistry:
    """Thin, explicit client over the Confluent-compatible REST API."""

    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": CONTENT_TYPE, "Accept": CONTENT_TYPE},
        )
        # id -> parsed schema. A message's schema id is immutable, so this can
        # be cached for the life of the process without staleness risk.
        self._by_id: dict[int, Any] = {}

    def __enter__(self) -> SchemaRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------ requests --

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise SchemaRegistryError(
                f"cannot reach the Schema Registry at {self.base_url}: {error}. "
                "Is the stack up? (`make up`)"
            ) from error
        if response.status_code >= 400:
            # The registry returns a structured error; surfacing its message is
            # the difference between a five-second fix and an afternoon.
            try:
                detail = response.json()
                message = detail.get("message", response.text)
                code = detail.get("error_code")
            except (json.JSONDecodeError, ValueError):
                message, code = response.text, None
            raise SchemaRegistryError(
                f"{method} {path} -> {response.status_code}"
                f"{f' (error_code {code})' if code else ''}: {message}"
            )
        return response.json() if response.content else None

    # -------------------------------------------------------------- health --

    def ping(self) -> bool:
        """True if the registry answers."""
        try:
            self._request("GET", "/subjects")
        except SchemaRegistryError:
            return False
        return True

    def subjects(self) -> list[str]:
        return list(self._request("GET", "/subjects") or [])

    # --------------------------------------------------------- registration --

    def set_compatibility(self, subject: str, level: str = COMPATIBILITY_LEVEL) -> None:
        """Pin a subject's compatibility level server-side.

        Belt and braces alongside the CI gate: CI stops a breaking change from
        being merged, and this stops one from being registered by anything that
        did not go through CI.
        """
        self._request("PUT", f"/config/{subject}", json={"compatibility": level})

    def register(self, subject: str, schema: dict[str, Any]) -> int:
        """Register a schema under *subject* and return its id.

        Idempotent: registering an identical schema returns the existing id
        rather than creating a new version.
        """
        payload = {"schema": json.dumps(schema), "schemaType": "AVRO"}
        result = self._request("POST", f"/subjects/{subject}/versions", json=payload)
        return int(result["id"])

    def register_contract(self, name: str, *, set_compatibility: bool = True) -> RegisteredSchema:
        """Register one of our Avro contracts by stem, e.g. ``"sct_inst_event.v1"``."""
        subject = subject_for(name)
        schema = load_schema_dict(name)
        if set_compatibility:
            self.set_compatibility(subject)
        schema_id = self.register(subject, schema)
        version = self.latest_version(subject)
        log.info(
            "registered contract",
            contract=name,
            subject=subject,
            schema_id=schema_id,
            version=version,
        )
        return RegisteredSchema(subject=subject, schema_id=schema_id, version=version)

    def latest_version(self, subject: str) -> int:
        versions = self._request("GET", f"/subjects/{subject}/versions") or []
        return int(max(versions)) if versions else 0

    # -------------------------------------------------------------- lookup --

    def schema_by_id(self, schema_id: int) -> Any:
        """Fetch and parse the schema a message was written with.

        Cached: a schema id never changes meaning, so this is safe forever.
        """
        cached = self._by_id.get(schema_id)
        if cached is not None:
            return cached
        import fastavro

        result = self._request("GET", f"/schemas/ids/{schema_id}")
        parsed = fastavro.parse_schema(json.loads(result["schema"]))
        self._by_id[schema_id] = parsed
        return parsed

    def check_compatibility(self, subject: str, schema: dict[str, Any]) -> bool:
        """Ask the registry whether *schema* would be accepted for *subject*."""
        payload = {"schema": json.dumps(schema), "schemaType": "AVRO"}
        try:
            result = self._request(
                "POST", f"/compatibility/subjects/{subject}/versions/latest", json=payload
            )
        except SchemaRegistryError as error:
            # A subject with no versions yet is trivially compatible.
            if "40401" in str(error) or "not found" in str(error).lower():
                return True
            raise
        return bool(result.get("is_compatible", False))


def register_all_contracts(base_url: str) -> dict[str, RegisteredSchema]:
    """Register every contract in the repository. Idempotent."""
    from kg_core.contracts import list_schemas

    with SchemaRegistry(base_url) as registry:
        return {name: registry.register_contract(name) for name in list_schemas()}
