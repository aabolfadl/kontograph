"""Data-quality gates.

The platform scope asks for quality checks as **pipeline-blocking assertions**, not
as a report somebody reads on Friday. That word — blocking — is the design
constraint: a failing expectation must stop the promotion from bronze to silver,
which means it runs on every batch, in every process, in CI.

**Why this is not Great Expectations.** GE was installed and measured:
approximately 10 s to import warm, 62 s cold. Paid once per process, on every
Dagster asset materialisation and every CI job, for an assertion that takes
milliseconds to evaluate. That is the wrong shape for an inline gate, so the
blocking path is implemented here over Arrow — zero import cost, exact, and
testable without a DataContext. GE remains available behind the ``quality``
extra for the rich published report, where a slow import costs nothing and its
rendering earns its place. Both read the *same* suite defined below, so the two
cannot disagree about what the rules are. See ADR-0011.

Expectations carry a **severity**:

``BLOCKING``   the batch does not get promoted. Structural violations only:
               things that mean the data is wrong, not merely surprising.
``QUARANTINE`` the offending rows are diverted to ``bronze.quarantine`` and the
               rest proceed. Row-level defects.
``WARN``       recorded in the report, does not affect the pipeline. Used for
               distributional drift, which is information rather than error.

The severity split matters. Making everything blocking produces a pipeline that
halts on a 0.01 % anomaly and trains everyone to bypass the gate; making nothing
blocking produces a report nobody reads.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from kg_core.ids import iban_checksum_valid
from kg_core.logging import get_logger

log = get_logger(__name__)


class Severity(enum.StrEnum):
    BLOCKING = "BLOCKING"
    QUARANTINE = "QUARANTINE"
    WARN = "WARN"


@dataclass(frozen=True, slots=True)
class Expectation:
    """One named, severity-tagged assertion over an Arrow table.

    ``predicate`` returns a boolean mask of *valid* rows, so a failure knows
    exactly which rows offended — which is what makes quarantine possible and
    what makes a failure message actionable.
    """

    name: str
    description: str
    severity: Severity
    predicate: Callable[[pa.Table], pa.ChunkedArray | pa.Array]
    #: Fraction of rows allowed to fail before the expectation itself fails.
    #: Non-zero only where a small rate is genuinely expected and not an error.
    tolerance: float = 0.0


@dataclass(slots=True)
class ExpectationResult:
    name: str
    description: str
    severity: Severity
    total_rows: int
    failed_rows: int
    passed: bool
    sample_failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed_rows / self.total_rows if self.total_rows else 0.0


@dataclass(slots=True)
class ValidationReport:
    """The outcome of running a suite against one batch."""

    suite: str
    total_rows: int
    results: list[ExpectationResult] = field(default_factory=list)

    @property
    def blocking_failures(self) -> list[ExpectationResult]:
        return [r for r in self.results if not r.passed and r.severity is Severity.BLOCKING]

    @property
    def quarantine_failures(self) -> list[ExpectationResult]:
        return [r for r in self.results if not r.passed and r.severity is Severity.QUARANTINE]

    @property
    def warnings(self) -> list[ExpectationResult]:
        return [r for r in self.results if not r.passed and r.severity is Severity.WARN]

    @property
    def ok(self) -> bool:
        """True iff nothing blocking failed."""
        return not self.blocking_failures

    def summary(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "total_rows": self.total_rows,
            "expectations": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "blocking_failures": len(self.blocking_failures),
            "quarantine_failures": len(self.quarantine_failures),
            "warnings": len(self.warnings),
            "ok": self.ok,
            "results": [
                {
                    "name": r.name,
                    "severity": r.severity.value,
                    "passed": r.passed,
                    "failed_rows": r.failed_rows,
                    "failure_rate": round(r.failure_rate, 6),
                }
                for r in self.results
            ],
        }

    def raise_if_blocked(self) -> None:
        """Stop the pipeline if any blocking expectation failed."""
        if self.ok:
            return
        lines = [
            f"  - {r.name}: {r.description} "
            f"({r.failed_rows}/{r.total_rows} rows, {r.failure_rate:.2%})"
            for r in self.blocking_failures
        ]
        samples = self.blocking_failures[0].sample_failures[:3]
        raise DataQualityError(
            f"{len(self.blocking_failures)} blocking expectation(s) failed on suite "
            f"{self.suite!r}; the batch was not promoted:\n"
            + "\n".join(lines)
            + (
                f"\n  sample offending rows: {json.dumps(samples, default=str)[:500]}"
                if samples
                else ""
            )
        )


class DataQualityError(RuntimeError):
    """Raised when a blocking expectation fails. Halts the pipeline."""


# --------------------------------------------------------------- helpers ----


def _not_null(column: str) -> Callable[[pa.Table], Any]:
    return lambda table: pc.is_valid(table.column(column))


def _positive(column: str) -> Callable[[pa.Table], Any]:
    return lambda table: pc.greater(table.column(column), 0)


def _in_set(column: str, allowed: set[str]) -> Callable[[pa.Table], Any]:
    return lambda table: pc.is_in(table.column(column), value_set=pa.array(sorted(allowed)))


def _unique(column: str) -> Callable[[pa.Table], Any]:
    """Mask marking the first occurrence of each value as valid.

    Marking only duplicates-after-the-first as invalid means quarantine removes
    the redundant copies and keeps one — which is what "deduplicate" should do.
    """

    def predicate(table: pa.Table) -> pa.Array:
        seen: set[Any] = set()
        mask = []
        for value in table.column(column).to_pylist():
            mask.append(value not in seen)
            seen.add(value)
        return pa.array(mask, type=pa.bool_())

    return predicate


def _valid_iban(column: str) -> Callable[[pa.Table], Any]:
    def predicate(table: pa.Table) -> pa.Array:
        return pa.array(
            [
                value is not None and iban_checksum_valid(value)
                for value in table.column(column).to_pylist()
            ],
            type=pa.bool_(),
        )

    return predicate


# ------------------------------------------------------------- the suite ----

#: Values the contract permits. Kept in sync with the Avro enums by a test.
_VOP_RESULTS = {"MATCH", "CLOSE_MATCH", "NO_MATCH", "NOT_APPLICABLE", "UNAVAILABLE"}
_CHANNELS = {
    "ONLINE_BANKING",
    "MOBILE_APP",
    "POS",
    "STANDING_ORDER",
    "BATCH_UPLOAD",
    "OPEN_BANKING_API",
    "BRANCH",
}
_SETTLEMENT = {"ACCP", "ACSC", "RJCT", "PDNG"}

PAYMENTS_SUITE: list[Expectation] = [
    # ---- structural: the data is wrong if these fail ----------------------
    Expectation(
        "tx_id_present",
        "every payment carries a transaction reference",
        Severity.BLOCKING,
        _not_null("tx_id"),
    ),
    Expectation(
        "acceptance_ts_present",
        "every payment carries an event time",
        Severity.BLOCKING,
        _not_null("acceptance_ts"),
    ),
    Expectation(
        "amount_is_positive",
        "amounts are positive integer cents",
        Severity.BLOCKING,
        _positive("amount_cents"),
    ),
    Expectation(
        "ingest_not_before_acceptance",
        "an event cannot be observed before it happened; a negative lag means a clock fault",
        Severity.BLOCKING,
        lambda t: pc.greater_equal(t.column("ingest_ts"), t.column("acceptance_ts")),
    ),
    Expectation(
        "debtor_and_creditor_differ",
        "a payment to oneself is not a transfer between two parties",
        Severity.BLOCKING,
        lambda t: pc.not_equal(t.column("debtor_account_id"), t.column("creditor_account_id")),
    ),
    Expectation(
        "currency_is_euro",
        "the SEPA Instant scheme settles in euro only",
        Severity.BLOCKING,
        lambda t: pc.equal(t.column("currency"), "EUR"),
    ),
    # ---- enum domains: a value outside these means a contract drift -------
    Expectation(
        "vop_result_in_domain",
        "vop_result is one of the values the contract declares",
        Severity.BLOCKING,
        _in_set("vop_result", _VOP_RESULTS),
    ),
    Expectation(
        "channel_in_domain",
        "channel is one of the values the contract declares",
        Severity.BLOCKING,
        _in_set("channel", _CHANNELS),
    ),
    Expectation(
        "settlement_status_in_domain",
        "settlement_status is a valid pacs.002 status",
        Severity.BLOCKING,
        _in_set("settlement_status", _SETTLEMENT),
    ),
    # ---- row-level defects: quarantine the row, keep the batch -----------
    Expectation(
        "tx_id_unique",
        "duplicate transaction references are redundant deliveries, not new payments",
        Severity.QUARANTINE,
        _unique("tx_id"),
    ),
    Expectation(
        "debtor_iban_valid",
        "debtor IBAN passes the ISO 7064 MOD-97-10 checksum",
        Severity.QUARANTINE,
        _valid_iban("debtor_iban"),
    ),
    Expectation(
        "creditor_iban_valid",
        "creditor IBAN passes the ISO 7064 MOD-97-10 checksum",
        Severity.QUARANTINE,
        _valid_iban("creditor_iban"),
    ),
    Expectation(
        "vop_similarity_in_unit_interval",
        "the name-similarity score is a value in [0, 1]",
        Severity.QUARANTINE,
        lambda t: pc.and_(
            pc.greater_equal(t.column("vop_name_similarity"), 0.0),
            pc.less_equal(t.column("vop_name_similarity"), 1.0),
        ),
    ),
    # ---- distributional: information, not error --------------------------
    Expectation(
        "amounts_within_scheme_expectations",
        "amounts above EUR 100k were impossible under the pre-2025 SCT Inst cap; "
        "they are legal now, but a surge is worth knowing about",
        Severity.WARN,
        lambda t: pc.less_equal(t.column("amount_cents"), 100_000_00),
        tolerance=0.01,
    ),
    Expectation(
        "vop_mostly_matches",
        "a collapse in the VoP match rate usually means a registry outage, not a fraud wave",
        Severity.WARN,
        lambda t: pc.equal(t.column("vop_result"), "MATCH"),
        tolerance=0.35,
    ),
    Expectation(
        "ingest_lag_is_sub_second",
        "SCT Inst settles in ten seconds; a sustained ingest lag erodes the whole budget",
        Severity.WARN,
        lambda t: pc.less(pc.subtract(t.column("ingest_ts"), t.column("acceptance_ts")), 1000),
        tolerance=0.05,
    ),
]

SUITES: dict[str, list[Expectation]] = {"payments": PAYMENTS_SUITE}


# -------------------------------------------------------------- execution ---


def validate(
    table: pa.Table, suite: str = "payments", *, sample_failures: int = 3
) -> ValidationReport:
    """Run a suite against an Arrow table."""
    expectations = SUITES[suite]
    report = ValidationReport(suite=suite, total_rows=table.num_rows)

    for expectation in expectations:
        if table.num_rows == 0:
            report.results.append(
                ExpectationResult(
                    expectation.name, expectation.description, expectation.severity, 0, 0, True
                )
            )
            continue

        mask = expectation.predicate(table)
        # A null result is a failure: "unknown" is not "valid".
        valid = pc.fill_null(mask, False)
        failed = int(pc.sum(pc.invert(valid)).as_py() or 0)
        rate = failed / table.num_rows
        passed = rate <= expectation.tolerance

        samples: list[dict[str, Any]] = []
        if not passed and sample_failures:
            offending = table.filter(pc.invert(valid)).slice(0, sample_failures)
            keep = [
                c
                for c in ("tx_id", "acceptance_ts", "amount_cents", "currency")
                if c in offending.column_names
            ]
            samples = offending.select(keep).to_pylist() if keep else []

        report.results.append(
            ExpectationResult(
                expectation.name,
                expectation.description,
                expectation.severity,
                table.num_rows,
                failed,
                passed,
                samples,
            )
        )
    return report


def partition_valid_rows(
    table: pa.Table, suite: str = "payments"
) -> tuple[pa.Table, pa.Table, list[str]]:
    """Split a table into (clean, quarantined, reasons).

    Quarantine expectations are applied cumulatively: a row failing any of them
    is diverted. Blocking expectations are *not* applied here — they are checked
    by :func:`validate` and stop the promotion entirely.
    """
    if table.num_rows == 0:
        return table, table.slice(0, 0), []

    keep = pa.array([True] * table.num_rows, type=pa.bool_())
    reasons: list[str] = []
    for expectation in SUITES[suite]:
        if expectation.severity is not Severity.QUARANTINE:
            continue
        valid = pc.fill_null(expectation.predicate(table), False)
        if int(pc.sum(pc.invert(valid)).as_py() or 0):
            reasons.append(expectation.name)
        keep = pc.and_(keep, valid)

    return table.filter(keep), table.filter(pc.invert(keep)), reasons
