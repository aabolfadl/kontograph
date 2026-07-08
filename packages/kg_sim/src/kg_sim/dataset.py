"""Dataset persistence.

Writes a simulation to Parquet with **explicit** schemas. Explicit rather than
inferred, because inference silently changes types when data changes: an
all-null ``purpose_code`` column in a small profile infers as ``null`` rather
than ``string``, and every downstream reader then breaks on the full profile.
Pinning the schema makes the dataset's shape a property of the code.

Layout:

    data/<profile>-<fingerprint>/
        manifest.json        identity card: config fingerprint, counts, digests
        accounts.parquet     node table
        events.parquet       the payment stream (the edges)
        ground_truth.parquet edge-level typology membership  [oracle only]
        account_roles.parquet account-level membership        [oracle only]
        labels.parquet       late, incomplete, biased outcomes

The two oracle tables are the ones a real bank would never have. They exist so
the evaluation protocol can report both label-based and oracle metrics and
publish the gap between them — see design principle P7. **Nothing in the training path may read
them**, and the directory name marks them clearly for that reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from kg_core.hashing import directory_digest
from kg_sim.config import SimulationConfig
from kg_sim.generator import SimulationResult

#: Timestamps are stored as int64 epoch millis rather than pyarrow timestamps.
#: Arrow timestamps carry a timezone that some engines drop on round-trip, and a
#: silently localised event time is the exact leakage bug design principle P2 is about. An
#: int64 cannot be misinterpreted.
_TS = pa.int64()

EVENTS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int32()),
        ("tx_id", pa.string()),
        ("end_to_end_id", pa.string()),
        ("acceptance_ts", _TS),
        ("ingest_ts", _TS),
        ("debtor_account_id", pa.string()),
        ("debtor_iban", pa.string()),
        ("debtor_bic", pa.string()),
        ("debtor_name", pa.string()),
        ("debtor_country", pa.string()),
        ("creditor_account_id", pa.string()),
        ("creditor_iban", pa.string()),
        ("creditor_bic", pa.string()),
        ("creditor_name", pa.string()),
        ("creditor_country", pa.string()),
        ("creditor_name_claimed", pa.string()),
        ("vop_result", pa.string()),
        ("vop_name_similarity", pa.float64()),
        ("vop_overridden", pa.bool_()),
        ("amount_cents", pa.int64()),
        ("currency", pa.string()),
        ("purpose_code", pa.string()),
        ("remittance_info", pa.string()),
        ("channel", pa.string()),
        ("initiation_source", pa.string()),
        ("device_id", pa.string()),
        ("session_ip_prefix", pa.string()),
        ("is_new_beneficiary", pa.bool_()),
        ("settlement_status", pa.string()),
    ]
)

ACCOUNTS_SCHEMA = pa.schema(
    [
        ("account_id", pa.string()),
        ("iban", pa.string()),
        ("bic", pa.string()),
        ("holder_name", pa.string()),
        ("kind", pa.string()),
        ("bank_blz", pa.string()),
        ("country", pa.string()),
        ("opened_ts", _TS),
    ]
)

GROUND_TRUTH_SCHEMA = pa.schema(
    [
        ("tx_id", pa.string()),
        ("acceptance_ts", _TS),
        ("campaign_id", pa.string()),
        ("typology", pa.string()),
        ("src_account_id", pa.string()),
        ("dst_account_id", pa.string()),
        ("src_role", pa.string()),
        ("dst_role", pa.string()),
        ("hop", pa.int32()),
        ("is_evasive", pa.bool_()),
        ("category", pa.string()),
    ]
)

ACCOUNT_ROLES_SCHEMA = pa.schema(
    [
        ("account_id", pa.string()),
        ("campaign_id", pa.string()),
        ("typology", pa.string()),
        ("role", pa.string()),
        ("is_evasive", pa.bool_()),
        ("first_ts", _TS),
        ("last_ts", _TS),
        ("edge_count", pa.int32()),
    ]
)

LABELS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int32()),
        ("label_id", pa.string()),
        ("subject_type", pa.string()),
        ("subject_id", pa.string()),
        ("event_ts", _TS),
        ("decided_ts", _TS),
        ("label", pa.string()),
        ("source", pa.string()),
        ("sar_filed", pa.bool_()),
        ("confidence", pa.float64()),
        ("notes", pa.string()),
    ]
)

TABLES: dict[str, pa.Schema] = {
    "events": EVENTS_SCHEMA,
    "accounts": ACCOUNTS_SCHEMA,
    "ground_truth": GROUND_TRUTH_SCHEMA,
    "account_roles": ACCOUNT_ROLES_SCHEMA,
    "labels": LABELS_SCHEMA,
}

#: Tables a bank would not possess. Guarded by :func:`assert_no_oracle_access`.
ORACLE_TABLES: frozenset[str] = frozenset({"ground_truth", "account_roles"})


class OracleAccessError(RuntimeError):
    """Raised when training-path code tries to read an oracle table."""


def dataset_name(config: SimulationConfig, profile: str = "custom") -> str:
    """Directory name for a dataset: profile plus config fingerprint.

    The fingerprint is in the name so two datasets from different configs cannot
    silently overwrite each other, and so a stale directory is visibly stale.
    """
    return f"{profile}-{config.fingerprint[:12]}"


def _to_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build a table with the declared schema, filling absent columns with null."""
    columns = {field.name: [row.get(field.name) for row in rows] for field in schema}
    return pa.Table.from_pydict(columns, schema=schema)


def write_dataset(result: SimulationResult, root: Path, profile: str = "custom") -> Path:
    """Write a simulation to ``root/<profile>-<fingerprint>/`` and return the path."""
    target = root / dataset_name(result.config, profile)
    target.mkdir(parents=True, exist_ok=True)

    payloads = {
        "events": result.events,
        "accounts": result.accounts,
        "ground_truth": result.ground_truth,
        "account_roles": result.account_roles,
        "labels": result.labels,
    }
    for name, rows in payloads.items():
        pq.write_table(
            _to_table(rows, TABLES[name]),
            target / f"{name}.parquet",
            compression="zstd",
            # Sorted by event time already; a modest row-group size keeps
            # time-ranged reads cheap without exploding metadata.
            row_group_size=50_000,
        )

    manifest = dict(result.manifest)
    manifest["profile_name"] = profile
    manifest["files"] = sorted(p.name for p in target.glob("*.parquet"))
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Directory digest is computed *after* the manifest is written, then appended
    # as a sidecar -- writing it into manifest.json would change the file it is
    # computed over. This is the ADR-0009 dataset identity.
    (target / "DIGEST").write_text(
        directory_digest(target, glob="*.parquet") + "\n", encoding="utf-8"
    )
    return target


def read_table(path: Path, name: str, *, allow_oracle: bool = False) -> pa.Table:
    """Read one table from a dataset directory.

    Reading an oracle table requires ``allow_oracle=True``. This is a guard rail,
    not security: it exists so that using ground truth is a deliberate, greppable
    act rather than an accident in a notebook at midnight. Evaluation code passes
    it; training code must not.
    """
    if name in ORACLE_TABLES and not allow_oracle:
        raise OracleAccessError(
            f"{name!r} is oracle ground truth that a real institution would not have. "
            "It is available to the evaluation path only. If you genuinely need it "
            "for evaluation, pass allow_oracle=True and say so at the call site."
        )
    return pq.read_table(path / f"{name}.parquet")


def read_manifest(path: Path) -> dict[str, Any]:
    """Load a dataset's manifest."""
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def verify_digest(path: Path) -> bool:
    """True iff the Parquet files still hash to the recorded digest.

    Called before any evaluation that quotes a number, so a result can never be
    attributed to data that has since changed on disk.
    """
    recorded = (path / "DIGEST").read_text(encoding="utf-8").strip()
    return directory_digest(path, glob="*.parquet") == recorded


def find_latest(root: Path, profile: str | None = None) -> Path | None:
    """Most recently modified dataset directory under *root*."""
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    if profile:
        candidates = [p for p in candidates if p.name.startswith(f"{profile}-")]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)
