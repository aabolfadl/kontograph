"""Content hashing, config fingerprints, and the decision-log hash chain.

The engineering invariants: *"No unreproducible number."* Reproducibility needs an
identity function for inputs — that is what :func:`content_hash` is. A dataset,
a feature-set definition and a simulator config each carry a fingerprint, and a
result is only comparable to another result with the same fingerprints.

The hash chain at the bottom backs the GDPR Art. 22 decision log: an
append-only record where tampering with entry *n* invalidates every entry after
it, which is what makes "we logged the decision" an auditable claim rather than
an assertion.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import hashlib
import json
from pathlib import Path
from typing import Any, Final

DIGEST_BYTES: Final = 16  # 128-bit digests: short enough to eyeball, long enough to be safe
GENESIS: Final = "0" * (DIGEST_BYTES * 2)


def _canonical(obj: Any) -> Any:
    """Reduce *obj* to JSON-serialisable primitives, deterministically.

    Sets are sorted, floats that are exactly integral are normalised, and
    dataclasses/enums/paths get a stable representation — so that two logically
    identical configs never differ by an accident of construction order.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # 3.0 and 3 must fingerprint identically; NaN must not silently pass.
        if obj != obj or obj in (float("inf"), float("-inf")):
            return repr(obj)
        return int(obj) if obj.is_integer() else obj
    if isinstance(obj, (bytes, bytearray)):
        return hashlib.blake2b(bytes(obj), digest_size=DIGEST_BYTES).hexdigest()
    if isinstance(obj, enum.Enum):
        return _canonical(obj.value)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(dataclasses.asdict(obj))
    if hasattr(obj, "model_dump"):  # pydantic v2
        return _canonical(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (set, frozenset)):
        return sorted((_canonical(v) for v in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return repr(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(obj: Any, *, prefix: str = "") -> str:
    """Stable fingerprint of any structured object.

    ``prefix`` is a domain separator: fingerprints of a *config* and of a
    *dataset* that happen to serialise identically should still differ.
    """
    payload = f"{prefix}\x00{canonical_json(obj)}".encode()
    return hashlib.blake2b(payload, digest_size=DIGEST_BYTES).hexdigest()


def short_hash(obj: Any, *, length: int = 8, prefix: str = "") -> str:
    """A human-quotable fingerprint, for run names and artifact directories."""
    return content_hash(obj, prefix=prefix)[:length]


def file_digest(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming digest of a file, safe for datasets larger than memory."""
    digest = hashlib.blake2b(digest_size=DIGEST_BYTES)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: str | Path, *, glob: str = "**/*") -> str:
    """Digest of a directory's contents, independent of filesystem ordering.

    This is the replacement for DVC (see ADR-0009): a dataset is identified by
    what it contains, not by which tool wrote it.
    """
    root = Path(path)
    entries = sorted(
        (p.relative_to(root).as_posix(), file_digest(p))
        for p in sorted(root.glob(glob))
        if p.is_file()
    )
    return content_hash(entries, prefix="directory")


def chain_link(previous: str, payload: Any) -> str:
    """Next link in an append-only hash chain.

    ``previous`` is the prior entry's digest (:data:`GENESIS` for the first).
    Altering any earlier payload changes every subsequent link, so the chain
    detects retro-active edits to the decision log.
    """
    return hashlib.blake2b(
        f"{previous}\x00{canonical_json(payload)}".encode(), digest_size=DIGEST_BYTES
    ).hexdigest()


def verify_chain(
    entries: list[dict[str, Any]],
    *,
    hash_key: str = "entry_hash",
    prev_key: str = "prev_hash",
    payload_key: str = "payload",
) -> int | None:
    """Verify a hash chain; return the index of the first broken entry, or None.

    Returning the index rather than a bool is deliberate — an auditor needs to
    know *where* the chain broke, which bounds the blast radius of the incident.
    """
    previous = GENESIS
    for index, entry in enumerate(entries):
        if entry.get(prev_key) != previous:
            return index
        expected = chain_link(previous, entry.get(payload_key))
        if entry.get(hash_key) != expected:
            return index
        previous = expected
    return None
