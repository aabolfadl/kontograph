"""Iceberg catalog.

A SQLite-backed SQL catalog over object storage (ADR-0004): real Iceberg tables
— snapshots, schema evolution, time travel — with zero extra containers.

Catalog construction is isolated in this one module and driven entirely by
configuration, so moving to a REST catalog later is a config change plus a
container, not a code change. That isolation is the whole reason the ADR can
claim the decision is reversible.
"""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path
from typing import Any

from pyiceberg.catalog import Catalog
from pyiceberg.catalog.sql import SqlCatalog

from kg_core.config import get_settings
from kg_core.logging import get_logger

log = get_logger(__name__)

CATALOG_NAME = "kontograph"

#: The medallion layers, as Iceberg namespaces.
#:
#: bronze  raw events exactly as they arrived, append-only, never rewritten.
#:         This is the audit substrate: if it can be edited, the lineage report
#:         is decoration.
#: silver  validated, deduplicated, conformed. Quality gates run on the
#:         promotion into this layer, so silver is the first table anything is
#:         allowed to trust.
#: gold    serving and regulatory marts, built by dbt.
NAMESPACES = ("bronze", "silver", "gold")


def catalog_properties(settings: Any = None) -> dict[str, str]:
    """Connection properties for the Iceberg catalog."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    catalog_path = settings.iceberg_catalog_path
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "uri": settings.iceberg_catalog_uri or f"sqlite:///{catalog_path.as_posix()}",
        "warehouse": f"s3://{settings.s3_bucket}/warehouse",
        "s3.endpoint": settings.s3_endpoint,
        "s3.access-key-id": settings.s3_access_key,
        "s3.secret-access-key": settings.s3_secret_key,
        # The local object store requires path-style addressing; virtual-host style would resolve
        # bucket.localhost and fail with a DNS error that says nothing useful.
        "s3.path-style-access": "true",
        "s3.region": "us-east-1",
    }


@functools.lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """The process-wide Iceberg catalog, with namespaces ensured."""
    catalog = SqlCatalog(CATALOG_NAME, **catalog_properties())
    for namespace in NAMESPACES:
        try:
            catalog.create_namespace(namespace)
            log.info("created namespace", namespace=namespace)
        except Exception:
            pass
    return catalog


def reset_catalog_cache() -> None:
    """Drop the cached catalog. Used by tests that point at a temp warehouse."""
    get_catalog.cache_clear()


def local_catalog(root: Path) -> Catalog:
    """A catalog backed entirely by the local filesystem.

    Used by tests and by anyone who wants the lakehouse without the object store running.
    Same Iceberg semantics, no S3 — which keeps the table code under test
    without the integration marker.
    """
    root.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "kontograph-local",
        **{
            "uri": f"sqlite:///{(root / 'catalog.db').as_posix()}",
            # "file://" + a POSIX path, deliberately, and correct on both
            # platforms: on Linux the POSIX path already begins with "/",
            # giving the usual file:///tmp/..., and on Windows it gives
            # file://C:/Users/... which is what pyarrow's resolver accepts.
            #
            # Path.as_uri() does NOT work here: on Windows it emits
            # file:///C:/... (three slashes), which pyarrow strips to the
            # invalid path /C:/... and fails with WinError 123. A bare native
            # path fails differently -- the drive letter is parsed as a URI
            # scheme ("Unrecognized filesystem type in URI: c").
            "warehouse": "file://" + root.as_posix(),
        },
    )
    for namespace in NAMESPACES:
        # Already-exists is the normal case after the first run.
        with contextlib.suppress(Exception):
            catalog.create_namespace(namespace)
    return catalog
