"""Avro producer.

Two decisions here are load-bearing for everything downstream.

**Partitioning by debtor account.** The message key is the debtor's account id,
so every payment from one account lands on one partition and is therefore
consumed in the order it was produced. Kafka guarantees ordering *per partition*
only, and the entire point-in-time story depends on per-account ordering — a
feature like "count of this account's transfers in the last hour" is meaningless
if two of that account's events can be processed out of order. Partitioning by
``tx_id`` would spread them and quietly break the invariant.

**Event time is carried explicitly.** The Kafka message timestamp is set to the
payment's ``acceptance_ts``, not to wall clock at produce time. Otherwise a
replay of historical data would stamp every event with today's date, and any
window computed from message time would be wrong — the classic way a backfill
silently stops matching the stream.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from confluent_kafka import Producer as _KafkaProducer

from kg_core.config import get_settings
from kg_core.contracts import load_schema
from kg_core.logging import get_logger
from kg_stream.registry import SchemaRegistry
from kg_stream.wire import encode

log = get_logger(__name__)


@dataclass(slots=True)
class ProduceStats:
    """Delivery accounting for one producer session."""

    produced: int = 0
    delivered: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.delivered == self.produced


class AvroProducer:
    """Produces contract records as Confluent-framed Avro."""

    def __init__(
        self,
        contract: str,
        topic: str,
        *,
        bootstrap: str | None = None,
        registry_url: str | None = None,
        key_field: str = "debtor_account_id",
        timestamp_field: str = "acceptance_ts",
        linger_ms: int = 20,
    ) -> None:
        settings = get_settings()
        self.contract = contract
        self.topic = topic
        self.key_field = key_field
        self.timestamp_field = timestamp_field
        self.schema = load_schema(contract)

        self._registry = SchemaRegistry(registry_url or settings.schema_registry_url)
        registered = self._registry.register_contract(contract)
        self.schema_id = registered.schema_id

        self._producer = _KafkaProducer(
            {
                "bootstrap.servers": bootstrap or settings.kafka_bootstrap,
                # Ordering and durability. Without idempotence a retry can
                # reorder messages within a partition, which would break the
                # per-account ordering the feature layer depends on.
                "enable.idempotence": True,
                "acks": "all",
                "linger.ms": linger_ms,
                "compression.type": "zstd",
                "client.id": "kontograph-producer",
            }
        )
        self.stats = ProduceStats()

    def __enter__(self) -> AvroProducer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _on_delivery(self, error: Any, message: Any) -> None:
        if error is not None:
            self.stats.failed += 1
            if len(self.stats.errors) < 10:
                self.stats.errors.append(str(error))
        else:
            self.stats.delivered += 1

    def produce(self, record: dict[str, Any]) -> None:
        """Produce one record. Non-blocking; call :meth:`flush` to settle."""
        payload = encode(self.schema, self.schema_id, record)
        key = str(record.get(self.key_field, "")).encode()
        timestamp = int(record.get(self.timestamp_field, 0))

        while True:
            try:
                self._producer.produce(
                    topic=self.topic,
                    key=key,
                    value=payload,
                    timestamp=timestamp,
                    on_delivery=self._on_delivery,
                )
                break
            except BufferError:
                # The local queue is full: serve delivery callbacks and retry
                # rather than dropping. Dropping here would silently truncate a
                # backfill, which is the kind of data loss nobody notices until
                # a metric looks slightly wrong months later.
                self._producer.poll(0.5)
        self.stats.produced += 1
        self._producer.poll(0)

    def produce_many(self, records: Iterable[dict[str, Any]]) -> ProduceStats:
        for record in records:
            self.produce(record)
        return self.flush()

    def flush(self, timeout: float = 30.0) -> ProduceStats:
        """Block until every outstanding message is delivered or has failed."""
        remaining = self._producer.flush(timeout)
        if remaining:
            log.warning("flush timed out", undelivered=remaining, topic=self.topic)
        return self.stats

    def close(self) -> None:
        self.flush()
        self._registry.close()


def stream_dataset_events(dataset_dir: Any, batch_size: int = 10_000) -> Iterator[dict[str, Any]]:
    """Yield events from a generated dataset in event-time order.

    Reads in row-group batches rather than materialising the whole table: the
    full profile is over half a million rows, and a producer should not need to
    hold the dataset in memory to send it.
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(dataset_dir / "events.parquet")
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()
