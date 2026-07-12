"""Avro consumer.

Decodes by resolving each message's schema id against the registry, rather than
assuming every message was written with the schema this process happens to have
compiled in. That distinction is the whole reason a registry exists: during a
rolling upgrade the topic contains both versions at once, and a consumer that
assumes one of them fails on the other.

Offsets are committed **manually, after** the batch has been durably handled.
Auto-commit acknowledges messages the moment they are polled, so a crash between
poll and write loses data silently — which for a bronze layer that is supposed
to be the audit substrate would be disqualifying.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from confluent_kafka import Consumer as _KafkaConsumer
from confluent_kafka import KafkaError, KafkaException, TopicPartition

from kg_core.config import get_settings
from kg_core.contracts import load_schema
from kg_core.logging import get_logger
from kg_stream.registry import SchemaRegistry
from kg_stream.wire import WireFormatError, decode, peek_schema_id

log = get_logger(__name__)


@dataclass(slots=True)
class ConsumedMessage:
    """One decoded record plus the Kafka coordinates it arrived on."""

    value: dict[str, Any]
    key: str | None
    topic: str
    partition: int
    offset: int
    timestamp_ms: int
    schema_id: int


@dataclass(slots=True)
class ConsumeStats:
    consumed: int = 0
    poison: int = 0
    poison_samples: list[str] = field(default_factory=list)


class AvroConsumer:
    """Consumes Confluent-framed Avro, resolving writer schemas by id."""

    def __init__(
        self,
        contract: str,
        topics: list[str],
        group_id: str,
        *,
        bootstrap: str | None = None,
        registry_url: str | None = None,
        from_beginning: bool = True,
        skip_poison: bool = True,
    ) -> None:
        settings = get_settings()
        self.reader_schema = load_schema(contract)
        self.topics = topics
        self.skip_poison = skip_poison
        self._registry = SchemaRegistry(registry_url or settings.schema_registry_url)
        self._consumer = _KafkaConsumer(
            {
                "bootstrap.servers": bootstrap or settings.kafka_bootstrap,
                "group.id": group_id,
                "auto.offset.reset": "earliest" if from_beginning else "latest",
                # Manual commit only: an offset is acknowledged after the data
                # has actually been written, never merely because it was read.
                "enable.auto.commit": False,
                "client.id": "kontograph-consumer",
            }
        )
        self._consumer.subscribe(topics)
        self.stats = ConsumeStats()

    def __enter__(self) -> AvroConsumer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _decode(self, raw: bytes) -> tuple[dict[str, Any], int]:
        schema_id = peek_schema_id(raw)
        writer_schema = self._registry.schema_by_id(schema_id)
        return decode(raw, self.reader_schema, writer_schema), schema_id

    def poll_batch(self, max_messages: int = 1_000, timeout: float = 2.0) -> list[ConsumedMessage]:
        """Poll up to *max_messages*, returning what arrived within *timeout*."""
        batch: list[ConsumedMessage] = []
        while len(batch) < max_messages:
            message = self._consumer.poll(timeout if not batch else 0.1)
            if message is None:
                break
            error = message.error()
            if error is not None:
                if error.code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(error)

            raw = message.value()
            if raw is None:
                # A tombstone: a null value against a key. Meaningful in a
                # compacted topic, meaningless on an append-only payment log,
                # so it is counted rather than silently skipped.
                self.stats.poison += 1
                continue

            try:
                value, schema_id = self._decode(raw)
            except (WireFormatError, ValueError) as error:
                # A message we cannot decode must not stall the pipeline, but it
                # must also never be silently discarded: it is counted, sampled
                # and logged so the data-quality report can surface it.
                self.stats.poison += 1
                if len(self.stats.poison_samples) < 5:
                    self.stats.poison_samples.append(
                        f"{message.topic()}/{message.partition()}@{message.offset()}: {error}"
                    )
                log.warning(
                    "poison message",
                    topic=message.topic(),
                    partition=message.partition(),
                    offset=message.offset(),
                    error=str(error),
                )
                if self.skip_poison:
                    continue
                raise

            key = message.key()
            batch.append(
                ConsumedMessage(
                    value=value,
                    key=key.decode() if isinstance(key, bytes) else key,
                    topic=message.topic() or "",
                    partition=message.partition() or 0,
                    offset=message.offset() or 0,
                    timestamp_ms=message.timestamp()[1],
                    schema_id=schema_id,
                )
            )
            self.stats.consumed += 1
        return batch

    def stream(
        self, max_messages: int | None = None, idle_timeout: float = 5.0
    ) -> Iterator[ConsumedMessage]:
        """Yield messages until *max_messages*, or until the topic goes idle."""
        seen = 0
        while max_messages is None or seen < max_messages:
            batch = self.poll_batch(
                max_messages=1_000 if max_messages is None else min(1_000, max_messages - seen),
                timeout=idle_timeout,
            )
            if not batch:
                break
            for message in batch:
                yield message
                seen += 1

    def commit(self, message: ConsumedMessage | None = None) -> None:
        """Commit offsets — only after the data is durably handled."""
        if message is None:
            self._consumer.commit(asynchronous=False)
        else:
            self._consumer.commit(
                offsets=[TopicPartition(message.topic, message.partition, message.offset + 1)],
                asynchronous=False,
            )

    def close(self) -> None:
        self._consumer.close()
        self._registry.close()
