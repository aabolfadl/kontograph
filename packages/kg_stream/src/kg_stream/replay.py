"""Dataset replay.

Turns a static generated dataset back into a live stream. This is what makes
the live console possible (design principle P10) — a stream of synthetic payments
flowing across a graph in real time — and it is also how the load test and the drift monitor get
a realistic arrival pattern instead of a uniform firehose.

Three modes, each answering a different question:

``as_fast_as_possible``
    Backfill. How quickly can the pipeline absorb history?

``wall_clock``
    Faithful replay at a speed multiplier, preserving the *relative* spacing
    between events. A 90-day dataset at 10 000x replays in about 13 minutes, and
    the payroll spike still arrives as a spike. This is the demo mode: the
    diurnal rhythm and the month-end burst are visible.

``constant_rate``
    A fixed events-per-second, for load testing where the arrival pattern should
    be controlled rather than realistic.

Event time is always the payment's own ``acceptance_ts``, never wall clock — see
the note in :mod:`kg_stream.producer` about why that matters.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from kg_core.logging import get_logger
from kg_core.timeutil import iso

log = get_logger(__name__)


class ReplayMode(enum.StrEnum):
    AS_FAST_AS_POSSIBLE = "as_fast_as_possible"
    WALL_CLOCK = "wall_clock"
    CONSTANT_RATE = "constant_rate"


@dataclass(slots=True)
class ReplayProgress:
    """Where a replay has got to."""

    sent: int
    event_ts: int
    elapsed_seconds: float

    @property
    def rate(self) -> float:
        return self.sent / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def pace(
    events: Iterable[dict[str, Any]],
    *,
    mode: ReplayMode = ReplayMode.AS_FAST_AS_POSSIBLE,
    speed: float = 10_000.0,
    rate_per_second: float = 500.0,
    timestamp_field: str = "acceptance_ts",
) -> Iterator[dict[str, Any]]:
    """Yield *events*, sleeping so they arrive according to *mode*.

    Pacing is deliberately separated from producing: the same function drives
    the Kafka producer, the load test, and any local consumer, so "replay at
    100x" means exactly one thing across the project.
    """
    started = time.perf_counter()
    first_event_ts: int | None = None
    interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0

    for index, event in enumerate(events):
        if mode is ReplayMode.WALL_CLOCK:
            event_ts = int(event[timestamp_field])
            if first_event_ts is None:
                first_event_ts = event_ts
            # Where this event *should* land on the wall clock, given the speed
            # multiplier. Computed against the first event rather than the
            # previous one so that sleep jitter cannot accumulate into drift.
            target = (event_ts - first_event_ts) / 1000.0 / speed
            delay = target - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
        elif mode is ReplayMode.CONSTANT_RATE and interval:
            target = index * interval
            delay = target - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
        yield event


def replay_to_kafka(
    events: Iterable[dict[str, Any]],
    contract: str = "sct_inst_event.v1",
    topic: str | None = None,
    *,
    mode: ReplayMode = ReplayMode.AS_FAST_AS_POSSIBLE,
    speed: float = 10_000.0,
    rate_per_second: float = 500.0,
    limit: int | None = None,
    progress_every: int = 25_000,
    on_progress: Any = None,
) -> ReplayProgress:
    """Replay events onto a Kafka topic and return final progress."""
    from kg_core.config import get_settings
    from kg_stream.producer import AvroProducer

    settings = get_settings()
    topic = topic or settings.topic_payments
    started = time.perf_counter()
    sent, last_ts = 0, 0

    with AvroProducer(contract, topic) as producer:
        source = pace(events, mode=mode, speed=speed, rate_per_second=rate_per_second)
        for event in source:
            producer.produce(event)
            sent += 1
            last_ts = int(event.get("acceptance_ts", last_ts))

            if progress_every and sent % progress_every == 0:
                progress = ReplayProgress(sent, last_ts, time.perf_counter() - started)
                log.info(
                    "replay progress",
                    sent=sent,
                    event_time=iso(last_ts),
                    rate=round(progress.rate, 1),
                )
                if on_progress is not None:
                    on_progress(progress)

            if limit is not None and sent >= limit:
                break

        stats = producer.flush()

    if not stats.ok:
        raise RuntimeError(
            f"replay produced {stats.produced} messages but only {stats.delivered} were "
            f"delivered ({stats.failed} failed): {stats.errors[:3]}"
        )
    return ReplayProgress(sent, last_ts, time.perf_counter() - started)
