# ADR-0001 — Redpanda as the log, not Kafka + Confluent Schema Registry

**Status:** accepted · **Date:** 2026-07-02

## Context

The platform needs a durable, partitioned, replayable log with a schema registry in front of it.
The platform scope specifies "Kafka / Redpanda". The target machine is a 4-core
laptop with ~32 GB free disk, and `make up` has to finish fast enough that a developer actually waits
for it.

A conventional Kafka setup is Kafka + (ZooKeeper or KRaft) + Schema Registry + a UI: three to four
JVM containers, roughly 2.5 GB of images, and about 4 GB of RAM before a single message is produced.

## Decision

Use **Redpanda** as the log. It speaks the Kafka wire protocol, so every client library, every
`rpk`/`kafka-*` command, and every piece of application code is unchanged if this is later swapped
for Kafka. It ships the Schema Registry **in the same process**, which removes a container and the
class of failure where the registry is up but the broker is not.

## Consequences

**Good.** One container instead of three or four. ~1.6 GB total stack. Sub-30-second cold start.
No JVM tuning. `rpk` makes the topic-creation init container trivial and readable.

**Bad.** Redpanda's Schema Registry has historically trailed Confluent's on the more exotic
compatibility modes. We do not depend on that: the backward-compatibility gate is implemented in
`kg_core.contracts.registry` and runs in CI *without a broker at all*, so the registry is a runtime
convenience rather than the enforcement point. That was worth doing anyway — a compatibility rule
that only exists as a setting in a running service is not reviewable in a pull request.

**Bad.** Redpanda is a single vendor's C++ implementation. Mitigated by the wire compatibility:
nothing in this repository imports a Redpanda-specific API.

## What would change our mind

- Needing Kafka Streams or a Kafka-Connect ecosystem plugin with no Redpanda equivalent.
- Deploying into an organisation that already operates Kafka — then matching their platform beats
  saving 1 GB on a laptop.
- Needing tiered storage or multi-region replication semantics that differ between the two.
