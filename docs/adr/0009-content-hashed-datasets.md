# ADR-0009 — Content-hashed dataset manifests, not DVC

**Status:** accepted · **Date:** 2026-07-08

## Context

The platform scope names "DVC-tracked datasets" under MLOps. The requirement underneath it is real: every
result must be traceable to the exact data that produced it (the engineering invariants, "no unreproducible number").

## Decision

Identify datasets by a **content hash of their contents**, recorded in a manifest committed to git.
No DVC, no separate remote, no `.dvc` files.

The manifest records, per dataset: the generator's config fingerprint, the seed, the row count, the
per-file digests, the directory digest, and the code commit. `kg_core.hashing.directory_digest`
computes it.

## Rationale

**Our data is generated, not collected.** This is the decisive asymmetry. A dataset here is a pure
function of `(simulator config, seed, code commit)`. Storing the *bytes* is redundant when the
*recipe* reproduces them exactly — and the recipe is three orders of magnitude smaller. `make sim`
regenerates a byte-identical dataset from a 2 KB config.

**DVC's core value is versioning data you cannot regenerate.** That value is genuine for scraped,
purchased or hand-labelled data. It is close to zero for deterministic simulator output, and it
costs a remote to host, a cache to keep in sync, and a second mental model of "what is checked in".

**Content addressing is what DVC does anyway.** We keep that idea and drop the tooling around it:
a digest that verifies whether the data on disk is the data a result was computed from.

**Fewer moving parts in `make up`.** Anyone cloning the repo runs one command and gets the data.
No `dvc pull`, no remote credentials, no cache-corruption path.

## Consequences

**Good.** Reproduction needs only the repository. The manifest is human-readable and diffs cleanly.

**Bad.** No `dvc checkout` to swap dataset versions in place; switching means regenerating, which
takes minutes rather than seconds.

**Bad.** If the simulator ever becomes non-deterministic — an unseeded RNG, a dict-ordering
dependency, a parallelism race — the recipe stops reproducing the bytes and this decision silently
breaks. That risk is met head-on with a determinism test: generate twice, assert identical directory
digests. It runs in CI, and it is the single test that keeps this ADR honest.

## What would change our mind

- Introducing any real, non-regenerable data — the moment that happens, DVC (or LakeFS, or Iceberg
  snapshots used as the dataset boundary) becomes the right answer immediately.
- Generation becoming slow enough that regenerating is worse than storing.
