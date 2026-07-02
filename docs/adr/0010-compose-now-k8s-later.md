# ADR-0010 — Docker Compose as the deployment target; Kubernetes and Terraform deferred

**Status:** accepted · **Date:** 2026-07-02

## Context

The platform scope lists "Terraform + Kubernetes (k3d locally, Hetzner/AWS-ready)" as the
infrastructure layer. Design principle P10 requires that `make up` brings up the full stack with
one command on a single developer machine.

These pull in opposite directions on a 4-core laptop with ~32 GB free disk. A k3d cluster plus this
stack means a Kubernetes control plane, a registry, an ingress controller and per-service manifests
before the first payment is generated.

## Decision

- **Docker Compose is the runtime** for local development, CI integration tests and the demo.
  `make up` starts it; CI runs the identical file.
- **Kubernetes manifests and Terraform are deferred**, not written. There is no `infra/` directory,
  and nothing in this repository claims a cloud deployment.
- Optional components (Flink, Prometheus/Grafana) are separate Compose files behind profiles, so the
  default stack stays small.

## Rationale

**Applying Terraform against a real cloud is a recurring bill and a live security surface.** For a
single-node reference implementation, a leaked cloud credential or silently accruing charges is a
liability with no offsetting benefit.

**A stack that does not start is worth nothing.** Design principle P10 asks for a first run measured
in seconds. Compose starts in under a minute; k3d plus image loading does not.

**Compose and Kubernetes express the same topology here.** Services, ports, volumes, health checks
and dependencies map across almost mechanically at this scale. Kubernetes earns its complexity with
autoscaling, rolling deploys across nodes and multi-tenancy — none of which a single-node deployment
exercises.

**No unvalidated infrastructure code.** The alternative is a `terraform/` folder nobody has ever
applied, which the engineering invariants forbid: no claimed capability without an artifact.
Writing IaC that is never executed would produce code that looks finished and has never met reality.

## Consequences

**Good.** Fast, reliable local loop. No cloud spend. No credentials in CI. The integration job in CI
exercises exactly the topology developers run.

**Bad.** There is no production deployment path in the repository. Moving to a managed cluster
requires translating the Compose services into manifests, adding secrets management, and replacing
the SQLite Iceberg catalog (ADR-0004) with a shared one.

## What would change our mind

- A target environment that already operates Kubernetes: then write the manifests against it and
  validate them there, rather than against a local imitation.
- Needing to demonstrate autoscaling or rolling deploys, which Compose genuinely cannot show.
