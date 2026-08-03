# KONTOGRAPH
#
# Design principle P10: "`make up` works. One command, full stack, local."
#
# Every target here is a thin shim over `uv run kg ...` or docker compose.
# The real logic lives in the kg CLI so that Windows (via make.ps1), WSL, and
# CI all execute the identical code path — a Makefile that only works on the
# maintainer's machine is the failure mode this indirection exists to avoid.

SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE := docker compose -f docker/compose.yml
UV := uv run

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------- environment --

.PHONY: setup
setup: ## Install the workspace into .venv
	uv sync --all-packages
	@echo "Run 'make doctor' to check the environment."

.PHONY: doctor
doctor: ## Check this machine can run the stack
	$(UV) kg doctor

.PHONY: clean
clean: ## Remove generated data and artifacts (keeps images)
	rm -rf data artifacts .pytest_cache .ruff_cache .mypy_cache .hypothesis
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ------------------------------------------------------------------ stack --

.PHONY: up
up: ## Start the local stack (Redpanda, object store, Redis, MLflow)
	$(COMPOSE) up -d
	@echo ""
	@echo "  Redpanda console  http://localhost:8080"
	@echo "  Object store      http://localhost:8888  (S3 API on :9000, kontograph / kontograph)"
	@echo "  MLflow            http://localhost:5000"

.PHONY: down
down: ## Stop the stack, keeping volumes
	$(COMPOSE) down

.PHONY: nuke
nuke: ## Stop the stack and delete its volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail stack logs
	$(COMPOSE) logs -f --tail=100

.PHONY: ps
ps: ## Show stack status
	$(COMPOSE) ps

# Flink is a separate profile: it is the specified streaming engine,
# but a JVM cluster is heavy on a laptop. See ADR-0002.
.PHONY: flink-up
flink-up: ## Start the optional Flink profile
	docker compose -f docker/compose.yml -f docker/compose.flink.yml --profile flink up -d

.PHONY: obs-up
obs-up: ## Start the optional observability profile (Prometheus + Grafana)
	docker compose -f docker/compose.yml -f docker/compose.obs.yml --profile obs up -d

# ------------------------------------------------------------------- code --

.PHONY: test
test: ## Run the full test suite
	$(UV) pytest

.PHONY: test-fast
test-fast: ## Run tests, skipping anything needing the stack
	$(UV) pytest -m "not integration and not slow"

.PHONY: lint
lint: ## Lint without modifying files
	$(UV) ruff check .
	$(UV) ruff format --check .

.PHONY: fmt
fmt: ## Auto-format and fix
	$(UV) ruff format .
	$(UV) ruff check --fix .

.PHONY: typecheck
typecheck: ## Static type check
	$(UV) mypy packages

.PHONY: contracts
contracts: ## Fail if a contract change would break existing consumers
	$(UV) kg contracts check

.PHONY: ci
ci: lint typecheck contracts test ## Everything CI runs

# ---------------------------------------------------------------- pipeline --

.PHONY: sim
sim: ## Generate the synthetic payments dataset
	$(UV) kg sim generate

.PHONY: register
register: ## Register the Avro contracts with the Schema Registry
	$(UV) kg stream register

.PHONY: replay
replay: ## Replay the dataset onto the payments topic
	$(UV) kg stream replay

.PHONY: ingest
ingest: ## Load payments into the bronze Iceberg layer
	$(UV) kg lake ingest

.PHONY: promote
promote: ## Promote bronze to silver through the quality gate
	$(UV) kg lake promote

.PHONY: marts
marts: ## Export silver to DuckDB and build the gold marts with dbt
	$(UV) kg lake export
	$(UV) kg lake dbt build

.PHONY: features
features: ## Build the training feature table from silver
	$(UV) kg features build

.PHONY: verify-pit
verify-pit: ## Prove the backends agree and nothing sees the future
	$(UV) kg features verify

# The whole pipeline, in order. This is the target that demonstrates the
# streaming spine end to end.
.PHONY: pipeline
pipeline: sim ingest promote marts features ## Simulator -> bronze -> silver -> gold -> features

.PHONY: dagster
dagster: ## Open the Dagster asset graph (needs the orchestration extra)
	$(UV) dagster dev -f orchestration/dagster/definitions.py

.PHONY: eval
eval: ## Regenerate every published number from artifacts
	$(UV) kg eval report

.PHONY: serve
serve: ## Run the decision gateway locally
	$(UV) kg serve run

.PHONY: loadtest
loadtest: ## k6 load test; fails if p99 exceeds the SLA
	$(UV) kg serve loadtest

.PHONY: govern
govern: ## Regenerate model cards, lineage and the decision-log audit
	$(UV) kg gov all
