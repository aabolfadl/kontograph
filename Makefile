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
