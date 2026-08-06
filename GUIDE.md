# Developer guide

Everything you need to run KONTOGRAPH, plus a map of the codebase. The
[README](README.md) explains *what* the project is and why; this file explains
*how to operate it*.

---

## Contents

1. [Hardware reality](#hardware-reality)
2. [Install](#install)
3. [The `kg` command](#the-kg-command)
4. [Running the whole thing](#running-the-whole-thing)
5. [Training on a GPU](#training-on-a-gpu)
6. [Codebase map](#codebase-map)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## Hardware reality

This was built and runs end to end on a **laptop**:

| | |
|---|---|
| CPU | Intel Core i7-8550U, 4 cores / 8 threads |
| RAM | 16 GB |
| Free disk | ~32 GB |
| GPU | **none** |
| OS | Windows 10 |

Every design decision that looks unusual traces back to that row saying "none".
The graph neural network trains on a free Kaggle GPU; everything else — the
simulator, the streaming pipeline, the feature engine, the gradient-boosted
baseline, the serving gateway — runs on the laptop's CPU.

Disk is the binding constraint, not compute. The default stack targets roughly
5 GB of container images. Observability (Prometheus, Grafana) and Apache Flink
live behind optional Docker Compose *profiles* so they are not pulled unless
asked for.

**Minimum to run the interesting parts:** any machine with Python 3.11+ and
8 GB of RAM. Docker is only needed for the streaming spine; the simulator,
feature engine, training and serving all work without it.

---

## Install

### 1. Prerequisites

- **Python 3.11 or 3.12**
- **[uv](https://github.com/astral-sh/uv)** — a Rust-based Python package
  manager, roughly 10-100x faster than `pip`. It resolves and installs the whole
  workspace from a single lockfile.
- **Docker Desktop** (optional — only for Kafka, object storage, Redis, MLflow)

```powershell
# Windows
winget install --id=astral-sh.uv -e

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. The project

```bash
git clone <this repo> && cd Kontograph

# Windows (PowerShell)      # macOS / Linux / WSL
.\make.ps1 setup            make setup
.\make.ps1 doctor           make doctor
```

`setup` creates a virtual environment and installs all ten packages in editable
mode. `doctor` checks the two things that actually block people: free disk space
and whether the Docker daemon is running.

> **Why both `Makefile` and `make.ps1`?** `make` is not installed on a stock
> Windows machine. Both files are thin shims that delegate to the same `kg`
> command-line tool, so neither can drift from the other — the CLI is the single
> source of truth for behaviour.

### 3. Optional extras

Installed on demand, because each is heavy and most work does not need them:

```bash
uv sync --extra torch          # PyTorch + PyTorch Geometric (graph neural nets)
uv sync --extra kaggle         # the Kaggle API client
uv sync --extra quality        # Great Expectations (rich data-quality reports)
uv sync --extra orchestration  # Dagster + dbt
```

---

## The `kg` command

Everything is driven by one command-line tool, built with
**[Typer](https://typer.tiangolo.com/)** (a library that turns typed Python
functions into a CLI). Run any subcommand with `--help`.

| Group | What it does |
|---|---|
| `kg sim` | Generate the synthetic payments dataset |
| `kg stream` | Register data contracts, replay events onto Kafka |
| `kg lake` | Load into the data lake, promote through quality gates, build marts |
| `kg features` | List, compile and **verify** features across backends |
| `kg train` | Train the baseline model |
| `kg eval` | Regenerate every published number from stored artifacts |
| `kg serve` | Run the decision gateway, measure latency, export to ONNX |
| `kg gov` | Generate model cards, lineage, audit the decision log |
| `kg kaggle` | Push data and notebooks to a GPU, pull results back |

---

## Running the whole thing

### Step 1 — bring up the infrastructure (optional)

```bash
.\make.ps1 up        # or: make up
```

| Service | URL | Login |
|---|---|---|
| Redpanda Console (inspect Kafka topics) | <http://localhost:8080> | — |
| SeaweedFS filer (object storage; S3 API on port 9000) | <http://localhost:8888> | S3 keys `kontograph` / `kontograph` |
| MLflow (experiment tracking) | <http://localhost:5000> | — |

Skip this if you only want to train and serve — those paths do not need Docker.

### Step 2 — generate data

```bash
kg sim profiles                    # smoke (8k events) | dev (35k) | full (1.5M)
kg sim generate --profile full
kg sim determinism                 # generate twice, prove the bytes match
```

The `full` profile produces **1,562,860 payments across 6,576 accounts over 270
simulated days**, taking a few minutes on a laptop CPU.

The dataset is a pure function of `(config, seed, code commit)`, so it is
identified by a content hash and *regenerated* rather than stored in git.
`kg sim determinism` is the test that keeps that claim honest.

### Step 3 — run the pipeline

```bash
.\make.ps1 pipeline    # simulator -> bronze -> silver -> gold, in order
```

Or step by step, which is worth doing once to see what each stage is:

```bash
kg stream register     # register data contracts with the Schema Registry
kg stream replay       # replay the dataset onto the payments topic
kg lake ingest         # load into the bronze (raw) layer
kg lake promote        # promote to silver THROUGH THE QUALITY GATE
kg lake export         # materialise silver into DuckDB
kg lake dbt build      # build the gold business tables
kg lake status         # row counts and lake snapshot history
```

### Step 4 — prove there is no leakage

```bash
kg features list       # 29 features; which can stream, and which cannot
kg features verify     # backfill == online, and nothing sees the future
```

This is the check the project is proudest of. See
[the README](README.md#idea-2-every-leakage-bug-is-a-time-travel-bug) for what
it means.

### Step 5 — train, serve, and open the console

```bash
kg train baseline                  # LightGBM + calibration + cost-optimal threshold
kg eval report                     # the results table, from stored artifacts
kg serve smoke --events 2000       # measure the latency budget, no server needed
kg serve run                       # the gateway, on http://localhost:8000
```

Then open **<http://localhost:8000>** and click **Replay**.

### Step 6 — governance

```bash
kg gov all                         # regenerate the whole governance bundle
kg gov sar --alert <id>            # draft a bilingual suspicious-activity narrative
kg gov decision-log                # audit the decision log's tamper-evident chain
make loadtest                      # k6 load test; fails the run if p99 > 200 ms
```

---

## Training on a GPU

The graph neural network is the one part that wants a GPU. It trains on
**Kaggle**, which gives roughly 30 GPU-hours per week free.

### One-time setup

1. Create a Kaggle account and **phone-verify it** — this gates both GPU access
   *and* notebook internet access, and the notebooks need internet to install
   PyTorch Geometric.
2. Authenticate. Two mechanisms exist and **the newer one is required for
   uploads**:

   ```bash
   kaggle auth login       # OAuth, opens a browser. Do this one.
   ```

   An older `~/.kaggle/kaggle.json` API key still works for *reading* but is
   rejected for uploads with a bare `401 Unauthorized`. Worse, the Kaggle SDK
   checks the legacy key *before* OAuth, so a stale `kaggle.json` silently
   shadows a fresh login. `kg kaggle` works around this by minting an access
   token from your OAuth credentials, which the SDK checks first.

   The tell that you are on the wrong credential: `kg kaggle status` shows a
   **dataset count of zero**.

### Per training run

```bash
kg kaggle status                        # checks credentials by actually calling the API
kg kaggle push-data                     # upload the dataset (~123 MB)
kg kaggle push-kernel 01_train_tgn      # upload + run headless on a GPU
kg kaggle wait 01_train_tgn             # poll until it finishes
kg kaggle pull-artifacts 01_train_tgn   # scores + metrics come back
kg eval ingest-gpu 01_train_tgn         # recompute the metrics LOCALLY and merge
kg eval report --markdown               # regenerate the published tables
```

### The notebooks

| # | Notebook | Needs | Produces | Runtime (T4) |
|---|---|---|---|---|
| 1 | `01_train_tgn` | the dataset | the detector, rungs 3 and 4 | ~20 min |
| 2 | `02_distill_explainer` | dataset + notebook 1's output | the explainer, rung 5 | ~10 min |
| 3 | `03_ablations` | the dataset | the full comparison table | ~30 min |

Notebook 3 is independent of the other two and can run in parallel with 1.

### Use the T4 graphics card, not the P100

This is not a preference — **the P100 does not work.** It is compute capability
`sm_60`, and current Kaggle PyTorch images ship code compiled for `sm_70` and
above. The result is the worst kind of failure: `torch.cuda.is_available()`
returns `True`, memory allocations succeed, and the run dies minutes later at
the first real computation.

`kg kaggle push-kernel` pins `"machine_shape": "NvidiaTeslaT4"` by default, and
the notebooks check the device's capability on startup and fail immediately with
an explanation if it is wrong.

| Kaggle UI setting | Metadata value |
|---|---|
| GPU T4 x2 | `NvidiaTeslaT4` |
| GPU P100 | `NvidiaTeslaP100` |
| TPU VM v3-8 | `Tpu1VmV38` |

> **Pushing a notebook rewrites its settings.** Without an explicit
> `machine_shape`, a push resets the accelerator to P100 *and discards a T4
> chosen in the browser*. Kaggle also has no upload-without-running mode, so
> every push spends a run. Pin it, and only push when the notebook has actually
> changed.

---

## Codebase map

Ten Python packages in a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) — one
repository, many packages, one lockfile. Roughly 22,000 lines across 93 files.

```
packages/
  kg_core/      data contracts, configuration, bank identifiers, time, hashing
  kg_sim/       the agent-based payments simulator
  kg_stream/    Kafka producer/consumer and schema registry
  kg_lake/      the data lake: bronze/silver/gold + quality gates
  kg_features/  the feature language and its three compilers
  kg_models/    baseline model, graph model, explainer
  kg_eval/      splits, metrics, cost model, confidence intervals
  kg_serve/     the decision gateway and investigator console
  kg_gov/       model cards, lineage, suspicious-activity narratives
  kg_cli/       the `kg` command
```

### What lives where

| Package | Contains | Notable file |
|---|---|---|
| **kg_core** | Avro schemas, IBAN generation and validation, UTC time handling, content hashing | `ids.py` — mod-97 checksum, country-aware bank codes |
| **kg_sim** | Households, merchants, banks, the German economic calendar, five crime typologies | `typologies.py` — each with a standard *and* an evasive variant |
| **kg_stream** | Confluent wire format, ordered replay, manual offset commits | `producer.py` — keyed by debtor for per-partition ordering |
| **kg_lake** | Apache Iceberg tables, blocking/quarantine/warn expectations | `quality.py` — the gate that blocks a bad promotion |
| **kg_features** | Feature definitions → typed intermediate representation → three compilers | `ir.py` — window semantics stated once |
| **kg_models** | LightGBM, calibration, temporal graph network, explainer teacher and student | `explain/` — the amortisation |
| **kg_eval** | Chronological splits with purge/embargo, cost matrix, day-blocked bootstrap | `cost.py` — where the operating point comes from |
| **kg_serve** | FastAPI gateway, latency budget, tamper-evident decision log, the console | `console/index.html` — one self-contained file |
| **kg_gov** | Jinja-generated model cards, lineage diagrams, bilingual narratives | `sar.py` — never claims evidence it lacks |

### Elsewhere

```
kaggle/          three GPU notebooks + their runbook
docs/
  adr/           11 architecture decision records
  writeup/       the explainability essay
  model-cards/   generated, not written
  evaluation-protocol.md   frozen before any model existed
docker/          Compose files (default, Flink profile, observability profile)
dbt/             business-layer table definitions
orchestration/   Dagster pipeline definitions
loadtest/        k6 load-test script
tests/
  unit/          fast, deterministic
  property/      Hypothesis-generated — the leakage suite lives here
```

---

## Testing

```bash
.\make.ps1 test        # unit + property suites
.\make.ps1 lint        # ruff
.\make.ps1 typecheck   # mypy
.\make.ps1 ci          # everything CI runs
```

**267 tests**, all passing, plus a strict `mypy` pass over all ten packages.

### The property tests are the interesting ones

Most tests check that code does what you expected. **Property-based tests**
check that a *rule* holds for inputs you never thought of. The library is
[Hypothesis](https://hypothesis.readthedocs.io/): you state a property, it
generates hundreds of random cases trying to break it, and when it succeeds it
shrinks the failure to the smallest example that still fails.

Two suites matter:

- `tests/property/test_point_in_time.py` — generates random event streams and
  asserts that no feature value can depend on an event at or after its own
  timestamp. This is the anti-leakage guarantee.
- `tests/property/test_identifiers.py` — generates account numbers and asserts
  the checksum rules. It found a real bug: the validator accepted forged check
  digits outside the legal 02–98 range.

---

## Troubleshooting

**`make: command not found` on Windows** — use `.\make.ps1` instead. Same
targets.

**Docker containers will not start** — run `.\make.ps1 doctor`. Almost always
free disk space or the Docker daemon not running.

**`kg lake` fails with a path error on Windows** — Iceberg wants a POSIX-style
URI. The catalog builds one explicitly; if you have edited it, note that
Python's `Path.as_uri()` produces `file:///C:/...`, which the library rejects.

**Kaggle upload returns 401 while reads work** — credential precedence, not
permissions. Run `kaggle auth login`. See
[Training on a GPU](#training-on-a-gpu).

**A notebook fails in its first cell with a compute-capability error** — you are
on a P100. Switch to T4 x2.

**Great Expectations takes 10 seconds to import** — it does, which is why it is
not on the blocking path.
[ADR-0011](docs/adr/0011-native-quality-gate.md) has the measurement.

**Out of memory during feature backfill** — the DuckDB path processes in chunks
with a lookback window. If you have changed the chunk size, that is the first
thing to check.
