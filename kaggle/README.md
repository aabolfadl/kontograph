# Training on Kaggle GPUs

The temporal GNN and its explainer head need a GPU. Everything else in this repository — the
simulator, the lakehouse, the feature platform, the LightGBM baseline, serving and evaluation — runs
on CPU locally. This directory is the bridge between the two halves.

> **Milestone status:** the automation described here lands in **M5**. The one-time setup below is
> worth doing now, because step 3 is the step that silently breaks things and it is better
> discovered early than at 2 a.m. mid-training.

---

## Why the split exists

A TGN trains sequentially over an event stream (see [ADR-0006](../docs/adr/0006-continuous-time-tgn.md)),
which makes it throughput-bound in a way LightGBM is not. On a 4-core laptop CPU that is hours; on a
Kaggle T4 it is well under one.

> **Use T4 x2, not P100.** This is not a preference — P100 does not work. It is compute capability
> `sm_60`, and current Kaggle PyTorch images ship kernels for `sm_70` and above, so
> `torch.cuda.is_available()` returns `True`, allocations succeed, and the run dies at the first
> real kernel launch. T4 is `sm_75`. The notebooks now check the device capability against
> `torch.cuda.get_arch_list()` in the environment cell and fail immediately rather than several
> minutes in.
>
> **Pin it in the metadata.** `kernel-metadata.json` takes a `machine_shape` field alongside
> `enable_gpu`, and `kg kaggle push-kernel` writes `"machine_shape": "NvidiaTeslaT4"` by default
> (override with `--accelerator p100|tpu`, though neither runs these images).
>
> | UI setting | `machine_shape` |
> |---|---|
> | GPU T4 x2 | `NvidiaTeslaT4` |
> | GPU P100 | `NvidiaTeslaP100` |
> | TPU VM v3-8 | `Tpu1VmV38` |
>
> This matters more than a default would suggest: **a push rewrites the kernel's settings**, so
> without an explicit `machine_shape` the API falls back to P100 *and discards a T4 chosen in the
> UI*. Every push then spends a run failing. Pinning it is what makes `push-kernel` safe to call
> repeatedly.

Kaggle gives roughly **30 GPU-hours per week** and a **12-hour session cap**, which is comfortably
more than the full ablation sweep needs.

---

## One-time setup

### 1. Create an API token

kaggle.com → your avatar → **Settings** → **API** → **Create New Token**. A `kaggle.json` downloads.

### 2. Put it where the client looks

```powershell
# Windows
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Move-Item -Force "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

```bash
# macOS / Linux
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

A repo-root `kaggle.json` also works and is gitignored — but the home directory is the better habit,
because a credential inside a project folder eventually gets committed by someone.

Confirm with `kg doctor`; the **kaggle credentials** row shows the path it found.

### 3. Phone-verify your Kaggle account — *this is the one that bites*

Kaggle gates **both GPU access and notebook internet access** behind phone verification
(Settings → Phone Verification). Our notebooks need internet to `pip install torch_geometric`.

Without it the symptom is not a clear error: the notebook runs, the install cell fails quietly, and
training dies on an import several cells later. Verify first.

---

## The workflow

```powershell
.\make.ps1 sim                          # generate the dataset locally (CPU)
kg kaggle push-data                     # version it as a Kaggle Dataset
kg kaggle push-kernel 01_train_tgn      # upload + run headless, GPU on
#                                       # then: Settings -> Accelerator -> GPU T4 x2
kg kaggle wait 01_train_tgn             # poll until complete
kg kaggle pull-artifacts 01_train_tgn   # checkpoints + ONNX + metrics.json
kg models register                      # register the pulled artifacts into local MLflow
.\make.ps1 eval                         # regenerate the ablation tables locally
```

`push-kernel` writes `kernel-metadata.json` with `enable_gpu: true`, `enable_internet: true`, the
dataset source **pinned to the exact version you pushed**, and — for `02_distill_explainer` — the
`kernel_sources` entry that mounts `01_train_tgn`'s checkpoint. A training run is therefore
reproducible from a commit hash plus a dataset version, which is what the engineering invariants require.

The one thing it cannot write is the accelerator type; see the note above.

**Where the dataset lands is not guaranteed.** Kaggle mounted it under a nested `datasets/`
directory rather than `/kaggle/input/<slug>/`, so the notebooks locate it by searching for
`features.parquet` anywhere beneath `/kaggle/input`. If it is missing they raise and list what *is*
mounted, rather than falling back to a path that cannot work — the earlier fallback turned "not
attached" into a `FileNotFoundError` deep inside pandas.

---

## The contract between the two halves

Deliberately narrow, so that nothing about GPU training leaks into the local stack:

| Direction | Payload |
|---|---|
| **local → Kaggle** | A versioned dataset directory: Parquet events, ground truth, split boundaries, and the dataset manifest carrying the config fingerprint and seed |
| **Kaggle → local** | A bundle: model checkpoints, ONNX exports, and a fixed-schema `metrics.json` |

Because the boundary is just files, the pulled bundle drops straight into MLflow and every
evaluation table regenerates on your CPU. Nothing needs a GPU to *report* results — only to produce
the checkpoints.

---

## Notebooks

| Notebook | Does |
|---|---|
| `01_train_tgn.ipynb` | Trains the TGN detector; emits checkpoints + `metrics.json` |
| `02_distill_explainer.ipynb` | Runs the slow perturbation teacher, distils the amortised student, measures the fidelity gap |
| `03_ablations.ipynb` | The full ladder from the evaluation protocol, plus label-delay and evasive-typology runs |

Each is self-contained and also usable by manual browser upload if you would rather not use the API.

---

## Install notes

The install cell **detects the notebook's torch version** rather than pinning a CUDA wheel index.
That matters: since **PyG 2.3**, `pip install torch_geometric` requires nothing but PyTorch —
`torch-scatter`, `torch-sparse` and `pyg-lib` are optional extras. Pinning wheels against a CUDA
version Kaggle has since moved past is the classic way these notebooks rot, and avoiding it is why
they should keep working as Kaggle's base image changes.

---

## Cost control

- Turn the GPU **off** for notebooks that only read results — the quota is per hour of GPU *session*,
  not per hour of compute.
- `03_ablations.ipynb` is the expensive one. Run `01` and `02` first and confirm the metrics look
  sane before spending quota on the full sweep.
- Kaggle kills a session at 12 hours. Every notebook checkpoints so a killed run resumes rather
  than restarting.
