"""``kg kaggle`` — push data, run notebooks on GPU, pull artifacts back.

The bridge between the CPU half of this project and the one part that needs a
GPU. The contract across it is deliberately narrow (see ``kaggle/README.md``):

    local -> Kaggle    a versioned dataset directory
    Kaggle -> local    checkpoints, ONNX exports, and a fixed-schema metrics.json

Nothing about GPU training reaches back into the local stack except through that
bundle, so a pulled result drops straight into the evaluation path and every
table regenerates on CPU.

Credentials are read from ``~/.kaggle/kaggle.json``. A repo-root copy is also
honoured — it is gitignored — because that is where the file lands if you
download it into the project folder.

**The Python API is used, not the ``kaggle`` command-line tool.** Shelling out
looked simpler and was worse in three separate ways, all of which bit:

- ``kaggle`` on ``PATH`` need not be the one in this virtualenv. Here it resolved
  to a user-level install whose authentication path differs from the library's,
  so the library authenticated fine and the subprocess did not.
- The CLI printed ``Authentication required to call the Kaggle API`` to *stdout*
  and exited **0**. Checking the return code — which the wrapper did — was not
  enough to notice the upload had failed.
- Errors arrive as prose meant for a human, so failures cannot be distinguished
  by kind.

The API raises, returns objects, and lives in the venv the rest of the project
uses.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from kg_core.config import get_settings

#: Newline for multi-part console messages.
NEWLINE = chr(10)

#: Kaggle's accelerator identifiers. NOT interchangeable: P100 is compute
#: capability sm_60, and current Kaggle PyTorch images ship kernels for sm_70
#: and above, so a P100 run reports CUDA available and then dies at the first
#: real kernel launch. T4 is sm_75.
#:
#: This is `machine_shape` in kernel-metadata.json (equivalently the `acc`
#: argument to kernels_push). Without it the API defaults to P100 — and because
#: a push rewrites the kernel's settings, it silently discards a T4 chosen in
#: the UI, which is worse than merely defaulting badly.
ACCELERATORS: dict[str, str] = {
    "t4": "NvidiaTeslaT4",
    "p100": "NvidiaTeslaP100",
    "tpu": "Tpu1VmV38",
}
DEFAULT_ACCELERATOR = "t4"

#: Which notebooks read another notebook's output. 02 loads the checkpoint that
#: 01 writes, so it cannot run until 01 has finished and must declare it as a
#: kernel source. 03 reads only the dataset, so it can run alongside 01.
KERNEL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "02_distill_explainer": ("01_train_tgn",),
}

kaggle_app = typer.Typer(
    name="kaggle", help="GPU training on Kaggle: push, run, pull.", no_args_is_help=True
)
console = Console()


def _credentials() -> Path | None:
    settings = get_settings()
    for candidate in (
        Path.home() / ".kaggle" / "kaggle.json",
        settings.repo_root / "kaggle.json",
        settings.repo_root / ".kaggle" / "kaggle.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _api() -> Any:
    """An authenticated Kaggle API client, or a clear exit explaining why not.

    ``KAGGLE_CONFIG_DIR`` is set before the import because the client reads it at
    import time, not at ``authenticate()``.
    """
    credentials = _credentials()
    if credentials is None:
        console.print(
            "[red]no kaggle.json found.[/]\n"
            "  1. kaggle.com -> Settings -> API -> Create New Token\n"
            "  2. move it to ~/.kaggle/kaggle.json\n"
            "  3. phone-verify your account, or GPU and notebook internet stay off\n"
            "See kaggle/README.md."
        )
        raise typer.Exit(1)
    os.environ["KAGGLE_CONFIG_DIR"] = str(credentials.parent)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        # `import kaggle` can succeed while the package is absent: this repo has
        # a `kaggle/` directory, and from the repo root Python binds that as a
        # namespace package. The import that matters is `kaggle.api`, which a
        # directory of notebooks cannot satisfy.
        console.print("[red]the kaggle package is not installed.[/] Run: uv sync --extra kaggle")
        raise typer.Exit(1) from None

    _prefer_oauth()

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as error:
        console.print(f"[red]kaggle authentication failed:[/] {error}")
        raise typer.Exit(1) from None

    # authenticate() succeeding is NOT evidence of working credentials. The
    # legacy path returns True merely because a username and key are present in
    # the config file - it never calls the service. So the check that matters is
    # an authenticated request: listing the account's own datasets.
    try:
        owned = len(list(api.dataset_list(mine=True)))
    except Exception as error:
        console.print(f"[red]kaggle credentials did not work:[/] {error}")
        _diagnose_write_failure(error)
        raise typer.Exit(1) from None

    method = str(api.config_values.get("auth_method", "unknown"))
    if method == "AuthMethod.LEGACY_API_KEY" and owned == 0:
        console.print(
            "[yellow]warning:[/] authenticated by legacy API key, and the account "
            "reports zero datasets. That combination usually means the key is "
            "being accepted locally but rejected by the service. Run "
            "`kaggle auth login`."
        )
    return api


def _prefer_oauth() -> None:
    """Put OAuth credentials ahead of a legacy key, if both exist.

    The SDK tries an access token, THEN a legacy API key, THEN OAuth. So a stale
    ``kaggle.json`` outranks a fresh `kaggle auth login`, and the login appears
    to have done nothing: reads still work (many are anonymous), while uploads
    return a bare 401 that names neither the cause nor the fix.

    Minting an access token from the OAuth credentials moves them to the front
    of that order, since an access token is the first thing the SDK looks for.
    Silent when there are no OAuth credentials - the legacy key is then the only
    thing available and should be used.
    """
    creds_file = Path.home() / ".kaggle" / "credentials.json"
    if not creds_file.exists() or os.environ.get("KAGGLE_API_TOKEN"):
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk import KaggleCredentials

        with KaggleApi().build_kaggle_client() as client:
            credentials = KaggleCredentials.load(client=client)
            if credentials:
                os.environ["KAGGLE_API_TOKEN"] = credentials.get_access_token()
    except Exception:
        # Deliberately quiet. This is an upgrade, not a requirement: if the
        # OAuth credentials cannot mint a token the legacy path still runs, and
        # the authenticated probe above reports the real problem.
        return


def _diagnose_write_failure(error: Exception) -> None:
    """Explain a write rejection, because the raw error names the wrong problem.

    A 401 from the blob service reads as "your token is wrong", which sends
    people to regenerate a credential that is often fine. The real cause,
    observed here, is credential *precedence*: the SDK tries an access token,
    then a legacy API key, then OAuth. A ``kaggle.json`` holding a username and
    key therefore outranks a fresh ``kaggle auth login``, and the legacy path
    reports success without ever calling the service - so reads appear to work
    (many are anonymous) while writes are refused.

    The tell is an account that authenticates but reports zero datasets.
    """
    text = str(error)
    if "401" not in text and "Unauthorized" not in text:
        return
    console.print(
        NEWLINE
        + "[yellow]A 401 on upload usually does not mean the token is invalid.[/]"
        + NEWLINE
        + "Kaggle's SDK prefers a legacy API key over OAuth, so a kaggle.json "
        + "with a username and key shadows `kaggle auth login`. The legacy path "
        + "reports success without contacting the service, which is why reads "
        + "look fine and only writes fail."
        + NEWLINE
        + NEWLINE
        + "  1. Run `kaggle auth login` (opens a browser)."
        + NEWLINE
        + "  2. Re-run `kg kaggle push-data`. It mints an access token from those "
        + "credentials, which the SDK checks first."
        + NEWLINE
        + "  3. `kg kaggle status` should then show a non-zero dataset count via "
        + "ACCESS_TOKEN. A count of zero means it is still on the legacy key."
        + NEWLINE
        + NEWLINE
        + "Separately: GPU and notebook internet need a phone-verified account. "
        + "That is a different gate and does not cause this 401."
    )


def _username() -> str:
    settings = get_settings()
    if settings.kaggle_username:
        return settings.kaggle_username
    credentials = _credentials()
    if credentials:
        return json.loads(credentials.read_text(encoding="utf-8")).get("username", "")
    return ""


@kaggle_app.command("status")
def status() -> None:
    """Check credentials and the local artifacts the GPU run needs."""
    from kg_sim.dataset import find_latest

    settings = get_settings()
    table = Table("check", "state", "detail")

    credentials = _credentials()
    table.add_row(
        "credentials",
        "[green]found[/]" if credentials else "[red]missing[/]",
        str(credentials) if credentials else "see kaggle/README.md",
    )

    # Actually call the API. A file on disk proves nothing: the token can be
    # revoked or rejected, `import kaggle` succeeds from the repo root
    # regardless because the notebooks live in a directory of that name, and the
    # SDK's legacy path returns success without ever contacting the service.
    # Routed through _api() so this reports the same auth the commands will use
    # - a status check that authenticates differently from the thing it is
    # checking is worse than none.
    reachable, detail = False, "not checked"
    if credentials:
        try:
            probe = _api()
            owned = list(probe.dataset_list(mine=True))
            method = str(probe.config_values.get("auth_method", "")).split(".")[-1]
            reachable = True
            detail = f"{len(owned)} dataset(s), via {method}"
        except SystemExit:
            detail = "authentication failed; see the message above"
        except Exception as error:
            detail = str(error)[:60]
    table.add_row("api reachable", "[green]yes[/]" if reachable else "[red]no[/]", detail)
    table.add_row("username", _username() or "[yellow]unknown[/]", "")

    dataset = find_latest(settings.data_dir)
    table.add_row(
        "dataset",
        "[green]ready[/]" if dataset else "[red]missing[/]",
        dataset.name if dataset else "run `kg sim generate --profile full`",
    )

    features = settings.artifacts_dir / "features" / f"{dataset.name}.parquet" if dataset else None
    ready = features is not None and features.exists()
    table.add_row(
        "features",
        "[green]ready[/]" if ready else "[red]missing[/]",
        f"{features.stat().st_size / 1e6:.0f} MB"
        if ready and features is not None
        else "run `kg features build`",
    )

    staged = settings.artifacts_dir / "kaggle" / "dataset"
    table.add_row(
        "staged upload",
        "[green]yes[/]" if staged.exists() else "[dim]not yet[/]",
        str(staged) if staged.exists() else "run `kg kaggle push-data`",
    )
    console.print(table)
    console.print(
        NEWLINE + "[dim]GPU and notebook internet both need a phone-verified account, and "
        "the install cell fails quietly without internet. A 401 on upload while reads work is the CREDENTIAL, not verification: run `kaggle auth login`.[/]"
    )


@kaggle_app.command("push-data")
def push_data(
    dataset: Path = typer.Option(None, "--dataset"),
    slug: str = typer.Option(None, "--slug", help="Kaggle dataset slug."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Stage the files, do not upload."),
) -> None:
    """Stage the training bundle and version it as a Kaggle Dataset.

    Only what training needs goes up: the feature table, the labels, and the
    ground truth for the honest-evaluation gap. The raw event stream stays local
    — it is large, and the GPU half never reads it.
    """
    import shutil

    from kg_sim.dataset import find_latest

    settings = get_settings()
    settings.ensure_dirs()
    source = dataset or find_latest(settings.data_dir)
    if source is None:
        console.print("[red]no dataset found.[/] Run `kg sim generate --profile full` first.")
        raise typer.Exit(1)

    features = settings.artifacts_dir / "features" / f"{source.name}.parquet"
    if not features.exists():
        console.print(f"[red]no feature table at {features}.[/] Run `kg features build`.")
        raise typer.Exit(1)

    staging = settings.artifacts_dir / "kaggle" / "dataset"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy2(features, staging / "features.parquet")
    for name in ("labels.parquet", "ground_truth.parquet", "manifest.json", "DIGEST"):
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, staging / name)

    slug = slug or settings.kaggle_dataset_slug
    username = _username()
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "KONTOGRAPH SEPA Instant features",
                "id": f"{username}/{slug}",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total = sum(p.stat().st_size for p in staging.iterdir()) / 1e6
    console.print(f"staged [bold]{total:.0f} MB[/] in {staging}")
    for path in sorted(staging.iterdir()):
        console.print(f"  {path.name:24} {path.stat().st_size / 1e6:8.1f} MB")

    if dry_run:
        console.print("\n[yellow]--dry-run:[/] nothing uploaded.")
        return

    api = _api()
    # `mine=True` and `user=` are mutually exclusive in the client, and asking
    # for both raises rather than being treated as a redundant filter.
    owned = {str(getattr(d, "ref", d)) for d in api.dataset_list(mine=True)}
    exists = f"{username}/{slug}" in owned

    console.print(f"\n[dim]uploading {total:.0f} MB, this takes a few minutes[/]")
    try:
        if exists:
            result = api.dataset_create_version(
                str(staging), version_notes=f"kontograph {source.name}", dir_mode="zip"
            )
        else:
            result = api.dataset_create_new(str(staging), dir_mode="zip")
    except Exception as error:
        console.print(f"[red]upload failed:[/] {error}")
        _diagnose_write_failure(error)
        raise typer.Exit(1) from None

    # The client reports some failures in the RESPONSE rather than by raising,
    # which is how an earlier version of this command printed "pushed" over a
    # failed upload. Treat anything carrying an error as a failure.
    rejection = getattr(result, "error", None) or getattr(result, "errorMessage", None)
    if rejection:
        console.print(f"[red]upload rejected:[/] {rejection}")
        raise typer.Exit(1)

    console.print(f"\n[green]pushed[/] {username}/{slug}")
    console.print(f"[dim]https://www.kaggle.com/datasets/{username}/{slug}[/]")


@kaggle_app.command("push-kernel")
def push_kernel(
    notebook: str = typer.Argument(..., help="Notebook stem, e.g. 01_train_tgn"),
    slug: str = typer.Option(None, "--slug", help="Dataset slug to attach."),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu"),
    accelerator: str = typer.Option(
        DEFAULT_ACCELERATOR,
        "--accelerator",
        help=f"Accelerator to request: {', '.join(ACCELERATORS)}. T4 is the only one that works.",
    ),
) -> None:
    """Upload a notebook and run it headless with the GPU on.

    Writes ``kernel-metadata.json`` with the dataset source pinned, so a run is
    reproducible from a commit hash plus a dataset version.

    The accelerator is pinned through ``machine_shape``, which defaults to T4
    here. It matters because a push rewrites the kernel's settings: without an
    explicit value the API falls back to P100 and silently discards a T4 chosen
    in the UI — and P100 cannot run these images at all, being sm_60 against an
    sm_70+ build. Kaggle has no upload-without-running mode, so a push that
    resets the accelerator also spends a run failing.
    """
    settings = get_settings()
    source = settings.repo_root / "kaggle" / f"{notebook}.ipynb"
    if not source.exists():
        available = sorted(p.stem for p in (settings.repo_root / "kaggle").glob("*.ipynb"))
        console.print(f"[red]no notebook {notebook!r}.[/] Available: {available}")
        raise typer.Exit(1)

    accelerator = accelerator.lower()
    if accelerator not in ACCELERATORS:
        console.print(
            f"[red]unknown accelerator {accelerator!r}.[/] Choose from: {', '.join(ACCELERATORS)}"
        )
        raise typer.Exit(1)

    username = _username()
    slug = slug or settings.kaggle_dataset_slug
    staging = settings.artifacts_dir / "kaggle" / "kernels" / notebook
    staging.mkdir(parents=True, exist_ok=True)

    import shutil

    shutil.copy2(source, staging / f"{notebook}.ipynb")
    (staging / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{username}/kontograph-{notebook.replace('_', '-')}",
                "title": f"KONTOGRAPH {notebook}",
                "code_file": f"{notebook}.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                # Both required. Internet is needed for `pip install
                # torch_geometric`, and Kaggle gates BOTH behind phone
                # verification — which is the step that silently breaks this.
                "enable_gpu": gpu,
                # Pinned, not left to the API default. See ACCELERATORS.
                "machine_shape": ACCELERATORS[accelerator],
                "enable_internet": True,
                "dataset_sources": [f"{username}/{slug}"],
                "competition_sources": [],
                # A notebook that reads another's output must declare it, or
                # Kaggle will not mount it and the run fails on a missing path
                # several minutes in.
                "kernel_sources": [
                    f"{username}/kontograph-{dep.replace('_', '-')}"
                    for dep in KERNEL_DEPENDENCIES.get(notebook, ())
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    api = _api()
    try:
        # Passed explicitly as well as written into the metadata: `acc` takes
        # precedence in the client, so the request cannot end up on a different
        # accelerator than the file records.
        result = api.kernels_push(str(staging), acc=ACCELERATORS[accelerator])
    except Exception as error:
        console.print(f"[red]kernel push failed:[/] {error}")
        _diagnose_write_failure(error)
        raise typer.Exit(1) from None

    rejection = getattr(result, "error", None)
    if rejection:
        console.print(f"[red]kernel rejected:[/] {rejection}")
        raise typer.Exit(1)
    if getattr(result, "url", None):
        console.print(f"[dim]{result.url}[/]")

    if gpu:
        console.print(
            f"[dim]accelerator: {ACCELERATORS[accelerator]} "
            f"({'the only one these images run on' if accelerator == 't4' else 'NOT sm_70+; expect the capability check to fail'})[/]"
        )


@kaggle_app.command("wait")
def wait(
    notebook: str = typer.Argument(...),
    poll_seconds: int = typer.Option(60, "--poll"),
    timeout_minutes: int = typer.Option(180, "--timeout"),
) -> None:
    """Poll until the kernel finishes."""
    kernel = f"{_username()}/kontograph-{notebook.replace('_', '-')}"
    api = _api()
    deadline = time.time() + timeout_minutes * 60

    # Matched against the API's status field, not against a substring of console
    # output. "error" appears in the word "errors" and in plenty of innocuous
    # log lines, so the substring test this replaced could call a healthy run
    # failed - and, worse, a failed one complete.
    finished = {"complete": True, "error": False, "cancelacknowledged": False, "cancelled": False}

    while time.time() < deadline:
        result = api.kernels_status(kernel)
        # The client returns an enum, so str() gives "KernelWorkerStatus.RUNNING"
        # rather than "running". Comparing that against bare state names matches
        # nothing, and `wait` would poll a finished kernel until it timed out.
        state = str(getattr(result, "status", "") or "").rsplit(".", 1)[-1].lower()

        if state in finished:
            if finished[state]:
                console.print("[green]complete[/]")
                return
            message = getattr(result, "failureMessage", None) or "no message given"
            console.print(f"[red]the kernel finished as {state}:[/] {message}")
            console.print(f"[dim]https://www.kaggle.com/code/{kernel}[/]")
            raise typer.Exit(1)

        remaining = int(deadline - time.time())
        console.print(
            f"[dim]{state or 'queued'}; next check in {poll_seconds}s ({remaining}s left)[/]"
        )
        time.sleep(poll_seconds)

    console.print(
        f"[yellow]still running after {timeout_minutes} minutes.[/] It is not lost - "
        f"check https://www.kaggle.com/code/{kernel} or re-run `kg kaggle wait {notebook}`."
    )
    raise typer.Exit(1)


@kaggle_app.command("pull-artifacts")
def pull_artifacts(
    notebook: str = typer.Argument(...),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Download a finished kernel's output and summarise its metrics."""
    settings = get_settings()
    destination = out or (settings.artifacts_dir / "kaggle" / "out" / notebook)
    destination.mkdir(parents=True, exist_ok=True)
    kernel = f"{_username()}/kontograph-{notebook.replace('_', '-')}"

    api = _api()
    try:
        api.kernels_output(kernel, str(destination))
    except Exception as error:
        console.print(f"[red]could not download the kernel output:[/] {error}")
        raise typer.Exit(1) from None

    table = Table("file", "size")
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            table.add_row(str(path.relative_to(destination)), f"{path.stat().st_size / 1e6:.2f} MB")
    console.print(table)

    metrics = destination / "metrics.json"
    if metrics.exists():
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        console.print("\n[bold]metrics.json[/]")
        console.print(json.dumps(payload, indent=2)[:2500])
    else:
        console.print(
            "\n[yellow]no metrics.json in the output.[/] The notebook writes it to "
            "/kaggle/working; check that the run reached its final cell."
        )
    console.print(f"\n[green]pulled[/] {destination}")


@kaggle_app.command("notebooks")
def list_notebooks() -> None:
    """List the notebooks in kaggle/ and what each does."""
    settings = get_settings()
    descriptions = {
        "01_train_tgn": "Train the TGN detector; emits checkpoints and metrics.json",
        "02_distill_explainer": "Perturbation teacher, amortised student, fidelity gap",
        "03_ablations": "The full ladder, label-delay and evasive-typology tables",
    }
    table = Table("notebook", "does")
    for path in sorted((settings.repo_root / "kaggle").glob("*.ipynb")):
        table.add_row(path.stem, descriptions.get(path.stem, ""))
    console.print(table)
