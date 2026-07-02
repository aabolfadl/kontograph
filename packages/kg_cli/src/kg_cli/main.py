"""``kg`` — the one entrypoint.

The Makefile and make.ps1 are thin shims over this module. That indirection is
deliberate: ``make`` does not exist on a stock Windows box, and a project whose
"one command to run it" only works on the maintainer's laptop fails
design principle P10 ("`make up` works") in the only way that matters.

Subcommands are grouped by milestone and filled in as those land. Each group
lives in its own module so this file stays a routing table.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kg_core.config import get_settings

app = typer.Typer(
    name="kg",
    help="KONTOGRAPH — real-time financial-crime intelligence for the German payments rail.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

contracts_app = typer.Typer(
    name="contracts", help="Data contracts and the compatibility gate.", no_args_is_help=True
)
app.add_typer(contracts_app)


# --------------------------------------------------------------- version ---


@app.command()
def version() -> None:
    """Print component versions and the commit this tree is at."""
    import kg_core

    table = Table(show_header=False, box=None)
    table.add_row("kontograph", kg_core.__version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("commit", git_commit())
    console.print(table)


def git_commit(short: bool = True) -> str:
    """The commit the current tree is at, or ``"unknown"`` outside a checkout.

    Stamped onto every decision and artifact — the engineering invariants, "reproducible
    from a commit hash". Reports a ``-dirty`` suffix rather than pretending an
    uncommitted tree is the commit it was branched from.
    """
    try:
        args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
        commit = subprocess.run(
            args, capture_output=True, text=True, check=True, cwd=get_settings().repo_root
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=get_settings().repo_root,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------- doctor ---


@app.command()
def doctor() -> None:
    """Check the local environment can actually run the stack.

    Written because this project targets a laptop, and the two things that
    actually stop people are free disk and a missing Docker daemon — neither of
    which produces a comprehensible error at the point of failure.
    """
    settings = get_settings()
    table = Table("check", "status", "detail")
    ok = True

    def row(name: str, passed: bool, detail: str, *, fatal: bool = True) -> None:
        nonlocal ok
        if not passed and fatal:
            ok = False
        mark = "[green]pass[/]" if passed else ("[red]FAIL[/]" if fatal else "[yellow]warn[/]")
        table.add_row(name, mark, detail)

    version_ok = sys.version_info >= (3, 11)
    row("python >= 3.11", version_ok, sys.version.split()[0])

    docker = shutil.which("docker")
    if docker:
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=30)
            row("docker daemon", True, "running")
        except Exception:
            row("docker daemon", False, "installed but not responding — start Docker Desktop")
    else:
        row("docker", False, "not found on PATH")

    total, _used, free = shutil.disk_usage(settings.repo_root)
    free_gb = free / 1024**3
    # The default compose stack pulls roughly 5 GB of images plus generated data.
    row(
        "free disk >= 15 GB",
        free_gb >= 15,
        f"{free_gb:.1f} GB free of {total / 1024**3:.0f} GB",
        fatal=False,
    )

    row("git checkout", git_commit() != "unknown", git_commit(), fatal=False)

    kaggle_json = _find_kaggle_credentials()
    row(
        "kaggle credentials",
        kaggle_json is not None,
        str(kaggle_json)
        if kaggle_json
        else "not found — needed only for GPU training (see kaggle/README.md)",
        fatal=False,
    )

    try:
        import torch  # noqa: F401
        import torch as _t

        row(
            "torch (optional)", True, f"{_t.__version__} cuda={_t.cuda.is_available()}", fatal=False
        )
    except ImportError:
        row(
            "torch (optional)",
            False,
            "not installed — the GNN spike trains on Kaggle; run `uv sync --extra torch` for local CPU work",
            fatal=False,
        )

    console.print(table)
    if not ok:
        console.print("\n[red]Environment is not ready.[/] Fix the FAIL rows above.")
        raise typer.Exit(1)
    console.print("\n[green]Environment looks good.[/]")


def _find_kaggle_credentials() -> Path | None:
    """Locate kaggle.json, preferring the standard location.

    A repo-local kaggle.json is supported (and gitignored) because that is
    where the file lands if you download it into the project folder.
    """
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        get_settings().repo_root / "kaggle.json",
        get_settings().repo_root / ".kaggle" / "kaggle.json",
    ]
    return next((p for p in candidates if p.exists()), None)


# ------------------------------------------------------------- contracts ---


@contracts_app.command("list")
def contracts_list() -> None:
    """List every data contract with its subject and fingerprint."""
    from kg_core.contracts import list_schemas, load_schema_dict, schema_fingerprint, subject_for

    table = Table("contract", "record", "subject", "fields", "fingerprint")
    for name in list_schemas():
        schema = load_schema_dict(name)
        table.add_row(
            name,
            f"{schema['namespace']}.{schema['name']}",
            subject_for(name),
            str(len(schema["fields"])),
            schema_fingerprint(name)[:12],
        )
    console.print(table)


@contracts_app.command("check")
def contracts_check(
    against: Path = typer.Option(
        None,
        "--against",
        help="Directory of previously released .avsc files to compare against. "
        "Defaults to the versions at git HEAD.",
    ),
) -> None:
    """Fail if any contract change would break existing consumers.

    This is the CI gate. Compatibility direction is BACKWARD: a consumer on the
    new schema must still read data written under the old one.
    """
    from kg_core.contracts import check_backward_compatible, list_schemas, load_schema_dict

    failures = 0
    for name in list_schemas():
        new = load_schema_dict(name)
        old = _previous_schema(name, against)
        if old is None:
            console.print(f"[yellow]new contract[/] {name} — no prior version to compare")
            continue
        problems = check_backward_compatible(new, old)
        if problems:
            failures += 1
            console.print(f"[red]INCOMPATIBLE[/] {name}")
            for problem in problems:
                console.print(f"    - {problem}")
        else:
            console.print(f"[green]compatible[/] {name}")

    if failures:
        console.print(f"\n[red]{failures} contract(s) would break existing consumers.[/]")
        raise typer.Exit(1)
    console.print("\n[green]All contracts are backward compatible.[/]")


def _previous_schema(name: str, against: Path | None) -> dict | None:
    """Load the prior version of a contract, from a directory or from git HEAD."""
    import json

    from kg_core.contracts.registry import SCHEMA_DIR

    if against is not None:
        path = against / f"{name}.avsc"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    settings = get_settings()
    relative = (SCHEMA_DIR / f"{name}.avsc").relative_to(settings.repo_root).as_posix()
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=settings.repo_root,
        ).stdout
        return json.loads(blob)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------- config --


@app.command("config")
def show_config() -> None:
    """Print the effective configuration, with secrets masked."""
    settings = get_settings()
    table = Table("setting", "value")
    secret_markers = ("secret", "key", "password", "token")
    for field, value in settings.model_dump().items():
        if any(marker in field for marker in secret_markers) and value:
            value = f"{str(value)[:2]}***"
        table.add_row(field, str(value))
    console.print(table)


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()
