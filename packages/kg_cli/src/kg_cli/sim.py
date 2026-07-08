"""``kg sim`` — generate and inspect the synthetic payments ecosystem."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kg_core.config import get_settings

sim_app = typer.Typer(
    name="sim",
    help="Agent-based synthetic German retail payments ecosystem.",
    no_args_is_help=True,
)
console = Console()


@sim_app.command("profiles")
def list_profiles() -> None:
    """List the available simulation profiles."""
    from kg_sim.config import PROFILES

    table = Table("profile", "accounts", "days", "campaigns", "fingerprint")
    for name, config in PROFILES.items():
        summary = config.profile_summary()
        table.add_row(
            name,
            f"{summary['accounts']:,}",
            str(summary["days"]),
            str(summary["campaigns"]),
            summary["fingerprint"][:12],
        )
    console.print(table)
    console.print(
        "\n[dim]The fingerprint is the content hash of the whole recipe. "
        "Two datasets with the same fingerprint are byte-identical.[/]"
    )


@sim_app.command("generate")
def generate(
    profile: str = typer.Option("dev", "--profile", "-p", help="smoke | dev | full"),
    seed: int = typer.Option(None, "--seed", help="Override the configured seed."),
    days: int = typer.Option(None, "--days", help="Override the horizon length."),
    out: Path = typer.Option(None, "--out", help="Output root (default: data/)."),
    verify: bool = typer.Option(
        True, "--verify/--no-verify", help="Validate a sample against the wire contract."
    ),
) -> None:
    """Generate a dataset and write it to Parquet.

    Output is a pure function of (profile, seed, code commit). Re-running with
    the same arguments reproduces byte-identical files, which is what lets the
    dataset be identified by a content hash instead of being version-controlled
    (ADR-0009).
    """
    from kg_sim.config import get_profile
    from kg_sim.dataset import write_dataset
    from kg_sim.generator import simulate, validate_sample

    settings = get_settings()
    settings.ensure_dirs()

    try:
        config = get_profile(profile)
    except KeyError as error:
        console.print(f"[red]{error}[/]")
        raise typer.Exit(1) from None

    overrides: dict = {}
    if seed is not None:
        overrides["seed"] = seed
    if days is not None:
        overrides["horizon"] = config.horizon.model_copy(update={"days": days})
    if overrides:
        config = config.model_copy(update=overrides)

    summary = config.profile_summary()
    console.print(
        f"[bold]Simulating[/] {summary['accounts']:,} accounts over {summary['days']} days "
        f"({summary['start']} to {summary['end']}), {summary['campaigns']} campaigns, "
        f"seed {summary['seed']}"
    )

    started = time.perf_counter()
    with console.status("generating…"):
        result = simulate(config)
    elapsed = time.perf_counter() - started

    if verify:
        checked = validate_sample(result)
        console.print(f"[green]{checked:,}[/] sampled events validate against the wire contract")

    target = write_dataset(result, out or settings.data_dir, profile=profile)
    _print_summary(result.manifest, target, elapsed)


def _print_summary(manifest: dict, target: Path, elapsed: float) -> None:
    counts, rates = manifest["counts"], manifest["rates"]

    table = Table("", "", show_header=False, box=None)
    table.add_row("accounts", f"{counts['accounts']:,}")
    table.add_row("events", f"{counts['events']:,}")
    table.add_row("criminal events", f"{counts['criminal_events']:,}")
    table.add_row("campaigns", str(counts["campaigns"]))
    table.add_row("labels", f"{counts['labels']:,}")
    console.print(table)

    console.print(
        f"\n  positive rate   [bold]{rates['positive_rate']:.4%}[/]  (oracle truth)\n"
        f"  labelled rate   {rates['labelled_rate']:.4%}\n"
        f"  label coverage  [bold]{rates['criminal_label_coverage']:.1%}[/] "
        "of criminal transactions are ever confirmed"
    )
    console.print(
        "\n[dim]Label coverage below 100% is deliberate (design principle P7): labels are late, "
        "incomplete, and biased towards what the incumbent rules engine already caught.[/]"
    )

    typology = Table("typology", "edges", "standard", "evasive")
    splits = manifest["evasive_split"]
    for name, total in sorted(manifest["typologies"].items(), key=lambda kv: -kv[1]):
        typology.add_row(
            name,
            str(total),
            str(splits.get(f"{name}:standard", 0)),
            str(splits.get(f"{name}:evasive", 0)),
        )
    console.print()
    console.print(typology)

    console.print(f"\n[green]written[/] {target}  [dim]({elapsed:.1f}s)[/]")
    console.print(f"[dim]digest {(target / 'DIGEST').read_text().strip()}[/]")


@sim_app.command("inspect")
def inspect(
    path: Path = typer.Argument(None, help="Dataset directory (default: latest under data/)."),
) -> None:
    """Show a dataset's manifest and verify its digest still matches."""
    from kg_sim.dataset import find_latest, read_manifest, verify_digest

    settings = get_settings()
    target = path or find_latest(settings.data_dir)
    if target is None:
        console.print("[red]no dataset found.[/] Run `kg sim generate` first.")
        raise typer.Exit(1)

    manifest = read_manifest(target)
    console.print(f"[bold]{target}[/]\n")
    console.print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if verify_digest(target):
        console.print("\n[green]digest verified[/] — the files match the recorded hash")
    else:
        console.print(
            "\n[red]DIGEST MISMATCH[/] — the Parquet files have changed since they were "
            "written. Any number computed from this dataset is unattributable; regenerate it."
        )
        raise typer.Exit(1)


@sim_app.command("determinism")
def determinism(
    profile: str = typer.Option("smoke", "--profile", "-p"),
) -> None:
    """Generate twice and prove the output is identical.

    This is the test that keeps ADR-0009 honest. Content-hashing a dataset only
    replaces DVC if the recipe genuinely reproduces the bytes; the moment an
    unseeded RNG or a dict-ordering dependency creeps in, this command fails and
    the decision to drop DVC becomes wrong.
    """
    from kg_sim.config import get_profile
    from kg_sim.generator import simulate

    config = get_profile(profile)
    console.print(f"generating {profile!r} twice…")
    first, second = simulate(config), simulate(config)

    table = Table("table", "run 1", "run 2", "match")
    ok = True
    for name in ("events", "ground_truth", "labels", "accounts"):
        left = first.manifest["content_hashes"][name]
        right = second.manifest["content_hashes"][name]
        matched = left == right
        ok &= matched
        table.add_row(name, left[:16], right[:16], "[green]yes[/]" if matched else "[red]NO[/]")
    console.print(table)

    if not ok:
        console.print(
            "\n[red]The simulator is not deterministic.[/] "
            "ADR-0009 replaces DVC with content hashing on the premise that the "
            "dataset regenerates exactly — that premise is currently false."
        )
        raise typer.Exit(1)
    console.print("\n[green]Deterministic.[/] The dataset is reproducible from (config, seed).")
