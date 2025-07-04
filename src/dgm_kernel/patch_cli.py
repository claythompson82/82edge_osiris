from __future__ import annotations

import json
from pathlib import Path

import click

PATCH_HISTORY_FILE = Path(__file__).resolve().parent.parent / "patch_history.json"


@click.group()
def main() -> None:
    """Patch history inspection commands."""
    pass


@main.command()
@click.option("--limit", default=10, show_default=True, type=int)
def list(limit: int) -> None:
    """List recent patch ids."""
    if not PATCH_HISTORY_FILE.exists():
        click.echo("No patch history found", err=True)
        raise SystemExit(1)

    history = json.loads(PATCH_HISTORY_FILE.read_text())
    for entry in history[-limit:]:
        click.echo(entry.get("patch_id", ""))


@main.command()
@click.option("--id", "patch_id", required=True)
def show(patch_id: str) -> None:
    """Show diff for a patch id."""
    if not PATCH_HISTORY_FILE.exists():
        click.echo("No patch history found", err=True)
        raise SystemExit(1)

    history = json.loads(PATCH_HISTORY_FILE.read_text())
    for entry in history:
        if entry.get("patch_id") == patch_id:
            click.echo(entry.get("diff", ""))
            break
    else:
        click.echo(f"Patch id not found: {patch_id}", err=True)
        raise SystemExit(1)


@main.command()
def hist() -> None:
    """Print histogram of pylint scores."""
    if not PATCH_HISTORY_FILE.exists():
        click.echo("No patch history found", err=True)
        raise SystemExit(1)

    history = json.loads(PATCH_HISTORY_FILE.read_text())
    scores = [entry.get("pylint_score") for entry in history if "pylint_score" in entry]
    if not scores:
        return

    buckets: dict[int, int] = {}
    for sc in scores:
        try:
            bucket = int(float(sc))
        except (TypeError, ValueError):
            continue
        buckets[bucket] = buckets.get(bucket, 0) + 1

    for b in sorted(buckets):
        label = f"{b}\u2013{b+1}"
        click.echo(f"{label}: {buckets[b]}")


if __name__ == "__main__":  # pragma: no cover - manual CLI
    main()
