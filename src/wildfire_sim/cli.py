"""Command-line interface: ``wildfire-sim``.

Only ``validate`` exists in Phase 0. ``run``, ``ensemble`` and ``plot`` arrive in WP 1.8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from wildfire_sim import __version__
from wildfire_sim.config import load_scenario
from wildfire_sim.data import build_grid

app = typer.Typer(help="Cellular-automaton wildfire spread and suppression model.", no_args_is_help=True)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"wildfire-sim {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit.")
    ] = False,
) -> None:
    """wildfire-sim command line."""


@app.command()
def validate(
    scenario: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Scenario YAML file.")],
    build: Annotated[bool, typer.Option(help="Also build the grid and report its size.")] = True,
) -> None:
    """Validate a scenario file and summarise it."""
    try:
        cfg = load_scenario(scenario)
    except (ValidationError, ValueError) as exc:
        console.print(f"[red]Invalid scenario:[/red] {exc}")
        raise typer.Exit(code=1) from None

    table = Table(title=f"Scenario: {cfg.name}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("grid", f"{cfg.grid.kind}")
    table.add_row("spread rule", cfg.spread.rule)
    table.add_row("dt / duration", f"{cfg.dt_s:g} s / {cfg.duration_s:g} s ({cfg.n_steps} steps)")
    table.add_row("stencil", str(cfg.stencil))
    table.add_row("weather points", str(len(cfg.weather.series)))
    table.add_row("suppression policy", cfg.suppression.policy)
    table.add_row("seed", str(cfg.seed))
    if build:
        try:
            grid = build_grid(cfg.grid)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            console.print(f"[red]Cannot build grid:[/red] {exc}")
            raise typer.Exit(code=1) from None
        h, w = grid.shape
        table.add_row("grid shape", f"{h} x {w} cells at {grid.dx:g} m ({grid.n_cells:,} cells)")
        table.add_row("burnable cells", f"{int(grid.burnable_mask().sum()):,}")
    console.print(table)
    console.print("[green]OK[/green]")


if __name__ == "__main__":
    app()
