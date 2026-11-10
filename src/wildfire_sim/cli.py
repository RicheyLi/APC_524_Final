"""Command-line interface: ``wildfire-sim``.

Commands: ``validate`` (check a scenario), ``run`` (one seeded run), ``ensemble`` (Monte Carlo
runs) and ``plot`` (redraw the figures of a saved run or ensemble directory). File writing and
reading live in :mod:`wildfire_sim.io`; this module only parses options and reports.

User mistakes (an invalid scenario, an unknown spread rule, an ignition on an unburnable cell,
a non-empty output directory without ``--force``, ...) print one red line and exit with code 1.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from wildfire_sim import __version__, io, sim
from wildfire_sim.config import ScenarioConfig, load_scenario
from wildfire_sim.data import build_grid
from wildfire_sim.ensemble import run_ensemble

app = typer.Typer(help="Cellular-automaton wildfire spread and suppression model.", no_args_is_help=True)
console = Console()

# Errors raised by bad input (scenario contents, grid files, output paths); anything else is a bug
# and keeps its traceback.
_USER_ERRORS = (ValueError, KeyError, TypeError, OSError)


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


# ----------------------------------------------------------------------------- shared helpers


def _fail(prefix: str, message: str) -> NoReturn:
    """Print ``prefix: message`` as one red line and exit with code 1."""
    console.print(f"[red]{escape(prefix)}:[/red] {escape(' '.join(message.split()))}", soft_wrap=True)
    raise typer.Exit(code=1)


def _message(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'scenario'}: {e['msg']}" for e in exc.errors())
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])  # str(KeyError) adds quotes
    return str(exc)


@contextmanager
def _user_errors(prefix: str) -> Generator[None]:
    """Turn the exceptions caused by bad input into a red one-line message and exit code 1."""
    try:
        yield
    except _USER_ERRORS as exc:
        _fail(prefix, _message(exc))


def _load(scenario: Path) -> ScenarioConfig:
    if not scenario.is_file():
        _fail("Error", f"scenario file {scenario} not found")
    with _user_errors("Invalid scenario"):
        return load_scenario(scenario)


def _report(title: str, rows: list[tuple[str, str]], out: Path) -> None:
    table = Table(title=title)
    table.add_column("quantity")
    table.add_column("value")
    for row in rows:
        table.add_row(*row)
    console.print(table)
    console.print(f"[green]Wrote[/green] {escape(str(out))}", soft_wrap=True)


_FORCE_HELP = "Overwrite the outputs in a non-empty output directory (other files are kept)."


# ----------------------------------------------------------------------------- run


@app.command()
def run(
    scenario: Annotated[Path, typer.Argument(metavar="SCENARIO", help="Scenario YAML file.")],
    seed: Annotated[int | None, typer.Option(min=0, help="Run seed (>= 0). Default: the scenario's seed.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output directory. Default: runs/<scenario name>-seed<SEED>.")
    ] = None,
    gif: Annotated[bool, typer.Option(help="Also write fire.gif, an animation of the spread.")] = False,
    frames: Annotated[int, typer.Option(min=1, help="Number of GIF frames (with --gif).")] = 40,
    fps: Annotated[float, typer.Option(min=0.1, help="GIF playback speed, frames per second (with --gif).")] = 10.0,
    force: Annotated[bool, typer.Option(help=_FORCE_HELP)] = False,
) -> None:
    """Run one simulation and save its results.

    Data: grid.npz, arrival_time.npy (s; NaN = never burned), events.csv, summary.json (areas in m^2 and ha).

    Records: provenance.json, scenario.yaml. Figures: final.png, arrival.png and, with --gif, fire.gif.
    """
    cfg = _load(scenario)
    seed = cfg.seed if seed is None else seed
    cfg = cfg.model_copy(update={"seed": seed})
    out = Path("runs") / f"{cfg.name}-seed{seed}" if out is None else out
    with _user_errors("Error"):
        io.check_output_dir(out, force=force)
    with _user_errors("Cannot run scenario"):
        result = sim.run(cfg, seed=seed)
    with _user_errors("Cannot write results"):
        io.prepare_output_dir(out, force=force, owned=io.RUN_OUTPUTS)
        summary = io.write_run(result, cfg, out, gif=gif, frames=frames, fps=fps)

    counts = summary["final_state_counts"]
    last = summary["last_ignition_s"]
    rows = [
        ("seed", str(seed)),
        ("burned cells", f"{summary['burned_cells']:,}"),
        ("burned area", f"{summary['burned_area_ha']:,.2f} ha ({summary['burned_area_m2']:,.0f} m^2)"),
        ("simulated time", f"{summary['simulated_time_s']:g} s ({summary['steps']} steps)"),
        ("last ignition", "-" if last is None else f"{last:,.0f} s"),
        ("still burning", f"{counts['BURNING']:,} cells"),
        ("wall clock", f"{summary['wall_clock_s']:.2f} s"),
    ]
    _report(f"Run: {cfg.name}", rows, out)


# ----------------------------------------------------------------------------- ensemble


@app.command()
def ensemble(
    scenario: Annotated[Path, typer.Argument(metavar="SCENARIO", help="Scenario YAML file.")],
    n: Annotated[
        int | None, typer.Option("--n", min=1, help="Number of runs. Default: the scenario's ensemble.n_runs.")
    ] = None,
    seed: Annotated[
        int | None, typer.Option(min=0, help="Base seed of the ensemble. Default: the scenario's ensemble.base_seed.")
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Number of worker processes.")] = 1,
    out: Annotated[Path | None, typer.Option(help="Output directory. Default: runs/<scenario name>-ensemble.")] = None,
    force: Annotated[bool, typer.Option(help=_FORCE_HELP)] = False,
) -> None:
    """Run a Monte Carlo ensemble of seeded, perturbed simulations and save the statistics.

    Data: ensemble.npz (burn probability, arrival times in s, burned areas in m^2, seeds), runs.csv, summary.json.

    Records: provenance.json, scenario.yaml. Figures: burn_probability.png, burned_area.png.

    Results are identical for any number of --workers.
    """
    cfg = _load(scenario)
    ens = cfg.ensemble.model_copy(
        update={
            "n_runs": cfg.ensemble.n_runs if n is None else n,
            "base_seed": cfg.ensemble.base_seed if seed is None else seed,
        }
    )
    cfg = cfg.model_copy(update={"ensemble": ens})
    out = Path("runs") / f"{cfg.name}-ensemble" if out is None else out
    with _user_errors("Error"):
        io.check_output_dir(out, force=force)
    with _user_errors("Cannot run scenario"):
        result = run_ensemble(cfg, n_runs=ens.n_runs, base_seed=ens.base_seed, n_workers=workers)
    with _user_errors("Cannot write results"):
        io.prepare_output_dir(out, force=force, owned=io.ENSEMBLE_OUTPUTS)
        summary = io.write_ensemble(result, cfg, out, workers=workers)

    ci = summary["ci95_burned_area_m2"]
    ci_text = "-" if ci is None else f"{ci[0] / io.M2_PER_HA:,.2f} - {ci[1] / io.M2_PER_HA:,.2f} ha"
    rows = [
        ("runs / base seed", f"{summary['n']} / {summary['base_seed']}"),
        ("mean burned area", f"{summary['mean_burned_area_ha']:,.2f} ha"),
        ("95 % CI of mean", ci_text),
        (
            "median / p90",
            f"{summary['median_burned_area_m2'] / io.M2_PER_HA:,.2f} / "
            f"{summary['p90_burned_area_m2'] / io.M2_PER_HA:,.2f} ha",
        ),
        (
            "min / max",
            f"{summary['min_burned_area_m2'] / io.M2_PER_HA:,.2f} / "
            f"{summary['max_burned_area_m2'] / io.M2_PER_HA:,.2f} ha",
        ),
        ("wall clock", f"{summary['wall_clock_s']:.2f} s ({workers} worker(s))"),
    ]
    _report(f"Ensemble: {cfg.name}", rows, out)


# ----------------------------------------------------------------------------- plot


@app.command()
def plot(
    directory: Annotated[Path, typer.Argument(metavar="DIR", help="Run or ensemble directory written by this CLI.")],
    gif: Annotated[bool, typer.Option("--gif", help="Run directories: also write fire.gif.")] = False,
    frames: Annotated[int, typer.Option(min=1, help="Number of GIF frames.")] = 40,
    fps: Annotated[float, typer.Option(min=0.1, help="GIF playback speed, frames per second.")] = 10.0,
) -> None:
    """Redraw the figures of a run or ensemble directory from its saved data, without re-running.

    Run directory: final.png, arrival.png (and fire.gif with --gif).

    Ensemble directory: burn_probability.png, burned_area.png.
    """
    with _user_errors("Error"):
        kind = io.dir_kind(directory)
    with _user_errors("Cannot plot"):
        if kind == "run":
            paths = io.plot_run_dir(directory, gif=gif, frames=frames, fps=fps)
        else:
            paths = io.plot_ensemble_dir(directory)
    for path in paths:
        console.print(f"[green]Wrote[/green] {escape(str(path))}", soft_wrap=True)


if __name__ == "__main__":
    app()
