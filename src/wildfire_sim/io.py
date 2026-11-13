"""Run and ensemble directories: writing results to disk and reading them back (WP 1.8).

The CLI (:mod:`wildfire_sim.cli`) is a thin layer over these functions; each can be used and
tested on its own.

A **run directory** (:func:`write_run`) holds:

* ``grid.npz`` - final grid (:meth:`wildfire_sim.grid.Grid.save`);
* ``arrival_time.npy`` - ``(H, W)`` float64 ignition time of each cell, s; NaN = never burned;
* ``events.csv`` - one row per ignition, header ``t_s,kind,row,col``;
* ``summary.json`` - burned cells and area, simulated time, steps, timings, final state counts
  and ``reached_edge_s`` (when the fire reached the edge of its grid, s; ``null`` if it did not);
* ``provenance.json`` - :attr:`wildfire_sim.sim.RunResult.provenance`;
* ``scenario.yaml`` - the resolved scenario (with the seed actually used);
* ``final.png``, ``arrival.png`` and, optionally, ``fire.gif`` - figures.

An **ensemble directory** (:func:`write_ensemble`) holds:

* ``ensemble.npz`` - compressed ``burn_probability`` ``(H, W)``, ``arrival_times`` ``(n, H, W)``
  float32 s, ``burned_area_m2`` ``(n,)``, ``seeds`` ``(n,)`` int64, ``reached_edge_s`` ``(n,)``
  (s; NaN for a run that stayed inside its grid) and ``dx_m`` (cell size, m);
* ``runs.csv`` - one row per run: index, seed, burned area (m^2, ha), ``reached_edge_s`` (empty
  if the run stayed inside its grid) and every perturbation;
* ``summary.json`` - burned-area statistics (mean with 95 % CI, median, min, max, 90th
  percentile), the number of runs that reached the grid edge and the earliest such time;
* ``provenance.json`` - resolved config, seeds, git commit, package versions, kernel settings;
* ``scenario.yaml`` - the resolved scenario (with ``ensemble.n_runs`` / ``base_seed`` used);
* ``burn_probability.png``, ``burned_area.png`` - figures.

All JSON is strict (NaN and infinities are written as ``null``). Figures are drawn off-screen
with :mod:`wildfire_sim.viz`; nothing opens a window.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from wildfire_sim import sim, viz
from wildfire_sim.config import ScenarioConfig, dump_yaml
from wildfire_sim.data import build_grid
from wildfire_sim.ensemble import EnsembleResult, mean_ci, running_mean
from wildfire_sim.grid import Grid
from wildfire_sim.types import CellState, FloatArray

__all__ = [
    "ENSEMBLE_OUTPUTS",
    "M2_PER_HA",
    "RUN_OUTPUTS",
    "OutputDirError",
    "SavedRun",
    "burned_area_figure",
    "check_output_dir",
    "dir_kind",
    "ensemble_summary",
    "jsonable",
    "load_run",
    "plot_ensemble_dir",
    "plot_run_dir",
    "prepare_output_dir",
    "run_summary",
    "write_ensemble",
    "write_json",
    "write_run",
]

M2_PER_HA = 10_000.0
"""Square metres per hectare."""

RUN_OUTPUTS: tuple[str, ...] = (
    "grid.npz",
    "arrival_time.npy",
    "events.csv",
    "summary.json",
    "provenance.json",
    "scenario.yaml",
    "final.png",
    "arrival.png",
    "fire.gif",
)
"""Files a run directory may contain (``fire.gif`` only when requested)."""

ENSEMBLE_OUTPUTS: tuple[str, ...] = (
    "ensemble.npz",
    "runs.csv",
    "summary.json",
    "provenance.json",
    "scenario.yaml",
    "burn_probability.png",
    "burned_area.png",
)
"""Files an ensemble directory contains."""

# Provenance entries that describe the code and kernel rather than one particular run.
_SHARED_PROVENANCE_KEYS = (
    "git_commit",
    "versions",
    "rule",
    "stencil",
    "arrival_shape",
    "kernel_correction",
    "kernel_correction_source",
    "residence_time_s",
    "holdover_s",
)


class OutputDirError(ValueError):
    """A directory cannot be used: not empty without ``force``, missing, or not a run directory."""


# ----------------------------------------------------------------------------- generic helpers


def jsonable(obj: Any) -> Any:
    """``obj`` converted to plain JSON types; NaN and +-inf become None (JSON ``null``).

    Mappings become dicts with string keys, tuples/lists/arrays become lists and numpy
    scalars become Python numbers.
    """
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, data: Any) -> Path:
    """Write ``data`` to ``path`` as indented strict JSON (see :func:`jsonable`); return the path."""
    path.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def check_output_dir(path: Path, *, force: bool) -> None:
    """Check, without touching anything, that results may be written to ``path``.

    ``path`` may be missing or an empty directory; a non-empty directory needs ``force``.

    Raises:
        OutputDirError: ``path`` exists and is not a directory, or is a non-empty directory and
            ``force`` is False.
    """
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise OutputDirError(f"output path {path} exists and is not a directory")
    if not force and path.is_dir() and any(path.iterdir()):
        raise OutputDirError(f"output directory {path} exists and is not empty; pass --force to overwrite")


def prepare_output_dir(path: Path, *, force: bool, owned: Iterable[str]) -> Path:
    """Create ``path`` if needed and make sure results may be written into it.

    Checks ``path`` with :func:`check_output_dir`. With ``force``, the files named in ``owned``
    (outputs of an earlier run) are deleted so no stale output survives; any other file in the
    directory is left alone.

    Raises:
        OutputDirError: As :func:`check_output_dir`.
    """
    path = Path(path)
    check_output_dir(path, force=force)
    if force and path.is_dir():
        for name in owned:
            target = path / name
            if target.is_file():
                target.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------------------------------------------------------- single runs


@dataclass  # not frozen: viz.HasArrival declares plain (writable) attributes
class SavedRun:
    """A run read back from disk; satisfies :class:`wildfire_sim.viz.HasArrival`.

    Attributes:
        grid: Final grid from ``grid.npz``.
        arrival_time: ``(H, W)`` float64 ignition times from ``arrival_time.npy``, s; NaN = never burned.
        name: Scenario name from ``summary.json`` ("" if unknown).
    """

    grid: Grid
    arrival_time: FloatArray
    name: str = ""


def run_summary(result: sim.RunResult, name: str) -> dict[str, Any]:
    """Headline numbers of one run.

    Returns:
        Dict with ``scenario``, ``seed``, ``burned_cells``, ``burned_area_m2``, ``burned_area_ha``,
        ``simulated_time_s``, ``steps``, ``last_ignition_s`` (None if nothing burned),
        ``reached_edge_s`` (time, s, at which the fire reached the edge band of its grid, or
        None; when set, the burned area is a lower bound), ``wall_clock_s`` and
        ``final_state_counts`` (cells per :class:`CellState` name).
    """
    burned = int(np.count_nonzero(result.burned_mask))
    area_m2 = burned * result.grid.dx**2
    finite = result.arrival_time[np.isfinite(result.arrival_time)]
    return {
        "scenario": name,
        "seed": int(result.provenance["seed"]),
        "burned_cells": burned,
        "burned_area_m2": area_m2,
        "burned_area_ha": area_m2 / M2_PER_HA,
        "simulated_time_s": float(result.provenance["final_time_s"]),
        "steps": int(result.provenance["n_steps"]),
        "last_ignition_s": float(finite.max()) if finite.size else None,
        "reached_edge_s": result.provenance.get("reached_edge_s"),
        "wall_clock_s": result.elapsed_s,
        "final_state_counts": {s.name: int(np.count_nonzero(result.grid.state == int(s))) for s in CellState},
    }


def _plot_run(run: SavedRun, out: Path, *, gif: bool, frames: int, fps: float) -> list[Path]:
    label = f": {run.name}" if run.name else ""
    paths = [
        viz.save_figure(viz.plot_state(run.grid, title=f"Final state{label}"), out / "final.png"),
        viz.save_figure(viz.plot_arrival(run, title=f"Fire arrival time{label}"), out / "arrival.png"),
    ]
    if gif:
        paths.append(viz.animate(run, out / "fire.gif", n_frames=frames, fps=fps))
    return paths


def write_run(
    result: sim.RunResult,
    config: ScenarioConfig,
    out: Path,
    *,
    gif: bool = False,
    frames: int = 40,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Write every file of a run directory into ``out`` (which must exist).

    Args:
        result: Output of :func:`wildfire_sim.sim.run`.
        config: The scenario that was run (written to ``scenario.yaml``).
        out: Target directory.
        gif: Also write ``fire.gif``.
        frames: Number of GIF frames.
        fps: GIF playback speed, frames per second.

    Returns:
        The summary written to ``summary.json`` (see :func:`run_summary`).
    """
    out = Path(out)
    result.grid.save(out / "grid.npz")
    np.save(out / "arrival_time.npy", np.asarray(result.arrival_time, dtype=np.float64))
    with (out / "events.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_s", "kind", "row", "col"])
        writer.writerows(result.events)
    summary = run_summary(result, config.name)
    write_json(out / "summary.json", summary)
    write_json(out / "provenance.json", result.provenance)
    dump_yaml(config, out / "scenario.yaml")
    saved = SavedRun(result.grid, np.asarray(result.arrival_time, dtype=np.float64), config.name)
    _plot_run(saved, out, gif=gif, frames=frames, fps=fps)
    return summary


def load_run(path: Path) -> SavedRun:
    """Read ``grid.npz``, ``arrival_time.npy`` and the scenario name of a run directory."""
    path = Path(path)
    grid = Grid.load(path / "grid.npz")
    arrival = np.load(path / "arrival_time.npy", allow_pickle=False).astype(np.float64)
    if arrival.shape != grid.shape:
        raise OutputDirError(f"{path}: arrival_time.npy has shape {arrival.shape}, grid has {grid.shape}")
    return SavedRun(grid, arrival, str(_read_json(path / "summary.json").get("scenario", "")))


def plot_run_dir(path: Path, *, gif: bool = False, frames: int = 40, fps: float = 10.0) -> list[Path]:
    """Redraw ``final.png``, ``arrival.png`` (and ``fire.gif`` if ``gif``) from a run directory's data."""
    return _plot_run(load_run(path), Path(path), gif=gif, frames=frames, fps=fps)


# ----------------------------------------------------------------------------- ensembles


def ensemble_summary(result: EnsembleResult, base_seed: int, workers: int) -> dict[str, Any]:
    """Burned-area statistics of an ensemble, m^2 (and the mean in ha).

    The 95 % confidence interval of the mean (Student t, :func:`wildfire_sim.ensemble.mean_ci`)
    is ``null`` for a single run. ``n_reached_edge`` counts the runs whose fire reached the edge
    band of the grid and ``first_reached_edge_s`` is the earliest time, s, at which one did
    (``null`` if none). When ``n_reached_edge > 0`` those runs are lower bounds, so every
    burned-area statistic here understates the fire.
    """
    area = np.asarray(result.burned_area_m2, dtype=np.float64)
    n = int(area.size)
    edge = np.asarray(result.reached_edge_s, dtype=np.float64)
    edge = edge[np.isfinite(edge)]
    ci: list[float] | None = list(mean_ci(area, 0.95)[1:]) if n >= 2 else None
    mean = float(area.mean())
    return {
        "n": n,
        "base_seed": int(base_seed),
        "mean_burned_area_m2": mean,
        "mean_burned_area_ha": mean / M2_PER_HA,
        "ci95_burned_area_m2": ci,
        "median_burned_area_m2": float(np.median(area)),
        "min_burned_area_m2": float(area.min()),
        "max_burned_area_m2": float(area.max()),
        "p90_burned_area_m2": float(np.percentile(area, 90)),
        "n_reached_edge": int(edge.size),
        "first_reached_edge_s": float(edge.min()) if edge.size else None,
        "workers": int(workers),
        "wall_clock_s": result.elapsed_s,
    }


def _shared_provenance(config: ScenarioConfig) -> tuple[dict[str, Any], float]:
    """Code/kernel provenance of ``config`` (see :func:`sim.model_provenance`) and its cell size, m."""
    return sim.model_provenance(config), float(build_grid(config.grid).dx)


def _burned_area_panels(areas_m2: FloatArray, fig: Figure) -> None:
    area_ha = np.asarray(areas_m2, dtype=np.float64) / M2_PER_HA
    hist_ax, mean_ax = fig.subplots(1, 2)
    hist_ax.hist(area_ha, bins="auto", color="#0072B2", edgecolor="white")
    hist_ax.axvline(float(area_ha.mean()), color="#D55E00", linestyle="--", label="mean")
    hist_ax.set_xlabel("Burned area (ha)")
    hist_ax.set_ylabel("Number of runs")
    hist_ax.set_title("Burned area per run")
    hist_ax.legend(fontsize="small")

    runs = np.arange(1, area_ha.size + 1)
    mean_ax.plot(runs, running_mean(area_ha), color="#0072B2", marker="." if area_ha.size <= 50 else None)
    if area_ha.size >= 2:
        _, low, high = mean_ci(area_ha)
        mean_ax.axhspan(low, high, color="#E69F00", alpha=0.3, label="95 % CI of final mean")
        mean_ax.legend(fontsize="small")
    mean_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    mean_ax.set_xlabel("Number of runs")
    mean_ax.set_ylabel("Running mean burned area (ha)")
    mean_ax.set_title("Convergence of the mean")


def burned_area_figure(areas_m2: FloatArray) -> Figure:
    """Off-screen figure: histogram of burned area (ha) and its running mean versus run count.

    Args:
        areas_m2: ``(n,)`` burned area of each run, m^2 (n >= 1).
    """
    fig = Figure(figsize=(10.0, 4.0), dpi=100.0, layout="constrained")
    FigureCanvasAgg(fig)
    _burned_area_panels(areas_m2, fig)
    return fig


def _plot_ensemble(probability: FloatArray, areas_m2: FloatArray, dx: float, out: Path, name: str) -> list[Path]:
    label = f": {name}" if name else ""
    ax = viz.plot_burn_probability(probability, dx, title=f"Burn probability{label}")
    return [
        viz.save_figure(ax, out / "burn_probability.png"),
        viz.save_figure(burned_area_figure(areas_m2), out / "burned_area.png"),
    ]


def write_ensemble(result: EnsembleResult, config: ScenarioConfig, out: Path, *, workers: int) -> dict[str, Any]:
    """Write every file of an ensemble directory into ``out`` (which must exist).

    Args:
        result: Output of :func:`wildfire_sim.ensemble.run_ensemble`.
        config: The scenario that was run; ``config.ensemble.base_seed`` must be the base seed used.
        out: Target directory.
        workers: Number of worker processes used (recorded only).

    Returns:
        The summary written to ``summary.json`` (see :func:`ensemble_summary`).
    """
    out = Path(out)
    base_seed = config.ensemble.base_seed
    shared, dx = _shared_provenance(config)
    seeds = np.array(result.seeds, dtype=np.int64)
    np.savez_compressed(
        out / "ensemble.npz",
        burn_probability=result.burn_probability,
        arrival_times=result.arrival_times,
        burned_area_m2=result.burned_area_m2,
        seeds=seeds,
        reached_edge_s=np.asarray(result.reached_edge_s, dtype=np.float64),
        dx_m=np.float64(dx),
    )
    keys = list(result.perturbations[0]) if result.perturbations else []
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "seed", "burned_area_m2", "burned_area_ha", "reached_edge_s", *keys])
        for i, (seed, area, edge, pert) in enumerate(
            zip(
                result.seeds,
                result.burned_area_m2.tolist(),
                np.asarray(result.reached_edge_s, dtype=np.float64).tolist(),
                result.perturbations,
                strict=True,
            )
        ):
            edge_text = edge if math.isfinite(edge) else ""
            writer.writerow([i, seed, area, area / M2_PER_HA, edge_text, *(pert[k] for k in keys)])
    summary = ensemble_summary(result, base_seed, workers)
    write_json(out / "summary.json", summary)
    provenance = {
        "config": config.model_dump(mode="json"),
        "base_seed": base_seed,
        "n": len(result.seeds),
        "seeds": list(result.seeds),
        "workers": workers,
        **shared,
    }
    write_json(out / "provenance.json", provenance)
    dump_yaml(config, out / "scenario.yaml")
    _plot_ensemble(result.burn_probability, result.burned_area_m2, dx, out, config.name)
    return summary


def plot_ensemble_dir(path: Path) -> list[Path]:
    """Redraw ``burn_probability.png`` and ``burned_area.png`` from an ensemble directory's data."""
    path = Path(path)
    with np.load(path / "ensemble.npz", allow_pickle=False) as data:
        probability = np.asarray(data["burn_probability"], dtype=np.float64)
        areas = np.asarray(data["burned_area_m2"], dtype=np.float64)
        dx = float(data["dx_m"])
    name = str(_read_json(path / "provenance.json").get("config", {}).get("name", ""))
    return _plot_ensemble(probability, areas, dx, path, name)


# ----------------------------------------------------------------------------- detection


def dir_kind(path: Path) -> Literal["run", "ensemble"]:
    """Whether ``path`` is a run directory (``grid.npz`` + ``arrival_time.npy``) or an ensemble one.

    Raises:
        OutputDirError: ``path`` does not exist, is not a directory, or holds neither set of files.
    """
    path = Path(path)
    if not path.exists():
        raise OutputDirError(f"directory {path} not found")
    if not path.is_dir():
        raise OutputDirError(f"{path} is not a directory")
    if (path / "grid.npz").is_file() and (path / "arrival_time.npy").is_file():
        return "run"
    if (path / "ensemble.npz").is_file():
        return "ensemble"
    raise OutputDirError(
        f"{path} is not a run or ensemble directory (expected grid.npz + arrival_time.npy, or ensemble.npz)"
    )
