"""Monte Carlo ensembles (WP 1.6): many seeded, perturbed runs of one scenario.

See docs/plan.md section 6. An ensemble of ``n`` runs of a :class:`ScenarioConfig` gives a
burn-probability map, per-run arrival times and burned areas, and the statistics helpers
:func:`running_mean`, :func:`mean_ci` and :func:`runs_for_precision` that decide how many runs
are enough. Each run also reports whether and when it reached the edge of its grid
(``EnsembleResult.reached_edge_s``, WP 1.11); such runs are lower bounds on their fires.

**Seeds.** ``children = SeedSequence(base_seed).spawn(n_runs)``; run ``i`` uses the integer seed
``int(children[i].generate_state(1, dtype=np.uint64)[0] >> 1)`` (a 63-bit non-negative integer)
and calls :func:`wildfire_sim.sim.run` with it. ``SeedSequence.spawn`` gives child ``i`` the spawn
key ``(i,)`` whatever ``n_runs`` is, so run ``i`` depends only on ``(base_seed, i)``: not on
``n_runs`` and not on the worker count. See :func:`derive_seeds`.

**Perturbation draws.** :func:`wildfire_sim.sim.run` splits its seed into three streams,
``SeedSequence(seed).spawn(3)`` = spread / perturbation / suppression, and does not use the
perturbation stream itself. The ensemble draws run ``i``'s input perturbation from
``default_rng(SeedSequence(seed_i).spawn(3)[1])`` (:func:`perturbation_rng`), in this fixed order:

1. ``z = rng.standard_normal(4)``, always all four, even when a sigma is 0, so switching one
   perturbation on does not change the others' values;
2. ``wind_speed_factor = exp(wind_speed_sigma_frac * z[0])`` (log-normal, median 1);
3. ``wind_direction_offset_deg = wind_direction_sigma_deg * z[1]`` (normal, degrees);
4. ``moisture_factor = exp(moisture_sigma_frac * z[2])`` (log-normal, median 1);
5. ``ros_multiplier = exp(ros_multiplier_sigma_frac * z[3])`` (log-normal, median 1);
6. only if ``ignition_jitter_cells = j > 0``: ``rng.integers(-j, j + 1, size=(n_cells, 2))``,
   one ``(d_row, d_col)`` per ignition cell (:func:`jitter_ignition`).

The spread stream is untouched, so the link thresholds of run ``i`` are the same whatever the
perturbation settings are. With every sigma 0 and no jitter, run ``i`` is bit-for-bit
``sim.run(config, seed=seed_i)``.

**Applying them.** Wind speed is multiplied by the factor, the direction offset is added and
wrapped into [0, 360), and the five fuel moistures (1 h, 10 h, 100 h, live herbaceous, live
woody) are multiplied by the moisture factor, at every point of the weather series. With a
per-cell wind field (``weather.wind_field``, WP 3.4) the same speed factor multiplies every
cell's speed and the same offset is added to every cell's direction
(:func:`perturbed_wind_field`), so the whole field moves together. The rate
multiplier wraps the spread rule (:class:`ScaledRateRule`) and scales ``SpreadOutput.ros`` only;
fireline intensity and flame length are left as the rule computed them (the multiplier stands
for uncertainty in the rate, not in the fire's energy release).
"""

from __future__ import annotations

import math
import multiprocessing
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from wildfire_sim import sim
from wildfire_sim.config import EnsembleConfig, ScenarioConfig
from wildfire_sim.data import build_grid
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.base import SpreadOutput, SpreadRule
from wildfire_sim.types import CellState, FloatArray, IntArray, Stencil
from wildfire_sim.wind_field import WindFieldSeries, wind_field_for

PERTURBATION_KEYS: tuple[str, ...] = (
    "wind_speed_factor",
    "wind_direction_offset_deg",
    "moisture_factor",
    "ros_multiplier",
)
"""Keys of the per-run factors returned by :func:`draw_factors`, in draw order."""

_MOISTURE_FIELDS = ("moisture_1h", "moisture_10h", "moisture_100h", "moisture_live_herb", "moisture_live_woody")


# ----------------------------------------------------------------------------- result


@dataclass(frozen=True)
class EnsembleResult:
    """Outcome of :func:`run_ensemble`. Run ``i`` is entry ``i`` of every per-run field.

    Attributes:
        burn_probability: ``(H, W)`` float64 in [0, 1], fraction of runs in which the cell
            burned (ignited at any time).
        arrival_times: ``(n, H, W)`` float32, s since run start; NaN where that run did not
            burn the cell.
        burned_area_m2: ``(n,)`` float64, burned area of each run, m^2.
        seeds: The integer seed passed to :func:`wildfire_sim.sim.run` for each run.
        perturbations: Per run, the factors and offsets actually applied: the keys of
            :data:`PERTURBATION_KEYS` plus ``ignition_drow_k`` / ``ignition_dcol_k`` (cells)
            for each ignition cell ``k`` (0 when there is no jitter or it fell back to the
            original cell). All neutral (1 or 0) when perturbation is off.
        elapsed_s: Wall-clock time of the whole ensemble, s.
        reached_edge_s: ``(n,)`` float64, the time, s, at which each run first reached the edge
            band of the grid (``RunResult.provenance["reached_edge_s"]``); NaN for a run that
            stayed inside. A run with a finite value is a lower bound on its fire: its burned
            area, and every statistic that includes it, understates what would have burned.
    """

    burn_probability: FloatArray
    arrival_times: FloatArray
    burned_area_m2: FloatArray
    seeds: tuple[int, ...]
    perturbations: tuple[dict[str, float], ...]
    elapsed_s: float
    reached_edge_s: FloatArray

    @property
    def n_reached_edge(self) -> int:
        """Number of runs that reached the edge band of the grid."""
        return int(np.count_nonzero(np.isfinite(self.reached_edge_s)))


# ----------------------------------------------------------------------------- seeds and draws


def derive_seeds(base_seed: int, n_runs: int) -> tuple[int, ...]:
    """Integer run seeds: ``int(SeedSequence(base_seed).spawn(n_runs)[i].generate_state(1, uint64)[0] >> 1)``.

    Seed ``i`` depends only on ``(base_seed, i)``, so ``derive_seeds(b, 8)[:4] == derive_seeds(b, 4)``.

    Raises:
        ValueError: ``base_seed < 0`` or ``n_runs < 1``.
    """
    if base_seed < 0:
        raise ValueError(f"base_seed must be >= 0, got {base_seed}")
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")
    children = np.random.SeedSequence(base_seed).spawn(n_runs)
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0] >> 1) for child in children)


def perturbation_rng(seed: int) -> np.random.Generator:
    """Generator on the perturbation stream of run seed ``seed``: ``SeedSequence(seed).spawn(3)[1]``.

    :func:`wildfire_sim.sim.run` uses streams 0 (spread) and 2 (suppression) of the same split.
    """
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1])


def draw_factors(rng: np.random.Generator, ensemble: EnsembleConfig) -> dict[str, float]:
    """Draw one run's input factors (steps 1-5 of the module docstring).

    Always consumes exactly four standard normals from ``rng``, whatever the sigmas are.

    Returns:
        ``wind_speed_factor`` (dimensionless, > 0), ``wind_direction_offset_deg`` (degrees),
        ``moisture_factor`` and ``ros_multiplier`` (dimensionless, > 0). Exactly 1, 0, 1, 1 when
        the corresponding sigma is 0.
    """
    z = rng.standard_normal(4)
    return {
        "wind_speed_factor": math.exp(ensemble.wind_speed_sigma_frac * float(z[0])),
        "wind_direction_offset_deg": ensemble.wind_direction_sigma_deg * float(z[1]) + 0.0,  # + 0.0: no -0.0
        "moisture_factor": math.exp(ensemble.moisture_sigma_frac * float(z[2])),
        "ros_multiplier": math.exp(ensemble.ros_multiplier_sigma_frac * float(z[3])),
    }


def jitter_ignition(
    rng: np.random.Generator, cells: Sequence[tuple[int, int]], grid: Grid, jitter: int
) -> tuple[list[tuple[int, int]], IntArray]:
    """Move each ignition cell by a uniform integer offset in ``[-jitter, jitter]`` cells.

    Draws ``rng.integers(-jitter, jitter + 1, size=(len(cells), 2))`` (nothing when ``jitter`` is
    0). The moved cell is clipped to the grid; if it is not FUEL in ``grid.state`` the original
    cell is kept.

    Args:
        rng: Perturbation-stream generator.
        cells: ``(row, col)`` ignition cells.
        grid: Grid whose ``state`` marks FUEL (call ``apply_unburnable`` first).
        jitter: Maximum offset, cells (>= 0).

    Returns:
        The new cells and the ``(len(cells), 2)`` int array of offsets actually applied.
    """
    if jitter < 0:
        raise ValueError(f"jitter must be >= 0 cells, got {jitter}")
    applied = np.zeros((len(cells), 2), dtype=np.int64)
    if jitter == 0 or not cells:
        return [(int(r), int(c)) for r, c in cells], applied
    h, w = grid.shape
    offsets = rng.integers(-jitter, jitter + 1, size=(len(cells), 2))
    out: list[tuple[int, int]] = []
    for k, ((r, c), (dr, dc)) in enumerate(zip(cells, offsets.tolist(), strict=True)):
        nr, nc = min(max(r + dr, 0), h - 1), min(max(c + dc, 0), w - 1)
        if grid.state[nr, nc] != int(CellState.FUEL):
            nr, nc = r, c
        out.append((int(nr), int(nc)))
        applied[k] = (nr - r, nc - c)
    return out, applied


# ----------------------------------------------------------------------------- applying perturbations


class ScaledRateRule:
    """Spread-rule wrapper multiplying the inner rule's ``ros`` by a constant.

    ``intensity`` and ``flame_length`` are passed through unchanged and ``heat_per_area`` is
    dropped, so the simulator scales the head intensity by ``R_n / max_k ros`` (WP 1.12). Both
    rates carry the multiplier, which cancels: a run's intensities do not depend on it.
    ``name`` is the inner rule's name.

    Args:
        inner: The rule to wrap.
        multiplier: Rate-of-spread factor, dimensionless; finite and > 0.
    """

    def __init__(self, inner: SpreadRule, multiplier: float) -> None:
        multiplier = float(multiplier)
        if not (math.isfinite(multiplier) and multiplier > 0):
            raise ValueError(f"multiplier must be finite and > 0, got {multiplier!r}")
        self.inner = inner
        self.multiplier = multiplier
        self.name = inner.name

    @property
    def lb_capped(self) -> np.ndarray | None:
        """The inner rule's ``(H, W)`` bool mask of cells under the lattice cap, or None.

        The simulator counts ``provenance["lb_capped_cell_hours"]`` from this attribute of the
        rule it is given (WP 2.1d); without it a scaled run would report 0.
        """
        return getattr(self.inner, "lb_capped", None)

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        """Inner output with ``ros`` (``(K, H, W)``, m/s) scaled by :attr:`multiplier`; no ``heat_per_area``."""
        out = self.inner.rate_of_spread(grid, weather, stencil)
        return SpreadOutput(
            ros=out.ros * self.multiplier, intensity=out.intensity, flame_length=out.flame_length, heat_per_area=None
        )


def perturb_config(
    config: ScenarioConfig, factors: dict[str, float], ignition_cells: list[tuple[int, int]] | None = None
) -> ScenarioConfig:
    """``config`` with the weather factors applied and, optionally, new ignition cells.

    Every weather point gets ``wind_speed * wind_speed_factor``, ``(wind_direction +
    wind_direction_offset_deg) mod 360`` and all five fuel moistures times ``moisture_factor``.
    ``ros_multiplier`` is not applied here (see :class:`ScaledRateRule`). When
    ``ignition_cells`` is given it replaces ``ignition.cells`` and ``ignition.center`` is
    cleared. The result is re-validated. ``weather.wind_field`` is kept as it is: the file is
    not perturbed; apply the factors to it with :func:`perturbed_wind_field`.
    """
    points = []
    for p in config.weather.series:
        direction = (p.wind_direction + factors["wind_direction_offset_deg"]) % 360.0
        if direction >= 360.0:  # a tiny negative value rounds up to 360.0
            direction = 0.0
        update: dict[str, Any] = {
            "wind_speed": p.wind_speed * factors["wind_speed_factor"],
            "wind_direction": direction,
        }
        for name in _MOISTURE_FIELDS:
            update[name] = getattr(p, name) * factors["moisture_factor"]
        points.append({**p.model_dump(), **update})
    data = config.model_dump()
    data["weather"] = {**data["weather"], "series": points}
    if ignition_cells is not None:
        data["ignition"] = {**data["ignition"], "cells": [list(c) for c in ignition_cells], "center": False}
    return ScenarioConfig.model_validate(data)


def perturbed_wind_field(config: ScenarioConfig, factors: dict[str, float]) -> WindFieldSeries | None:
    """The scenario's wind field with the wind factors of ``factors`` applied, or None.

    Every cell's speed is multiplied by ``wind_speed_factor`` and ``wind_direction_offset_deg``
    (degrees) is added to every cell's "from" bearing, as :func:`perturb_config` does to the
    weather series. None when the scenario has no ``weather.wind_field``.

    Raises:
        FileNotFoundError: The scenario's wind field file does not exist.
    """
    wind = wind_field_for(config)
    if wind is None:
        return None
    speed, offset = factors.get("wind_speed_factor", 1.0), factors.get("wind_direction_offset_deg", 0.0)
    return wind if speed == 1.0 and offset == 0.0 else wind.perturbed(speed, offset)


def _resolved_ignition_cells(config: ScenarioConfig, shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Explicit ignition cells, then the grid centre if ``center`` is set (the simulator's order).

    Not de-duplicated, so entry ``k`` always matches ``ignition_drow_k`` / ``ignition_dcol_k``.
    """
    cells = [(int(r), int(c)) for r, c in config.ignition.cells]
    if config.ignition.center:
        cells.append((shape[0] // 2, shape[1] // 2))
    return cells


# ----------------------------------------------------------------------------- one run


_RunOutput = tuple[np.ndarray, float, int, dict[str, float], float]


def _run_member(config_data: dict[str, Any], seed: int) -> _RunOutput:
    """One ensemble member. Module level so that a ``spawn`` pool can pickle it.

    Args:
        config_data: ``ScenarioConfig.model_dump(mode="json")``.
        seed: Run seed.

    Returns:
        ``(arrival (H, W) float32 s, burned area m^2, seed, perturbation dict, time the run
        reached the grid edge in s or NaN)``.
    """
    config = ScenarioConfig.model_validate(config_data)
    ens = config.ensemble
    rng = perturbation_rng(seed)
    factors = draw_factors(rng, ens)
    perturbation: dict[str, float] = dict(factors)

    grid: Grid | None = None
    new_cells: list[tuple[int, int]] | None = None
    if ens.ignition_jitter_cells > 0:
        grid = build_grid(config.grid)
        grid.apply_unburnable()
        cells = _resolved_ignition_cells(config, grid.shape)
        new_cells, applied = jitter_ignition(rng, cells, grid, ens.ignition_jitter_cells)
    else:
        applied = np.zeros((len(config.ignition.cells) + int(config.ignition.center), 2), dtype=np.int64)
    for k, (dr, dc) in enumerate(applied.tolist()):
        perturbation[f"ignition_drow_{k}"] = float(dr)
        perturbation[f"ignition_dcol_{k}"] = float(dc)

    neutral_weather = factors["wind_speed_factor"] == 1.0 and factors["wind_direction_offset_deg"] == 0.0
    neutral_weather = neutral_weather and factors["moisture_factor"] == 1.0
    run_config = config if neutral_weather and new_cells is None else perturb_config(config, factors, new_cells)

    rule: SpreadRule = make_spread_rule(config.spread.rule, **config.spread.params)
    if factors["ros_multiplier"] != 1.0:
        rule = ScaledRateRule(rule, factors["ros_multiplier"])

    wind = perturbed_wind_field(config, factors)
    result = sim.run(run_config, seed=seed, rule=rule, grid=grid, record_events=False, wind_field=wind)
    arrival = result.arrival_time.astype(np.float32)
    area = float(np.count_nonzero(result.burned_mask)) * result.grid.dx * result.grid.dx
    edge = result.provenance["reached_edge_s"]
    return arrival, area, seed, perturbation, float("nan") if edge is None else float(edge)


def _run_member_star(args: tuple[dict[str, Any], int]) -> _RunOutput:
    return _run_member(*args)


# ----------------------------------------------------------------------------- ensemble


def run_ensemble(
    config: ScenarioConfig,
    n_runs: int | None = None,
    base_seed: int | None = None,
    n_workers: int = 1,
) -> EnsembleResult:
    """Run ``n_runs`` seeded, perturbed simulations of ``config``.

    Seeds and perturbations are derived as described in the module docstring; perturbation
    settings come from ``config.ensemble``. ``n_workers == 1`` runs serially in this process;
    ``n_workers > 1`` uses a ``multiprocessing`` pool with the ``spawn`` start method (Windows
    and Linux) of ``min(n_workers, n_runs)`` processes. Results are in run order and identical
    bit-for-bit for any worker count.

    Args:
        config: Scenario.
        n_runs: Number of runs (>= 1); None uses ``config.ensemble.n_runs``.
        base_seed: Ensemble seed (>= 0); None uses ``config.ensemble.base_seed``.
        n_workers: Number of processes (>= 1).

    Returns:
        An :class:`EnsembleResult` with ``(n, H, W)`` arrival times.

    Raises:
        ValueError: ``n_runs < 1``, ``base_seed < 0`` or ``n_workers < 1``; or a run fails (for
            example an ignition cell that is not fuel).
    """
    start = time.perf_counter()
    n = config.ensemble.n_runs if n_runs is None else int(n_runs)
    base = config.ensemble.base_seed if base_seed is None else int(base_seed)
    if n < 1:
        raise ValueError(f"n_runs must be >= 1, got {n}")
    if base < 0:
        raise ValueError(f"base_seed must be >= 0, got {base}")
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    data = config.model_dump(mode="json")
    tasks = [(data, seed) for seed in derive_seeds(base, n)]
    if n_workers == 1:
        outputs = [_run_member_star(task) for task in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=min(n_workers, n)) as pool:
            outputs = pool.map(_run_member_star, tasks, chunksize=1)

    arrival_times = np.stack([o[0] for o in outputs]).astype(np.float32, copy=False)
    burned_area = np.array([o[1] for o in outputs], dtype=np.float64)
    burn_probability = np.mean(~np.isnan(arrival_times), axis=0, dtype=np.float64)
    return EnsembleResult(
        burn_probability=burn_probability,
        arrival_times=arrival_times,
        burned_area_m2=burned_area,
        seeds=tuple(o[2] for o in outputs),
        perturbations=tuple(o[3] for o in outputs),
        elapsed_s=time.perf_counter() - start,
        reached_edge_s=np.array([o[4] for o in outputs], dtype=np.float64),
    )


# ----------------------------------------------------------------------------- statistics


def _as_sample(values: FloatArray, minimum: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < minimum:
        raise ValueError(f"need at least {minimum} value(s), got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def running_mean(values: FloatArray) -> FloatArray:
    """Cumulative mean: ``out[i] = mean(values[: i + 1])``.

    Args:
        values: ``(n,)`` per-run values (any units), n >= 1.

    Returns:
        ``(n,)`` float64 in the units of ``values``.
    """
    arr = _as_sample(values, 1)
    return np.cumsum(arr) / np.arange(1, arr.size + 1)


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError(f"level must lie in (0, 1), got {level}")


def mean_ci(values: FloatArray, level: float = 0.95) -> tuple[float, float, float]:
    """Sample mean and its two-sided Student-t confidence interval.

    ``mean +- t_{(1 + level) / 2, n - 1} * s / sqrt(n)`` with ``s`` the sample standard
    deviation (``ddof = 1``).

    Args:
        values: ``(n,)`` per-run values (any units), n >= 2.
        level: Confidence level in (0, 1).

    Returns:
        ``(mean, lower, upper)`` in the units of ``values``.
    """
    _check_level(level)
    arr = _as_sample(values, 2)
    # Work on values scaled to [-1, 1] so neither the sum nor the variance can overflow.
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        return 0.0, 0.0, 0.0
    unit = arr / scale
    mean = float(unit.mean())
    sd = float(unit.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2, arr.size - 1)) * sd / math.sqrt(arr.size) if sd > 0.0 else 0.0
    return mean * scale, (mean - half) * scale, (mean + half) * scale


def runs_for_precision(values: FloatArray, rel_half_width: float, level: float = 0.95) -> int:
    """Smallest number of runs whose CI half-width is at most ``rel_half_width * |mean|``.

    Mean and standard deviation are estimated from ``values`` (a pilot ensemble) and held
    fixed; the answer is the smallest ``n >= 2`` with ``t_{(1 + level) / 2, n - 1} * s / sqrt(n)
    <= rel_half_width * |mean|``. It may be smaller or larger than ``len(values)``.

    Args:
        values: ``(m,)`` pilot values, m >= 2.
        rel_half_width: Target half-width as a fraction of ``|mean|`` (> 0).
        level: Confidence level in (0, 1).

    Raises:
        ValueError: Bad inputs, or the mean is 0 while the spread is not (no finite answer).
    """
    _check_level(level)
    if not rel_half_width > 0:
        raise ValueError(f"rel_half_width must be > 0, got {rel_half_width}")
    arr = _as_sample(values, 2)
    sd = float(arr.std(ddof=1))
    target = rel_half_width * abs(float(arr.mean()))
    if sd == 0.0:
        return 2
    if target == 0.0:
        raise ValueError("the mean is 0, so no number of runs reaches a relative precision")
    q = 0.5 + level / 2
    # t > z, so the normal-theory count is a lower bound; step up from there.
    n = max(2, math.floor((float(stats.norm.ppf(q)) * sd / target) ** 2))
    while float(stats.t.ppf(q, n - 1)) * sd / math.sqrt(n) > target:
        n += 1
    return n


__all__ = [
    "PERTURBATION_KEYS",
    "EnsembleResult",
    "ScaledRateRule",
    "derive_seeds",
    "draw_factors",
    "jitter_ignition",
    "mean_ci",
    "perturb_config",
    "perturbation_rng",
    "run_ensemble",
    "running_mean",
    "runs_for_precision",
]
