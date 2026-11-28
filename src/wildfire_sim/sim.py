"""The simulator time loop (WP 1.3): turns a scenario and a seed into a burned grid.

The model is described in docs/plan.md section 2. In short:

* Every cell-to-neighbour *link* gets one random dimensionless threshold for the whole run
  (:func:`~wildfire_sim.spread.base.draw_crossing_thresholds`).
* A burning cell accumulates *progress* along each of its links at ``R_link / d`` per second,
  where ``R_link`` is the harmonic-mean link rate (:func:`~wildfire_sim.spread.base.link_rate_of_spread`)
  divided by the per-stencil front-speed correction and ``d`` the centre-to-centre distance.
* A FUEL neighbour ignites when the progress of one of its incoming links reaches the
  threshold. The arrival time is interpolated inside the step, and a newly lit cell spends the
  rest of the step spreading (a *cascade* of rounds), so a fast fire is not capped at one cell
  per step.
* A cell stays BURNING while it is a *source* (some link into a FUEL neighbour can still
  complete), still *flaming* (``burn_timer > 0``, set at ignition to the crossing time plus
  the fuel's residence time) or *dormant* (below); then it becomes BURNED for good.
* **Dormant sources** (WP 1.10, docs/plan.md section 2.3). A link whose rate is zero *now*
  (extinction moisture overnight, a full-strength retardant drop) is *stalled*, not dead: the
  rate can come back. A cell that is neither a source nor flaming but has a stalled link into
  a FUEL neighbour stays BURNING as a dormant source. It keeps its link progress, has zero
  ``intensity``, and spreads again from where it stopped when the rate returns. Once it has
  been dormant without a break for its fuel's holdover time (``configs/holdover.yaml``) it
  becomes BURNED. A link blocked by barrier geometry is not stalled, since that cannot end.
  The dormant clock is counted in whole steps, from the end of the first step that leaves the
  cell dormant, and restarts whenever the cell is a source again.

Steps 5-7 of the loop touch only the burning cells (index arrays), never the whole grid; the
spread rule's output and everything derived from it is cached until the weather changes or a
hook reports a change.

The in-step cascade is exact, not a fixed number of rounds: within one step the link rates are
constant and a newly lit cell starts at progress 0, so arrival times inside the step are
shortest-path times, found by label correction (a cell's tentative arrival may still be
revised downward by a later round before the step commits). Hence, for a given seed, arrival
times do not depend on ``dt`` beyond floating-point rounding as long as the rates are constant
over each step.

**Urban cells** (WP 1.13, docs/plan.md section 2.6, :mod:`wildfire_sim.urban`). Urban cells
burn like wildland cells, scaled: after the spread rule has run, the rate of spread of every
urban cell is multiplied by ``ros_factor`` in all directions, and its residence time by
``residence_factor`` (``configs/urban.yaml``). A link between a wildland and an urban cell gets
the harmonic mean of the two, like any other link. With ``ros_factor = 0`` urban cells are
made UNBURNABLE at the start of the run and the town holds.

**Front-normal intensity** (WP 1.12, docs/plan.md section 2.2). A spread rule reports the
fireline intensity of a head fire, but Byram's intensity is heat per unit area times the rate
at which the front advances *normal to itself*, so flanks and the back are far cooler. When a
cell ignites, the engine estimates that rate as ``R_n = 1 / |grad T|`` from a least-squares
plane through the arrival times of the cell and of its stencil neighbours that burned no later
than it (:func:`front_normal_rate`). The cell's intensity is ``heat_per_area * R_n`` when the
rule gives ``SpreadOutput.heat_per_area`` and ``intensity * R_n / max_k ros`` otherwise, in
both cases capped at the rule's head ``intensity``. With fewer than two burned neighbours that
are not collinear with the cell the plane is undetermined and ``R_n`` is the rate of the link
that ignited the cell. Initial ignition cells get the head value. The value is fixed at
ignition, shown in ``grid.intensity`` while the cell flames, and returned for every burned
cell as :attr:`RunResult.front_intensity`. Arrival times do not depend on any of this.

**Reaching the grid edge** (WP 1.11, docs/plan.md section 2). The model cannot say what
would have burned beyond the grid, so a fire that reaches the edge is a lower bound from then
on. A cell within ``stencil.max_offset`` cells of the grid edge has a link that leaves the grid;
the earliest arrival time at such a cell is recorded as ``provenance["reached_edge_s"]``
(None if the fire stayed inside) together with an ``"edge"`` event (:func:`first_edge_arrival`).
The run itself is not changed or stopped.

What does depend on ``dt``: weather is piecewise constant and looked up at the start of each
step (a breakpoint that is not a multiple of ``dt`` takes effect at the next step start; a
per-cell wind field, WP 3.4, is looked up the same way and attached to the step's weather);
hooks act at step starts; the spread rule is evaluated once per step, so a rule that reads
``grid.state`` sees the state at the step start.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.metadata
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import scipy
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wildfire_sim import __version__
from wildfire_sim.breaks import BreakState, draw_breach_thresholds
from wildfire_sim.config import CitedValue, IgnitionConfig, ScenarioConfig, configs_dir, load_yaml
from wildfire_sim.data import build_grid
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.base import (
    SpreadOutput,
    SpreadRule,
    _cells_crossed,
    draw_crossing_thresholds,
    link_rate_of_spread,
)
from wildfire_sim.types import BoolArray, CellState, FloatArray, IntArray, LandUse, Stencil, StencilSize, get_stencil
from wildfire_sim.urban import UrbanModel, load_urban_model, residence_scale, ros_scale, urban_mask
from wildfire_sim.wind_field import WindFieldSeries, wind_field_for

StepHook = Callable[[Grid, float, float, np.random.Generator], bool]
"""Called once per step, before spread, as ``hook(grid, t_s, dt_s, suppression_rng)``.

A hook may edit the suppression layers (``fireline``, ``retardant``) and ``state`` (for example
call ``grid.apply_unburnable()`` after adding a fireline, or set an extinguished cell to
BURNED). It must not ignite cells. It returns True if it changed anything the spread rule or
link blocking depends on (static layers, suppression layers, unburnable cells), which makes
the simulator recompute the spread rule output and the blocked-link mask.
"""

_FUEL = int(CellState.FUEL)
_BURNING = int(CellState.BURNING)
_BURNED = int(CellState.BURNED)
_UNBURNABLE = int(CellState.UNBURNABLE)


# ----------------------------------------------------------------------------- parameter tables


class _FuelModelTable(BaseModel):
    """A duration, s, per fuel model with a default: the schema of the per-fuel parameter tables.

    Attributes:
        default: Value, s, for fuel models without an entry.
        by_fuel_model: Fuel model number (as in ``Grid.fuel_model``) -> value, s.
    """

    model_config = ConfigDict(extra="forbid")

    default: CitedValue
    by_fuel_model: dict[int, CitedValue] = Field(default_factory=dict)

    def _cited(self) -> list[tuple[str | int, CitedValue]]:
        return [("default", self.default), *self.by_fuel_model.items()]

    def lookup(self, fuel_model: IntArray) -> FloatArray:
        """Value, s, for each entry of ``fuel_model`` (any shape; output has the same shape)."""
        fm = np.asarray(fuel_model)
        out = np.full(fm.shape, self.default.value, dtype=np.float64)
        for key, cited in self.by_fuel_model.items():
            out[fm == key] = cited.value
        return out


class ResidenceTimeTable(_FuelModelTable):
    """Flame residence time per fuel model, s (``configs/residence_time.yaml``); every value > 0."""

    @model_validator(mode="after")
    def _positive(self) -> ResidenceTimeTable:
        for key, cited in self._cited():
            if not (cited.value > 0 and np.isfinite(cited.value)):
                raise ValueError(f"residence time for {key!r} must be finite and > 0 s, got {cited.value}")
        return self


class HoldoverTable(_FuelModelTable):
    """Holdover time per fuel model, s (``configs/holdover.yaml``): how long a dormant source lasts.

    A BURNING cell whose only remaining links into FUEL neighbours have a zero rate (see the
    module docstring) becomes BURNED once it has been dormant for this long. Every value is
    finite and >= 0; 0 switches dormancy off for that fuel (the cell burns out at once).
    """

    @model_validator(mode="after")
    def _non_negative(self) -> HoldoverTable:
        for key, cited in self._cited():
            if not (cited.value >= 0 and np.isfinite(cited.value)):
                raise ValueError(f"holdover time for {key!r} must be finite and >= 0 s, got {cited.value}")
        return self


class KernelCorrectionEntry(BaseModel):
    """Front-speed correction (dimensionless) measured for one (stencil, arrival_shape) pair."""

    model_config = ConfigDict(extra="forbid")

    stencil: StencilSize
    arrival_shape: Annotated[float, Field(ge=1)]
    value: Annotated[float, Field(gt=0)]
    source: Annotated[str, Field(min_length=1)]
    note: str = ""


class KernelCorrectionTable(BaseModel):
    """Front-speed corrections (``configs/kernel_correction.yaml``); link rates are divided by them."""

    model_config = ConfigDict(extra="forbid")

    entries: list[KernelCorrectionEntry] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def _unique(cls, entries: list[KernelCorrectionEntry]) -> list[KernelCorrectionEntry]:
        keys = [(e.stencil, e.arrival_shape) for e in entries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate (stencil, arrival_shape) entries in the kernel correction table")
        return entries

    def entry_for(self, stencil: int, arrival_shape: float) -> KernelCorrectionEntry | None:
        """The entry matching ``stencil`` and ``arrival_shape`` exactly, or None."""
        for entry in self.entries:
            if entry.stencil == stencil and entry.arrival_shape == arrival_shape:
                return entry
        return None

    def value_for(self, stencil: int, arrival_shape: float) -> float:
        """Correction (dimensionless) for this pair; 1.0 when the table has no exact match."""
        entry = self.entry_for(stencil, arrival_shape)
        return 1.0 if entry is None else entry.value


def load_residence_times(path: str | Path | None = None) -> ResidenceTimeTable:
    """Load the residence-time table; default ``configs_dir() / "residence_time.yaml"``."""
    return load_yaml(configs_dir() / "residence_time.yaml" if path is None else path, ResidenceTimeTable)


class FlameLengthModel(BaseModel):
    """Byram's (1959) flame length ``L = coefficient * I ** exponent`` (``configs/flame_length.yaml``).

    Attributes:
        coefficient: Flame length, m, at an intensity of 1 kW/m.
        exponent: Dimensionless power of the fireline intensity (kW/m).
    """

    model_config = ConfigDict(extra="forbid")

    coefficient: CitedValue
    exponent: CitedValue

    @model_validator(mode="after")
    def _positive(self) -> FlameLengthModel:
        for key, cited in (("coefficient", self.coefficient), ("exponent", self.exponent)):
            if not (cited.value > 0 and np.isfinite(cited.value)):
                raise ValueError(f"flame length {key} must be finite and > 0, got {cited.value}")
        return self

    def flame_length(self, intensity: FloatArray | float) -> FloatArray:
        """Flame length, m, for fireline intensity ``intensity``, kW/m (>= 0; any shape, same shape out)."""
        value = np.asarray(intensity, dtype=np.float64)
        if np.any(value < 0) or np.any(np.isnan(value)):
            raise ValueError("intensity must be >= 0 kW/m and not NaN")
        return self.coefficient.value * value**self.exponent.value


def load_flame_length_model(path: str | Path | None = None) -> FlameLengthModel:
    """Load Byram's flame-length constants; default ``configs_dir() / "flame_length.yaml"``."""
    return load_yaml(configs_dir() / "flame_length.yaml" if path is None else path, FlameLengthModel)


def flame_length_from_intensity(intensity: FloatArray | float, model: FlameLengthModel | None = None) -> FloatArray:
    """Flame length from fireline intensity (Byram 1959): ``L = 0.0775 * I ** 0.46``.

    Use it on the local, front-normal intensity (``grid.intensity`` or
    :attr:`RunResult.front_intensity`) to get the local flame length that the crew safety limit
    needs (WP 4.3): a flank can be workable while the head is not.

    Args:
        intensity: Fireline intensity, kW/m (>= 0), any shape.
        model: The constants; None loads ``configs/flame_length.yaml``.

    Returns:
        Flame length, m, float64, same shape as ``intensity`` (0 where the intensity is 0).
    """
    return (load_flame_length_model() if model is None else model).flame_length(intensity)


def load_holdover_times(path: str | Path | None = None) -> HoldoverTable:
    """Load the holdover-time table; default ``configs_dir() / "holdover.yaml"``."""
    return load_yaml(configs_dir() / "holdover.yaml" if path is None else path, HoldoverTable)


def load_kernel_corrections(path: str | Path | None = None) -> KernelCorrectionTable:
    """Load the front-speed correction table; default ``configs_dir() / "kernel_correction.yaml"``."""
    return load_yaml(configs_dir() / "kernel_correction.yaml" if path is None else path, KernelCorrectionTable)


# ----------------------------------------------------------------------------- result


@dataclass(frozen=True)
class RunResult:
    """Outcome of one simulation run.

    Attributes:
        grid: Final grid; ``state``, ``burn_timer`` (s), ``intensity`` (kW/m) and
            ``arrival_time`` (s, float32) reflect the end of the run.
        arrival_time: ``(H, W)`` float64, seconds since run start at which each cell ignited
            (interpolated inside the step); NaN exactly where the cell never ignited.
        burned_mask: ``(H, W)`` bool, True where the cell ignited at any time.
        events: ``(t_s, kind, row, col)`` in time order. One per ignition, with ``kind``
            ``"ignition"`` (initial ignition) or ``"spread"``, plus at most one ``"edge"`` event
            directly after the ignition of the first cell to burn within ``stencil.max_offset``
            cells of the grid edge. Empty if events were not recorded.
        front_intensity: ``(H, W)`` float64, fireline intensity, kW/m, each cell had when the
            front crossed it: the rule's head value scaled to the local front-normal rate (see
            the module docstring). NaN exactly where the cell never ignited. ``grid.intensity``
            holds the same value only while a cell is flaming.
        elapsed_s: Wall-clock time of the run, s.
        provenance: JSON-serializable record of what produced the result (resolved config,
            seed, ``start_time_utc`` (ISO 8601 wall-clock instant of simulated time 0, or None),
            git commit, package versions, kernel settings, steps executed, final time) and
            ``reached_edge_s``: the time, s, at which the fire first reached the edge band of
            the grid (see :func:`first_edge_arrival`), or None if it never did. A run with a
            value here is a lower bound on the fire: the model cannot spread beyond the grid.
            ``lb_capped_cell_hours``: burning cells whose fire ellipse the spread rule capped at
            what the lattice represents (WP 2.1d), times the time they burned capped, cell-hours;
            0 for rules that do not cap. Read from the rule's ``lb_capped`` attribute (the
            ``(H, W)`` mask of its last evaluation) when it has one. Where it is above 0 the
            fire was drawn rounder than its wind and slope imply.
    """

    grid: Grid
    arrival_time: FloatArray
    burned_mask: BoolArray
    events: list[tuple[float, str, int, int]]
    front_intensity: FloatArray
    elapsed_s: float
    provenance: dict[str, Any]


# ----------------------------------------------------------------------------- helpers


def _blocked_links(barrier: BoolArray, stencil: Stencil) -> BoolArray:
    """Links whose straight segment squeezes between two barrier cells.

    A diagonal move ``(dr, dc)`` is blocked when both ``(r + dr, c)`` and ``(r, c + dc)`` are
    barriers; a knight move ``(+-2, +-1)`` when both cells of the row it steps through first are,
    i.e. ``(r + sign(dr), c)`` and ``(r + sign(dr), c + dc)``; and symmetrically for ``(+-1, +-2)``.
    Cells beyond the grid edge are not barriers.

    Args:
        barrier: ``(H, W)`` bool, True for unburnable cells.
        stencil: Stencil with ``K`` directions.

    Returns:
        ``(K, H, W)`` bool; ``out[k, r, c]`` is True when the link from ``(r, c)`` in direction
        ``k`` is blocked.
    """
    h, w = barrier.shape
    m = stencil.max_offset
    padded = np.pad(barrier, m, constant_values=False)

    def at(dr: int, dc: int) -> BoolArray:
        return padded[m + dr : m + dr + h, m + dc : m + dc + w]

    blocked = np.zeros((stencil.size, h, w), dtype=bool)
    for k, (dr, dc) in enumerate(stencil.offsets.tolist()):
        sr, sc = int(np.sign(dr)), int(np.sign(dc))
        if abs(dr) == 1 and abs(dc) == 1:
            blocked[k] = at(dr, 0) & at(0, dc)
        elif abs(dr) == 2 and abs(dc) == 1:
            blocked[k] = at(sr, 0) & at(sr, dc)
        elif abs(dr) == 1 and abs(dc) == 2:
            blocked[k] = at(0, sc) & at(dr, sc)
    return blocked


def _earliest(idx: IntArray, times: FloatArray, speeds: FloatArray) -> tuple[IntArray, FloatArray, FloatArray]:
    """Reduce candidate ``(cell, time, link speed)`` triples to the earliest time per cell (cells ascending).

    Among candidates with the same earliest time the fastest link is kept, so the choice does
    not depend on the order in which candidates were found.
    """
    order = np.lexsort((-speeds, times, idx))  # by cell, then time, then speed descending
    idx, times, speeds = idx[order], times[order], speeds[order]
    first = np.ones(idx.size, dtype=bool)
    first[1:] = idx[1:] != idx[:-1]
    return idx[first], times[first], speeds[first]


def _ignition_cells(ignition: IgnitionConfig, grid: Grid) -> list[tuple[int, int]]:
    """Validated, de-duplicated ``(row, col)`` cells of the initial ignition."""
    h, w = grid.shape
    cells = [(int(r), int(c)) for r, c in ignition.cells]
    if ignition.center:
        cells.append((h // 2, w // 2))
    unique = list(dict.fromkeys(cells))
    for r, c in unique:
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError(f"ignition cell ({r}, {c}) is outside the grid of shape ({h}, {w})")
        if grid.state[r, c] != _FUEL:
            state = CellState(int(grid.state[r, c])).name
            raise ValueError(f"ignition cell ({r}, {c}) is not burnable fuel (state {state})")
    return unique


_COLLINEAR_TOL = 1e-9  # sim_fast._step_py repeats this literal
"""Relative size of the determinant below which the burned neighbours count as collinear."""


def _plane_fit_rate(
    arrival: FloatArray, idx: IntArray, offsets: IntArray, h: int, w: int, dx: float
) -> tuple[FloatArray, BoolArray]:
    """Front-normal rate of spread at cells ``idx`` from a plane fitted to nearby arrival times.

    For each cell, the points are the cell itself and every stencil neighbour inside the grid
    whose arrival time is finite and not later than the cell's. A plane ``T = c + g . x`` is
    fitted to them by least squares (``x`` in metres, east and south, relative to the cell;
    ``T`` relative to the cell's arrival), and the rate is ``1 / |g|``. The compiled engine
    (``sim_fast._step_py``, step 4b) evaluates the same formulas cell by cell.

    Args:
        arrival: ``(H * W,)`` float64 arrival times, s; NaN where never ignited.
        idx: ``(m,)`` flat indices of the cells (each with a finite arrival time).
        offsets: ``(K, 2)`` int ``(d_row, d_col)`` per stencil direction.
        h, w: Grid shape.
        dx: Cell size, m.

    Returns:
        ``(rate, ok)``: ``(m,)`` float64 rate, m/s (``inf`` where the fitted gradient is zero,
        as where two fronts meet), and ``(m,)`` bool, False where fewer than two neighbours
        qualify or they are collinear with the cell (``rate`` is then NaN).
    """
    rows, cols = np.divmod(idx, w)
    tr = rows[None, :] + offsets[:, 0:1]  # (K, m)
    tc = cols[None, :] + offsets[:, 1:2]
    inside = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
    tn = arrival[np.where(inside, tr * w + tc, 0)]
    with np.errstate(invalid="ignore"):
        use = inside & (tn <= arrival[idx])  # False for NaN (not burned yet)
    tau = np.where(use, tn - arrival[idx], 0.0)  # (K, m), s, <= 0
    used = use.astype(np.float64)
    x = offsets[:, 1].astype(np.float64) * dx  # (K,) east, m
    y = offsets[:, 0].astype(np.float64) * dx  # (K,) south, m

    n = 1.0 + used.sum(axis=0)  # the cell itself is the point x = y = t = 0
    sx, sy, st = x @ used, y @ used, tau.sum(axis=0)
    sxx, sxy, syy = (x * x) @ used, (x * y) @ used, (y * y) @ used
    sxt, syt = x @ tau, y @ tau
    cxx = sxx - sx * sx / n
    cxy = sxy - sx * sy / n
    cyy = syy - sy * sy / n
    cxt = sxt - sx * st / n
    cyt = syt - sy * st / n
    det = cxx * cyy - cxy * cxy
    ok = (n >= 3.0) & (det > _COLLINEAR_TOL * (cxx + cyy) ** 2)
    safe = np.where(ok, det, 1.0)
    gx = (cxt * cyy - cyt * cxy) / safe
    gy = (cyt * cxx - cxt * cxy) / safe
    slowness = np.sqrt(gx * gx + gy * gy)  # s/m
    rate = np.full(idx.shape, np.inf)
    np.divide(1.0, slowness, out=rate, where=slowness > 0)
    rate[~ok] = np.nan
    return rate, ok


def front_normal_rate(arrival_time: FloatArray, stencil: Stencil, dx: float) -> FloatArray:
    """Rate at which the front advanced normal to itself through each burned cell, m/s.

    ``R_n = 1 / |grad T|`` with the gradient of the arrival time taken from a least-squares
    plane through each cell and its stencil neighbours that burned no later than it, the
    estimate the simulator uses for the fireline intensity (docs/plan.md section 2.2). It needs
    only the arrival times, so it can be applied to any finished run.

    Args:
        arrival_time: ``(H, W)`` arrival times, s; NaN where the cell never burned.
        stencil: The stencil the run used.
        dx: Cell size, m.

    Returns:
        ``(H, W)`` float64, m/s. NaN where the cell never burned or the plane is undetermined
        (fewer than two earlier-burned neighbours not collinear with the cell: the ignition
        point and cells reached through a one-cell gap); ``inf`` where the gradient is zero.
    """
    arrival = np.asarray(arrival_time, dtype=np.float64)
    h, w = int(arrival.shape[0]), int(arrival.shape[1])
    flat = np.ascontiguousarray(arrival).reshape(-1)
    idx = np.flatnonzero(~np.isnan(flat))
    rate, _ = _plane_fit_rate(flat, idx, np.asarray(stencil.offsets, dtype=np.int64), h, w, dx)
    out = np.full(h * w, np.nan)
    out[idx] = rate
    return out.reshape(h, w)


def edge_band_mask(shape: tuple[int, int], width: int) -> BoolArray:
    """Cells within ``width`` cells of the grid edge.

    Args:
        shape: ``(H, W)``.
        width: Band width, cells (>= 0). ``stencil.max_offset`` gives the cells that have at
            least one stencil link leaving the grid.

    Returns:
        ``(H, W)`` bool, True in the outermost ``width`` rows and columns (everywhere when the
        grid is no larger than ``2 * width`` in either direction).
    """
    if width < 0:
        raise ValueError(f"width must be >= 0 cells, got {width}")
    h, w = shape
    band = np.ones((h, w), dtype=bool)
    if h > 2 * width and w > 2 * width:
        band[width : h - width, width : w - width] = False
    return band


def first_edge_arrival(arrival_time: FloatArray, stencil: Stencil) -> tuple[float, int, int] | None:
    """When and where the fire first reached the edge band of the grid.

    The edge band is the cells within ``stencil.max_offset`` cells of the grid edge
    (:func:`edge_band_mask`): the cells from which a stencil link points outside the grid, where
    the model loses fire it cannot follow.

    Args:
        arrival_time: ``(H, W)`` arrival times, s; NaN where the cell never burned.
        stencil: The stencil the run used.

    Returns:
        ``(t_s, row, col)`` of the earliest arrival in the band (ties go to the lowest row, then
        column), or None if no cell of the band burned.
    """
    arrival = np.asarray(arrival_time, dtype=np.float64)
    band = edge_band_mask((int(arrival.shape[0]), int(arrival.shape[1])), stencil.max_offset)
    times = np.where(band & ~np.isnan(arrival), arrival, np.inf)
    flat = int(np.argmin(times))  # first minimum in row-major order
    row, col = divmod(flat, arrival.shape[1])
    t = float(times[row, col])
    return None if np.isinf(t) else (t, row, col)


def _insert_edge_event(events: list[tuple[float, str, int, int]], edge: tuple[float, int, int]) -> None:
    """Add the ``"edge"`` event directly after the ignition event of the edge cell (in place)."""
    t, row, col = edge
    for i, (_, _, r, c) in enumerate(events):
        if r == row and c == col:
            events.insert(i + 1, (t, "edge", row, col))
            return
    raise AssertionError(f"no ignition event for edge cell ({row}, {col})")  # pragma: no cover


@functools.cache
def _git_commit() -> str | None:
    """Commit hash of the checkout this package runs from, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


# ----------------------------------------------------------------------------- engine


class _FireEngine:
    """Per-run fire state and the per-step update. Private: use :func:`run`.

    Own arrays are flat over cells (index ``r * W + c``); grid layers are indexed ``[r, c]``
    afresh each time, so a hook that replaces a layer array does not leave stale views.
    """

    def __init__(
        self,
        grid: Grid,
        rule: SpreadRule,
        stencil: Stencil,
        thresholds: FloatArray,
        correction: float,
        residence: ResidenceTimeTable,
        max_source_age_s: float | None,
        record_events: bool,
        holdover: HoldoverTable,
        urban: UrbanModel,
    ) -> None:
        self.grid = grid
        self.rule = rule
        self.stencil = stencil
        self.h, self.w = grid.shape
        self.k = stencil.size
        self.offsets = [(int(dr), int(dc)) for dr, dc in stencil.offsets.tolist()]
        self._offsets_array = np.asarray(stencil.offsets, dtype=np.int64)  # (K, 2)
        self.link_length_m = grid.dx * np.asarray(stencil.distances, dtype=np.float64)  # (K,)
        self.correction = correction
        self.residence = residence
        self.holdover = holdover
        self.urban = urban
        self.has_urban = False  # any urban cell on the grid (set by refresh)
        self.residence_scale = np.ones(self.h * self.w, dtype=np.float64)  # urban residence factor, else 1
        self.max_age = max_source_age_s
        self.record_events = record_events

        n = self.h * self.w
        self.thresholds = np.asarray(thresholds, dtype=np.float64).reshape(self.k, n)
        self.progress = np.zeros((self.k, n), dtype=np.float64)
        self.arrival = np.full(n, np.nan, dtype=np.float64)
        self.timer = np.zeros(n, dtype=np.float64)  # remaining flaming time, s
        self.dormant_since = np.full(n, np.nan, dtype=np.float64)  # s; NaN while not dormant
        self.holdover_s = np.zeros(n, dtype=np.float64)  # holdover time of each cell's fuel, s
        self.may_stall = np.zeros(n, dtype=bool)  # has a zero-rate, unblocked link to a burnable cell
        self.burning = np.empty(0, dtype=np.intp)  # flat indices of BURNING cells
        self._tentative = np.full(n, np.inf, dtype=np.float64)  # scratch for _cascade, kept all-inf
        self._tentative_speed = np.zeros(n, dtype=np.float64)  # igniting link speed of _tentative, m/s
        self.front_intensity = np.full(n, np.nan, dtype=np.float64)  # kW/m, fixed at ignition
        self.link_speed_scale = self.link_length_m * correction  # (K,) rate (1/s) -> link rate, m/s
        self.ros_head = np.zeros(n, dtype=np.float64)  # max_k ros of the rule output, m/s
        self.head_intensity = np.zeros(n, dtype=np.float64)  # rule output, kW/m
        self.heat: FloatArray | None = None  # rule output heat per unit area, kJ/m^2, or None
        self.events: list[tuple[float, str, int, int]] = []

        self.blocked = self._compute_blocked()
        self.weather: Weather | None = None
        self.out: SpreadOutput | None = None
        self.rate = np.zeros((self.k, n), dtype=np.float64)  # progress per second, 1/s

    @property
    def progress_kn(self) -> FloatArray:
        """Link progress as ``(K, H * W)`` (a view; the Numba engine stores it transposed)."""
        return self.progress

    @property
    def thresholds_kn(self) -> FloatArray:
        """Link thresholds as ``(K, H * W)`` (a view; the Numba engine stores them transposed)."""
        return self.thresholds

    # -- cached spread-rule output --------------------------------------------------------

    def _compute_blocked(self) -> BoolArray:
        barrier = (self.grid.state == _UNBURNABLE) | ~self.grid.burnable_mask()
        return _blocked_links(barrier, self.stencil)

    def _compute_may_stall(self) -> BoolArray:
        """``(H * W,)`` bool: cells with a link that could be stalled, whatever burns later.

        True where some link inside the grid has a rate of 0, is not blocked by barrier geometry
        and leads to a cell that is not UNBURNABLE. Only these cells can ever be dormant, so
        the burn-out of an ordinary front never has to look at its links twice. Recomputed
        with the link rates (a hook that makes cells UNBURNABLE reports a change).
        """
        state = self.grid.state
        m = self.stencil.max_offset
        burnable = np.pad(state != _UNBURNABLE, m, constant_values=False)  # outside the grid: no target
        rate = self.rate.reshape(self.k, self.h, self.w)
        may = np.zeros((self.h, self.w), dtype=bool)
        for k, (dr, dc) in enumerate(self.offsets):
            target = burnable[m + dr : m + dr + self.h, m + dc : m + dc + self.w]
            may |= target & ~(rate[k] > 0) & ~self.blocked[k]
        return may.reshape(-1)

    def refresh(self, weather: Weather, changed: bool) -> None:
        """Recompute the rule output and link rates if the weather or (``changed``) the grid changed."""
        if changed:
            self.blocked = self._compute_blocked()
        if not changed and self.out is not None and weather == self.weather:
            return
        out = self.rule.rate_of_spread(self.grid, weather, self.stencil)
        out.validate(self.stencil, self.grid.shape)
        self.has_urban = bool(urban_mask(self.grid).any())
        if self.has_urban:  # scale urban cells after the rule, so every rule is scaled the same way
            out = dataclasses.replace(out, ros=out.ros * ros_scale(self.grid, self.urban)[None])
            self.residence_scale = residence_scale(self.grid, self.urban).reshape(-1)
        link = link_rate_of_spread(out.ros, self.stencil) / self.correction  # (K, H, W), m/s
        link[self.blocked] = 0.0
        self.rate = (link / self.link_length_m[:, None, None]).reshape(self.k, -1)
        self.holdover_s = self.holdover.lookup(self.grid.fuel_model).reshape(-1)
        self.may_stall = self._compute_may_stall()
        self.ros_head = np.ascontiguousarray(out.ros.max(axis=0), dtype=np.float64).reshape(-1)
        self.head_intensity = np.ascontiguousarray(out.intensity, dtype=np.float64).reshape(-1)
        self.heat = (
            None if out.heat_per_area is None else np.ascontiguousarray(out.heat_per_area, dtype=np.float64).reshape(-1)
        )
        self.out = out
        self.weather = weather

    def update_barriers(self, cells: IntArray) -> None:
        """Barriers were removed at flat ``cells`` and nothing else changed (breached breaks, WP 2.9).

        Gives the same blocked links, link rates and stall flags as ``refresh(weather,
        changed=True)`` with the current weather, at a cost that grows with the number of cells,
        not the grid: a cell's state only enters the links of sources within ``max_offset`` of it
        (as target or as a cell a diagonal or knight move squeezes past), and the spread rule's
        output does not depend on it. A link that becomes unblocked gets its rate from the cached
        rule output by the formula of :func:`~wildfire_sim.spread.base.link_rate_of_spread`, in
        the same order of operations, so the result is bit-identical.
        """
        assert self.out is not None
        idx = np.unique(np.asarray(cells, dtype=np.int64))
        if idx.size == 0:
            return
        m = self.stencil.max_offset
        rows, cols = np.divmod(idx, self.w)
        span = np.arange(-m, m + 1)
        sr = (rows[:, None, None] + span[None, :, None] + 0 * span[None, None, :]).ravel()
        sc = (cols[:, None, None] + 0 * span[None, :, None] + span[None, None, :]).ravel()
        inside = (sr >= 0) & (sr < self.h) & (sc >= 0) & (sc < self.w)
        src = np.unique(sr[inside] * self.w + sc[inside])
        srow, scol = np.divmod(src, self.w)
        grid = self.grid
        burnable_lu = np.isin(grid.land_use, [int(lu) for lu in LandUse if lu.burnable])

        def barrier(r: IntArray, c: IntArray) -> BoolArray:
            ok = (r >= 0) & (r < self.h) & (c >= 0) & (c < self.w)
            out = np.zeros(r.shape, dtype=bool)
            rr, cc = r[ok], c[ok]
            unburnable = (grid.state[rr, cc] == _UNBURNABLE) | ~(
                burnable_lu[rr, cc] & (grid.fuel_model[rr, cc] > 0) & ~grid.fireline[rr, cc]
            )
            out[ok] = unburnable
            return out

        may = np.zeros(src.size, dtype=bool)
        for k, (dr, dc) in enumerate(self.offsets):
            sgr, sgc = int(np.sign(dr)), int(np.sign(dc))
            if abs(dr) == 1 and abs(dc) == 1:
                blocked = barrier(srow + dr, scol) & barrier(srow, scol + dc)
            elif abs(dr) == 2 and abs(dc) == 1:
                blocked = barrier(srow + sgr, scol) & barrier(srow + sgr, scol + dc)
            elif abs(dr) == 1 and abs(dc) == 2:
                blocked = barrier(srow, scol + sgc) & barrier(srow + dr, scol + sgc)
            else:
                blocked = np.zeros(src.size, dtype=bool)
            tr, tc = srow + dr, scol + dc
            target_in = (tr >= 0) & (tr < self.h) & (tc >= 0) & (tc < self.w)
            rate = self.rate[k, src]
            freed = target_in & self.blocked[k, srow, scol] & ~blocked
            if freed.any():
                rate[freed] = self._free_link_rate(k, srow[freed], scol[freed])
            rate[blocked] = 0.0
            self.rate[k, src] = rate
            self.blocked[k, srow, scol] = blocked
            target_ok = np.zeros(src.size, dtype=bool)
            target_ok[target_in] = grid.state[tr[target_in], tc[target_in]] != _UNBURNABLE
            may |= target_ok & ~(rate > 0) & ~blocked
        self.may_stall[src] = may
        self._links_updated(src)

    def _free_link_rate(self, k: int, rows: IntArray, cols: IntArray) -> FloatArray:
        """Unblocked link rate (progress per second) of direction ``k`` from cells ``(rows, cols)``."""
        assert self.out is not None
        dr, dc = self.offsets[k]
        crossed = [np.asarray(self.out.ros[k, rows + r, cols + c], dtype=np.float64) for r, c in _cells_crossed(dr, dc)]
        passable = np.all([cell > 0 for cell in crossed], axis=0)
        slowness = np.zeros(rows.shape, dtype=np.float64)
        for cell in crossed:
            slowness += np.divide(1.0, cell, out=np.zeros_like(slowness), where=passable)
        link = np.divide(len(crossed), slowness, out=np.zeros_like(slowness), where=passable) / self.correction
        return link / self.link_length_m[k]

    def _links_updated(self, src: IntArray) -> None:
        """Hook: the link rates and blocked flags of sources ``src`` changed (the Numba engine mirrors them)."""

    # -- life cycle -----------------------------------------------------------------------

    def ignite(
        self,
        idx: IntArray,
        times: FloatArray,
        kind: str,
        step_events: list[tuple[float, str, int, int]],
        link_speed: FloatArray | None = None,
    ) -> None:
        """Set cells ``idx`` (flat) BURNING at ``times`` (s); start their flaming clock and fix their intensity.

        ``link_speed`` is the rate, m/s, of the link that ignited each cell (the fallback for
        the front-normal rate); None marks an initial ignition, which gets the head intensity.
        """
        assert self.out is not None
        rows, cols = np.divmod(idx, self.w)
        grid = self.grid
        grid.state[rows, cols] = _BURNING
        self.arrival[idx] = times
        grid.arrival_time[rows, cols] = times
        ros_head = self.out.ros[:, rows, cols].max(axis=0)
        crossing = np.divide(grid.dx, ros_head, out=np.zeros_like(ros_head, dtype=np.float64), where=ros_head > 0)
        residence = self.residence.lookup(grid.fuel_model[rows, cols])
        if self.has_urban:
            residence = residence * self.residence_scale[idx]
        self.timer[idx] = crossing + residence
        self.front_intensity[idx] = self._front_intensity(idx, link_speed)
        if self.record_events:
            step_events.extend(
                (float(t), kind, int(r), int(c))
                for t, r, c in zip(times.tolist(), rows.tolist(), cols.tolist(), strict=True)
            )

    def _front_intensity(self, idx: IntArray, link_speed: FloatArray | None) -> FloatArray:
        """Fireline intensity, kW/m, of cells ``idx`` just ignited (arrival times already stored).

        Head intensity of the rule scaled to the front-normal rate (module docstring); the head
        value itself for an initial ignition (``link_speed`` None).
        """
        head = self.head_intensity[idx]
        if link_speed is None or not head.any():  # initial ignition, or a rule without intensity (cap of 0)
            return head.copy()
        rate, ok = _plane_fit_rate(self.arrival, idx, self._offsets_array, self.h, self.w, self.grid.dx)
        rate = np.where(ok, rate, link_speed)
        if self.heat is not None:
            raw = self.heat[idx] * np.where(np.isinf(rate), 0.0, rate)
        else:
            ros_head = self.ros_head[idx]
            ratio = np.ones_like(rate)
            np.divide(np.where(np.isinf(rate), 0.0, rate), ros_head, out=ratio, where=ros_head > 0)
            raw = head * ratio
        return np.where(np.isinf(rate), head, np.minimum(raw, head))

    def _time_window(
        self, src: IntArray, t: float, t_end: float, arrival: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Start time (s) and spreading time available (s) in ``[t, t_end)`` for sources ``src``.

        ``arrival`` overrides the stored arrival times (s) of ``src`` (tentative cascade times).
        """
        if arrival is None:
            arrival = self.arrival[src]
        start = np.maximum(t, arrival)
        stop = np.full(src.shape, t_end) if self.max_age is None else np.minimum(t_end, arrival + self.max_age)
        return start, np.maximum(stop - start, 0.0)

    def _targets(self, src: IntArray, k: int) -> tuple[BoolArray, IntArray]:
        """For direction ``k``: mask of sources with a live link into FUEL, and those targets (flat)."""
        dr, dc = self.offsets[k]
        rows, cols = np.divmod(src, self.w)
        tr, tc = rows + dr, cols + dc
        ok = (tr >= 0) & (tr < self.h) & (tc >= 0) & (tc < self.w) & (self.rate[k, src] > 0)
        ok[ok] = self.grid.state[tr[ok], tc[ok]] == _FUEL
        return ok, tr[ok] * self.w + tc[ok]

    def _crossings(
        self, src: IntArray, start: FloatArray, tau: FloatArray, *, fresh: bool
    ) -> tuple[IntArray, FloatArray, FloatArray]:
        """FUEL cells reached from ``src`` within their windows: earliest crossing time (s) and link rate (m/s).

        The link rate is that of the link that got there first: the harmonic-mean rate of
        spread along it, without the front-speed correction.

        Args:
            src: Flat indices of the sources.
            start: Time (s) from which each source spreads in this step.
            tau: Spreading time (s) each source has from ``start``.
            fresh: True for cells lit during this step, whose progress starts at 0 at their
                arrival; False for cells burning at the start of the step (stored progress).
        """
        cand_idx: list[IntArray] = []
        cand_t: list[FloatArray] = []
        cand_v: list[FloatArray] = []
        for k in range(self.k):
            ok, targets = self._targets(src, k)
            if targets.size == 0:
                continue
            s = src[ok]
            rate = self.rate[k, s]
            need = self.thresholds[k, s] if fresh else np.maximum(self.thresholds[k, s] - self.progress[k, s], 0.0)
            hit = rate * tau[ok] >= need
            if not hit.any():
                continue
            s_start, s_stop = start[ok][hit], start[ok][hit] + tau[ok][hit]
            cand_idx.append(targets[hit])
            cand_t.append(np.minimum(s_start + need[hit] / rate[hit], s_stop))  # min() guards rounding
            cand_v.append(rate[hit] * self.link_speed_scale[k])
        if not cand_idx:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        return _earliest(np.concatenate(cand_idx), np.concatenate(cand_t), np.concatenate(cand_v))

    def _cascade(
        self, idx: IntArray, times: FloatArray, speeds: FloatArray, t: float, t_end: float
    ) -> tuple[IntArray, FloatArray, FloatArray]:
        """Exact in-step propagation from cells first reached at ``times`` (s).

        Inside one step the link rates are constant and a newly lit cell's progress starts at
        0, so its link to ``j`` completes ``threshold / rate`` after its own arrival. The
        arrival times inside the step are therefore shortest-path times, found here by label
        correction: re-propagate from every cell whose tentative arrival improved, until
        nothing improves. Cells stay FUEL until the end, so an arrival can still be revised.

        Args:
            idx, times, speeds: Cells first reached from the cells burning at the step start,
                their crossing times (s) and the rates (m/s) of the links that reached them.

        Returns:
            Flat indices of every cell lit in this step, their arrival times, s, and the rate,
            m/s, of the link that ignited each.
        """
        best = self._tentative  # persistent (n,) array of +inf; entries touched here are reset
        best_v = self._tentative_speed
        best[idx] = times
        best_v[idx] = speeds
        touched = [idx]
        frontier = idx
        while frontier.size:
            start, tau = self._time_window(frontier, t, t_end, arrival=best[frontier])
            cand, cand_t, cand_v = self._crossings(frontier, start, tau, fresh=True)
            tie = (cand_t == best[cand]) & (cand_v > best_v[cand])  # same time by a faster link
            best_v[cand[tie]] = cand_v[tie]
            better = cand_t < best[cand]
            frontier = cand[better]
            best[frontier] = cand_t[better]
            best_v[frontier] = cand_v[better]
            touched.append(frontier)
        lit = np.unique(np.concatenate(touched))
        lit_t = best[lit].copy()
        lit_v = best_v[lit].copy()
        best[lit] = np.inf
        return lit, lit_t, lit_v

    def _has_live_link(self, src: IntArray) -> BoolArray:
        """Whether each source still has an unblocked link with rate > 0 into a FUEL neighbour."""
        live = np.zeros(src.shape, dtype=bool)
        for k in range(self.k):
            live |= self._targets(src, k)[0]
        return live

    def _has_stalled_link(self, src: IntArray) -> BoolArray:
        """Whether each cell of ``src`` has a *stalled* link into a FUEL neighbour inside the grid.

        A stalled link has a rate of exactly 0 without being blocked by barrier geometry: one of
        the cells it crosses cannot spread under the current weather or retardant, which can
        change. (A live link, by contrast, has a rate > 0 and can still complete.)
        """
        stalled = np.zeros(src.shape, dtype=bool)
        rows, cols = np.divmod(src, self.w)
        blocked = self.blocked.reshape(self.k, -1)
        for k, (dr, dc) in enumerate(self.offsets):
            tr, tc = rows + dr, cols + dc
            ok = (tr >= 0) & (tr < self.h) & (tc >= 0) & (tc < self.w)
            ok &= ~(self.rate[k, src] > 0) & ~blocked[k, src]
            ok[ok] = self.grid.state[tr[ok], tc[ok]] == _FUEL
            stalled |= ok
        return stalled

    def advance(self, t: float, t_end: float) -> None:
        """Spread over the step ``[t, t_end)``: crossings, cascade, progress, burn-out."""
        grid = self.grid
        rows, cols = np.divmod(self.burning, self.w)
        still = grid.state[rows, cols] == _BURNING  # a hook may have extinguished some
        grid.intensity[rows[~still], cols[~still]] = 0.0
        grid.burn_timer[rows[~still], cols[~still]] = 0.0
        self.burning = self.burning[still]

        src = self.burning
        start, tau = self._time_window(src, t, t_end)
        first_idx, first_t, first_v = self._crossings(src, start, tau, fresh=False)
        self.progress[:, src] += self.rate[:, src] * tau

        step_events: list[tuple[float, str, int, int]] = []
        if first_idx.size:
            new_idx, new_t, new_v = self._cascade(first_idx, first_t, first_v, t, t_end)
            self.ignite(new_idx, new_t, "spread", step_events, link_speed=new_v)
            _, new_tau = self._time_window(new_idx, t, t_end)
            self.progress[:, new_idx] = self.rate[:, new_idx] * new_tau  # progress starts at 0 at ignition
            self.burning = np.concatenate([src, new_idx])
        self._burn_out(t, t_end)
        step_events.sort()
        self.events.extend(step_events)

    def _burn_out(self, t: float, t_end: float) -> None:
        """Count down flaming clocks; BURNING cells that are not source, flaming or dormant become BURNED."""
        assert self.out is not None
        src = self.burning
        arrival = self.arrival[src]
        self.timer[src] = np.maximum(self.timer[src] - (t_end - np.maximum(t, arrival)), 0.0)
        source = self._has_live_link(src)
        flaming = self.timer[src] > 0
        idle = ~source & ~flaming & self.may_stall[src]  # candidates for dormancy; none on an ordinary front
        if self.max_age is not None:
            young = (t_end - arrival) < self.max_age
            source &= young
            idle &= young
        # Dormant: nothing left to do now, but a stalled link may come back. The clock starts
        # at the end of the first step that leaves the cell dormant and stops when it spreads.
        dormant = idle
        if idle.any():
            dormant[idle] = self._has_stalled_link(src[idle])
        since = np.where(dormant, self.dormant_since[src], np.nan)
        since[dormant & np.isnan(since)] = t_end
        self.dormant_since[src] = since
        dormant &= ~((t_end - since) >= self.holdover_s[src])  # NaN compares False
        keep = source | flaming | dormant

        grid = self.grid
        rows, cols = np.divmod(src, self.w)
        out_r, out_c = rows[~keep], cols[~keep]
        grid.state[out_r, out_c] = _BURNED
        grid.burn_timer[out_r, out_c] = 0.0
        grid.intensity[out_r, out_c] = 0.0
        in_r, in_c = rows[keep], cols[keep]
        grid.burn_timer[in_r, in_c] = self.timer[src[keep]]
        grid.intensity[in_r, in_c] = np.where(flaming[keep], self.front_intensity[src[keep]], 0.0)
        self.burning = src[keep]


# ----------------------------------------------------------------------------- public entry point


def model_provenance(config: ScenarioConfig, rule_name: str | None = None) -> dict[str, Any]:
    """Code and kernel provenance shared by every run of ``config`` (JSON-serializable).

    Keys: ``git_commit`` (or None), ``versions`` (python, numpy, scipy, wildfire_sim, and numba
    when the numba engine is selected), ``engine`` (``config.engine``), ``rule`` (class name of
    the spread rule; the configured registry name when ``rule_name`` is None), ``stencil``,
    ``arrival_shape``, ``kernel_correction`` and its ``kernel_correction_source`` for that
    stencil and shape, ``residence_time_s`` and ``holdover_s`` (the whole tables, seconds),
    ``urban`` (the urban-cell parameters) and ``wind_field`` (None, or the file, SHA-256 and
    sizes of the scenario's per-cell wind, WP 3.4).
    :func:`run` adds the per-run entries. Used by ensembles, which have no single run to ask.
    """
    return _model_provenance(
        config,
        rule_name,
        load_kernel_corrections(),
        load_residence_times(),
        load_holdover_times(),
        load_urban_model(),
        wind_field_for(config),
    )


def _model_provenance(
    config: ScenarioConfig,
    rule_name: str | None,
    corrections: KernelCorrectionTable,
    residence: ResidenceTimeTable,
    holdover: HoldoverTable,
    urban: UrbanModel,
    wind: WindFieldSeries | None = None,
) -> dict[str, Any]:
    """:func:`model_provenance` from tables already loaded (a run loads each YAML file once)."""
    entry = corrections.entry_for(config.stencil, config.spread.arrival_shape)
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "wildfire_sim": __version__,
    }
    if config.engine == "numba":
        versions["numba"] = _distribution_version("numba")
    return {
        "git_commit": _git_commit(),
        "versions": versions,
        "engine": config.engine,
        "rule": config.spread.rule if rule_name is None else rule_name,
        "stencil": config.stencil,
        "arrival_shape": config.spread.arrival_shape,
        "kernel_correction": corrections.value_for(config.stencil, config.spread.arrival_shape),
        "kernel_correction_source": None if entry is None else entry.source,
        "residence_time_s": residence.model_dump(mode="json"),
        "holdover_s": holdover.model_dump(mode="json"),
        "urban": urban.model_dump(mode="json"),
        "wind_field": None if wind is None else wind.provenance(),
    }


def _distribution_version(name: str) -> str | None:
    """Installed version of distribution ``name`` without importing it, or None if not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _engine_class(name: str) -> type[_FireEngine]:
    """The engine class for ``ScenarioConfig.engine``.

    ``"numba"`` imports :mod:`wildfire_sim.sim_fast`, which imports numba lazily and raises a
    clear ``ImportError`` when it is not installed.
    """
    if name == "numpy":
        return _FireEngine
    if name == "numba":
        from wildfire_sim import sim_fast

        return sim_fast.numba_engine_class()
    raise ValueError(f"unknown engine {name!r}; expected 'numpy' or 'numba'")


def run(
    config: ScenarioConfig,
    seed: int | None = None,
    rule: SpreadRule | None = None,
    *,
    grid: Grid | None = None,
    hooks: Sequence[StepHook] = (),
    record_events: bool = True,
    max_source_age_s: float | None = None,
    wind_field: WindFieldSeries | None = None,
) -> RunResult:
    """Run one simulation of ``config``.

    Random streams: ``SeedSequence(seed).spawn(4)`` gives the spread, perturbation (reserved
    for WP 1.6), suppression and break streams (the first three are those of ``spawn(3)``, so
    earlier runs keep their draws). The spread stream draws the ``(K, H, W)`` link thresholds
    once; hooks receive a generator on the suppression stream, so suppression can never change
    the fire's random draws; the break stream draws one breach threshold per cell
    (:mod:`wildfire_sim.breaks`, WP 2.9).

    Break cells (``break_width_m > 0``, and constructed lines with a width) hold as barriers
    until breached (:class:`wildfire_sim.breaks.BreakState`); breaches are ``"breach"`` events
    and counted in ``provenance["breaks"]``.

    Each step ``[t, t + dt)``: weather lookup at ``t``; hooks; spread rule (cached); the
    initial ignition if ``config.ignition.time_s`` falls in the step; link crossings with
    interpolated arrival times and an in-step cascade; burn-out. The run stops after
    ``config.n_steps`` steps, or earlier once the initial ignition has happened and no cell is
    BURNING. An ignition time at or after ``n_steps * dt_s`` never happens.

    A fire whose rate of spread falls to zero for a while (weather, retardant) does not end:
    its edge cells wait as dormant sources for up to their fuel's holdover time and resume when
    the rate returns (see the module docstring). Such a run keeps stepping through the stall.

    Args:
        config: Scenario. Uses ``grid``, ``weather``, ``spread``, ``ignition``, ``dt_s`` (s),
            ``duration_s`` (s), ``stencil``, ``seed`` and ``engine`` (``"numpy"``, the reference,
            or ``"numba"``, the compiled per-step kernel of :mod:`wildfire_sim.sim_fast`; both
            share all setup here and agree to floating-point rounding).
        seed: Run seed (>= 0); None uses ``config.seed``.
        rule: Spread rule; None builds ``make_spread_rule(config.spread.rule, **config.spread.params)``.
        grid: Starting grid; None builds it from ``config.grid``. A passed grid is copied, never
            mutated; it must contain no BURNING or BURNED cells. ``config.grid`` is then ignored.
        hooks: Callables run once per step before spread (see :data:`StepHook`).
        record_events: Whether to fill :attr:`RunResult.events`.
        max_source_age_s: Optional limit, s, on how long after ignition a cell may spread fire.
        wind_field: Per-cell wind (WP 3.4) to use in place of the file named by
            ``config.weather.wind_field``, for example one perturbed by an ensemble. None loads
            that file when the scenario names one; with neither, the series' scalar wind
            applies everywhere. At each step the field in force is attached to the weather as
            ``Weather.wind_field``; the scalar wind stays the series' (centroid) value.

    Returns:
        A :class:`RunResult` with ``(H, W)`` arrays.

    Raises:
        ValueError: An ignition cell is outside the grid or not FUEL; the grid already burns;
            ``seed`` is negative; ``max_source_age_s`` is not positive; the wind field is for a
            grid of another shape.
        FileNotFoundError: The scenario's wind field file does not exist.
        ImportError: ``config.engine`` is ``"numba"`` and numba is not installed.
    """
    wall_start = time.perf_counter()
    seed = config.seed if seed is None else int(seed)
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if max_source_age_s is not None and not max_source_age_s > 0:
        raise ValueError(f"max_source_age_s must be > 0 s, got {max_source_age_s}")
    engine_cls = _engine_class(config.engine)  # fails fast if the numba engine is not installed
    if rule is None:
        rule = make_spread_rule(config.spread.rule, **config.spread.params)

    urban = load_urban_model()
    g = build_grid(config.grid) if grid is None else grid.copy()
    g.validate()
    wind = wind_field_for(config) if wind_field is None else wind_field
    if wind is not None and wind.shape != g.shape:
        raise ValueError(
            f"the wind field {'' if wind.path is None else str(wind.path) + ' '}is for a {wind.shape} grid, "
            f"but the grid is {g.shape}"
        )
    g.apply_unburnable()
    if not urban.burnable:  # ros_factor 0: the town holds, as before urban cells could burn
        g.state[urban_mask(g) & (g.state == _FUEL)] = _UNBURNABLE
    if np.any((g.state == _BURNING) | (g.state == _BURNED)):
        raise ValueError("the starting grid must not contain BURNING or BURNED cells")
    ignition_cells = _ignition_cells(config.ignition, g)

    stencil = get_stencil(config.stencil)
    shape_k = config.spread.arrival_shape
    spread_ss, _perturb_ss, suppression_ss, breaks_ss = np.random.SeedSequence(seed).spawn(4)
    thresholds = draw_crossing_thresholds(np.random.default_rng(spread_ss), shape_k, stencil, g.shape)
    suppression_rng = np.random.default_rng(suppression_ss)
    breaks = BreakState(g, stencil, draw_breach_thresholds(breaks_ss, g.shape), ignition_cells)
    corrections = load_kernel_corrections()
    correction = corrections.value_for(config.stencil, shape_k)
    residence = load_residence_times()
    holdover = load_holdover_times()

    engine = engine_cls(
        g, rule, stencil, thresholds, correction, residence, max_source_age_s, record_events, holdover, urban
    )
    series = config.weather.to_series()
    dt = config.dt_s
    t_ign = config.ignition.time_s
    ignited = False
    steps_done = 0
    capped_cell_s = 0.0  # burning cells whose ellipse the rule capped (WP 2.1d) x time, cell-s
    grid_changed = False  # hook edits not yet seen by the engine (before the fire, it is not refreshed)
    for i in range(config.n_steps):
        t, t_end = i * dt, (i + 1) * dt
        weather = series.at(t)
        if wind is not None:
            weather = dataclasses.replace(weather, wind_field=wind.at(t))
        changed = False
        for hook in hooks:
            changed = bool(hook(g, t, dt, suppression_rng)) or changed
        if changed:
            breaks.add_new(g)  # lines a hook built with a width are held like any break
        grid_changed = grid_changed or changed
        steps_done = i + 1
        if not ignited and t_ign >= t_end:
            continue  # nothing burns yet: the rule output is not needed until the ignition step
        breached, full_refresh = breaks.take_pending()
        engine.refresh(weather, grid_changed or full_refresh)
        if breached.size and not (grid_changed or full_refresh):
            engine.update_barriers(breached)  # local: only links near the breached cells change
        grid_changed = False
        if not ignited:
            ignited = True
            _ignite_initial(engine, ignition_cells, t_ign)
        breaks.ignite_due(engine, t, t_end)
        capped = getattr(rule, "lb_capped", None)  # set by rules that cap the ellipse, e.g. "rothermel"
        if capped is not None and engine.burning.size:
            capped_cell_s += float(np.count_nonzero(np.asarray(capped).reshape(-1)[engine.burning])) * dt
        engine.advance(t, t_end)
        breaks.evaluate(engine)
        if ignited and engine.burning.size == 0 and not breaks.active_schedule:
            break
    g.state[breaks.held & ~g.fireline] = _FUEL  # breaks that held were never burnable barriers

    arrival = engine.arrival.reshape(g.shape)
    edge = first_edge_arrival(arrival, stencil)
    if edge is not None and record_events:
        _insert_edge_event(engine.events, edge)
    elapsed = time.perf_counter() - wall_start
    provenance: dict[str, Any] = {
        "config": config.model_dump(mode="json"),
        "seed": seed,
        "start_time_utc": None if config.start_time_utc is None else config.start_time_utc.isoformat(),
        **_model_provenance(config, type(rule).__name__, corrections, residence, holdover, urban, wind),
        "grid_from": "config" if grid is None else "argument",
        "max_source_age_s": max_source_age_s,
        "n_hooks": len(hooks),
        "n_steps": steps_done,
        "final_time_s": steps_done * dt,
        "reached_edge_s": None if edge is None else edge[0],
        "lb_capped_cell_hours": capped_cell_s / 3600.0,
        "breaks": breaks.provenance(),
    }
    return RunResult(
        grid=g,
        arrival_time=arrival,
        burned_mask=~np.isnan(arrival),
        events=engine.events,
        front_intensity=engine.front_intensity.reshape(g.shape),
        elapsed_s=elapsed,
        provenance=provenance,
    )


def _ignite_initial(engine: _FireEngine, cells: list[tuple[int, int]], t_ign: float) -> None:
    """Light the initial ignition cells that are still FUEL at ``t_ign`` (s) and make them sources."""
    fuel = [(r, c) for r, c in cells if engine.grid.state[r, c] == _FUEL]
    if not fuel:
        return
    idx = np.array([r * engine.w + c for r, c in fuel], dtype=np.intp)
    times = np.full(idx.size, t_ign, dtype=np.float64)
    step_events: list[tuple[float, str, int, int]] = []
    engine.ignite(idx, times, "ignition", step_events)
    engine.events.extend(step_events)
    engine.burning = np.concatenate([engine.burning, idx])


__all__ = [
    "FlameLengthModel",
    "HoldoverTable",
    "KernelCorrectionEntry",
    "KernelCorrectionTable",
    "ResidenceTimeTable",
    "RunResult",
    "StepHook",
    "edge_band_mask",
    "first_edge_arrival",
    "flame_length_from_intensity",
    "front_normal_rate",
    "load_flame_length_model",
    "load_holdover_times",
    "load_kernel_corrections",
    "load_residence_times",
    "model_provenance",
    "run",
]
