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

**Reaching the grid edge** (WP 1.11, docs/plan.md section 2). The model cannot say what
would have burned beyond the grid, so a fire that reaches the edge is a lower bound from then
on. A cell within ``stencil.max_offset`` cells of the grid edge has a link that leaves the grid;
the earliest arrival time at such a cell is recorded as ``provenance["reached_edge_s"]``
(None if the fire stayed inside) together with an ``"edge"`` event (:func:`first_edge_arrival`).
The run itself is not changed or stopped.

What does depend on ``dt``: weather is piecewise constant and looked up at the start of each
step (a breakpoint that is not a multiple of ``dt`` takes effect at the next step start);
hooks act at step starts; the spread rule is evaluated once per step, so a rule that reads
``grid.state`` sees the state at the step start.
"""

from __future__ import annotations

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
from wildfire_sim.config import CitedValue, IgnitionConfig, ScenarioConfig, configs_dir, load_yaml
from wildfire_sim.data import build_grid
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.base import SpreadOutput, SpreadRule, draw_crossing_thresholds, link_rate_of_spread
from wildfire_sim.types import BoolArray, CellState, FloatArray, IntArray, Stencil, StencilSize, get_stencil

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
        elapsed_s: Wall-clock time of the run, s.
        provenance: JSON-serializable record of what produced the result (resolved config,
            seed, ``start_time_utc`` (ISO 8601 wall-clock instant of simulated time 0, or None),
            git commit, package versions, kernel settings, steps executed, final time) and
            ``reached_edge_s``: the time, s, at which the fire first reached the edge band of
            the grid (see :func:`first_edge_arrival`), or None if it never did. A run with a
            value here is a lower bound on the fire: the model cannot spread beyond the grid.
    """

    grid: Grid
    arrival_time: FloatArray
    burned_mask: BoolArray
    events: list[tuple[float, str, int, int]]
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


def _earliest(idx: IntArray, times: FloatArray) -> tuple[IntArray, FloatArray]:
    """Reduce candidate ``(cell, time)`` pairs to the earliest time per cell (cells ascending)."""
    order = np.lexsort((times, idx))  # by cell, then time
    idx, times = idx[order], times[order]
    first = np.ones(idx.size, dtype=bool)
    first[1:] = idx[1:] != idx[:-1]
    return idx[first], times[first]


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
    ) -> None:
        self.grid = grid
        self.rule = rule
        self.stencil = stencil
        self.h, self.w = grid.shape
        self.k = stencil.size
        self.offsets = [(int(dr), int(dc)) for dr, dc in stencil.offsets.tolist()]
        self.link_length_m = grid.dx * np.asarray(stencil.distances, dtype=np.float64)  # (K,)
        self.correction = correction
        self.residence = residence
        self.holdover = holdover
        self.max_age = max_source_age_s
        self.record_events = record_events

        n = self.h * self.w
        self.thresholds = np.asarray(thresholds, dtype=np.float64).reshape(self.k, n)
        self.progress = np.zeros((self.k, n), dtype=np.float64)
        self.arrival = np.full(n, np.nan, dtype=np.float64)
        self.timer = np.zeros(n, dtype=np.float64)  # remaining flaming time, s
        self.dormant_since = np.full(n, np.nan, dtype=np.float64)  # s; NaN while not dormant
        self.holdover_s = np.zeros(n, dtype=np.float64)  # holdover time of each cell's fuel, s
        self.burning = np.empty(0, dtype=np.intp)  # flat indices of BURNING cells
        self._tentative = np.full(n, np.inf, dtype=np.float64)  # scratch for _cascade, kept all-inf
        self.events: list[tuple[float, str, int, int]] = []

        self.blocked = self._compute_blocked()
        self.weather: Weather | None = None
        self.out: SpreadOutput | None = None
        self.rate = np.zeros((self.k, n), dtype=np.float64)  # progress per second, 1/s

    # -- cached spread-rule output --------------------------------------------------------

    def _compute_blocked(self) -> BoolArray:
        barrier = (self.grid.state == _UNBURNABLE) | ~self.grid.burnable_mask()
        return _blocked_links(barrier, self.stencil)

    def refresh(self, weather: Weather, changed: bool) -> None:
        """Recompute the rule output and link rates if the weather or (``changed``) the grid changed."""
        if changed:
            self.blocked = self._compute_blocked()
        if not changed and self.out is not None and weather == self.weather:
            return
        out = self.rule.rate_of_spread(self.grid, weather, self.stencil)
        out.validate(self.stencil, self.grid.shape)
        link = link_rate_of_spread(out.ros, self.stencil) / self.correction  # (K, H, W), m/s
        link[self.blocked] = 0.0
        self.rate = (link / self.link_length_m[:, None, None]).reshape(self.k, -1)
        self.holdover_s = self.holdover.lookup(self.grid.fuel_model).reshape(-1)
        self.out = out
        self.weather = weather

    # -- life cycle -----------------------------------------------------------------------

    def ignite(
        self, idx: IntArray, times: FloatArray, kind: str, step_events: list[tuple[float, str, int, int]]
    ) -> None:
        """Set cells ``idx`` (flat) BURNING at ``times`` (s) and start their flaming clock."""
        assert self.out is not None
        rows, cols = np.divmod(idx, self.w)
        grid = self.grid
        grid.state[rows, cols] = _BURNING
        self.arrival[idx] = times
        grid.arrival_time[rows, cols] = times
        ros_head = self.out.ros[:, rows, cols].max(axis=0)
        crossing = np.divide(grid.dx, ros_head, out=np.zeros_like(ros_head, dtype=np.float64), where=ros_head > 0)
        self.timer[idx] = crossing + self.residence.lookup(grid.fuel_model[rows, cols])
        if self.record_events:
            step_events.extend(
                (float(t), kind, int(r), int(c))
                for t, r, c in zip(times.tolist(), rows.tolist(), cols.tolist(), strict=True)
            )

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
    ) -> tuple[IntArray, FloatArray]:
        """FUEL cells reached from ``src`` within their windows, with the earliest crossing time (s).

        Args:
            src: Flat indices of the sources.
            start: Time (s) from which each source spreads in this step.
            tau: Spreading time (s) each source has from ``start``.
            fresh: True for cells lit during this step, whose progress starts at 0 at their
                arrival; False for cells burning at the start of the step (stored progress).
        """
        cand_idx: list[IntArray] = []
        cand_t: list[FloatArray] = []
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
        if not cand_idx:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64)
        return _earliest(np.concatenate(cand_idx), np.concatenate(cand_t))

    def _cascade(self, idx: IntArray, times: FloatArray, t: float, t_end: float) -> tuple[IntArray, FloatArray]:
        """Exact in-step propagation from cells first reached at ``times`` (s).

        Inside one step the link rates are constant and a newly lit cell's progress starts at
        0, so its link to ``j`` completes ``threshold / rate`` after its own arrival. The
        arrival times inside the step are therefore shortest-path times, found here by label
        correction: re-propagate from every cell whose tentative arrival improved, until
        nothing improves. Cells stay FUEL until the end, so an arrival can still be revised.

        Returns:
            Flat indices of every cell lit in this step and their arrival times, s.
        """
        best = self._tentative  # persistent (n,) array of +inf; entries touched here are reset
        best[idx] = times
        touched = [idx]
        frontier = idx
        while frontier.size:
            start, tau = self._time_window(frontier, t, t_end, arrival=best[frontier])
            cand, cand_t = self._crossings(frontier, start, tau, fresh=True)
            better = cand_t < best[cand]
            frontier = cand[better]
            best[frontier] = cand_t[better]
            touched.append(frontier)
        lit = np.unique(np.concatenate(touched))
        lit_t = best[lit].copy()
        best[lit] = np.inf
        return lit, lit_t

    def _link_status(self, src: IntArray) -> tuple[BoolArray, BoolArray]:
        """For each cell of ``src``: has it a *live* link, and has it a *stalled* link?

        Both kinds of link lead to a FUEL neighbour inside the grid. A live link has a rate
        > 0 and can still complete. A stalled link has a rate of exactly 0 without being
        blocked by barrier geometry: one of the cells it crosses cannot spread under the
        current weather or retardant, which can change.
        """
        live = np.zeros(src.shape, dtype=bool)
        stalled = np.zeros(src.shape, dtype=bool)
        rows, cols = np.divmod(src, self.w)
        blocked = self.blocked.reshape(self.k, -1)
        for k, (dr, dc) in enumerate(self.offsets):
            tr, tc = rows + dr, cols + dc
            fuel = (tr >= 0) & (tr < self.h) & (tc >= 0) & (tc < self.w)
            fuel[fuel] = self.grid.state[tr[fuel], tc[fuel]] == _FUEL
            moving = self.rate[k, src] > 0
            live |= fuel & moving
            stalled |= fuel & ~moving & ~blocked[k, src]
        return live, stalled

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
        first_idx, first_t = self._crossings(src, start, tau, fresh=False)
        self.progress[:, src] += self.rate[:, src] * tau

        step_events: list[tuple[float, str, int, int]] = []
        if first_idx.size:
            new_idx, new_t = self._cascade(first_idx, first_t, t, t_end)
            self.ignite(new_idx, new_t, "spread", step_events)
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
        source, stalled = self._link_status(src)
        if self.max_age is not None:
            young = (t_end - arrival) < self.max_age
            source &= young
            stalled &= young
        flaming = self.timer[src] > 0
        # Dormant: nothing left to do now, but a stalled link may come back. The clock starts
        # at the end of the first step that leaves the cell dormant and stops when it spreads.
        dormant = stalled & ~source & ~flaming
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
        grid.intensity[in_r, in_c] = np.where(flaming[keep], self.out.intensity[in_r, in_c], 0.0)
        self.burning = src[keep]


# ----------------------------------------------------------------------------- public entry point


def model_provenance(config: ScenarioConfig, rule_name: str | None = None) -> dict[str, Any]:
    """Code and kernel provenance shared by every run of ``config`` (JSON-serializable).

    Keys: ``git_commit`` (or None), ``versions`` (python, numpy, scipy, wildfire_sim, and numba
    when the numba engine is selected), ``engine`` (``config.engine``), ``rule`` (class name of
    the spread rule; the configured registry name when ``rule_name`` is None), ``stencil``,
    ``arrival_shape``, ``kernel_correction`` and its ``kernel_correction_source`` for that
    stencil and shape, ``residence_time_s`` and ``holdover_s`` (the whole tables, seconds).
    :func:`run` adds the per-run entries. Used by ensembles, which have no single run to ask.
    """
    return _model_provenance(
        config, rule_name, load_kernel_corrections(), load_residence_times(), load_holdover_times()
    )


def _model_provenance(
    config: ScenarioConfig,
    rule_name: str | None,
    corrections: KernelCorrectionTable,
    residence: ResidenceTimeTable,
    holdover: HoldoverTable,
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
) -> RunResult:
    """Run one simulation of ``config``.

    Random streams: ``SeedSequence(seed).spawn(3)`` gives the spread, perturbation (reserved
    for WP 1.6) and suppression streams. The spread stream draws the ``(K, H, W)`` link
    thresholds once; hooks receive a generator on the suppression stream, so suppression can
    never change the fire's random draws.

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

    Returns:
        A :class:`RunResult` with ``(H, W)`` arrays.

    Raises:
        ValueError: An ignition cell is outside the grid or not FUEL; the grid already burns;
            ``seed`` is negative; ``max_source_age_s`` is not positive.
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

    g = build_grid(config.grid) if grid is None else grid.copy()
    g.validate()
    g.apply_unburnable()
    if np.any((g.state == _BURNING) | (g.state == _BURNED)):
        raise ValueError("the starting grid must not contain BURNING or BURNED cells")
    ignition_cells = _ignition_cells(config.ignition, g)

    stencil = get_stencil(config.stencil)
    shape_k = config.spread.arrival_shape
    spread_ss, _perturb_ss, suppression_ss = np.random.SeedSequence(seed).spawn(3)
    thresholds = draw_crossing_thresholds(np.random.default_rng(spread_ss), shape_k, stencil, g.shape)
    suppression_rng = np.random.default_rng(suppression_ss)
    corrections = load_kernel_corrections()
    correction = corrections.value_for(config.stencil, shape_k)
    residence = load_residence_times()
    holdover = load_holdover_times()

    engine = engine_cls(g, rule, stencil, thresholds, correction, residence, max_source_age_s, record_events, holdover)
    series = config.weather.to_series()
    dt = config.dt_s
    t_ign = config.ignition.time_s
    ignited = False
    steps_done = 0
    for i in range(config.n_steps):
        t, t_end = i * dt, (i + 1) * dt
        weather = series.at(t)
        changed = False
        for hook in hooks:
            changed = bool(hook(g, t, dt, suppression_rng)) or changed
        engine.refresh(weather, changed)
        if not ignited and t_ign < t_end:
            ignited = True
            _ignite_initial(engine, ignition_cells, t_ign)
        engine.advance(t, t_end)
        steps_done = i + 1
        if ignited and engine.burning.size == 0:
            break

    arrival = engine.arrival.reshape(g.shape)
    edge = first_edge_arrival(arrival, stencil)
    if edge is not None and record_events:
        _insert_edge_event(engine.events, edge)
    elapsed = time.perf_counter() - wall_start
    provenance: dict[str, Any] = {
        "config": config.model_dump(mode="json"),
        "seed": seed,
        "start_time_utc": None if config.start_time_utc is None else config.start_time_utc.isoformat(),
        **_model_provenance(config, type(rule).__name__, corrections, residence, holdover),
        "grid_from": "config" if grid is None else "argument",
        "max_source_age_s": max_source_age_s,
        "n_hooks": len(hooks),
        "n_steps": steps_done,
        "final_time_s": steps_done * dt,
        "reached_edge_s": None if edge is None else edge[0],
    }
    return RunResult(
        grid=g,
        arrival_time=arrival,
        burned_mask=~np.isnan(arrival),
        events=engine.events,
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
    "HoldoverTable",
    "KernelCorrectionEntry",
    "KernelCorrectionTable",
    "ResidenceTimeTable",
    "RunResult",
    "StepHook",
    "edge_band_mask",
    "first_edge_arrival",
    "load_holdover_times",
    "load_kernel_corrections",
    "load_residence_times",
    "model_provenance",
    "run",
]
