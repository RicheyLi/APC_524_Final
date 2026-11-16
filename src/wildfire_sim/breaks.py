"""Break crossing: roads, streams and built lines that hold or are breached (WP 2.9).

A *break* (docs/plan.md section 3) is a road or stream narrower than the barrier threshold
(WP 2.5) or a line a crew built (WP 4.3): its cell keeps its fuel and stores the break's width
in ``Grid.break_width_m``. It does not slow the fire; it holds or it is crossed.

**The breach model.** Wilson's (1988) logistic model of the probability that a head fire of
intensity ``I`` (MW/m) crosses a break ``W`` m wide, with ``T = 1`` when trees stand near it
(:func:`breach_probability`; coefficients, fitted range and the minimum intensity in
``configs/breach.yaml``):

.. math::

    p_b = \\frac{e^b}{1 + e^b}, \\qquad b = 1.36 + 0.36\\,I + (0.61\\,T - 0.99)\\,W

Below the minimum intensity (a placeholder, section 13) a break always holds, so a rule that
reports no intensity never breaches anything. Trees count as present when the break cell or
one of its 8 neighbours has canopy cover above ``tree_cover_min``.

**No chance per time step** (section 4). Each cell gets one uniform number ``u_c`` at the start
of the run, from the fourth child of the run's ``SeedSequence`` (:func:`draw_breach_thresholds`;
the first three children are those of earlier versions, so existing runs keep their draws). A
break is breached the first time ``p_b(I_peak, W, T) >= u_c``, where ``I_peak`` is the highest
intensity any adjacent (8-neighbour) cell has burned with. The engine fixes a cell's intensity
when it ignites (WP 1.12), and ``p_b`` rises with intensity, so the breach happens when the
first adjacent cell whose own intensity reaches the cell's threshold ignites: that cell's
arrival time is the breach time, whatever the time step.

**What happens in the run** (:class:`BreakState`, driven by :func:`wildfire_sim.sim.run` for both
engines):

1. At setup, every FUEL cell with ``break_width_m > 0`` (except initial ignition cells) and
   every constructed line (``fireline``) cell with a width is *held*: it is UNBURNABLE, so it
   stops spread and blocks diagonal and knight moves past it like any barrier. A fireline cell
   without a width stays an unconditional barrier, as before.
2. After each step, held cells next to newly ignited cells are checked. A breached cell turns
   back into fuel (a breached line is removed) and is scheduled to ignite when the fire would
   have crossed it from the neighbour that breached it: that neighbour's arrival plus the
   crossing time of their link (its threshold over its rate). Scheduling matters for fast
   fires: the neighbour may have burned out by the end of the step.
3. At the start of each step, scheduled cells whose time has come and that are still FUEL are
   ignited at their scheduled time, which may lie inside an earlier step; their links are
   credited with the spread time since then, as the engine does for cells lit inside a step.

A grid without breaks never creates a :class:`BreakState` that does anything, so its runs are
bit-identical to earlier versions.

References:
    Wilson, A. A. G. (1988). Width of firebreak that is necessary to stop grass fires: some
    field experiments. Can. J. For. Res. 18, 682-687.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildfire_sim.config import CitedValue, configs_dir, load_yaml
from wildfire_sim.grid import Grid
from wildfire_sim.types import BoolArray, CellState, FloatArray, LandUse, Stencil

if TYPE_CHECKING:
    from wildfire_sim.sim import _FireEngine

_FUEL = int(CellState.FUEL)
_UNBURNABLE = int(CellState.UNBURNABLE)
NEIGHBOURS_8: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# ----------------------------------------------------------------------------- the model


class BreachParams(BaseModel):
    """Schema of ``configs/breach.yaml``."""

    model_config = ConfigDict(extra="forbid")

    intercept: CitedValue
    intensity_coef: CitedValue
    trees_width_coef: CitedValue
    width_coef: CitedValue
    fitted_width_m: tuple[float, float]
    fitted_intensity_mw_m: tuple[float, float]
    fitted_range_source: str = Field(min_length=1)
    min_intensity_mw_m: CitedValue
    tree_cover_min: CitedValue

    @model_validator(mode="after")
    def _check(self) -> BreachParams:
        if self.intensity_coef.value <= 0:
            raise ValueError("intensity_coef must be > 0 (crossing becomes likelier with intensity)")
        if self.width_coef.value >= 0 or self.width_coef.value + self.trees_width_coef.value >= 0:
            raise ValueError("crossing must become less likely with width, with or without trees")
        if self.min_intensity_mw_m.value < 0 or not 0 <= self.tree_cover_min.value < 1:
            raise ValueError("min_intensity_mw_m must be >= 0 and tree_cover_min in [0, 1)")
        return self


def load_breach_params(path: str | Path | None = None) -> BreachParams:
    """Load the parameters; default ``configs_dir() / "breach.yaml"``."""
    return load_yaml(configs_dir() / "breach.yaml" if path is None else path, BreachParams)


def breach_probability(
    intensity_kw_m: npt.ArrayLike,
    width_m: npt.ArrayLike,
    trees: npt.ArrayLike,
    params: BreachParams | None = None,
) -> FloatArray:
    """Probability that a fire of fireline intensity ``intensity_kw_m`` crosses a break (Wilson 1988).

    Args:
        intensity_kw_m: Fireline intensity, kW/m (converted to MW/m); NaN counts as 0.
        width_m: Break width, m.
        trees: Whether trees stand near the break.
        params: Parameters; None loads ``configs/breach.yaml``.

    Returns:
        Probability in [0, 1], broadcast shape of the inputs; 0 below the minimum intensity.
    """
    p = load_breach_params() if params is None else params
    mw = np.nan_to_num(np.asarray(intensity_kw_m, dtype=np.float64), nan=0.0) / 1000.0
    w = np.asarray(width_m, dtype=np.float64)
    t = np.asarray(trees, dtype=np.float64)
    b = p.intercept.value + p.intensity_coef.value * mw + (p.trees_width_coef.value * t + p.width_coef.value) * w
    prob = 1.0 / (1.0 + np.exp(-b))
    return np.where(mw >= p.min_intensity_mw_m.value, prob, 0.0)


def in_fitted_range(intensity_kw_m: npt.ArrayLike, width_m: npt.ArrayLike, params: BreachParams) -> BoolArray:
    """Whether (intensity, width) lies inside the range Wilson's data cover (else an extrapolation)."""
    mw = np.asarray(intensity_kw_m, dtype=np.float64) / 1000.0
    w = np.asarray(width_m, dtype=np.float64)
    (w0, w1), (i0, i1) = params.fitted_width_m, params.fitted_intensity_mw_m
    return (w >= w0) & (w <= w1) & (mw >= i0) & (mw <= i1)


def trees_near(canopy_cover: FloatArray, threshold: float) -> BoolArray:
    """True where the cell or one of its 8 neighbours has canopy cover above ``threshold``."""
    tree = np.asarray(canopy_cover) > threshold
    padded = np.pad(tree, 1, constant_values=False)
    h, w = tree.shape
    out = tree.copy()
    for dr, dc in NEIGHBOURS_8:
        out |= padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
    return out


def draw_breach_thresholds(seed_sequence: np.random.SeedSequence, shape: tuple[int, int]) -> FloatArray:
    """One uniform number in [0, 1) per cell, from its own random stream."""
    return np.random.default_rng(seed_sequence).random(shape)


# ----------------------------------------------------------------------------- in a run


def _has_burnable_fuel(grid: Grid) -> BoolArray:
    burnable_lu = np.isin(grid.land_use, [int(lu) for lu in LandUse if lu.burnable])
    return burnable_lu & (grid.fuel_model > 0)


class BreakState:
    """Held and breached break cells of one run (see the module docstring).

    Args:
        grid: The run's grid (states already set by ``apply_unburnable``); held cells are set
            UNBURNABLE here.
        stencil: The run's stencil (it contains the 8 adjacent directions).
        u: ``(H, W)`` breach thresholds (:func:`draw_breach_thresholds`).
        ignition_cells: Initial ignition cells, which are never held.
        params: Parameters; None loads ``configs/breach.yaml``.
    """

    def __init__(
        self,
        grid: Grid,
        stencil: Stencil,
        u: FloatArray,
        ignition_cells: list[tuple[int, int]],
        params: BreachParams | None = None,
    ) -> None:
        self.params = load_breach_params() if params is None else params
        self.u = np.asarray(u, dtype=np.float64)
        self.h, self.w = grid.shape
        self.k_of = {tuple(map(int, o)): k for k, o in enumerate(stencil.offsets.tolist())}
        missing = [o for o in NEIGHBOURS_8 if o not in self.k_of]
        if missing:
            raise ValueError(f"the stencil lacks the adjacent directions {missing}")
        self.held = np.zeros(grid.shape, dtype=bool)
        self.breached = np.zeros(grid.shape, dtype=bool)
        self.scheduled = np.full(self.h * self.w, np.inf)  # ignition time of breached cells, s
        self.scheduled_speed = np.zeros(self.h * self.w)  # rate of the crossing link, m/s
        self.pending_change = False
        self.n_breached = 0
        self._ignition = np.zeros(grid.shape, dtype=bool)
        for r, c in ignition_cells:
            self._ignition[r, c] = True
        self.trees = trees_near(grid.canopy_cover, self.params.tree_cover_min.value)
        self.n_held_at_start = self.add_new(grid, initial=True)

    @property
    def active_schedule(self) -> bool:
        """Whether a breached cell is still waiting to ignite (the run must not stop yet)."""
        return bool(np.isfinite(self.scheduled).any())

    def add_new(self, grid: Grid, initial: bool = False) -> int:
        """Hold break cells and newly built lines with a width; returns how many were added."""
        width = grid.break_width_m > 0
        candidates = width & ~self.held & ~self.breached & ~self._ignition & _has_burnable_fuel(grid)
        fuel_breaks = candidates & (grid.state == _FUEL) & ~grid.fireline if initial else np.zeros_like(candidates)
        lines = candidates & grid.fireline & (grid.state == _UNBURNABLE)
        new = fuel_breaks | lines
        grid.state[fuel_breaks] = _UNBURNABLE
        self.held |= new
        return int(new.sum())

    def ignite_due(self, engine: _FireEngine, t: float, t_end: float) -> None:
        """Ignite breached cells whose scheduled time is before ``t_end`` and that are still FUEL."""
        due = np.flatnonzero(self.scheduled < t_end)
        if due.size == 0:
            return
        times, speeds = self.scheduled[due].copy(), self.scheduled_speed[due].copy()
        self.scheduled[due] = np.inf
        still = engine.grid.state.reshape(-1)[due] == _FUEL  # the fire may have got there another way
        due, times, speeds = due[still], times[still], speeds[still]
        if due.size == 0:
            return
        step_events: list[tuple[float, str, int, int]] = []
        engine.ignite(due, times, "spread", step_events, link_speed=speeds)
        engine.events.extend(step_events)
        credit = np.maximum(t - times, 0.0)  # spread time already elapsed inside earlier steps
        engine.progress_kn[:, due] = engine.rate[:, due] * credit
        engine.burning = np.concatenate([engine.burning, due.astype(engine.burning.dtype)])

    def evaluate(self, engine: _FireEngine) -> None:
        """Breach held cells next to cells that have ignited (after a step)."""
        held_idx = np.flatnonzero(self.held.reshape(-1))
        if held_idx.size == 0:
            return
        rows, cols = np.divmod(held_idx, self.w)
        grid = engine.grid
        width = grid.break_width_m.reshape(-1)[held_idx]
        trees = self.trees.reshape(-1)[held_idx]
        u = self.u.reshape(-1)[held_idx]
        t_breach = np.full(held_idx.size, np.inf)
        t_cross = np.full(held_idx.size, np.inf)
        speed = np.zeros(held_idx.size)
        thresholds = engine.thresholds_kn
        for dr, dc in NEIGHBOURS_8:
            nr, nc = rows + dr, cols + dc
            ok = (nr >= 0) & (nr < self.h) & (nc >= 0) & (nc < self.w)
            nb = np.where(ok, nr * self.w + nc, 0)
            arrival = np.where(ok, engine.arrival[nb], np.nan)
            ignited = np.isfinite(arrival)
            if not ignited.any():
                continue
            p = breach_probability(engine.front_intensity[nb], width, trees, self.params)
            qualifies = ignited & (p >= u)
            if not qualifies.any():
                continue
            k = self.k_of[(-dr, -dc)]  # link from the neighbour into the held cell
            rate = engine.rate[k, nb]
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = np.where(rate > 0, arrival + thresholds[k, nb] / rate, np.inf)
            t_breach = np.where(qualifies, np.minimum(t_breach, arrival), t_breach)
            better = qualifies & (cross < t_cross)
            t_cross = np.where(better, cross, t_cross)
            speed = np.where(better, rate * engine.link_speed_scale[k], speed)
        hit = np.isfinite(t_breach)
        if not hit.any():
            return
        cells = held_idx[hit]
        r, c = rows[hit], cols[hit]
        self.held[r, c] = False
        self.breached[r, c] = True
        grid.fireline[r, c] = False
        grid.state[r, c] = _FUEL
        self._schedule(cells, t_cross[hit], speed[hit])
        self.n_breached += int(hit.sum())
        self.pending_change = True
        if engine.record_events:
            engine.events.extend(
                (float(tb), "breach", int(rr), int(cc))
                for tb, rr, cc in zip(t_breach[hit].tolist(), r.tolist(), c.tolist(), strict=True)
            )

    def _schedule(self, cells: np.ndarray, times: np.ndarray, speeds: np.ndarray) -> None:
        """Ignite ``cells`` at ``times`` (s); an infinite time (no link can cross) schedules nothing."""
        finite = np.isfinite(times)
        self.scheduled[cells[finite]] = times[finite]
        self.scheduled_speed[cells[finite]] = speeds[finite]

    def provenance(self) -> dict[str, Any]:
        """Counts for ``RunResult.provenance``."""
        return {"break_cells_held": self.n_held_at_start, "breaches": self.n_breached}
