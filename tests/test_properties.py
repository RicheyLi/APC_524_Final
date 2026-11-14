"""Property-based tests (WP 1.9): invariants of the kernel, grid, simulator, config and ensemble.

Hypothesis generates small random inputs (simulator grids of at most 10 x 10 cells, so each
example takes milliseconds) and checks that an invariant holds for every one of them. The
invariants are those of docs/plan.md section 8 plus the exactness claims made in the docstrings
of :mod:`wildfire_sim.sim` and :mod:`wildfire_sim.spread.base`.

Determinism: every test uses the registered settings profile ``wildfire-ci``: ``derandomize=True``
(the same examples on every run and machine) and no example database, so a failure found once
is not silently replayed from local state. (Hypothesis still caches Unicode tables under the
git-ignored ``.hypothesis/``.) ``deadline=None`` because the first
simulator call of a session pays for imports and config parsing.

Simulator examples use :class:`FieldRule`, a rule with an arbitrary per-cell rate field defined
here, so the properties do not depend on the physics of the shipped rules.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from wildfire_sim.config import ScenarioConfig, dump_yaml, load_scenario
from wildfire_sim.ensemble import derive_seeds, mean_ci
from wildfire_sim.grid import LAYER_SPECS, Grid, Weather
from wildfire_sim.sim import RunResult, edge_band_mask, first_edge_arrival, run
from wildfire_sim.spread.base import (
    SpreadOutput,
    draw_crossing_thresholds,
    ignition_probability,
    link_rate_of_spread,
    neighbour_ignition_probability,
)
from wildfire_sim.types import CellState, LandUse, Stencil, get_stencil

settings.register_profile(
    "wildfire-ci",
    deadline=None,
    max_examples=40,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
PROFILE = settings.get_profile("wildfire-ci")
SIM_PROFILE = settings(PROFILE, max_examples=30)
"""Simulator properties run up to three simulations per example, so they use fewer examples."""

STENCIL_SIZES = st.sampled_from([8, 16])


def _stencil(size: int) -> Stencil:
    return get_stencil(8 if size == 8 else 16)


def _offsets(stencil: Stencil) -> list[tuple[int, int]]:
    return [(int(dr), int(dc)) for dr, dc in stencil.offsets.tolist()]


def _opposite(stencil: Stencil) -> list[int]:
    """Index of the direction opposite to each direction of ``stencil``."""
    offsets = _offsets(stencil)
    return [offsets.index((-dr, -dc)) for dr, dc in offsets]


def _cells_crossed(dr: int, dc: int) -> list[tuple[int, int]]:
    """Cells a link's straight segment passes through, end cells included (re-derived here)."""
    if max(abs(dr), abs(dc)) == 1:
        return [(0, 0), (dr, dc)]
    if abs(dr) == 2:
        return [(0, 0), (dr // 2, 0), (dr // 2, dc), (dr, dc)]
    return [(0, 0), (0, dc // 2), (dr, dc // 2), (dr, dc)]


def _inside(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


# ============================================================================= kernel


@st.composite
def kernel_inputs(draw: st.DrawFn) -> dict[str, Any]:
    """Random ``ignition_probability`` arguments on a grid of at most 4 x 4 cells."""
    stencil = _stencil(draw(STENCIL_SIZES))
    h, w = draw(st.integers(1, 4)), draw(st.integers(1, 4))
    ros = arrays(np.float64, (stencil.size, h, w), elements=st.floats(0.0, 5.0))
    return {
        "stencil": stencil,
        "ros": draw(ros),
        "delta": draw(ros),
        "age": draw(arrays(np.float64, (h, w), elements=st.floats(0.0, 3600.0))),
        "dt": draw(st.floats(1e-3, 600.0)),
        "dx": draw(st.floats(1.0, 100.0)),
        "shape": draw(st.one_of(st.just(1.0), st.floats(1.0, 200.0))),
    }


@PROFILE
@given(kernel_inputs())
def test_ignition_probability_in_unit_interval_and_monotone_in_ros(inp: dict[str, Any]) -> None:
    """Probabilities lie in [0, 1] and do not decrease when any rate of spread increases."""
    kw = {"age": inp["age"], "shape": inp["shape"]}
    p = ignition_probability(inp["ros"], inp["dt"], inp["dx"], inp["stencil"], **kw)
    p_faster = ignition_probability(inp["ros"] + inp["delta"], inp["dt"], inp["dx"], inp["stencil"], **kw)
    assert p.shape == inp["ros"].shape
    assert np.all((p >= 0) & (p <= 1))
    assert np.all((p_faster >= 0) & (p_faster <= 1))
    assert np.all(p_faster >= p - 1e-12)


@PROFILE
@given(kernel_inputs(), st.floats(0.01, 0.99))
def test_splitting_a_step_keeps_the_survival_probability(inp: dict[str, Any], frac: float) -> None:
    """Survival over ``[a, a + dt)`` equals survival over ``[a, a + f dt)`` times ``[a + f dt, a + dt)``."""
    ros, dx, stencil, shape, age, dt = inp["ros"], inp["dx"], inp["stencil"], inp["shape"], inp["age"], inp["dt"]
    dt1 = frac * dt
    dt2 = dt - dt1
    whole = 1.0 - ignition_probability(ros, dt, dx, stencil, age=age, shape=shape)
    first = 1.0 - ignition_probability(ros, dt1, dx, stencil, age=age, shape=shape)
    second = 1.0 - ignition_probability(ros, dt2, dx, stencil, age=age + dt1, shape=shape)
    np.testing.assert_allclose(first * second, whole, rtol=1e-9, atol=1e-12)


@st.composite
def neighbour_inputs(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, Stencil]:
    stencil = _stencil(draw(STENCIL_SIZES))
    h, w = draw(st.integers(1, 6)), draw(st.integers(1, 6))
    p = draw(arrays(np.float64, (stencil.size, h, w), elements=st.floats(0.0, 1.0)))
    burning = draw(arrays(np.bool_, (h, w), elements=st.booleans()))
    return p, burning, stencil


@PROFILE
@given(neighbour_inputs())
def test_neighbour_ignition_probability_bounds_and_support(args: tuple[np.ndarray, np.ndarray, Stencil]) -> None:
    """Combined probabilities lie in [0, 1] and are exactly 0 where no stencil neighbour burns."""
    p, burning, stencil = args
    h, w = burning.shape
    out = neighbour_ignition_probability(p, burning, stencil)
    assert out.shape == (h, w)
    assert np.all((out >= 0) & (out <= 1))
    has_burning_nbr = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(burning):
        for dr, dc in _offsets(stencil):
            if _inside(r + dr, c + dc, h, w):
                has_burning_nbr[r + dr, c + dc] = True
    assert np.all(out[~has_burning_nbr] == 0.0)


@st.composite
def link_inputs(draw: st.DrawFn) -> tuple[np.ndarray, Stencil]:
    """Directional rates with some exact zeros, on grids of at most 6 x 6 cells."""
    stencil = _stencil(draw(STENCIL_SIZES))
    h, w = draw(st.integers(1, 6)), draw(st.integers(1, 6))
    elements = st.one_of(st.just(0.0), st.floats(1e-3, 10.0))
    return draw(arrays(np.float64, (stencil.size, h, w), elements=elements)), stencil


@PROFILE
@given(link_inputs())
def test_link_rate_bounded_by_the_cells_it_crosses(args: tuple[np.ndarray, Stencil]) -> None:
    """Link rates are >= 0, between the smallest and largest rate of the cells crossed, and 0 if
    the source or target rate is 0 or the neighbour is off the grid."""
    ros, stencil = args
    _, h, w = ros.shape
    out = link_rate_of_spread(ros, stencil)
    assert out.shape == ros.shape
    assert np.all(out >= 0)
    for k, (dr, dc) in enumerate(_offsets(stencil)):
        for r in range(h):
            for c in range(w):
                if not _inside(r + dr, c + dc, h, w):
                    assert out[k, r, c] == 0.0
                    continue
                crossed = [ros[k, r + a, c + b] for a, b in _cells_crossed(dr, dc)]
                if ros[k, r, c] == 0 or ros[k, r + dr, c + dc] == 0:
                    assert out[k, r, c] == 0.0
                assert out[k, r, c] <= max(crossed) * (1 + 1e-12)
                if min(crossed) > 0:
                    assert out[k, r, c] >= min(crossed) * (1 - 1e-12)


@st.composite
def symmetric_link_inputs(draw: st.DrawFn) -> tuple[np.ndarray, Stencil]:
    """Directional rates whose field in each direction equals the field in the opposite one."""
    ros, stencil = draw(link_inputs())
    for k, opp in enumerate(_opposite(stencil)):
        if opp > k:
            ros[opp] = ros[k]
    return ros, stencil


@PROFILE
@given(symmetric_link_inputs())
def test_link_rate_is_symmetric(args: tuple[np.ndarray, Stencil]) -> None:
    """With equal rate fields in opposite directions, link i -> j equals link j -> i."""
    ros, stencil = args
    _, h, w = ros.shape
    out = link_rate_of_spread(ros, stencil)
    opposite = _opposite(stencil)
    for k, (dr, dc) in enumerate(_offsets(stencil)):
        for r in range(h):
            for c in range(w):
                if _inside(r + dr, c + dc, h, w):
                    back = out[opposite[k], r + dr, c + dc]
                    assert out[k, r, c] == pytest.approx(back, rel=1e-12, abs=0.0)


@PROFILE
@given(STENCIL_SIZES, st.integers(1, 7), st.integers(1, 7), st.floats(1e-3, 10.0))
def test_link_rate_equals_a_uniform_input(size: int, h: int, w: int, rate: float) -> None:
    """On a uniform field every in-grid link has the cell rate and every off-grid link 0."""
    stencil = _stencil(size)
    out = link_rate_of_spread(np.full((stencil.size, h, w), rate), stencil)
    for k, (dr, dc) in enumerate(_offsets(stencil)):
        for r in range(h):
            for c in range(w):
                expected = rate if _inside(r + dr, c + dc, h, w) else 0.0
                assert out[k, r, c] == pytest.approx(expected, rel=1e-12, abs=0.0)


@PROFILE
@given(st.integers(0, 2**63), st.floats(1.0, 1000.0), STENCIL_SIZES, st.integers(1, 8), st.integers(1, 8))
def test_crossing_thresholds_positive_finite_and_seeded(seed: int, shape: float, size: int, h: int, w: int) -> None:
    """Thresholds are float32, > 0, finite, and the same for the same seed."""
    stencil = _stencil(size)
    a = draw_crossing_thresholds(np.random.default_rng(seed), shape, stencil, (h, w))
    b = draw_crossing_thresholds(np.random.default_rng(seed), shape, stencil, (h, w))
    assert a.shape == (stencil.size, h, w)
    assert a.dtype == np.float32
    assert np.all(a > 0) and np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)


# ============================================================================= grid


def _f32(lo: float, hi: float, **kw: Any) -> st.SearchStrategy[float]:
    """Float32 values in ``[lo, hi]`` (bounds rounded to float32 first)."""
    return st.floats(float(np.float32(lo)), float(np.float32(hi)), width=32, **kw)


@st.composite
def grids(draw: st.DrawFn) -> Grid:
    """A grid whose every layer holds random values inside the range :meth:`Grid.validate` accepts."""
    shape = (draw(st.integers(1, 6)), draw(st.integers(1, 6)))
    elements: dict[str, st.SearchStrategy[Any]] = {
        "fuel_model": st.integers(0, 204),
        "canopy_cover": _f32(0.0, 1.0),
        "elevation": _f32(-500.0, 9000.0),
        "slope": _f32(0.0, 10.0),
        "aspect": _f32(0.0, 359.9),
        "land_use": st.sampled_from([int(v) for v in LandUse]),
        "asset_value": _f32(0.0, 1e9),
        "fireline": st.booleans(),
        "retardant": _f32(0.0, 1.0),
        "state": st.sampled_from([int(v) for v in CellState]),
        "burn_timer": _f32(0.0, 1e5),
        "intensity": _f32(0.0, 1e5),
        "arrival_time": st.one_of(st.just(math.nan), _f32(0.0, 1e6)),
    }
    layers: dict[str, Any] = {}
    for name, (dtype, _) in LAYER_SPECS.items():
        layers[name] = draw(arrays(np.dtype(dtype), shape, elements=elements[name]))
    # NUL is excluded: numpy's fixed-width str arrays (used by Grid.save) strip trailing NULs.
    text = st.text(st.characters(exclude_categories=("Cs",), exclude_characters="\x00"), max_size=8)
    metadata = draw(st.dictionaries(text, text, max_size=3))
    return Grid(dx=draw(st.floats(0.1, 1000.0)), metadata=metadata, **layers)


@PROFILE
@given(grids())
def test_generated_grids_validate_and_save_load_round_trip(grid: Grid) -> None:
    """:meth:`Grid.validate` accepts the generated grids, and ``save`` then ``load`` gives an equal grid."""
    grid.validate()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grid.npz"
        grid.save(path)
        loaded = Grid.load(path)
    assert loaded == grid
    for name, (dtype, _) in LAYER_SPECS.items():
        assert getattr(loaded, name).dtype == np.dtype(dtype)


@PROFILE
@given(grids())
def test_copy_is_deep(grid: Grid) -> None:
    """Changing every layer and the metadata of a copy leaves the original untouched."""
    snapshot = {name: getattr(grid, name).copy() for name in LAYER_SPECS}
    meta = dict(grid.metadata)
    dup = grid.copy()
    assert dup == grid
    for name in LAYER_SPECS:
        arr = getattr(dup, name)
        assert not np.shares_memory(arr, getattr(grid, name))
        if arr.dtype == np.bool_:
            arr[...] = ~arr
        elif name == "arrival_time":
            arr[...] = np.where(np.isnan(arr), 1.0, np.nan)
        else:
            arr[...] = arr + 1
    dup.metadata["__changed__"] = "yes"
    for name in LAYER_SPECS:
        np.testing.assert_array_equal(getattr(grid, name), snapshot[name])
    assert grid.metadata == meta


@PROFILE
@given(grids())
def test_apply_unburnable_is_idempotent_and_spares_burning_cells(grid: Grid) -> None:
    """``apply_unburnable`` only turns unburnable FUEL cells UNBURNABLE, and a second call changes nothing."""
    before = grid.state.copy()
    burnable = grid.burnable_mask()
    grid.apply_unburnable()
    once = grid.copy()
    grid.apply_unburnable()
    assert grid == once
    after = once.state
    fire = (before == CellState.BURNING) | (before == CellState.BURNED)
    np.testing.assert_array_equal(after[fire], before[fire])
    fuel = before == CellState.FUEL
    assert np.all(after[fuel & ~burnable] == CellState.UNBURNABLE)
    assert np.all(after[fuel & burnable] == CellState.FUEL)
    assert np.all(after[before == CellState.UNBURNABLE] == CellState.UNBURNABLE)


# ============================================================================= simulator


class FieldRule:
    """Rate ``rates[r, c]`` (m/s) in every direction from burnable cells, 0 elsewhere."""

    name = "field"

    def __init__(self, rates: np.ndarray) -> None:
        self.rates = rates

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        h, w = grid.shape
        cell = np.where(grid.burnable_mask(), self.rates * (1.0 - grid.retardant.astype(np.float64)), 0.0)
        return SpreadOutput(
            ros=np.broadcast_to(cell, (stencil.size, h, w)).copy(),
            intensity=np.zeros((h, w)),
            flame_length=np.zeros((h, w)),
        )


@dataclass(frozen=True)
class Landscape:
    """A random small simulation input: grid, rate field and run settings (s, m/s)."""

    grid: Grid
    rates: np.ndarray
    ignition: tuple[int, int]
    ignition_time: float
    stencil: int
    arrival_shape: float
    dt: float
    duration: float
    seed: int

    def config(self, *, dt: float | None = None, duration: float | None = None) -> ScenarioConfig:
        h, w = self.grid.shape
        return ScenarioConfig.model_validate(
            {
                "name": "property",
                "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [h, w]},
                "spread": {"rule": "constant", "params": {"ros": 0.0}, "arrival_shape": self.arrival_shape},
                "ignition": {"cells": [list(self.ignition)], "time_s": self.ignition_time},
                "dt_s": self.dt if dt is None else dt,
                "duration_s": self.duration if duration is None else duration,
                "stencil": self.stencil,
                "seed": self.seed,
            }
        )

    def run(self, *, scale: float = 1.0, hooks: tuple[Any, ...] = (), **overrides: float) -> RunResult:
        return run(self.config(**overrides), rule=FieldRule(self.rates * scale), grid=self.grid, hooks=hooks)


class StateWatcher:
    """Step hook that checks, at every step start, that no cell left the fire or re-ignited.

    A cell once BURNING or BURNED stays so with the same arrival time, and a BURNED cell stays
    BURNED. Changes nothing, so it returns False.
    """

    def __init__(self) -> None:
        self.state: np.ndarray | None = None
        self.arrival: np.ndarray | None = None
        self.steps = 0

    def __call__(self, grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        state, arrival = grid.state.copy(), grid.arrival_time.copy()
        if self.state is not None and self.arrival is not None:
            was_fire = (self.state == CellState.BURNING) | (self.state == CellState.BURNED)
            assert np.all((state[was_fire] == CellState.BURNING) | (state[was_fire] == CellState.BURNED))
            assert np.all(state[self.state == CellState.BURNED] == CellState.BURNED)
            np.testing.assert_array_equal(arrival[was_fire], self.arrival[was_fire])
        self.state, self.arrival = state, arrival
        self.steps += 1
        return False


@st.composite
def landscapes(draw: st.DrawFn) -> Landscape:
    """Random unburnable cells (water, about one in five), random rates (some zero), one FUEL ignition."""
    h, w = draw(st.integers(3, 10)), draw(st.integers(3, 10))
    unburnable = draw(arrays(np.int8, (h, w), elements=st.integers(0, 4))) == 4
    ignition = (draw(st.integers(0, h - 1)), draw(st.integers(0, w - 1)))
    unburnable[ignition] = False
    grid = Grid.empty((h, w), 30.0)
    grid.fuel_model[:] = 1
    grid.fuel_model[unburnable] = 0
    grid.land_use[unburnable] = int(LandUse.WATER)
    grid.apply_unburnable()
    grid.validate()
    rates = draw(arrays(np.float64, (h, w), elements=st.floats(0.05, 1.0)))
    # about one burnable cell in ten carries no fire (rate 0), never the ignition cell
    no_fire = draw(arrays(np.int8, (h, w), elements=st.integers(0, 9))) == 9
    no_fire[ignition] = False
    rates[no_fire] = 0.0
    dt = draw(st.floats(15.0, 120.0))
    ignition_time = draw(st.floats(0.0, 200.0))
    return Landscape(
        grid=grid,
        rates=rates,
        ignition=ignition,
        ignition_time=ignition_time,
        stencil=draw(STENCIL_SIZES),
        arrival_shape=draw(st.sampled_from([5.0, 20.0, 200.0, 37.5])),
        dt=dt,
        duration=draw(st.floats(max(dt, ignition_time) + 900.0, 3000.0)),
        seed=draw(st.integers(0, 2**32 - 1)),
    )


@SIM_PROFILE
@given(landscapes())
def test_single_run_invariants(land: Landscape) -> None:
    """Every invariant one run must satisfy on its own.

    * a cell never ignites twice and never un-burns: checked at every step by
      :class:`StateWatcher`; at the end the burned mask is exactly the cells whose final state
      is BURNING or BURNED, and ``arrival_time`` is NaN exactly off it;
    * unburnable cells never burn;
    * causality: every burned cell except the ignition has a stencil neighbour that burned
      strictly earlier;
    * arrival times lie in ``[ignition time, final simulated time)``;
    * the event log has one entry per burned cell, in time order, matching ``arrival_time``.
    """
    watcher = StateWatcher()
    result = land.run(hooks=(watcher,))
    assert watcher.steps == result.provenance["n_steps"]
    burned = result.burned_mask
    state = result.grid.state
    arrival = result.arrival_time
    n_burned = int(burned.sum())
    event("burned cells: " + ("1" if n_burned <= 1 else "2-9" if n_burned < 10 else ">= 10"))

    final_fire = (state == CellState.BURNING) | (state == CellState.BURNED)
    np.testing.assert_array_equal(burned, final_fire)
    np.testing.assert_array_equal(np.isnan(arrival), ~burned)
    np.testing.assert_array_equal(np.isnan(result.grid.arrival_time), ~burned)

    unburnable = ~land.grid.burnable_mask()
    assert not burned[unburnable].any()
    assert np.all(state[unburnable] == CellState.UNBURNABLE)

    h, w = burned.shape
    offsets = _offsets(_stencil(land.stencil))
    for r, c in np.argwhere(burned):
        if (r, c) == land.ignition:
            continue
        earlier = [
            arrival[r - dr, c - dc] < arrival[r, c]
            for dr, dc in offsets
            if _inside(r - dr, c - dc, h, w) and burned[r - dr, c - dc]
        ]
        assert any(earlier), f"cell ({r}, {c}) burned at {arrival[r, c]} s with no earlier neighbour"

    final_time = result.provenance["final_time_s"]
    if burned.any():
        assert burned[land.ignition]
        assert np.nanmin(arrival) >= land.ignition_time
        assert np.nanmax(arrival) < final_time

    # At most one "edge" event, directly after the ignition of the first cell burned in the edge band.
    edge = first_edge_arrival(arrival, _stencil(land.stencil))
    edge_events = [e for e in result.events if e[1] == "edge"]
    if edge is None:
        assert edge_events == [] and result.provenance["reached_edge_s"] is None
    else:
        assert edge_events == [(edge[0], "edge", edge[1], edge[2])]
        assert result.provenance["reached_edge_s"] == edge[0] == arrival[edge[1], edge[2]]
        at = result.events.index(edge_events[0])
        assert result.events[at - 1][0] == edge[0] and result.events[at - 1][2:] == edge[1:]
        band = edge_band_mask(arrival.shape, _stencil(land.stencil).max_offset)
        assert edge[0] == np.nanmin(arrival[band])

    events = [e for e in result.events if e[1] != "edge"]
    assert len(events) == int(burned.sum())
    times = [e[0] for e in events]
    assert times == sorted(times)
    cells = {(e[2], e[3]) for e in events}
    assert len(cells) == len(events)
    for t, kind, r, c in events:
        assert arrival[r, c] == t
        assert kind == ("ignition" if (r, c) == land.ignition else "spread")


@SIM_PROFILE
@given(landscapes(), st.floats(15.0, 120.0))
def test_same_seed_is_reproducible_and_dt_independent(land: Landscape, other_dt: float) -> None:
    """Same seed twice gives identical results; another dt gives the same arrival times (rates
    are constant in time) for every cell that burned in both runs."""
    a = land.run()
    b = land.run()
    np.testing.assert_array_equal(a.arrival_time, b.arrival_time)
    np.testing.assert_array_equal(a.grid.state, b.grid.state)
    assert a.events == b.events

    duration = max(land.duration, other_dt)
    c = land.run(dt=other_dt, duration=duration)
    both = a.burned_mask & c.burned_mask
    np.testing.assert_allclose(c.arrival_time[both], a.arrival_time[both], rtol=1e-9, atol=1e-9)


@SIM_PROFILE
@given(landscapes(), st.floats(1.0, 3000.0))
def test_burned_area_is_monotone_in_duration(land: Landscape, extra: float) -> None:
    """Running longer with the same seed and dt burns a superset, with the same arrival times."""
    short = land.run()
    long = land.run(duration=land.duration + extra)
    assert np.all(long.burned_mask[short.burned_mask])
    np.testing.assert_array_equal(long.arrival_time[short.burned_mask], short.arrival_time[short.burned_mask])


@SIM_PROFILE
@given(landscapes(), st.floats(0.5, 2.0))
def test_scaling_every_rate_scales_arrival_times(land: Landscape, factor: float) -> None:
    """Multiplying every rate by ``c`` divides every time since ignition by ``c`` (same seed).

    Arrival times do not depend on dt or on when the run stops, so cells burned in both runs are
    compared; the faster run must burn every cell the slower run burned.
    """
    duration = 20000.0
    base = land.run(duration=duration, dt=120.0)
    scaled = land.run(duration=duration, dt=120.0, scale=factor)
    slow, fast = (base, scaled) if factor >= 1 else (scaled, base)
    assert np.all(fast.burned_mask[slow.burned_mask])
    both = base.burned_mask & scaled.burned_mask
    t0 = land.ignition_time
    np.testing.assert_allclose(
        scaled.arrival_time[both] - t0, (base.arrival_time[both] - t0) / factor, rtol=1e-9, atol=1e-9
    )


# ============================================================================= config

_TEXT = st.text(st.characters(exclude_categories=("Cs",)), max_size=12)
_POS = st.floats(1e-3, 1e6)
_NONNEG = st.floats(0.0, 1e6)


@st.composite
def scenario_data(draw: st.DrawFn) -> dict[str, Any]:
    """Raw data of a valid :class:`ScenarioConfig` with random values in every field."""
    shape = (draw(st.integers(1, 300)), draw(st.integers(1, 300)))
    if draw(st.booleans()):
        grid: dict[str, Any] = {
            "kind": "synthetic",
            "generator": draw(st.sampled_from(["flat_uniform", "slope_ramp", "with_river", "with_town"])),
            "shape": list(shape),
            "dx": draw(_POS),
            "fuel_model": draw(st.integers(1, 204)),
            "params": draw(st.dictionaries(_TEXT, st.one_of(st.integers(-1000, 1000), _NONNEG, _TEXT), max_size=3)),
        }
    else:
        grid = {"kind": "cached", "path": draw(st.text(st.characters(categories=("L", "N")), min_size=1, max_size=12))}

    increments = draw(st.lists(st.floats(1e-3, 1e5), max_size=3))
    times = [0.0, *np.cumsum(increments).tolist()]
    series = [
        {
            "time_s": t,
            "wind_speed": draw(st.floats(0.0, 50.0)),
            "wind_direction": draw(st.floats(0.0, 359.999)),
            "moisture_1h": draw(st.floats(0.0, 3.0)),
            "moisture_10h": draw(st.floats(0.0, 3.0)),
            "moisture_100h": draw(st.floats(0.0, 3.0)),
            "moisture_live_herb": draw(st.floats(0.0, 3.0)),
            "moisture_live_woody": draw(st.floats(0.0, 3.0)),
            "temperature_c": draw(st.floats(-50.0, 60.0)),
            "relative_humidity": draw(st.floats(0.0, 1.0)),
        }
        for t in times
    ]
    dt = draw(_POS)
    duration = draw(st.floats(dt, 2e6))
    cells = draw(st.lists(st.tuples(st.integers(0, shape[0] - 1), st.integers(0, shape[1] - 1)), max_size=3))
    resource = st.fixed_dictionaries(
        {
            "type": st.sampled_from(["crew", "dozer", "engine", "aircraft"]),
            "count": st.integers(0, 100),
            "dispatch_delay_s": _NONNEG,
            "work_rate": _NONNEG,
            "hourly_cost_usd": _NONNEG,
            "max_flame_length_m": _NONNEG,
            "source": _TEXT,
        }
    )
    return {
        "name": draw(st.text(st.characters(exclude_categories=("Cs",)), min_size=1, max_size=12)),
        "description": draw(_TEXT),
        "grid": grid,
        "weather": {"series": series},
        "spread": {
            "rule": draw(st.text(st.characters(categories=("L",)), min_size=1, max_size=12)),
            "params": draw(st.dictionaries(_TEXT, st.one_of(_NONNEG, st.booleans()), max_size=3)),
            "arrival_shape": draw(st.floats(1.0, 1e6)),
        },
        "ignition": {
            "cells": [list(c) for c in cells],
            "center": draw(st.booleans()) or not cells,
            "time_s": draw(st.floats(0.0, duration)),
        },
        "suppression": {
            "policy": draw(st.text(st.characters(categories=("L",)), min_size=1, max_size=12)),
            "params": draw(st.dictionaries(_TEXT, _NONNEG, max_size=3)),
            "resources": draw(st.lists(resource, max_size=2)),
            "decision_interval_s": draw(_POS),
        },
        "dt_s": dt,
        "duration_s": duration,
        "stencil": draw(STENCIL_SIZES),
        "seed": draw(st.integers(0, 2**63)),
        "ensemble": {
            "wind_speed_sigma_frac": draw(st.floats(0.0, 2.0)),
            "wind_direction_sigma_deg": draw(st.floats(0.0, 90.0)),
            "moisture_sigma_frac": draw(st.floats(0.0, 2.0)),
            "ros_multiplier_sigma_frac": draw(st.floats(0.0, 2.0)),
            "ignition_jitter_cells": draw(st.integers(0, 10)),
            "n_runs": draw(st.integers(1, 1000)),
            "base_seed": draw(st.integers(0, 2**63)),
        },
    }


@PROFILE
@given(scenario_data())
def test_scenario_config_survives_a_yaml_round_trip(data: dict[str, Any]) -> None:
    """``dump_yaml`` then ``load_scenario`` gives back an equal :class:`ScenarioConfig`."""
    config = ScenarioConfig.model_validate(data)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scenario.yaml"
        dump_yaml(config, path)
        loaded = load_scenario(path)
    assert loaded == config


def _steps_config(dt: float, duration: float) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "name": "steps",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [3, 3]},
            "ignition": {"center": True},
            "dt_s": dt,
            "duration_s": max(duration, dt),
        }
    )


_RATIOS = st.one_of(
    st.floats(1.0, 1e6),
    # ratios just below or above a whole number, where the rounding tolerance matters
    st.builds(lambda n, eps: max(n + eps, 1.0), st.integers(1, 10**6), st.floats(-1e-6, 1e-6)),
)


@PROFILE
@given(st.floats(1e-3, 1e4), _RATIOS)
@example(dt=1.0, ratio_hint=999999.9995)
def test_n_steps_is_floor_of_duration_over_dt(dt: float, ratio_hint: float) -> None:
    """``n_steps`` is ``floor(duration / dt)``, or the nearest whole number when the ratio is within
    a relative ``1e-9`` of it, so it never exceeds ``(duration / dt) (1 + 1e-9)``.

    The bound is relative, as :attr:`ScenarioConfig.n_steps` documents: an absolute bound of
    ``duration / dt + 1e-9`` does not hold (``duration = 999999.9995 s``, ``dt = 1 s`` gives
    1000000 steps, the ``@example`` above), and should not, since 0.0005 of a step in a
    million is floating-point noise in the inputs.
    """
    config = _steps_config(dt, dt * ratio_hint)
    ratio = config.duration_s / config.dt_s
    n = config.n_steps
    assert n >= 1
    assert n in {math.floor(ratio), round(ratio)}
    assert n <= ratio * (1 + 1e-9)
    if n != math.floor(ratio):
        assert abs(ratio - n) <= 1e-9 * max(1.0, ratio)


# ============================================================================= ensemble


@PROFILE
@given(st.integers(0, 2**64), st.integers(1, 30), st.integers(0, 30))
def test_derive_seeds_is_prefix_stable(base: int, n: int, more: int) -> None:
    """``derive_seeds(b, n)`` is a prefix of ``derive_seeds(b, n + more)``."""
    short = derive_seeds(base, n)
    long = derive_seeds(base, n + more)
    assert len(short) == n
    assert long[:n] == short
    assert all(s >= 0 for s in long)


_LEVELS = st.floats(0.0, 1.0, exclude_min=True, exclude_max=True)


@PROFILE
@given(st.lists(st.floats(-1e100, 1e100), min_size=2, max_size=30), _LEVELS)
def test_mean_ci_brackets_the_mean(values: list[float], level: float) -> None:
    """For any finite sample of size >= 2 with ``|x| <= 1e100``, ``lower <= mean <= upper``.

    The magnitude bound keeps the sample variance representable; beyond it ``mean_ci``
    overflows (see the xfail test below).
    """
    mean, lower, upper = mean_ci(np.array(values), level)
    assert lower <= mean <= upper


@pytest.mark.parametrize(
    ("values", "level"),
    [
        ([1e308, 1e308], 0.95),  # a naive sum overflows: mean = inf, lower = inf - inf = nan
        ([0.0, 1.9e154], 1e-17),  # a naive variance overflows to inf, t quantile is 0: 0 * inf = nan
    ],
)
def test_mean_ci_brackets_the_mean_for_any_finite_sample(values: list[float], level: float) -> None:
    """Minimal reproductions Hypothesis found for the unbounded version of the property above."""
    mean, lower, upper = mean_ci(np.array(values), level)
    assert lower <= mean <= upper
