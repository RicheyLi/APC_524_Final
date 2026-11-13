"""Tests for dormant sources (WP 1.10, docs/plan.md section 2.3).

A fire whose rate of spread drops to zero for a while (a full-strength retardant drop, fuel
above its moisture of extinction overnight) must wait and resume, not end. The checks compare
a stalled run with the same run without the stall: the link thresholds are the same (same
seed), so every cell first reached after the stall began must arrive exactly one stall length
later. ``SHIFT_RTOL`` covers the rounding of ``start + need / rate`` evaluated at two different
``start`` times; nothing else differs between the two runs.

Every test runs on both engines (the Numba ones are skipped when numba is not installed).
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wildfire_sim import sim
from wildfire_sim.config import ScenarioConfig, load_scenario
from wildfire_sim.data.synthetic import flat_uniform
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import HoldoverTable, RunResult, load_holdover_times, run
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.types import CellState, LandUse, Stencil

SCENARIOS = Path(__file__).resolve().parent.parent / "configs" / "scenarios"
SHIFT_RTOL = 1e-9  # relative, on arrival times shifted by a stall (s)
STALL_START, STALL_END = 1200.0, 1800.0  # the stall of the WP 1.10 acceptance test, s

_BURNING, _BURNED, _FUEL = int(CellState.BURNING), int(CellState.BURNED), int(CellState.FUEL)


@pytest.fixture(params=["numpy", "numba"])
def engine(request: pytest.FixtureRequest) -> str:
    if request.param == "numba" and importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    return request.param


def flat_no_wind(engine: str, duration: float) -> ScenarioConfig:
    """The shipped ``flat_no_wind`` scenario (101 x 101, constant 0.2 m/s, dt 30 s) with a new duration."""
    cfg = load_scenario(SCENARIOS / "flat_no_wind.yaml")
    return cfg.model_copy(update={"engine": engine, "duration_s": duration})


def retardant_stall(*windows: tuple[float, float]) -> Callable[[Grid, float, float, np.random.Generator], bool]:
    """Hook: retardant 1 on every cell while ``start <= t < end`` for one of ``windows`` (s), else 0."""

    def hook(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        want = 1.0 if any(start <= t < end for start, end in windows) else 0.0
        if float(grid.retardant[0, 0]) == want:
            return False
        grid.retardant[:] = want
        return True

    return hook


def use_holdover(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Make every run use one holdover time, s, for all fuels."""
    table = HoldoverTable.model_validate({"default": {"value": seconds, "units": "s", "source": "test"}})
    monkeypatch.setattr(sim, "load_holdover_times", lambda path=None: table)


def assert_shifted(stalled: RunResult, free: RunResult, stall_start: float, shift: float) -> None:
    """``stalled`` is ``free`` with every arrival after ``stall_start`` (s) delayed by ``shift`` (s)."""
    np.testing.assert_array_equal(stalled.burned_mask, free.burned_mask)
    before = free.arrival_time <= stall_start
    after = free.arrival_time > stall_start
    assert before.sum() > 10 and after.sum() > 10
    np.testing.assert_array_equal(stalled.arrival_time[before], free.arrival_time[before])
    np.testing.assert_allclose(stalled.arrival_time[after], free.arrival_time[after] + shift, rtol=SHIFT_RTOL, atol=0.0)


class MoistureRule:
    """0.2 m/s times ``(1 - retardant)`` while the 1-h dead fuel moisture is below 0.25, else no spread.

    Reports a head intensity of 500 kW/m everywhere, so the tests can tell flaming from dormant.
    """

    name = "moisture"
    extinction = 0.25

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        ros = 0.2 if weather.moisture_1h < self.extinction else 0.0
        cell = np.where(grid.burnable_mask(), ros * (1.0 - grid.retardant.astype(np.float64)), 0.0)
        h, w = grid.shape
        return SpreadOutput(
            ros=np.broadcast_to(cell, (stencil.size, h, w)).copy(),
            intensity=np.full((h, w), 500.0),
            flame_length=np.ones((h, w)),
        )


# ----------------------------------------------------------------------------- the holdover table


def test_shipped_holdover_times_validate() -> None:
    table = load_holdover_times()
    assert isinstance(table, HoldoverTable)
    assert table.default.source
    # the placeholder of docs/plan.md section 13: 12 h for every fuel
    np.testing.assert_array_equal(table.lookup(np.array([[1, 102], [183, 9999]])), np.full((2, 2), 43200.0))


def test_holdover_table_lookup_and_bounds() -> None:
    table = HoldoverTable.model_validate(
        {"default": {"value": 100.0, "source": "x"}, "by_fuel_model": {7: {"value": 0.0, "source": "x"}}}
    )
    np.testing.assert_array_equal(table.lookup(np.array([1, 7, 7, 2])), [100.0, 0.0, 0.0, 100.0])
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="holdover"):
            HoldoverTable.model_validate({"default": {"value": bad, "source": "x"}})


def test_holdover_table_is_in_the_provenance(engine: str) -> None:
    result = run(flat_no_wind(engine, 300.0))
    assert result.provenance["holdover_s"]["default"]["value"] == 43200.0
    assert sim.model_provenance(flat_no_wind(engine, 300.0))["holdover_s"] == result.provenance["holdover_s"]
    json.dumps(result.provenance, allow_nan=False)


# ----------------------------------------------------------------------------- acceptance: retardant stall


def test_ten_minute_stall_shifts_later_arrivals_by_ten_minutes(engine: str) -> None:
    """WP 1.10 acceptance: retardant 1 everywhere from 1200 s to 1800 s on ``flat_no_wind``.

    Every cell first reached after 1200 s arrives exactly 600 s later than without the stall,
    and the fire burns the same cells. Before WP 1.10 the stalled run ended during the stall.
    """
    free = run(flat_no_wind(engine, 5400.0))
    stalled = run(flat_no_wind(engine, 6000.0), hooks=[retardant_stall((STALL_START, STALL_END))])
    assert_shifted(stalled, free, STALL_START, STALL_END - STALL_START)
    assert stalled.provenance["n_steps"] == 200  # ran to the end; the fire is still spreading
    assert np.count_nonzero(stalled.grid.state == _BURNING) > 0


def test_dormant_cells_are_the_stalled_edge_and_do_not_flame(engine: str) -> None:
    """Mid-stall: BURNING cells are exactly the burned cells next to FUEL, with zero intensity and timer."""
    cfg = flat_no_wind(engine, 1620.0)  # ends 420 s into the stall; grass flames for at most 160 s
    result = run(cfg, hooks=[retardant_stall((STALL_START, STALL_END))], rule=MoistureRule())
    state = result.grid.state
    assert result.provenance["n_steps"] == cfg.n_steps
    assert np.nanmax(result.arrival_time) < STALL_START + cfg.dt_s

    burning = state == _BURNING
    padded = np.pad(state == _FUEL, 2, constant_values=False)
    h, w = state.shape
    near_fuel = np.zeros_like(burning)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if (abs(dr), abs(dc)) in {(1, 0), (0, 1), (1, 1), (1, 2), (2, 1)}:  # the 16-neighbour stencil
                near_fuel |= padded[2 + dr : 2 + dr + h, 2 + dc : 2 + dc + w]
    np.testing.assert_array_equal(burning, result.burned_mask & near_fuel)
    assert burning.sum() > 20
    assert not result.grid.intensity[burning].any()
    assert not result.grid.burn_timer[burning].any()


def test_moisture_rule_in_the_stall_test_is_not_what_stalls(engine: str) -> None:
    """Guard for the test above: its rule spreads at the default moisture, so the hook causes the stall."""
    result = run(flat_no_wind(engine, 1620.0), rule=MoistureRule())
    assert np.nanmax(result.arrival_time) > 1500.0
    assert result.grid.intensity[result.grid.state == _BURNING].max() == 500.0


# ----------------------------------------------------------------------------- holdover limit


def test_stall_longer_than_the_holdover_ends_the_fire(engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a 300 s holdover a 600 s stall puts the fire out for good: nothing burns after 1200 s."""
    use_holdover(monkeypatch, 300.0)
    free = run(flat_no_wind(engine, 5400.0))
    cfg = flat_no_wind(engine, 6000.0)
    stalled = run(cfg, hooks=[retardant_stall((STALL_START, STALL_END))])
    np.testing.assert_array_equal(stalled.burned_mask, free.arrival_time <= STALL_START)
    assert not np.any(stalled.grid.state == _BURNING)
    assert np.all(stalled.grid.state[stalled.burned_mask] == _BURNED)
    # the last cells lit keep flaming for 160 s, are then dormant for 300 s, and the run stops
    assert STALL_START + 300.0 < stalled.provenance["final_time_s"] < STALL_END
    assert stalled.provenance["n_steps"] < cfg.n_steps


def test_stall_shorter_than_the_holdover_survives(engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same stall with a holdover just long enough: the fire resumes."""
    use_holdover(monkeypatch, 600.0)
    free = run(flat_no_wind(engine, 5400.0))
    stalled = run(flat_no_wind(engine, 6000.0), hooks=[retardant_stall((STALL_START, STALL_END))])
    assert_shifted(stalled, free, STALL_START, 600.0)


def test_zero_holdover_ends_the_fire_when_the_flames_go_out(engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Holdover 0 is the behaviour before WP 1.10: the run ends once the last cell stops flaming.

    A grass cell flames for 150 s (30 m at 0.2 m/s) + 10 s residence, so the fire is out by
    1200 + 160 s, rounded up to a 30 s step: 1380 s, the time recorded in docs/plan.md section 2.3.
    """
    use_holdover(monkeypatch, 0.0)
    stalled = run(flat_no_wind(engine, 6000.0), hooks=[retardant_stall((STALL_START, STALL_END))])
    assert stalled.provenance["final_time_s"] == 1380.0
    assert not np.any(stalled.grid.state == _BURNING)
    assert np.nanmax(stalled.arrival_time) <= STALL_START


def test_dormant_clock_restarts_when_the_fire_spreads_again(engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two 240 s stalls separated by 60 s of spread outlast a 300 s holdover: the clock is per stall.

    Only cells that were sources between the stalls get a fresh clock, which every edge cell of
    a round fire is, so the fire resumes after the second stall and ends up 480 s late.
    """
    use_holdover(monkeypatch, 300.0)
    free = run(flat_no_wind(engine, 5400.0))
    hook = retardant_stall((1200.0, 1440.0), (1500.0, 1740.0))
    stalled = run(flat_no_wind(engine, 5880.0), hooks=[hook])
    np.testing.assert_array_equal(stalled.burned_mask, free.burned_mask)
    late = free.arrival_time > 1260.0  # reached after the 60 s between the stalls had passed
    np.testing.assert_allclose(stalled.arrival_time[late], free.arrival_time[late] + 480.0, rtol=SHIFT_RTOL)


# ----------------------------------------------------------------------------- weather


def test_fire_resumes_after_a_night_above_extinction_moisture(engine: str) -> None:
    """Zero rate overnight from the weather series: the fire holds for 8 h and resumes in the morning."""
    night_start, night = 1800.0, 8 * 3600.0
    base: dict[str, Any] = {
        "name": "overnight",
        "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [61, 61], "dx": 30.0},
        "ignition": {"center": True},
        "dt_s": 60.0,
        "engine": engine,
    }
    day = ScenarioConfig.model_validate({**base, "duration_s": 3600.0})
    series = [
        {"time_s": 0.0, "moisture_1h": 0.06},
        {"time_s": night_start, "moisture_1h": 0.30, "relative_humidity": 0.95},
        {"time_s": night_start + night, "moisture_1h": 0.06},
    ]
    overnight = ScenarioConfig.model_validate({**base, "duration_s": 3600.0 + night, "weather": {"series": series}})
    free = run(day, rule=MoistureRule(), seed=3)
    stalled = run(overnight, rule=MoistureRule(), seed=3)
    assert_shifted(stalled, free, night_start, night)
    assert stalled.provenance["n_steps"] == overnight.n_steps


def test_a_rule_that_never_spreads_holds_the_ignition_cell_until_the_holdover(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rate 0 everywhere: the ignition cell flames for its residence time, waits, then goes out."""
    use_holdover(monkeypatch, 900.0)
    cfg = flat_no_wind(engine, 3600.0)
    cfg = cfg.model_copy(update={"spread": cfg.spread.model_copy(update={"params": {"ros": 0.0}})})
    short = run(cfg.model_copy(update={"duration_s": 600.0}))
    assert short.burned_mask.sum() == 1
    assert short.grid.state[50, 50] == _BURNING and short.grid.intensity[50, 50] == 0.0
    full = run(cfg)
    assert full.burned_mask.sum() == 1 and full.grid.state[50, 50] == _BURNED
    # flames for the 10 s residence time (first step), dormant from 30 s, out 900 s later
    assert full.provenance["final_time_s"] == 930.0


def test_max_source_age_ends_dormancy(engine: str) -> None:
    """A cell older than ``max_source_age_s`` is no source of any kind, dormant included."""
    cfg = flat_no_wind(engine, 3600.0)
    cfg = cfg.model_copy(update={"spread": cfg.spread.model_copy(update={"params": {"ros": 0.0}})})
    result = run(cfg, max_source_age_s=120.0)
    assert result.grid.state[50, 50] == _BURNED
    assert result.provenance["final_time_s"] == 120.0


# ----------------------------------------------------------------------------- what is not dormant


def test_links_blocked_by_barrier_geometry_do_not_make_a_cell_dormant(engine: str) -> None:
    """Fuel beyond a diagonal wall is reachable only through blocked links; the fire must end, not wait.

    A wall of water cells touching at corners splits the grid. With 8 neighbours the cells
    beside it have diagonal links to FUEL across the wall, blocked for good by the geometry.
    """
    grid = flat_uniform((21, 21), 30.0)
    idx = np.arange(21)
    grid.land_use[idx, 20 - idx] = int(LandUse.WATER)
    grid.fuel_model[idx, 20 - idx] = 0
    cfg = ScenarioConfig.model_validate(
        {
            "name": "wall",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [21, 21]},
            "spread": {"rule": "constant", "params": {"ros": 0.2}},
            "ignition": {"cells": [[3, 3]]},
            "duration_s": 36000.0,
            "stencil": 8,
            "engine": engine,
        }
    )
    result = run(cfg, grid=grid)
    rows, cols = np.nonzero(result.burned_mask)
    assert np.all(rows + cols < 20)  # the far side never burns
    assert result.burned_mask.sum() == 210  # the whole near side: 20 * 21 / 2 cells
    assert not np.any(result.grid.state == _BURNING)
    assert result.provenance["final_time_s"] < 7200.0  # ended when the near side burned out


def test_an_ordinary_fire_still_ends_when_everything_burned(engine: str) -> None:
    grid_cfg = {"kind": "synthetic", "generator": "with_river", "shape": [15, 31], "params": {"col": 20}}
    cfg = ScenarioConfig.model_validate(
        {
            "name": "small",
            "grid": grid_cfg,
            "spread": {"rule": "constant", "params": {"ros": 0.2}},
            "ignition": {"cells": [[7, 3]]},
            "duration_s": 36000.0,
            "engine": engine,
        }
    )
    result = run(cfg)
    assert not np.any(result.grid.state == _BURNING)
    assert result.provenance["n_steps"] < cfg.n_steps


# ----------------------------------------------------------------------------- local stalls


def test_a_retardant_strip_holds_the_fire_and_releases_it(engine: str) -> None:
    """A full-strength drop across the fire's path: the edge waits at the strip and crosses when it fades.

    The strip is three cells wide so that knight moves cannot step over it. The west side burns
    out while the strip holds; only the cells beside the strip stay BURNING, dormant.
    """
    cfg = ScenarioConfig.model_validate(
        {
            "name": "strip",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [11, 41]},
            "spread": {"rule": "constant", "params": {"ros": 0.2}},
            "ignition": {"cells": [[5, 2]]},
            "duration_s": 14400.0,
            "engine": engine,
        }
    )
    fade = 9000.0

    def drop(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        want = 1.0 if t < fade else 0.0
        if float(grid.retardant[0, 20]) == want:
            return False
        grid.retardant[:, 20:23] = want
        return True

    held = run(cfg.model_copy(update={"duration_s": fade}), hooks=[drop])
    assert not held.burned_mask[:, 20:].any()
    assert held.burned_mask[:, :20].all()
    burning = held.grid.state == _BURNING
    assert burning[:, 18:20].any() and not burning[:, :18].any()
    assert not held.grid.intensity.any()

    released = run(cfg, hooks=[drop])
    assert released.burned_mask.all()
    assert np.nanmin(released.arrival_time[:, 20:]) > fade
