"""Tests for wildfire_sim.breaks and break crossing in the engines (WP 2.9).

Acceptance (docs/plan.md WP 2.9): breach probability rises with intensity and falls with width;
it reproduces Wilson's 29 %, 55 % and 88 % for a 3 m break without trees at 2, 5 and 10 MW/m and
72 % with trees at 2 MW/m; below the minimum intensity no break is breached, so a rule with zero
intensity never breaches; a grid without breaks is bit-identical to before (the golden runs and
engine-parity tests, unchanged); the set of breached cells is the same for dt and dt / 2; two
runs with one seed and different hooks see the same ``u_c``.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from wildfire_sim.breaks import (
    BreachParams,
    breach_probability,
    draw_breach_thresholds,
    in_fitted_range,
    load_breach_params,
    trees_near,
)
from wildfire_sim.config import ScenarioConfig
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import RunResult, run
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.types import CellState, Stencil

ROS = 0.5  # m/s, every direction
SIZE = 70
LINE_COL = 35
IGNITION = (35, 12)


@pytest.fixture(scope="module")
def params() -> BreachParams:
    return load_breach_params()


@pytest.fixture(params=["numpy", "numba"])
def engine(request: pytest.FixtureRequest) -> str:
    if request.param == "numba" and importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    return request.param


# ----------------------------------------------------------------------------- the model


def test_reproduces_wilson(params: BreachParams) -> None:
    """Acceptance: 29, 55 and 88 % for 3 m without trees at 2, 5, 10 MW/m; 72 % with trees at 2 MW/m."""
    p = breach_probability(np.array([2000.0, 5000.0, 10000.0]), 3.0, False, params)
    assert np.round(p * 100).tolist() == [29.0, 55.0, 88.0]
    assert round(float(breach_probability(2000.0, 3.0, True, params)) * 100) == 72


def test_monotone_in_intensity_and_width(params: BreachParams) -> None:
    """Acceptance: rises with intensity, falls with width, with and without trees."""
    intensity = np.linspace(200.0, 17000.0, 50)
    for trees in (False, True):
        p_i = breach_probability(intensity, 4.0, trees, params)
        assert np.all(np.diff(p_i) > 0)
        p_w = breach_probability(5000.0, np.linspace(1.5, 15.0, 50), trees, params)
        assert np.all(np.diff(p_w) < 0)


def test_minimum_intensity(params: BreachParams) -> None:
    """Acceptance: below the minimum intensity (0.18 MW/m) no break is breached."""
    floor = params.min_intensity_mw_m.value * 1000.0
    p = breach_probability(np.array([0.0, floor - 1.0, floor, np.nan]), 1.5, True, params)
    assert p[0] == 0.0 and p[1] == 0.0 and p[2] > 0.5 and p[3] == 0.0


def test_fitted_range_and_trees(params: BreachParams) -> None:
    assert in_fitted_range(5000.0, 3.0, params) and not in_fitted_range(5000.0, 20.0, params)
    assert not in_fitted_range(20000.0, 3.0, params) and not in_fitted_range(5000.0, 1.0, params)
    cover = np.zeros((5, 5))
    cover[2, 2] = 0.5
    near = trees_near(cover, 0.1)
    assert near[1:4, 1:4].all() and near.sum() == 9


def test_params_validation(params: BreachParams) -> None:
    raw = params.model_dump()
    raw["width_coef"]["value"] = 0.5
    with pytest.raises(ValueError, match="width"):
        BreachParams.model_validate(raw)


def test_breach_thresholds_are_their_own_stream() -> None:
    a = draw_breach_thresholds(np.random.SeedSequence(3).spawn(4)[3], (4, 5))
    b = draw_breach_thresholds(np.random.SeedSequence(3).spawn(4)[3], (4, 5))
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() < 1
    # spawn(4)'s first three children are those of spawn(3): earlier runs keep their draws
    old = [s.generate_state(2) for s in np.random.SeedSequence(3).spawn(3)]
    new = [s.generate_state(2) for s in np.random.SeedSequence(3).spawn(4)[:3]]
    assert all(np.array_equal(x, y) for x, y in zip(old, new, strict=True))


# ----------------------------------------------------------------------------- in a run


class UniformRule:
    """The same rate in every direction and a uniform head intensity (kW/m)."""

    name = "uniform"

    def __init__(self, intensity_kw_m: float) -> None:
        self.intensity = intensity_kw_m

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        mask = grid.burnable_mask().astype(np.float64)
        h, w = grid.shape
        return SpreadOutput(
            ros=np.full(stencil.size, ROS)[:, None, None] * mask[None],
            intensity=np.full((h, w), self.intensity),
            flame_length=np.zeros((h, w)),
        )


def _grid(width_m: float, *, fireline: bool = False, trees: bool = False) -> Grid:
    grid = Grid.empty((SIZE, SIZE), 30.0)
    grid.fuel_model[:] = 102
    grid.break_width_m[:, LINE_COL] = width_m
    if fireline:
        grid.fireline[:, LINE_COL] = True
    if trees:
        grid.canopy_cover[:] = 0.5
    return grid


def _config(engine: str = "numpy", dt: float = 30.0, duration: float = 6000.0) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "name": "break test",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [SIZE, SIZE]},
            "ignition": {"cells": [list(IGNITION)]},
            "dt_s": dt,
            "duration_s": duration,
            "engine": engine,
        }
    )


def _run(
    width_m: float, intensity: float, *, engine: str = "numpy", dt: float = 30.0, seed: int = 4, **kw: Any
) -> RunResult:
    grid = _grid(width_m, fireline=kw.pop("fireline", False), trees=kw.pop("trees", False))
    return run(_config(engine, dt), seed=seed, rule=UniformRule(intensity), grid=grid, **kw)


def _breached(result: RunResult) -> set[tuple[int, int]]:
    return {(r, c) for _, kind, r, c in result.events if kind == "breach"}


def _east(result: RunResult) -> np.ndarray:
    return result.burned_mask[:, LINE_COL + 1 :]


def test_wide_break_holds_a_moderate_fire(engine: str) -> None:
    result = _run(15.0, 2000.0, engine=engine)  # p about 3e-6
    assert result.provenance["breaks"] == {"break_cells_held": SIZE, "breaches": 0}
    assert not _east(result).any() and not result.burned_mask[:, LINE_COL].any()
    assert (result.grid.state[:, LINE_COL] == int(CellState.FUEL)).all()  # restored, never burned


def test_zero_intensity_never_breaches(engine: str) -> None:
    """Acceptance: a rule with zero intensity (or below the floor) never breaches a break."""
    for intensity in (0.0, 100.0):
        result = _run(1.5, intensity, engine=engine)
        assert not _breached(result) and not _east(result).any()


def test_breached_set_follows_the_thresholds(engine: str, params: BreachParams) -> None:
    """A line cell is breached exactly when p_b of its hottest ignited neighbour reaches its u_c."""
    seed = 11
    result = _run(3.0, 5000.0, engine=engine, seed=seed)  # p about 0.55
    breached = _breached(result)
    assert 0 < len(breached) < SIZE and _east(result).any()
    u = draw_breach_thresholds(np.random.SeedSequence(seed).spawn(4)[3], (SIZE, SIZE))
    fi = np.nan_to_num(result.front_intensity, nan=0.0)
    expected = set()
    for r in range(SIZE):
        rows = slice(max(r - 1, 0), r + 2)
        hottest = max(fi[rows, LINE_COL - 1].max(), fi[rows, LINE_COL + 1].max(), fi[rows, LINE_COL].max())
        if breach_probability(hottest, 3.0, False, params) >= u[r, LINE_COL]:
            expected.add((r, LINE_COL))
    assert breached == expected


def test_trees_make_breaching_likelier() -> None:
    open_ground = _breached(_run(4.0, 3000.0, seed=2))
    wooded = _breached(_run(4.0, 3000.0, seed=2, trees=True))
    assert open_ground < wooded  # the same u_c, a higher p_b everywhere


def test_breach_lets_the_fire_cross_at_the_normal_rate(engine: str) -> None:
    """With p_b about 1 every line cell is breached and the fire beyond is barely delayed."""
    with_line = _run(1.5, 30000.0, engine=engine)
    assert len(_breached(with_line)) == SIZE
    grid = _grid(0.0)
    free = run(_config(engine), seed=4, rule=UniformRule(30000.0), grid=grid)
    east = slice(LINE_COL + 1, LINE_COL + 15)
    delay = with_line.arrival_time[:, east] - free.arrival_time[:, east]
    assert np.nanmax(np.abs(delay)) < 2 * 30.0  # within two steps


def test_same_breaches_for_dt_and_half_dt() -> None:
    """Acceptance: the set of breached cells is the same for dt and dt / 2."""
    a = _breached(_run(3.0, 5000.0, dt=30.0, seed=7))
    b = _breached(_run(3.0, 5000.0, dt=15.0, seed=7))
    assert a and a == b


def test_hooks_do_not_change_the_draws() -> None:
    """Acceptance: two runs with one seed and different hooks see the same u_c."""

    def noisy(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        rng.random(1000)  # consumes the suppression stream
        return False

    quiet = _run(3.0, 5000.0, seed=9)
    busy = _run(3.0, 5000.0, seed=9, hooks=[noisy])
    assert _breached(quiet) == _breached(busy) and _breached(quiet)
    assert np.array_equal(quiet.arrival_time, busy.arrival_time, equal_nan=True)


def test_engines_agree(params: BreachParams) -> None:
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    a = _run(3.0, 5000.0, engine="numpy", seed=5)
    b = _run(3.0, 5000.0, engine="numba", seed=5)
    assert _breached(a) == _breached(b)
    np.testing.assert_allclose(a.arrival_time, b.arrival_time, rtol=1e-9, atol=1e-6, equal_nan=True)


@pytest.mark.slow
def test_local_update_after_a_breach_equals_the_full_refresh(engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The local barrier update gives bit-identical runs to recomputing everything after each breach.

    On the committed 5 km crop of fire A: Rothermel, real weather, urban cells, roads and streams;
    the fire meets well over a hundred breaks in four hours.
    """
    from pathlib import Path

    from wildfire_sim.breaks import BreakState
    from wildfire_sim.config import WeatherConfig, load_yaml

    fixtures = Path(__file__).parent / "fixtures"
    weather = load_yaml(fixtures / "lake_2024_weather.yaml", WeatherConfig)
    config = ScenarioConfig.model_validate(
        {
            "name": "local update",
            "grid": {"kind": "cached", "path": str(fixtures / "lake_2024_5km.npz")},
            "weather": weather.model_dump(),
            "spread": {"rule": "rothermel"},
            "ignition": {"center": True, "time_s": 56880},
            "dt_s": 60,
            "duration_s": 56880 + 4 * 3600,
            "engine": engine,
        }
    )
    local = run(config, seed=1)
    monkeypatch.setattr(BreakState, "local_updates", False)
    full = run(config, seed=1)
    assert local.provenance["breaks"]["breaches"] > 100
    assert local.provenance["breaks"] == full.provenance["breaks"]
    assert np.array_equal(local.arrival_time, full.arrival_time, equal_nan=True)
    assert np.array_equal(local.front_intensity, full.front_intensity, equal_nan=True)
    assert local.events == full.events
    assert np.array_equal(local.grid.state, full.grid.state)


def test_lines_without_a_width_still_hold_and_lines_with_one_can_breach() -> None:
    held = _run(0.0, 30000.0, fireline=True)
    assert not _east(held).any() and not _breached(held)
    crossed = _run(1.5, 30000.0, fireline=True)
    assert _breached(crossed) and _east(crossed).any()
    assert not crossed.grid.fireline[sorted(_breached(crossed))[0]]  # a breached line is gone


def test_a_hook_can_build_a_line_with_a_width() -> None:
    """Lines added during the run (WP 4.3) are held and can be breached like roads."""

    def build(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        if t == 0.0:
            grid.fireline[:, LINE_COL] = True
            grid.break_width_m[:, LINE_COL] = 15.0
            grid.apply_unburnable()
            return True
        return False

    result = run(_config(), seed=4, rule=UniformRule(2000.0), grid=_grid(0.0), hooks=[build])
    assert not _east(result).any()
