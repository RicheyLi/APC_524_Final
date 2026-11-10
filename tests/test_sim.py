"""Tests for the simulator time loop (WP 1.3).

Deterministic-limit tests use ``arrival_shape = 1e6``: every link threshold is then 1 within
about 0.1 %, so a link of length ``d`` and rate ``R`` takes ``d / R`` to cross and arrival times
can be predicted exactly. Tolerances (1-2 %) cover that residual threshold noise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wildfire_sim import sim
from wildfire_sim.config import ScenarioConfig, load_scenario
from wildfire_sim.data.synthetic import flat_uniform
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import (
    KernelCorrectionTable,
    ResidenceTimeTable,
    RunResult,
    load_kernel_corrections,
    load_residence_times,
    run,
)
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.types import STENCIL_8, STENCIL_16, CellState, LandUse, Stencil

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = REPO_ROOT / "configs" / "scenarios"
DETERMINISTIC = 1e6  # arrival_shape giving thresholds of 1 within ~0.1 %


def make_config(
    *,
    shape: tuple[int, int] = (21, 21),
    generator: str = "flat_uniform",
    grid_params: dict[str, Any] | None = None,
    ros: float = 0.2,
    dt: float = 30.0,
    duration: float = 3600.0,
    stencil: int = 8,
    arrival_shape: float = 20.0,
    cells: list[tuple[int, int]] | None = None,
    center: bool = False,
    ignition_time: float = 0.0,
    weather: list[dict[str, float]] | None = None,
    seed: int = 0,
) -> ScenarioConfig:
    """Scenario on a synthetic 30 m grid with the constant rule, built in code."""
    data: dict[str, Any] = {
        "name": "test",
        "grid": {
            "kind": "synthetic",
            "generator": generator,
            "shape": list(shape),
            "dx": 30.0,
            "params": grid_params or {},
        },
        "spread": {"rule": "constant", "params": {"ros": ros}, "arrival_shape": arrival_shape},
        "ignition": {"cells": [list(c) for c in (cells or [])], "center": center, "time_s": ignition_time},
        "dt_s": dt,
        "duration_s": duration,
        "stencil": stencil,
        "seed": seed,
    }
    if weather is not None:
        data["weather"] = {"series": weather}
    return ScenarioConfig.model_validate(data)


def _uniform_output(cell_ros: np.ndarray, stencil: Stencil) -> SpreadOutput:
    h, w = cell_ros.shape
    return SpreadOutput(
        ros=np.broadcast_to(cell_ros, (stencil.size, h, w)).copy(),
        intensity=np.zeros((h, w)),
        flame_length=np.zeros((h, w)),
    )


class UniformRule:
    """Same rate everywhere on burnable cells (independent of the shipped constant rule)."""

    name = "uniform"

    def __init__(self, ros: float) -> None:
        self.ros = ros

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        cell = np.where(grid.burnable_mask(), self.ros * (1.0 - grid.retardant.astype(np.float64)), 0.0)
        return _uniform_output(cell, stencil)


class StripRule:
    """``ros`` everywhere, ``ros / factor`` in columns ``[c0, c1)``."""

    name = "strip"

    def __init__(self, ros: float, c0: int, c1: int, factor: float) -> None:
        self.ros, self.c0, self.c1, self.factor = ros, c0, c1, factor

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        cell = np.where(grid.burnable_mask(), self.ros, 0.0)
        cell[:, self.c0 : self.c1] /= self.factor
        return _uniform_output(cell, stencil)


class WindRule:
    """``ros = base + per_wind * wind_speed`` in every direction."""

    name = "wind"

    def __init__(self, base: float, per_wind: float) -> None:
        self.base, self.per_wind = base, per_wind

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        cell = np.where(grid.burnable_mask(), self.base + self.per_wind * weather.wind_speed, 0.0)
        return _uniform_output(cell, stencil)


def _burned_cols(result: RunResult) -> np.ndarray:
    return np.nonzero(result.burned_mask.any(axis=0))[0]


# ----------------------------------------------------------------------------- config files


def test_shipped_residence_times_validate() -> None:
    table = load_residence_times()
    assert isinstance(table, ResidenceTimeTable)
    assert table.default.value == 60.0
    assert "PLACEHOLDER" in table.default.source
    assert table.by_fuel_model, "ship at least a few per-fuel-model entries"
    assert all("PLACEHOLDER" in v.source for v in table.by_fuel_model.values())
    fm = next(iter(table.by_fuel_model))
    times = table.lookup(np.array([fm, 9999], dtype=np.int16))
    assert times.tolist() == [table.by_fuel_model[fm].value, table.default.value]


def test_shipped_kernel_corrections_validate() -> None:
    table = load_kernel_corrections()
    assert isinstance(table, KernelCorrectionTable)
    assert table.value_for(8, 20.0) == 1.0
    assert table.value_for(16, 20.0) == 1.0
    assert table.value_for(8, 3.0) == 1.0  # no entry -> 1.0


def test_kernel_correction_table_rejects_duplicates() -> None:
    entry = {"stencil": 8, "arrival_shape": 20.0, "value": 1.1, "source": "x"}
    with pytest.raises(ValueError):
        KernelCorrectionTable.model_validate({"entries": [entry, entry]})
    table = KernelCorrectionTable.model_validate({"entries": [entry]})
    assert table.value_for(8, 20.0) == 1.1


def test_kernel_correction_divides_link_rates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A correction of 2 halves the front speed (deterministic limit)."""
    for name in ("residence_time.yaml",):
        (tmp_path / name).write_text((REPO_ROOT / "configs" / name).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "kernel_correction.yaml").write_text(
        f"entries:\n  - {{stencil: 8, arrival_shape: {DETERMINISTIC}, value: 2.0, source: test}}\n", encoding="utf-8"
    )
    monkeypatch.setenv("WILDFIRE_SIM_CONFIGS", str(tmp_path))
    cfg = make_config(shape=(5, 21), cells=[(2, 0)], arrival_shape=DETERMINISTIC, duration=7200)
    result = run(cfg)
    assert result.arrival_time[2, 10] == pytest.approx(2 * 10 * 30 / 0.2, rel=0.01)
    assert result.provenance["kernel_correction"] == 2.0


def test_residence_table_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        ResidenceTimeTable.model_validate({"default": {"value": 0.0, "source": "x"}})


# ----------------------------------------------------------------------------- barriers and invariants


def test_water_and_urban_never_burn() -> None:
    for generator, params in (("with_river", {"col": 12, "width": 2}), ("with_town", {"size": 5})):
        cfg = make_config(generator=generator, grid_params=params, cells=[(3, 3)], duration=7200)
        result = run(cfg)
        lu = result.grid.land_use
        assert not result.burned_mask[lu != int(LandUse.WILDLAND)].any()
        assert result.burned_mask.sum() > 10


@pytest.mark.parametrize("stencil", [8, 16])
def test_river_scenario_no_ignition_east_of_river(stencil: int) -> None:
    cfg = load_scenario(SCENARIOS / "river_barrier.yaml").model_copy(update={"stencil": stencil})
    result = run(cfg)
    assert result.burned_mask[:, :60].any()
    assert result.burned_mask[:, 59].any(), "fire should reach the river bank"
    assert not result.burned_mask[:, 60:].any()


def test_invariants_on_flat_scenario(flat_scenario: ScenarioConfig) -> None:
    cfg = flat_scenario.model_copy(update={"duration_s": 1800.0})
    result = run(cfg)
    assert result.arrival_time.dtype == np.float64
    assert result.arrival_time.shape == result.grid.shape
    np.testing.assert_array_equal(np.isnan(result.arrival_time), ~result.burned_mask)
    np.testing.assert_array_equal(result.burned_mask, result.grid.burned_mask())
    result.grid.validate()
    # float32 layer mirrors the float64 result
    np.testing.assert_allclose(result.grid.arrival_time, result.arrival_time.astype(np.float32), equal_nan=True)
    # intensity: zero wherever nothing is flaming
    assert np.all(result.grid.intensity[~result.burned_mask] == 0)
    assert np.all(result.grid.burn_timer >= 0)
    assert np.all(result.grid.burn_timer[result.grid.state == int(CellState.BURNED)] == 0)


def test_same_seed_same_output_different_seed_differs(flat_scenario: ScenarioConfig) -> None:
    cfg = flat_scenario.model_copy(update={"duration_s": 1800.0})
    a = run(cfg, seed=7)
    b = run(cfg, seed=7)
    c = run(cfg, seed=8)
    np.testing.assert_array_equal(a.arrival_time, b.arrival_time)
    assert a.events == b.events
    assert a.grid == b.grid
    assert not np.array_equal(a.arrival_time, c.arrival_time, equal_nan=True)


def test_seed_none_uses_config_seed(flat_scenario: ScenarioConfig) -> None:
    cfg = flat_scenario.model_copy(update={"duration_s": 900.0, "seed": 3})
    np.testing.assert_array_equal(run(cfg).arrival_time, run(cfg, seed=3).arrival_time)
    assert run(cfg).provenance["seed"] == 3


def test_fire_ends_and_burns_everything_reachable() -> None:
    cfg = make_config(shape=(15, 15), center=True, duration=36000.0)
    result = run(cfg)
    assert result.provenance["n_steps"] < cfg.n_steps
    assert not (result.grid.state == int(CellState.BURNING)).any()
    assert result.burned_mask[result.grid.burnable_mask()].all()
    assert result.provenance["final_time_s"] == pytest.approx(result.provenance["n_steps"] * cfg.dt_s)


def test_arrival_times_are_consistent(flat_scenario: ScenarioConfig) -> None:
    cfg = flat_scenario.model_copy(update={"duration_s": 2400.0})
    result = run(cfg)
    arr = result.arrival_time
    h, w = arr.shape
    padded = np.pad(arr, 1, constant_values=np.nan)
    earliest_nbr = np.full(arr.shape, np.inf)
    for dr, dc in STENCIL_8.offsets:
        nbr = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        earliest_nbr = np.fmin(earliest_nbr, np.where(np.isnan(nbr), np.inf, nbr))
    burned = result.burned_mask.copy()
    burned[h // 2, w // 2] = False  # the ignition cell
    assert np.all(earliest_nbr[burned] < arr[burned])
    remainders = np.mod(arr[result.burned_mask], cfg.dt_s)
    assert np.count_nonzero(remainders > 1e-6) > 0.9 * remainders.size, "arrival times must be interpolated"


# ----------------------------------------------------------------------------- kernel accuracy


@pytest.mark.parametrize("stencil", [8, 16])
def test_deterministic_limit_matches_distance_over_ros(stencil: int) -> None:
    """With thresholds ~1, arrival = path length / R along straight stencil lines, for any dt."""
    ros, dx, c = 0.2, 30.0, 20
    results = []
    for dt in (30.0, 7.0):
        cfg = make_config(
            shape=(41, 41),
            cells=[(c, c)],
            ros=ros,
            dt=dt,
            duration=4800.0,
            stencil=stencil,
            arrival_shape=DETERMINISTIC,
        )
        arr = run(cfg).arrival_time
        n = np.arange(1, 20)
        np.testing.assert_allclose(arr[c, c + n], n * dx / ros, rtol=0.01)
        np.testing.assert_allclose(arr[c - n, c], n * dx / ros, rtol=0.01)
        np.testing.assert_allclose(arr[c + n, c + n], math.sqrt(2) * n * dx / ros, rtol=0.01)
        if stencil == 16:
            m = np.arange(1, 10)
            np.testing.assert_allclose(arr[c + m, c + 2 * m], math.sqrt(5) * m * dx / ros, rtol=0.01)
        results.append(arr)
    np.testing.assert_allclose(results[0], results[1], rtol=0.01)


def test_fast_fire_advances_several_cells_per_step() -> None:
    """``ros * dt = 5 dx``: the leftover time carries over, so the front moves ~5 cells per step."""
    ros, dt, c = 1.0, 150.0, 20
    cfg = make_config(shape=(41, 41), cells=[(c, c)], ros=ros, dt=dt, duration=2 * dt, arrival_shape=DETERMINISTIC)
    result = run(cfg)
    extent = int(np.nonzero(result.burned_mask[c, c:])[0].max())
    assert 9 <= extent <= 10
    n = np.arange(1, extent)
    np.testing.assert_allclose(result.arrival_time[c, c + n], n * 30.0 / ros, rtol=0.01)
    # same with the default (stochastic) kernel
    stochastic = run(cfg.model_copy(update={"spread": cfg.spread.model_copy(update={"arrival_shape": 20.0})}))
    extent_s = int(np.nonzero(stochastic.burned_mask[c, c:])[0].max())
    assert 7 <= extent_s <= 13


@pytest.mark.parametrize("stencil", [8, 16])
def test_same_seed_arrival_times_do_not_depend_on_dt(stencil: int) -> None:
    """Rates constant within steps: the in-step cascade is exact, so dt changes only rounding.

    Covers slow steps (R dt = dx / 20) and fast ones (R dt = 5 dx, several crossings per step).
    """
    ros = 1.0
    arrays = []
    for dt in (1.5, 30.0, 150.0):
        cfg = make_config(shape=(41, 41), center=True, ros=ros, dt=dt, duration=450.0, stencil=stencil)
        arrays.append(run(cfg, seed=4).arrival_time)
    for other in arrays[1:]:
        np.testing.assert_array_equal(np.isnan(arrays[0]), np.isnan(other))
        np.testing.assert_allclose(other, arrays[0], rtol=1e-9, atol=1e-9)


def test_thin_front_on_flat_no_wind(flat_scenario: ScenarioConfig) -> None:
    """At mid-run the BURNING set is a ring, a small fraction of the burned area."""
    cfg = flat_scenario.model_copy(update={"duration_s": 3600.0})
    result = run(cfg)
    n_burning = int((result.grid.state == int(CellState.BURNING)).sum())
    n_burned = int(result.burned_mask.sum())
    assert n_burned > 1000
    assert n_burning < 0.3 * n_burned, (n_burning, n_burned)


def test_slow_strip_delays_front_by_harmonic_prediction() -> None:
    """A strip of ``w`` cells at R/4 adds ``3 w dx / R`` to the arrival of a planar front."""
    ros, dx, w, c0 = 0.2, 30.0, 5, 15
    cfg = make_config(
        shape=(7, 41), cells=[(r, 0) for r in range(7)], ros=ros, duration=8000.0, arrival_shape=DETERMINISTIC
    )
    result = run(cfg, rule=StripRule(ros, c0, c0 + w, 4.0))
    arr = result.arrival_time
    assert arr[3, 10] == pytest.approx(10 * dx / ros, rel=0.02)
    expected = (30 + 3 * w) * dx / ros
    np.testing.assert_allclose(arr[:, 30], expected, rtol=0.02)


def test_diagonal_barrier_blocks_stencil_8(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = flat_uniform((21, 21), 30.0)
    idx = np.arange(21)
    grid.fuel_model[idx, idx] = 0
    grid.apply_unburnable()
    cfg = make_config(cells=[(15, 3)], duration=10800.0, stencil=8)
    result = run(cfg, grid=grid)
    rows, cols = np.nonzero(result.burned_mask)
    assert rows.size > 50
    assert np.all(cols < rows), "fire leaked across the corner-contact barrier"
    # without link blocking the same barrier leaks: the test is meaningful
    monkeypatch.setattr(sim, "_blocked_links", lambda barrier, stencil: np.zeros((stencil.size, *barrier.shape), bool))
    leaky = run(cfg, grid=grid)
    rows, cols = np.nonzero(leaky.burned_mask)
    assert np.any(cols > rows)


def test_one_cell_barrier_blocks_stencil_16(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = flat_uniform((21, 21), 30.0)
    grid.fuel_model[:, 10] = 0
    grid.apply_unburnable()
    cfg = make_config(cells=[(10, 3)], duration=10800.0, stencil=16)
    result = run(cfg, grid=grid)
    assert result.burned_mask[:, 9].any()
    assert not result.burned_mask[:, 11:].any()
    monkeypatch.setattr(sim, "_blocked_links", lambda barrier, stencil: np.zeros((stencil.size, *barrier.shape), bool))
    assert run(cfg, grid=grid).burned_mask[:, 11:].any()


def test_blocked_links_mask() -> None:
    barrier = np.zeros((5, 5), bool)
    barrier[2, 1] = barrier[1, 2] = True  # corner contact around (1, 1)-(2, 2)
    blocked = sim._blocked_links(barrier, STENCIL_8)
    se = [tuple(o) for o in STENCIL_8.offsets.tolist()].index((1, 1))
    nw = [tuple(o) for o in STENCIL_8.offsets.tolist()].index((-1, -1))
    assert blocked[se, 1, 1] and blocked[nw, 2, 2]
    assert blocked.sum() == 2  # only the two directions of the (1, 1) <-> (2, 2) link
    barrier = np.zeros((5, 5), bool)
    barrier[1:4, 2] = True
    blocked16 = sim._blocked_links(barrier, STENCIL_16)
    knight = [tuple(o) for o in STENCIL_16.offsets.tolist()].index((1, 2))
    assert blocked16[knight, 1, 1]
    assert not blocked16[knight, 0, 0]  # (0, 1) is not a barrier


# ----------------------------------------------------------------------------- hooks and random streams


def test_noop_hook_is_bit_identical(flat_scenario: ScenarioConfig) -> None:
    cfg = flat_scenario.model_copy(update={"duration_s": 1800.0})
    base = run(cfg)
    calls: list[float] = []

    def noop(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        calls.append(t)
        return False

    def random_noop(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        rng.random(100)
        return False

    def forced_refresh(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        rng.normal(size=10)
        return True

    for hooks in ((noop,), (random_noop,), (forced_refresh,), (noop, random_noop)):
        calls.clear()
        other = run(cfg, hooks=hooks)
        np.testing.assert_array_equal(base.arrival_time, other.arrival_time)
        assert base.events == other.events
        assert base.grid == other.grid
    assert calls == [i * cfg.dt_s for i in range(len(calls))]
    assert len(calls) == base.provenance["n_steps"]


@pytest.mark.parametrize("stencil", [8, 16])
def test_fireline_hook_stops_fire(stencil: int) -> None:
    cfg = make_config(shape=(21, 41), cells=[(10, 2)], duration=9000.0, stencil=stencil)
    wall = 25

    def build_line(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        if t == 600.0:
            grid.fireline[:, wall] = True
            grid.apply_unburnable()
            return True
        return False

    result = run(cfg, hooks=[build_line])
    assert result.burned_mask[:, wall - 1].any()
    assert not result.burned_mask[:, wall:].any()
    assert run(cfg).burned_mask[:, wall + 1 :].any()  # without the hook the fire crosses


def test_weather_change_changes_front_speed() -> None:
    """R = 0.1 m/s until t = 1800 s, then 0.3 m/s; planar front, deterministic limit."""
    cfg = make_config(
        shape=(5, 30),
        cells=[(r, 0) for r in range(5)],
        dt=60.0,
        duration=4000.0,
        arrival_shape=DETERMINISTIC,
        weather=[{"time_s": 0.0, "wind_speed": 0.0}, {"time_s": 1800.0, "wind_speed": 4.0}],
    )
    arr = run(cfg, rule=WindRule(0.1, 0.05)).arrival_time
    assert arr[2, 3] == pytest.approx(900.0, rel=0.01)
    assert arr[2, 6] == pytest.approx(1800.0, rel=0.01)
    assert arr[2, 10] == pytest.approx(2200.0, rel=0.01)
    assert arr[2, 20] == pytest.approx(3200.0, rel=0.01)


# ----------------------------------------------------------------------------- ignition


def test_ignition_outside_explicit_grid() -> None:
    cfg = make_config(cells=[(15, 15)])
    with pytest.raises(ValueError, match=r"\(15, 15\)"):
        run(cfg, grid=flat_uniform((10, 10), 30.0))


def test_ignition_outside_cached_grid(tmp_path: Path) -> None:
    path = tmp_path / "g.npz"
    flat_uniform((10, 10), 30.0).save(path)
    cfg = ScenarioConfig.model_validate(
        {
            "name": "c",
            "grid": {"kind": "cached", "path": str(path)},
            "spread": {"rule": "constant", "params": {"ros": 0.2}},
            "ignition": {"cells": [[3, 12]]},
        }
    )
    with pytest.raises(ValueError, match=r"\(3, 12\)"):
        run(cfg)


def test_ignition_on_unburnable_cell() -> None:
    cfg = make_config(generator="with_river", grid_params={"col": 10}, cells=[(4, 10)])
    with pytest.raises(ValueError, match=r"\(4, 10\)"):
        run(cfg)


def test_delayed_ignition() -> None:
    seen_burning: list[tuple[float, int]] = []

    def watch(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        seen_burning.append((t, int(grid.burning_mask().sum())))
        return False

    cfg = make_config(center=True, ignition_time=95.0, duration=1200.0)
    result = run(cfg, hooks=[watch])
    assert result.arrival_time[10, 10] == 95.0
    assert np.nanmin(result.arrival_time) == 95.0
    assert result.events[0] == (95.0, "ignition", 10, 10)
    assert all(n == 0 for t, n in seen_burning if t < 90.0)
    assert result.burned_mask.sum() > 5


def test_rejects_grid_already_burning() -> None:
    grid = flat_uniform((10, 10), 30.0)
    grid.state[0, 0] = int(CellState.BURNED)
    with pytest.raises(ValueError):
        run(make_config(cells=[(5, 5)]), grid=grid)


# ----------------------------------------------------------------------------- outputs


def test_passed_grid_is_not_mutated() -> None:
    grid = flat_uniform((21, 21), 30.0)
    before = grid.copy()
    result = run(make_config(center=True, duration=1200.0), grid=grid)
    assert grid == before
    assert result.burned_mask.sum() > 1


def test_record_events_false_and_events_content() -> None:
    cfg = make_config(center=True, duration=1200.0)
    assert run(cfg, record_events=False).events == []
    result = run(cfg)
    assert result.events[0] == (0.0, "ignition", 10, 10)
    assert all(kind == "spread" for _, kind, _, _ in result.events[1:])
    assert len(result.events) == int(result.burned_mask.sum())
    times = [e[0] for e in result.events]
    assert times == sorted(times)
    for t, _, r, c in result.events:
        assert result.arrival_time[r, c] == t


def test_provenance_is_json_serializable(flat_scenario: ScenarioConfig) -> None:
    cfg = flat_scenario.model_copy(update={"duration_s": 600.0})
    result = run(cfg, seed=11)
    text = json.dumps(result.provenance)
    back = json.loads(text)
    assert back["seed"] == 11
    assert back["config"] == cfg.model_dump(mode="json")
    for key in ("git_commit", "versions", "stencil", "arrival_shape", "kernel_correction", "n_steps", "final_time_s"):
        assert key in back
    assert set(back["versions"]) >= {"python", "numpy", "scipy", "wildfire_sim"}
    assert result.elapsed_s >= 0


def test_max_source_age_limits_spread() -> None:
    """A source that may burn for less than one crossing time never ignites a neighbour."""
    cfg = make_config(center=True, duration=1200.0, arrival_shape=DETERMINISTIC)
    result = run(cfg, max_source_age_s=100.0)  # crossing takes 150 s
    assert result.burned_mask.sum() == 1
    assert run(cfg, max_source_age_s=200.0).burned_mask.sum() > 1


def test_burn_timer_set_from_crossing_and_residence() -> None:
    """Right after ignition the timer is dx / ros_head + residence time, minus elapsed time."""
    cfg = make_config(center=True, dt=30.0, duration=30.0)
    result = run(cfg)
    residence = load_residence_times().lookup(np.array([1]))[0]
    assert result.grid.burn_timer[10, 10] == pytest.approx(30.0 / 0.2 + residence - 30.0, rel=1e-5)
    assert result.grid.state[10, 10] == int(CellState.BURNING)


def test_uses_custom_rule_and_retardant() -> None:
    """Retardant 0.5 on a passed grid halves the deterministic front speed."""
    grid = flat_uniform((5, 21), 30.0)
    grid.retardant[:] = 0.5
    cfg = make_config(shape=(5, 21), cells=[(2, 0)], duration=7200.0, arrival_shape=DETERMINISTIC)
    arr = run(cfg, rule=UniformRule(0.2), grid=grid).arrival_time
    assert arr[2, 10] == pytest.approx(10 * 30.0 / 0.1, rel=0.01)


def test_model_provenance_matches_a_run() -> None:
    """The shared provenance helper returns exactly the code/kernel entries a run records."""
    import json

    from wildfire_sim.sim import model_provenance

    cfg = ScenarioConfig.model_validate(
        {
            "name": "prov",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [11, 11]},
            "spread": {"rule": "constant", "params": {"ros": 0.2}},
            "ignition": {"center": True},
            "duration_s": 60.0,
        }
    )
    shared = model_provenance(cfg, "ConstantRule")
    json.dumps(shared)
    recorded = run(cfg, seed=0).provenance
    assert {key: recorded[key] for key in shared} == shared
    assert model_provenance(cfg)["rule"] == "constant"
