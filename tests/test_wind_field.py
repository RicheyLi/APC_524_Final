"""Tests for the per-cell wind field (WP 3.4): grid.WindField, wind_field.WindFieldSeries, the rules,
the simulator, the ensemble and sensitivity perturbations, the config and the sweep harness.

Offline and deterministic: every field is built in memory or written to ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from wildfire_sim.calibration import Target
from wildfire_sim.config import ScenarioConfig, WeatherConfig
from wildfire_sim.ensemble import perturb_config, perturbed_wind_field, run_ensemble
from wildfire_sim.grid import Grid, Weather, WindField, wind_to_bearing_field
from wildfire_sim.sim import model_provenance, run
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.rothermel import RothermelRule
from wildfire_sim.types import StencilSize, get_stencil
from wildfire_sim.wind_field import (
    WindFieldSeries,
    load_wind_field,
    save_wind_field,
    wind_field_for,
    wind_from_components,
)

REPO = Path(__file__).resolve().parents[1]
HAS_NUMBA = importlib.util.find_spec("numba") is not None


def components(speed: float, direction_from: float) -> tuple[float, float]:
    """East and north components, m/s, of a wind of ``speed`` from ``direction_from`` degrees."""
    b = math.radians(direction_from)
    return -speed * math.sin(b), -speed * math.cos(b)


def uniform_series(shape: tuple[int, int], u: float, v: float, hours: int = 1) -> WindFieldSeries:
    """A 2 x 2 node window with the same wind everywhere, cells spread over it."""
    h, w = shape
    ci, cj = np.meshgrid(np.linspace(0.1, 0.9, h), np.linspace(0.1, 0.9, w), indexing="ij")
    return WindFieldSeries(
        np.arange(hours) * 3600.0,
        np.full((hours, 2, 2), u, dtype=np.float32),
        np.full((hours, 2, 2), v, dtype=np.float32),
        ci.astype(np.float32),
        cj.astype(np.float32),
    )


# ----------------------------------------------------------------------------- WindField


def test_wind_field_validation() -> None:
    f = WindField(np.full((2, 3), 4.0), np.array([[0.0, 90.0, 180.0], [270.0, 359.5, 10.0]]))
    assert f.shape == (2, 3) and f.speed.dtype == np.float64
    assert np.array_equal(f.to_bearing, [[180.0, 270.0, 0.0], [90.0, 179.5, 190.0]])
    assert np.array_equal(wind_to_bearing_field(f.direction_from), f.to_bearing)
    assert f == WindField(f.speed.copy(), f.direction_from.copy()) and hash(f) == hash(WindField.uniform((2, 3), 1, 2))
    assert f != WindField.uniform((2, 3), 4.0, 0.0)
    with pytest.raises(ValueError, match="shapes"):
        WindField(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        WindField(np.array([[np.nan]]), np.zeros((1, 1)))
    with pytest.raises(ValueError, match=">= 0"):
        WindField(np.array([[-1.0]]), np.zeros((1, 1)))
    with pytest.raises(ValueError, match=r"\[0, 360\)"):
        WindField(np.ones((1, 1)), np.full((1, 1), 360.0))
    with pytest.raises(ValueError, match="WindField"):
        Weather(wind_field=cast(Any, np.zeros((2, 2))))
    # the scalars are untouched and weather with a field compares by value
    w = Weather(wind_speed=3.0, wind_direction=90.0, wind_field=f)
    assert w.wind_to_bearing == 270.0 and w == Weather(wind_speed=3.0, wind_direction=90.0, wind_field=f)


# ----------------------------------------------------------------------------- WindFieldSeries


def test_bilinear_interpolation_by_hand() -> None:
    """2 x 2 nodes; u and v interpolated as components, speed and direction derived from them."""
    u = np.array([[[0.0, 2.0], [4.0, 6.0]], [[-3.0, -3.0], [-3.0, -3.0]]], dtype=np.float32)
    v = np.array([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    ci = np.array([[0.0, 0.25], [1.0, 0.5]], dtype=np.float32)
    cj = np.array([[0.0, 0.5], [1.0, 0.75]], dtype=np.float32)
    s = WindFieldSeries(np.array([0.0, 3600.0]), u, v, ci, cj)
    ue, ve = s.components(0)
    # u = 2 j + 4 i on the nodes, bilinear is exact for it
    assert np.allclose(ue, [[0.0, 2 * 0.5 + 4 * 0.25], [6.0, 2 * 0.75 + 4 * 0.5]]) and np.allclose(ve, 1.0)
    f0 = s.at(0.0, (2, 2))
    assert f0.speed[0, 1] == pytest.approx(math.hypot(2.0, 1.0))
    # blowing toward the east-north-east: from the west-south-west, as hrrr.wind_from
    assert f0.direction_from[0, 1] == pytest.approx((math.degrees(math.atan2(-2.0, -1.0))) % 360)
    assert f0.speed[0, 0] == pytest.approx(1.0) and f0.direction_from[0, 0] == pytest.approx(180.0)  # from the south
    # time: piecewise constant with the WeatherSeries rule
    assert s.at(3599.0) is f0 and s.at(1800.0) is f0  # cached within the hour
    f1 = s.at(3600.0)
    assert np.allclose(f1.speed, 3.0) and np.allclose(f1.direction_from, 90.0)  # blowing west: from the east
    assert s.at(1e6) is f1
    with pytest.raises(ValueError, match="t must be"):
        s.at(-1.0)
    with pytest.raises(ValueError, match=r"\(2, 2\) grid"):
        s.at(0.0, (3, 3))


def test_wind_from_components_matches_the_scalar_convention() -> None:
    from wildfire_sim.data.hrrr import wind_from

    u = np.array([0.0, 5.0, -3.0, 0.0, 1e-9])
    v = np.array([-5.0, 0.0, 4.0, 0.0, -1.0])
    speed, direction = wind_from_components(u, v)
    for k in range(u.size):
        assert (speed[k], direction[k]) == pytest.approx(wind_from(float(u[k]), float(v[k])))
    assert np.all((direction >= 0) & (direction < 360))


def test_series_validation_and_perturbation() -> None:
    s = uniform_series((3, 3), *components(4.0, 350.0))
    with pytest.raises(ValueError, match="first wind field time"):
        WindFieldSeries(np.array([10.0]), s.u, s.v, s.cell_i, s.cell_j)
    with pytest.raises(ValueError, match="outside"):
        WindFieldSeries(s.times_s, s.u, s.v, s.cell_i + 1.5, s.cell_j)
    with pytest.raises(ValueError, match="T ="):
        WindFieldSeries(np.array([0.0, 1.0]), s.u, s.v, s.cell_i, s.cell_j)
    with pytest.raises(ValueError, match="at least 2 x 2"):
        WindFieldSeries(s.times_s, s.u[:, :1], s.v[:, :1], s.cell_i * 0, s.cell_j)
    p = s.perturbed(1.5, 20.0)
    base, moved = s.at(0), p.at(0)
    assert np.allclose(moved.speed, 1.5 * base.speed)
    assert np.allclose(moved.direction_from, (base.direction_from + 20.0) % 360)  # wraps past north
    assert np.all(moved.direction_from < 30)
    assert p.perturbed(2.0, -20.0).speed_factor == 3.0 and p.perturbed(2.0, -20.0).direction_offset_deg == 0.0
    with pytest.raises(ValueError, match="speed_factor"):
        s.perturbed(0.0)


def test_save_load_and_missing_file(tmp_path: Path) -> None:
    s = uniform_series((4, 5), 1.0, -2.0, hours=3)
    path = tmp_path / "wind_field.npz"
    save_wind_field(path, s.times_s, s.u, s.v, s.cell_i, s.cell_j)
    loaded = load_wind_field(path)
    assert loaded.shape == (4, 5) and loaded.n_times == 3 and loaded.window_shape == (2, 2)
    assert np.array_equal(loaded.u, s.u) and loaded.path == path and loaded.sha256 is not None
    assert len(loaded.sha256) == 64
    assert loaded.provenance()["ny"] == 2 and loaded.provenance()["grid_shape"] == [4, 5]
    with pytest.raises(FileNotFoundError, match=r"download_data\.py <region> --wind-field"):
        load_wind_field(tmp_path / "nope.npz")
    np.savez(tmp_path / "old.npz", _format_version=np.int64(99), times_s=s.times_s)
    with pytest.raises(ValueError, match="version 99"):
        load_wind_field(tmp_path / "old.npz")


# ----------------------------------------------------------------------------- rules


@pytest.mark.parametrize("rule_name", ["rothermel", "alexandridis"])
@pytest.mark.parametrize("stencil_size", [8, 16])
def test_uniform_field_is_bit_identical_to_the_scalar(rule_name: str, stencil_size: StencilSize) -> None:
    from wildfire_sim.data.synthetic import slope_ramp

    grid = slope_ramp((21, 21), 30.0, fuel_model=102)
    rule = make_spread_rule(rule_name)
    stencil = get_stencil(stencil_size)
    scalar = Weather(wind_speed=5.3, wind_direction=247.3)
    field = dataclasses.replace(scalar, wind_field=WindField.uniform(grid.shape, 5.3, 247.3))
    a, b = rule.rate_of_spread(grid, scalar, stencil), rule.rate_of_spread(grid, field, stencil)
    assert np.array_equal(a.ros, b.ros) and np.array_equal(a.intensity, b.intensity)
    # with no field the scalar path is the one the golden tests pin (tests/test_golden.py)
    assert np.array_equal(rule.rate_of_spread(grid, Weather(wind_speed=5.3, wind_direction=247.3), stencil).ros, a.ros)
    with pytest.raises(ValueError, match="wind field shape"):
        rule.rate_of_spread(grid, dataclasses.replace(scalar, wind_field=WindField.uniform((3, 3), 1, 1)), stencil)


def test_rothermel_uses_each_cells_wind() -> None:
    from wildfire_sim.data.synthetic import flat_uniform

    grid = flat_uniform((4, 6), 30.0, fuel_model=102)
    speed = np.where(np.arange(6)[None, :] < 3, 2.0, 6.0) * np.ones((4, 1))
    direction = np.where(np.arange(6)[None, :] < 3, 90.0, 270.0) * np.ones((4, 1))
    fire = RothermelRule().fire(grid, Weather(wind_field=WindField(speed, direction)))
    assert np.allclose(fire.head_bearing[:, :3], 270.0) and np.allclose(fire.head_bearing[:, 3:], 90.0)
    assert np.all(fire.ros_head[:, 3:] > fire.ros_head[:, :3])
    windy = RothermelRule().fire(grid, Weather(wind_speed=6.0, wind_direction=270.0))
    assert np.array_equal(windy.ros_head[:, 3:], fire.ros_head[:, 3:])


# ----------------------------------------------------------------------------- simulator


def _scenario(grid_path: Path, rule: str, wind_speed: float, wind_direction: float, **extra: Any) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "name": "wind field test",
            "grid": {"kind": "cached", "path": str(grid_path)},
            "weather": {"series": [{"wind_speed": wind_speed, "wind_direction": wind_direction}], **extra},
            "spread": {"rule": rule},
            "ignition": {"cells": [[10, 10]]},
            "dt_s": 10.0,
            "duration_s": 600.0,
            **({"engine": extra.pop("engine")} if "engine" in extra else {}),
        }
    )


ENGINES = ["numpy", pytest.param("numba", marks=pytest.mark.skipif(not HAS_NUMBA, reason="needs the fast extra"))]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("rule", ["rothermel", "alexandridis"])
def test_uniform_field_run_is_bit_identical(tmp_path: Path, fixture_grid_path: Path, rule: str, engine: str) -> None:
    grid = Grid.load(fixture_grid_path)
    series = uniform_series(grid.shape, *components(6.0, 300.0))
    path = tmp_path / "wind_field.npz"
    save_wind_field(path, series.times_s, series.u, series.v, series.cell_i, series.cell_j)
    f = load_wind_field(path).at(0.0)
    speed, direction = float(f.speed[0, 0]), float(f.direction_from[0, 0])
    assert np.all(f.speed == speed) and np.all(f.direction_from == direction)

    plain = _scenario(fixture_grid_path, rule, speed, direction)
    with_field = _scenario(fixture_grid_path, rule, speed, direction, wind_field=str(path))
    update = {"engine": engine, "duration_s": 600.0 if rule == "rothermel" else 3600.0}  # Alexandridis is slower
    plain = plain.model_copy(update=update)
    with_field = with_field.model_copy(update=update)
    a, b = run(plain, seed=3), run(with_field, seed=3)
    assert a.burned_mask.sum() > 10
    assert np.array_equal(a.arrival_time, b.arrival_time, equal_nan=True)
    assert np.array_equal(a.front_intensity, b.front_intensity, equal_nan=True)
    assert a.provenance["wind_field"] is None
    assert b.provenance["wind_field"]["sha256"] == load_wind_field(path).sha256
    assert model_provenance(with_field)["wind_field"]["n_times"] == 1


def _split_series(shape: tuple[int, int], speed: float = 6.0) -> WindFieldSeries:
    """Top half of the grid: wind from the east; bottom half: from the west (4 node rows, no blending)."""
    h, w = shape
    u = np.zeros((1, 4, 2), dtype=np.float32)
    u[0, :2] = components(speed, 90.0)[0]
    u[0, 2:] = components(speed, 270.0)[0]
    rows = np.arange(h)[:, None] * np.ones((1, w))
    ci = np.where(rows < h // 2, 0.5 * rows / (h // 2), 2.0 + 0.5 * (rows - h // 2) / (h - h // 2))
    cj = np.ones((h, w)) * 0.5
    return WindFieldSeries(np.zeros(1), u, np.zeros_like(u), ci.astype(np.float32), cj.astype(np.float32))


def test_opposite_winds_drive_the_halves_opposite_ways(tmp_path: Path) -> None:
    from wildfire_sim.data.synthetic import flat_uniform

    grid = flat_uniform((40, 61), 30.0, fuel_model=1)
    series = _split_series(grid.shape)
    field = series.at(0.0)
    assert np.allclose(field.direction_from[:20], 90.0) and np.allclose(field.direction_from[20:], 270.0)
    config = ScenarioConfig.model_validate(
        {
            "name": "split",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [40, 61], "fuel_model": 1},
            "weather": {"series": [{"wind_speed": 5.0, "wind_direction": 0.0}]},
            "spread": {"rule": "rothermel"},
            "ignition": {"cells": [[9, 30], [30, 30]]},
            "dt_s": 10.0,
            "duration_s": 900.0,
        }
    )
    res = run(config, seed=0, grid=grid, wind_field=series)
    burned = res.burned_mask
    cols = np.arange(61)
    top_west = int(burned[:20, cols < 30].sum())
    top_east = int(burned[:20, cols > 30].sum())
    bottom_west = int(burned[20:, cols < 30].sum())
    bottom_east = int(burned[20:, cols > 30].sum())
    assert top_west > 3 * max(top_east, 1) and bottom_east > 3 * max(bottom_west, 1)
    # the head of each half: furthest burned column from the ignition
    assert cols[burned[9]].min() < 30 - 5 and cols[burned[30]].max() > 30 + 5


def test_field_shape_must_match_the_grid(tmp_path: Path, fixture_grid_path: Path) -> None:
    series = uniform_series((10, 10), 1.0, 1.0)
    path = tmp_path / "wf.npz"
    save_wind_field(path, series.times_s, series.u, series.v, series.cell_i, series.cell_j)
    with pytest.raises(ValueError, match=r"is for a \(10, 10\) grid, but the grid is \(21, 21\)"):
        run(_scenario(fixture_grid_path, "rothermel", 1.0, 0.0, wind_field=str(path)))


# ----------------------------------------------------------------------------- config


def test_config_wind_field_path(tmp_path: Path, fixture_grid_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert WeatherConfig().wind_field is None
    series = uniform_series((21, 21), 0.0, -3.0)
    save_wind_field(tmp_path / "wf.npz", series.times_s, series.u, series.v, series.cell_i, series.cell_j)
    monkeypatch.chdir(tmp_path)  # a relative path resolves against the working directory, like grid.path
    config = _scenario(fixture_grid_path, "rothermel", 3.0, 0.0, wind_field="wf.npz")
    wind = wind_field_for(config)
    assert wind is not None and wind.shape == (21, 21) and wind.path == Path("wf.npz")
    missing = _scenario(fixture_grid_path, "rothermel", 3.0, 0.0, wind_field="nowhere/wind_field.npz")
    with pytest.raises(FileNotFoundError, match=r"nowhere.wind_field\.npz not found"):
        run(missing)
    # the field survives the round trip through the ensemble's weather perturbation
    kept = perturb_config(
        config, {"wind_speed_factor": 2.0, "wind_direction_offset_deg": 0.0, "moisture_factor": 1.0}
    ).weather
    assert kept.wind_field == "wf.npz" and kept.series[0].wind_speed == 6.0


# ----------------------------------------------------------------------------- ensemble and sensitivity


def _ensemble_config(path: Path, **ensemble: Any) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "name": "field ensemble",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [31, 31], "fuel_model": 1},
            "weather": {"series": [{"wind_speed": 6.0, "wind_direction": 270.0}], "wind_field": str(path)},
            "spread": {"rule": "rothermel"},
            "ignition": {"center": True},
            "dt_s": 10.0,
            "duration_s": 600.0,
            "ensemble": ensemble,
        }
    )


def test_ensemble_perturbs_the_field(tmp_path: Path) -> None:
    series = uniform_series((31, 31), *components(6.0, 270.0))  # from the west: the head runs east
    path = tmp_path / "wf.npz"
    save_wind_field(path, series.times_s, series.u, series.v, series.cell_i, series.cell_j)
    config = _ensemble_config(path)
    neutral = {"wind_speed_factor": 1.0, "wind_direction_offset_deg": 0.0}
    assert perturbed_wind_field(config, neutral) is not None
    turned = perturbed_wind_field(config, {"wind_speed_factor": 1.0, "wind_direction_offset_deg": 90.0})
    assert turned is not None
    # the direction offset turns the head: from the north now (270 + 90 wraps to 0), so it runs south
    d = turned.at(0).direction_from
    assert np.all(np.minimum(d, 360.0 - d) < 1e-4)
    grid = Grid.empty((31, 31), 30.0)
    grid.fuel_model[:] = 1
    head = RothermelRule().fire(grid, Weather(wind_field=turned.at(0))).head_bearing
    assert np.allclose(np.cos(np.radians(head - 180.0)), 1.0, atol=1e-6)
    # the speed factor changes the extent
    base = run(config, seed=1)
    faster = run(config, seed=1, wind_field=perturbed_wind_field(config, {"wind_speed_factor": 1.6}))
    assert faster.burned_mask.sum() > base.burned_mask.sum()
    assert faster.provenance["wind_field"]["speed_factor"] == 1.6
    south = run(config, seed=1, wind_field=turned)
    rows = np.nonzero(south.burned_mask)[0]
    assert rows.max() - 15 > 15 - rows.min() + 2  # ran south


def test_run_ensemble_with_a_field_is_reproducible(tmp_path: Path) -> None:
    series = uniform_series((31, 31), *components(6.0, 270.0))
    path = tmp_path / "wf.npz"
    save_wind_field(path, series.times_s, series.u, series.v, series.cell_i, series.cell_j)
    config = _ensemble_config(path, wind_speed_sigma_frac=0.3, wind_direction_sigma_deg=40.0)
    a = run_ensemble(config, n_runs=3, base_seed=5)
    b = run_ensemble(config, n_runs=3, base_seed=5)
    assert np.array_equal(a.arrival_times, b.arrival_times, equal_nan=True)
    # without the perturbation applied to the field, the members would only differ through the
    # scalar (which the rule ignores when a field is present) and the seed's thresholds
    plain = _ensemble_config(path)
    c = run_ensemble(plain, n_runs=3, base_seed=5)
    assert not np.array_equal(a.arrival_times, c.arrival_times, equal_nan=True)


def test_sensitivity_run_point_perturbs_the_field(tmp_path: Path) -> None:
    from wildfire_sim.sensitivity import SensitivityInput, run_point

    series = uniform_series((31, 31), *components(6.0, 270.0))
    path = tmp_path / "wf.npz"
    save_wind_field(path, series.times_s, series.u, series.v, series.cell_i, series.cell_j)
    base = _ensemble_config(path)
    spec = SensitivityInput.model_validate(
        {"target": "wind_speed_factor", "range": [0.7, 1.6], "units": "factor", "source": "test"}
    )
    slow = run_point(base, {"wind": spec}, {"wind": 0.7}, seed=1, window_s=600.0, head_distance_m=200.0)
    fast = run_point(base, {"wind": spec}, {"wind": 1.6}, seed=1, window_s=600.0, head_distance_m=200.0)
    assert fast.burned_area_ha > slow.burned_area_ha


# ----------------------------------------------------------------------------- sweep harness


def _sweep() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sweep_eval", REPO / "scripts" / "sweep_eval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("no_field", [False, True])
def test_sweep_eval_smoke(
    tmp_path: Path, fixture_grid_path: Path, monkeypatch: pytest.MonkeyPatch, no_field: bool
) -> None:
    grid = Grid.load(fixture_grid_path)
    series = uniform_series(grid.shape, *components(3.0, 270.0))
    path = tmp_path / "wind_field.npz"
    save_wind_field(path, series.times_s, series.u, series.v, series.cell_i, series.cell_j)
    module = _sweep()
    asked: dict[str, Any] = {}

    def fake_real_fire(name: str, hours: float, **kwargs: Any) -> Any:
        asked.update(kwargs)
        extra = {} if not kwargs.get("wind_field", True) else {"wind_field": str(path)}
        config = _scenario(fixture_grid_path, "rothermel", 3.0, 270.0, **extra)
        config = config.model_copy(update={"duration_s": 600.0})
        return SimpleNamespace(config=config, grid=grid)

    observed = np.zeros(grid.shape, dtype=bool)
    observed[8:13, 10:18] = True

    def fake_target(fire: Any, footprint_m: float) -> Target:
        return Target(observed, (), 0.0, 600.0, footprint_m, "fake")

    monkeypatch.setattr(module, "real_fire", fake_real_fire)
    monkeypatch.setattr(module, "observed_target", fake_target)
    out = tmp_path / "sweep.json"
    argv = ["lake_2024", "--multipliers", "1.0", "--dt", "10", "--out", str(out), "--wind-scale", "1.2"]
    assert module.main([*argv, "--no-wind-field"] if no_field else argv) == 0
    import json

    record = json.loads(out.read_text(encoding="utf-8"))
    assert asked["wind_field"] is (not no_field)
    assert record["best"]["ros_multiplier"] == 1.0 and 0 <= record["best"]["jaccard"] <= 1
    assert set(record["observed_sectors"]) == set(module.SECTORS)
    if no_field:
        assert record["wind_field"] is None
    else:
        assert record["wind_field"]["speed_factor"] == 1.2 and record["wind_field"]["path"] == str(path)
