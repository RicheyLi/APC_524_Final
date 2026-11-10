"""Tests for the Alexandridis et al. (2008) spread rule (WP 1.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from wildfire_sim.config import CitedValue, configs_dir, load_scenario, load_yaml
from wildfire_sim.data.synthetic import flat_uniform, slope_ramp
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread import SpreadRule, make_spread_rule
from wildfire_sim.spread.alexandridis import AlexandridisParams, AlexandridisRule
from wildfire_sim.types import STENCIL_8, STENCIL_16, LandUse

PARAMS_PATH = configs_dir() / "alexandridis.yaml"
E, W, N, S = 2, 6, 0, 4  # STENCIL_8 indices


@pytest.fixture
def params() -> AlexandridisParams:
    return load_yaml(PARAMS_PATH, AlexandridisParams)


def _modified_yaml(tmp_path: Path, **changes: float) -> Path:
    raw = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    for key, value in changes.items():
        raw[key]["value"] = value
    out = tmp_path / "alexandridis.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- the YAML


def test_shipped_yaml_validates_and_every_value_is_cited(params: AlexandridisParams) -> None:
    def cited(obj: object) -> list[CitedValue]:
        if isinstance(obj, CitedValue):
            return [obj]
        if hasattr(obj, "__dict__"):
            return [c for v in vars(obj).values() for c in cited(v)]
        if isinstance(obj, list | tuple):
            return [c for v in obj for c in cited(v)]
        return []

    values = cited(params)
    assert len(values) >= 12  # ros_base, a, c1, c2, 3 p_veg, 3 p_den, 2 thresholds
    assert all(v.source.strip() for v in values)
    assert params.vegetation_map.source.strip()
    assert params.a.value == pytest.approx(0.078)
    assert params.c1.value == pytest.approx(0.045)
    assert params.c2.value == pytest.approx(0.131)


def test_shipped_yaml_maps_every_scott_burgan_and_anderson_model(params: AlexandridisParams) -> None:
    expected = {
        1: "agricultural",
        3: "agricultural",
        4: "thickets",
        7: "thickets",
        8: "forest",
        13: "forest",
        101: "agricultural",
        109: "agricultural",
        121: "thickets",
        145: "thickets",
        161: "forest",
        185: "forest",
        204: "forest",
    }
    for fuel, veg in expected.items():
        assert params.vegetation_class(fuel) == veg, fuel
    assert params.vegetation_class(50) is None


def test_schema_rejects_unknown_keys_and_overlapping_ranges(tmp_path: Path) -> None:
    raw = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    raw["typo"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(bad, AlexandridisParams)

    raw = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    raw["vegetation_map"]["ranges"].append({"first": 2, "last": 2, "vegetation": "forest"})
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_yaml(bad, AlexandridisParams)


# ----------------------------------------------------------------------------- registry / contract


def test_registered_and_satisfies_protocol() -> None:
    rule = make_spread_rule("alexandridis", ros_base=0.2)
    assert isinstance(rule, AlexandridisRule)
    assert isinstance(rule, SpreadRule)
    assert rule.name == "alexandridis"


@pytest.mark.parametrize("stencil", [STENCIL_8, STENCIL_16])
def test_output_validates(stencil, small_grid: Grid, windy_weather: Weather) -> None:
    out = AlexandridisRule().rate_of_spread(small_grid, windy_weather, stencil)
    out.validate(stencil, small_grid.shape)
    assert np.all(out.intensity == 0) and np.all(out.flame_length == 0)


def test_steady_wind_scenario_uses_rule() -> None:
    cfg = load_scenario(configs_dir() / "scenarios" / "steady_wind.yaml")
    assert cfg.spread.rule == "alexandridis"
    rule = make_spread_rule(cfg.spread.rule, **cfg.spread.params)
    assert isinstance(rule, AlexandridisRule)


# ----------------------------------------------------------------------------- physics


def test_wind_from_west_fastest_east_slowest_west(small_grid: Grid, params: AlexandridisParams) -> None:
    v = 5.0
    out = AlexandridisRule().rate_of_spread(small_grid, Weather(wind_speed=v, wind_direction=270.0), STENCIL_8)
    centre = out.ros[:, 10, 10]
    assert int(np.argmax(centre)) == E
    assert int(np.argmin(centre)) == W
    assert centre[E] / centre[W] == pytest.approx(np.exp(2 * params.c2.value * v))
    # symmetric about the wind axis
    assert centre[N] == pytest.approx(centre[S])


def test_no_wind_flat_all_directions_equal(small_grid: Grid, calm_weather: Weather) -> None:
    for stencil in (STENCIL_8, STENCIL_16):
        out = AlexandridisRule().rate_of_spread(small_grid, calm_weather, stencil)
        np.testing.assert_allclose(out.ros, out.ros[0, 0, 0])
        assert out.ros[0, 0, 0] > 0


def test_no_wind_slope_uphill_fastest(calm_weather: Weather, params: AlexandridisParams) -> None:
    grid = slope_ramp((21, 21), 30.0, fuel_model=1, slope=0.3, aspect_deg=180.0)  # faces south, uphill north
    out = AlexandridisRule().rate_of_spread(grid, calm_weather, STENCIL_8)
    centre = out.ros[:, 10, 10]
    assert int(np.argmax(centre)) == N
    assert int(np.argmin(centre)) == S
    theta_s = np.degrees(np.arctan(0.3))
    assert centre[N] / centre[S] == pytest.approx(np.exp(2 * params.a.value * theta_s), rel=1e-5)


def test_edge_cells_finite_and_zero_slope_toward_outside(calm_weather: Weather) -> None:
    grid = slope_ramp((5, 5), 30.0, fuel_model=1, slope=0.5, aspect_deg=180.0)
    out = AlexandridisRule().rate_of_spread(grid, calm_weather, STENCIL_16)
    assert np.all(np.isfinite(out.ros))
    flat = AlexandridisRule().rate_of_spread(flat_uniform((5, 5), 30.0), calm_weather, STENCIL_16)
    # north neighbour of the top row is outside -> no slope factor
    assert out.ros[0, 0, 2] == pytest.approx(flat.ros[0, 0, 2])


def test_extreme_inputs_stay_finite() -> None:
    grid = flat_uniform((5, 5), 1.0)
    grid.elevation[:, 2:] = 1e6
    out = AlexandridisRule().rate_of_spread(grid, Weather(wind_speed=1e4, wind_direction=0.0), STENCIL_8)
    out.validate(STENCIL_8, grid.shape)


# ----------------------------------------------------------------------------- inputs and masks


def test_non_burnable_cells_are_zero(small_grid: Grid, windy_weather: Weather) -> None:
    small_grid.fuel_model[3, 3] = 0
    small_grid.land_use[4, 4] = int(LandUse.WATER)
    small_grid.fireline[5, 5] = True
    out = AlexandridisRule().rate_of_spread(small_grid, windy_weather, STENCIL_8)
    for r, c in [(3, 3), (4, 4), (5, 5)]:
        assert np.all(out.ros[:, r, c] == 0)
    assert np.all(out.ros[:, 10, 10] > 0)


def test_retardant_halves_ros(small_grid: Grid, windy_weather: Weather) -> None:
    base = AlexandridisRule().rate_of_spread(small_grid, windy_weather, STENCIL_8)
    small_grid.retardant[10, 10] = 0.5
    out = AlexandridisRule().rate_of_spread(small_grid, windy_weather, STENCIL_8)
    np.testing.assert_allclose(out.ros[:, 10, 10], 0.5 * base.ros[:, 10, 10])
    np.testing.assert_allclose(out.ros[:, 9, 9], base.ros[:, 9, 9])


def test_vegetation_and_density_factors(small_grid: Grid, calm_weather: Weather, params: AlexandridisParams) -> None:
    small_grid.fuel_model[2, 2] = 165  # TU5 -> forest, canopy 0 -> sparse
    small_grid.canopy_cover[4, 4] = 0.9  # grass, dense
    out = AlexandridisRule().rate_of_spread(small_grid, calm_weather, STENCIL_8)
    grass_sparse = out.ros[0, 10, 10]
    forest_sparse = out.ros[0, 2, 2]
    grass_dense = out.ros[0, 4, 4]
    assert forest_sparse > grass_sparse
    assert grass_dense > grass_sparse
    pv, pd = params.p_veg, params.p_den
    assert forest_sparse / grass_sparse == pytest.approx((1 + pv.forest.value) / (1 + pv.agricultural.value))
    assert grass_dense / grass_sparse == pytest.approx((1 + pd.dense.value) / (1 + pd.sparse.value))
    expected = 0.1 * (1 + pv.agricultural.value) * (1 + pd.sparse.value)
    assert AlexandridisRule(ros_base=0.1).rate_of_spread(small_grid, calm_weather, STENCIL_8).ros[
        0, 10, 10
    ] == pytest.approx(expected)


def test_unmapped_burnable_fuel_model_raises(small_grid: Grid, calm_weather: Weather) -> None:
    small_grid.fuel_model[0, 0] = 50
    with pytest.raises(ValueError, match="50"):
        AlexandridisRule().rate_of_spread(small_grid, calm_weather, STENCIL_8)


def test_constants_come_from_yaml(tmp_path: Path, small_grid: Grid, windy_weather: Weather) -> None:
    path = _modified_yaml(tmp_path, ros_base=0.5, c2=0.3)
    out = AlexandridisRule(params_path=path).rate_of_spread(small_grid, windy_weather, STENCIL_8)
    centre = out.ros[:, 10, 10]
    assert centre[E] / centre[W] == pytest.approx(np.exp(2 * 0.3 * 5.0))
    ref = AlexandridisRule(ros_base=0.5).rate_of_spread(small_grid, Weather(), STENCIL_8).ros[0, 10, 10]
    calm = AlexandridisRule(params_path=path).rate_of_spread(small_grid, Weather(), STENCIL_8).ros[0, 10, 10]
    assert calm == pytest.approx(ref)


def test_constructor_overrides(small_grid: Grid, windy_weather: Weather, params: AlexandridisParams) -> None:
    base = AlexandridisRule().rate_of_spread(small_grid, windy_weather, STENCIL_8).ros
    doubled = AlexandridisRule(ros_base=2 * params.ros_base.value).rate_of_spread(small_grid, windy_weather, STENCIL_8)
    np.testing.assert_allclose(doubled.ros, 2 * base)
    no_wind = AlexandridisRule(c1=0.0, c2=0.0).rate_of_spread(small_grid, windy_weather, STENCIL_8)
    np.testing.assert_allclose(no_wind.ros, no_wind.ros[0, 0, 0])
    grid = slope_ramp((9, 9), 30.0, slope=0.3)
    no_slope = AlexandridisRule(a=0.0).rate_of_spread(grid, Weather(), STENCIL_8)
    np.testing.assert_allclose(no_slope.ros, no_slope.ros[0, 0, 0])


def test_negative_ros_base_rejected() -> None:
    with pytest.raises(ValueError):
        AlexandridisRule(ros_base=-0.1)


def test_does_not_modify_grid_and_is_deterministic(windy_weather: Weather) -> None:
    grid = slope_ramp((15, 15), 30.0, slope=0.2, aspect_deg=45.0)
    grid.retardant[3, 3] = 0.4
    before = grid.copy()
    rule = AlexandridisRule()
    a = rule.rate_of_spread(grid, windy_weather, STENCIL_16)
    b = rule.rate_of_spread(grid, windy_weather, STENCIL_16)
    assert grid == before
    np.testing.assert_array_equal(a.ros, b.ros)
