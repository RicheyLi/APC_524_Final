"""Tests for wildfire_sim.grid: Grid container, Weather, WeatherSeries."""

from pathlib import Path

import numpy as np
import pytest

from wildfire_sim.grid import LAYER_SPECS, Grid, Weather, WeatherSeries
from wildfire_sim.types import CellState, LandUse


def test_empty_grid_has_all_layers_with_right_dtypes() -> None:
    g = Grid.empty((5, 7), 30.0)
    assert g.shape == (5, 7)
    assert g.n_cells == 35
    for name, (dtype, _) in LAYER_SPECS.items():
        arr = getattr(g, name)
        assert arr.shape == (5, 7), name
        assert arr.dtype == np.dtype(dtype), name
    assert np.isnan(g.arrival_time).all()
    assert (g.state == int(CellState.FUEL)).all()


@pytest.mark.parametrize("shape,dx", [((0, 5), 30.0), ((5,), 30.0), ((5, 5), 0.0), ((5, 5), -1.0)])
def test_empty_rejects_bad_args(shape, dx) -> None:
    with pytest.raises(ValueError):
        Grid.empty(shape, dx)  # type: ignore[arg-type]


def test_validate_catches_shape_dtype_and_range(small_grid: Grid) -> None:
    small_grid.validate()
    bad = small_grid.copy()
    bad.slope = bad.slope[:-1]
    with pytest.raises(ValueError, match="shape"):
        bad.validate()
    bad = small_grid.copy()
    bad.state = bad.state.astype(np.int32)
    with pytest.raises(ValueError, match="dtype"):
        bad.validate()
    bad = small_grid.copy()
    bad.canopy_cover[0, 0] = 1.5
    with pytest.raises(ValueError, match="canopy_cover"):
        bad.validate()
    bad = small_grid.copy()
    bad.state[0, 0] = 9
    with pytest.raises(ValueError, match="state"):
        bad.validate()
    bad = small_grid.copy()
    bad.aspect[0, 0] = 360.0
    with pytest.raises(ValueError, match="aspect"):
        bad.validate()


def test_burnable_mask_and_apply_unburnable() -> None:
    g = Grid.empty((4, 4), 30.0)
    g.fuel_model[:] = 1
    g.land_use[0, :] = int(LandUse.WATER)
    g.fuel_model[1, 0] = 0
    g.fireline[2, 2] = True
    mask = g.burnable_mask()
    assert not mask[0].any()
    assert not mask[1, 0]
    assert not mask[2, 2]
    assert mask[3].all()
    g.apply_unburnable()
    assert (g.state[0] == int(CellState.UNBURNABLE)).all()
    assert g.state[3, 3] == int(CellState.FUEL)
    # idempotent
    before = g.state.copy()
    g.apply_unburnable()
    assert np.array_equal(before, g.state)


def test_burned_area(small_grid: Grid) -> None:
    small_grid.state[0, 0] = int(CellState.BURNING)
    small_grid.state[0, 1] = int(CellState.BURNED)
    assert small_grid.burning_mask().sum() == 1
    assert small_grid.burned_mask().sum() == 2
    assert small_grid.burned_area_m2() == pytest.approx(2 * 30.0 * 30.0)


def test_copy_is_deep(small_grid: Grid) -> None:
    c = small_grid.copy()
    assert c == small_grid
    c.state[0, 0] = int(CellState.BURNING)
    assert small_grid.state[0, 0] == int(CellState.FUEL)
    assert c != small_grid


def test_save_load_roundtrip(small_grid: Grid, tmp_path: Path) -> None:
    small_grid.metadata["note"] = "roundtrip"
    small_grid.arrival_time[3, 3] = 120.0
    p = tmp_path / "g.npz"
    small_grid.save(p)
    loaded = Grid.load(p)
    assert loaded == small_grid
    assert loaded.metadata["note"] == "roundtrip"


def test_committed_fixture_loads(fixture_grid_path: Path) -> None:
    g = Grid.load(fixture_grid_path)
    assert g.shape == (21, 21)
    assert g.dx == 30.0
    assert g.burnable_mask().all()


def test_weather_defaults_and_validation() -> None:
    w = Weather()
    assert w.wind_speed == 0.0
    assert Weather(wind_direction=270.0).wind_to_bearing == 90.0
    with pytest.raises(ValueError):
        Weather(wind_speed=-1)
    with pytest.raises(ValueError):
        Weather(wind_direction=360)
    with pytest.raises(ValueError):
        Weather(moisture_1h=-0.1)


def test_weather_series_lookup() -> None:
    a, b, c = Weather(wind_speed=1), Weather(wind_speed=2), Weather(wind_speed=3)
    s = WeatherSeries(times_s=(0.0, 3600.0, 7200.0), values=(a, b, c))
    assert s.at(0) is a
    assert s.at(3599.9) is a
    assert s.at(3600) is b
    assert s.at(1e9) is c
    assert WeatherSeries.constant(a).at(5000) is a
    with pytest.raises(ValueError):
        s.at(-1)
    with pytest.raises(ValueError):
        WeatherSeries(times_s=(10.0,), values=(a,))
    with pytest.raises(ValueError):
        WeatherSeries(times_s=(0.0, 0.0), values=(a, b))
