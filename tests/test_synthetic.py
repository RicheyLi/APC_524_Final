"""Tests for synthetic landscapes and build_grid."""

from pathlib import Path
from typing import get_args

import numpy as np
import pytest

from wildfire_sim.config import CachedGridConfig, SyntheticGridConfig
from wildfire_sim.data import build_grid
from wildfire_sim.data.synthetic import GENERATORS, flat_uniform, slope_ramp, with_river, with_town
from wildfire_sim.grid import Grid
from wildfire_sim.types import CellState, LandUse


def test_flat_uniform() -> None:
    g = flat_uniform((11, 13), 30.0, fuel_model=2)
    g.validate()
    assert g.shape == (11, 13)
    assert (g.fuel_model == 2).all()
    assert g.burnable_mask().all()
    assert (g.state == int(CellState.FUEL)).all()
    with pytest.raises(ValueError):
        flat_uniform(fuel_model=0)


def test_slope_ramp_elevation_consistent_with_slope_and_aspect() -> None:
    g = slope_ramp((21, 21), 30.0, slope=0.25, aspect_deg=180.0)
    g.validate()
    assert (g.slope == np.float32(0.25)).all()
    assert (g.aspect == 180.0).all()
    # aspect 180 = faces south = downhill toward increasing row, so elevation decreases with row
    col_profile = g.elevation[:, 10]
    assert np.all(np.diff(col_profile) < 0)
    assert np.allclose(np.diff(col_profile), -0.25 * 30.0, atol=1e-3)
    assert np.allclose(g.elevation[5, :], g.elevation[5, 0])
    # east-facing ramp: elevation decreases with column
    g2 = slope_ramp((5, 9), 10.0, slope=0.1, aspect_deg=90.0)
    assert np.allclose(np.diff(g2.elevation[2, :]), -1.0, atol=1e-4)
    with pytest.raises(ValueError):
        slope_ramp(slope=-0.1)


def test_with_river_is_unburnable_strip() -> None:
    g = with_river((9, 9), 30.0, col=4, width=2)
    g.validate()
    water = g.land_use == int(LandUse.WATER)
    assert water[:, 4:6].all()
    assert water.sum() == 18
    assert (g.state[:, 4:6] == int(CellState.UNBURNABLE)).all()
    assert g.burnable_mask()[:, :4].all() and g.burnable_mask()[:, 6:].all()
    with pytest.raises(ValueError):
        with_river((9, 9), col=20)


def test_with_town_block_and_assets() -> None:
    g = with_town((15, 15), 30.0, center=(7, 7), size=3, value_per_cell=1000.0)
    g.validate()
    urban = g.land_use == int(LandUse.URBAN)
    assert urban.sum() == 9
    assert urban[6:9, 6:9].all()
    assert g.asset_value.sum() == pytest.approx(9000.0)
    assert (g.state[6:9, 6:9] == int(CellState.UNBURNABLE)).all()
    # block clipped at the edge still works
    g_edge = with_town((15, 15), center=(0, 0), size=3)
    assert (g_edge.land_use == int(LandUse.URBAN)).sum() == 4
    with pytest.raises(ValueError):
        with_town((5, 5), center=(50, 50), size=3)


def test_generators_registry_matches_config_literal() -> None:
    allowed = get_args(SyntheticGridConfig.model_fields["generator"].annotation)
    assert set(GENERATORS) == set(allowed)


def test_build_grid_synthetic_and_cached(tmp_path: Path) -> None:
    cfg = SyntheticGridConfig(generator="with_river", shape=(7, 7), dx=10.0, params={"col": 3, "width": 1})
    g = build_grid(cfg)
    assert g.shape == (7, 7) and g.dx == 10.0
    assert (g.land_use[:, 3] == int(LandUse.WATER)).all()
    p = tmp_path / "g.npz"
    g.save(p)
    g2 = build_grid(CachedGridConfig(path=p))
    assert isinstance(g2, Grid) and g2 == g
