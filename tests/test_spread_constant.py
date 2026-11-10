"""Tests for the constant-rate spread rule (WP 1.1)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread.base import SpreadRule
from wildfire_sim.spread.constant import ConstantRule
from wildfire_sim.types import STENCIL_8, STENCIL_16, LandUse, Stencil


@pytest.mark.parametrize("stencil", [STENCIL_8, STENCIL_16], ids=["stencil8", "stencil16"])
def test_output_validates_and_is_uniform(small_grid: Grid, calm_weather: Weather, stencil: Stencil) -> None:
    out = ConstantRule(ros=0.2).rate_of_spread(small_grid, calm_weather, stencil)
    out.validate(stencil, small_grid.shape)
    assert out.ros.shape == (stencil.size, *small_grid.shape)
    np.testing.assert_array_equal(out.ros, 0.2)
    np.testing.assert_array_equal(out.intensity, np.zeros(small_grid.shape))
    np.testing.assert_array_equal(out.flame_length, np.zeros(small_grid.shape))


def test_rule_satisfies_protocol() -> None:
    rule = ConstantRule(ros=0.1)
    assert isinstance(rule, SpreadRule)
    assert rule.name == "constant"


def test_non_burnable_cells_are_zero(small_grid: Grid, calm_weather: Weather) -> None:
    grid = small_grid
    grid.land_use[1, 1] = int(LandUse.WATER)
    grid.land_use[2, 2] = int(LandUse.URBAN)
    grid.fireline[3, 3] = True
    grid.fuel_model[4, 4] = 0
    out = ConstantRule(ros=0.3).rate_of_spread(grid, calm_weather, STENCIL_8)
    for r, c in [(1, 1), (2, 2), (3, 3), (4, 4)]:
        assert np.all(out.ros[:, r, c] == 0.0)
    burnable = grid.burnable_mask()
    assert np.all(out.ros[:, burnable] == 0.3)
    assert np.all(out.ros[:, ~burnable] == 0.0)


def test_retardant_scales_ros(small_grid: Grid, calm_weather: Weather) -> None:
    grid = small_grid
    grid.retardant[5, 5] = 0.5
    grid.retardant[6, 6] = 1.0
    out = ConstantRule(ros=0.4).rate_of_spread(grid, calm_weather, STENCIL_16)
    np.testing.assert_allclose(out.ros[:, 5, 5], 0.2)
    assert np.all(out.ros[:, 6, 6] == 0.0)
    assert np.all(out.ros[:, 7, 7] == 0.4)


def test_ignores_weather(small_grid: Grid, calm_weather: Weather, windy_weather: Weather) -> None:
    rule = ConstantRule(ros=0.25)
    a = rule.rate_of_spread(small_grid, calm_weather, STENCIL_8)
    b = rule.rate_of_spread(small_grid, windy_weather, STENCIL_8)
    np.testing.assert_array_equal(a.ros, b.ros)


def test_grid_is_unchanged(small_grid: Grid, calm_weather: Weather) -> None:
    small_grid.retardant[2, 3] = 0.7
    small_grid.fireline[4, 5] = True
    before = small_grid.copy()
    out = ConstantRule(ros=0.2).rate_of_spread(small_grid, calm_weather, STENCIL_8)
    assert small_grid == before
    # the output must not alias grid layers either
    out.ros[...] = 9.0
    out.intensity[...] = 9.0
    assert small_grid == before


def test_zero_ros_is_allowed(small_grid: Grid, calm_weather: Weather) -> None:
    out = ConstantRule(ros=0.0).rate_of_spread(small_grid, calm_weather, STENCIL_8)
    out.validate(STENCIL_8, small_grid.shape)
    assert out.ros.max() == 0.0


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_rejects_negative_or_non_finite_ros(bad: float) -> None:
    with pytest.raises(ValueError, match="ros"):
        ConstantRule(ros=bad)


def test_make_spread_rule_in_fresh_interpreter() -> None:
    """The registry finds the rule by name without the caller importing its module."""
    code = (
        "import sys\n"
        "from wildfire_sim import spread\n"
        "assert 'wildfire_sim.spread.constant' not in sys.modules\n"
        "rule = spread.make_spread_rule('constant', ros=0.2)\n"
        "assert isinstance(rule, spread.SpreadRule)\n"
        "assert rule.name == 'constant'\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
