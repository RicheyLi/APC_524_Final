"""Tests for the elliptical fire shape and the ``"rothermel"`` spread rule (WP 2.1b).

Covers ``wildfire_sim.spread.ellipse`` (midflame wind, wind-slope vector combination, the
ellipse) and ``wildfire_sim.spread.rothermel.RothermelRule``. Simulation runs with the rule are in
``tests/test_physics.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from wildfire_sim.config import configs_dir
from wildfire_sim.data.synthetic import flat_uniform, slope_ramp
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.ellipse import (
    combine_wind_and_slope,
    directional_rates,
    eccentricity,
    length_to_breadth,
    twenty_foot_wind,
    wind_adjustment_factor,
)
from wildfire_sim.spread.rothermel import RothermelRule, load_rothermel_params
from wildfire_sim.types import CellState, LandUse, get_stencil

FT, MPH = 0.3048, 0.44704
RULE = RothermelRule()


def cell(fuel_model: int = 102, slope: float = 0.0, aspect: float = 0.0, canopy_cover: float = 0.0) -> Grid:
    """A 1 x 1 grid of one fuel, slope (rise/run), aspect (deg) and canopy cover (fraction)."""
    g = Grid.empty((1, 1), 30.0)
    g.fuel_model[:] = fuel_model
    g.slope[:] = slope
    g.aspect[:] = aspect
    g.canopy_cover[:] = canopy_cover
    return g


# ----------------------------------------------------------------------------- midflame wind


def test_a_10m_wind_of_11_5_is_a_20ft_wind_of_10() -> None:
    """WP 2.1b acceptance: 11.5 m/s at 10 m gives the midflame wind of 10 m/s at 20 ft."""
    factor = load_rothermel_params().ten_m_to_twenty_ft.value
    assert factor == 1.15
    assert float(twenty_foot_wind(11.5, factor)) == pytest.approx(10.0)
    g = cell()
    at_10m = RULE.fire(g, Weather(wind_speed=11.5)).midflame_wind[0, 0]
    waf = wind_adjustment_factor(1.0 * FT, 0.0, 15.0, 0.4)  # GR2 is 1 ft deep, no canopy
    assert at_10m == pytest.approx(10.0 * float(waf))


def test_unsheltered_waf_is_albini_and_baughman() -> None:
    """``WAF = 1.83 / ln((20 + 0.36 H) / (0.13 H))``, H in ft: 0.362 for a 1 ft fuel bed."""
    assert float(wind_adjustment_factor(FT, 0.0, 15.0, 0.4)) == pytest.approx(0.36210, abs=1e-5)
    deeper = wind_adjustment_factor(6 * FT, 0.0, 15.0, 0.4)
    assert float(deeper) > 0.36210  # a deep fuel bed feels more of the wind


def test_sheltered_waf_needs_crown_fill_above_five_percent() -> None:
    """Sheltered (Andrews 2012) only where ``cc * cr / 3 > 0.05``; then it depends on canopy height only."""
    open_waf = float(wind_adjustment_factor(FT, 0.0, 18.288, 0.5))
    assert float(wind_adjustment_factor(FT, 0.3, 18.288, 0.5)) == open_waf  # f = 0.05, not above
    sheltered = float(wind_adjustment_factor(FT, 0.5, 18.288, 0.5))  # f = 0.083, H_c = 60 ft
    assert sheltered == pytest.approx(
        0.555 / (math.sqrt(0.5 * 0.5 / 3 * 60) * math.log((20 + 0.36 * 60) / (0.13 * 60)))
    )
    assert sheltered == pytest.approx(0.14827, abs=1e-5)  # BehavePlus engine, same inputs
    assert float(wind_adjustment_factor(6 * FT, 0.5, 18.288, 0.5)) == sheltered


def test_waf_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="fuel_depth_m"):
        wind_adjustment_factor(0.0, 0.0, 15.0, 0.4)
    with pytest.raises(ValueError, match="canopy_height_m"):
        wind_adjustment_factor(FT, 0.9, 0.0, 0.9)
    with pytest.raises(ValueError, match="factor"):
        twenty_foot_wind(5.0, 0.0)


# ----------------------------------------------------------------------------- wind and slope


def test_upslope_with_no_wind_spreads_uphill() -> None:
    """WP 2.1b acceptance: on a slope facing south (aspect 180) with no wind, the head runs north."""
    g = slope_ramp((5, 5), 30.0, fuel_model=102, slope=0.4, aspect_deg=180.0)
    stencil = get_stencil(16)
    out = RULE.rate_of_spread(g, Weather(), stencil)
    fastest = int(np.argmax(out.ros[:, 2, 2]))
    assert stencil.bearings_deg[fastest] == 0.0
    assert RULE.fire(g, Weather()).head_bearing[2, 2] == pytest.approx(0.0)


@pytest.mark.parametrize("wind_speed", [1.0, 3.0, 8.0])
def test_cross_slope_wind_heads_between_wind_and_slope(wind_speed: float) -> None:
    """WP 2.1b acceptance: wind blowing east across a north-facing upslope gives a head between 0 and 90 deg.

    And the stronger the wind, the closer the head is to it.
    """
    g = cell(slope=math.tan(math.radians(20.0)), aspect=180.0)
    head = RULE.fire(g, Weather(wind_speed=wind_speed, wind_direction=270.0)).head_bearing[0, 0]
    assert 0.0 < head < 90.0
    stronger = RULE.fire(g, Weather(wind_speed=2 * wind_speed, wind_direction=270.0)).head_bearing[0, 0]
    assert head < stronger < 90.0


def test_vector_sum_reduces_to_addition_when_aligned() -> None:
    phi, head = combine_wind_and_slope(2.0, 45.0, 1.5, 45.0)
    assert float(phi) == pytest.approx(3.5) and float(head) == pytest.approx(45.0)
    phi, head = combine_wind_and_slope(2.0, 90.0, 2.0, 270.0)
    assert float(phi) == pytest.approx(0.0, abs=1e-12) and float(head) == 0.0
    phi, head = combine_wind_and_slope(3.0, 90.0, 4.0, 0.0)
    assert float(phi) == pytest.approx(5.0) and float(head) == pytest.approx(math.degrees(math.atan2(3, 4)))


# ----------------------------------------------------------------------------- ellipse


def test_length_to_breadth_is_andersons_curve_capped_at_8() -> None:
    """``LB(0) = 1``; Anderson (1983) with mph coefficients 0.1147 / 0.0692; at most 8 (BehavePlus)."""
    assert float(length_to_breadth(0.0)) == pytest.approx(1.0)
    for mph, lb in ((1.0, 1.0829), (5.0, 1.5901), (10.0, 2.7810)):  # BehavePlus engine
        assert float(length_to_breadth(mph * MPH)) == pytest.approx(lb, abs=1e-4)
    assert float(length_to_breadth(20 * MPH)) == 8.0
    assert float(length_to_breadth(20 * MPH, lb_max=3.0)) == 3.0
    with pytest.raises(ValueError):
        length_to_breadth(1.0, lb_max=0.5)
    with pytest.raises(ValueError):
        eccentricity(0.9)


def test_back_rate_is_head_times_1_minus_e_over_1_plus_e() -> None:
    """WP 2.1b acceptance: the rate opposite the head is ``R_head (1 - e) / (1 + e)``."""
    stencil = get_stencil(16)
    bearings = [float(b) for b in stencil.bearings_deg]
    for bearing in (0.0, 45.0, 90.0, math.degrees(math.atan2(1, 2))):  # axis, diagonal, knight move
        k_head = int(np.argmin(np.abs(((stencil.bearings_deg - bearing + 180) % 360) - 180)))
        k_back = int(np.argmin(np.abs(((stencil.bearings_deg - bearing) % 360) - 180)))
        assert bearings[k_head] == pytest.approx(bearing)
        assert bearings[k_back] == pytest.approx((bearing + 180) % 360)
        for lb in (1.0, 1.5, 3.0, 8.0):
            rates = directional_rates(np.array(0.3), bearing, lb, stencil.bearings_deg)
            e = float(eccentricity(lb))
            assert float(rates[k_head]) == pytest.approx(0.3)
            assert float(rates[k_back]) == pytest.approx(0.3 * (1 - e) / (1 + e))
    # a rule cell: the back of a wind-driven fire on flat ground
    g = flat_uniform((3, 3), 30.0, fuel_model=145)
    out = RULE.rate_of_spread(g, Weather(wind_speed=6.0, wind_direction=270.0), stencil)  # head east
    fire = RULE.fire(g, Weather(wind_speed=6.0, wind_direction=270.0))
    e = float(eccentricity(fire.length_to_breadth[1, 1]))
    east, west = list(stencil.bearings_deg).index(90.0), list(stencil.bearings_deg).index(270.0)
    assert out.ros[east, 1, 1] == pytest.approx(fire.ros_head[1, 1])
    assert out.ros[west, 1, 1] == pytest.approx(fire.ros_head[1, 1] * (1 - e) / (1 + e))


def test_no_wind_no_slope_spreads_equally_everywhere() -> None:
    g = flat_uniform((3, 3), 30.0, fuel_model=183)
    out = RULE.rate_of_spread(g, Weather(), get_stencil(16))
    assert np.allclose(out.ros[:, 1, 1], out.ros[0, 1, 1]) and out.ros[0, 1, 1] > 0


# ----------------------------------------------------------------------------- the rule on a grid


def test_registered_as_rothermel_and_matches_the_contract() -> None:
    rule = make_spread_rule("rothermel")
    assert isinstance(rule, RothermelRule) and rule.name == "rothermel"
    g = slope_ramp((6, 7), 30.0, fuel_model=145, slope=0.3, aspect_deg=90.0)
    stencil = get_stencil(16)
    out = rule.rate_of_spread(g, Weather(wind_speed=4.0, wind_direction=200.0), stencil)
    out.validate(stencil, g.shape)
    assert out.heat_per_area is not None and np.all(out.heat_per_area > 0)
    assert np.all(out.intensity > 0) and np.all(out.flame_length > 0)
    head = rule.fire(g, Weather(wind_speed=4.0, wind_direction=200.0)).ros_head
    np.testing.assert_allclose(out.intensity, out.heat_per_area * head, rtol=1e-12)
    assert np.all(out.ros.max(axis=0) <= head * (1 + 1e-12))  # no stencil direction beats the head


def test_cells_that_cannot_burn_do_not_spread() -> None:
    g = flat_uniform((3, 4), 30.0, fuel_model=102)
    g.fuel_model[0, 0] = 0  # no fuel model
    g.fuel_model[0, 1] = 98  # non-burnable code
    g.land_use[0, 2] = int(LandUse.WATER)
    out = RULE.rate_of_spread(g, Weather(wind_speed=5.0), get_stencil(8))
    assert np.all(out.ros[:, 0, :3] == 0.0) and np.all(out.intensity[0, :3] == 0.0)
    assert np.all(out.ros[:, 1:, :] > 0.0)


def test_retardant_scales_rates_and_intensity() -> None:
    g = flat_uniform((1, 3), 30.0, fuel_model=102)
    g.retardant[0, :] = [0.0, 0.5, 1.0]
    out = RULE.rate_of_spread(g, Weather(wind_speed=3.0), get_stencil(8))
    np.testing.assert_allclose(out.ros[:, 0, 1], 0.5 * out.ros[:, 0, 0])
    assert np.all(out.ros[:, 0, 2] == 0.0)
    assert out.intensity[0, 1] == pytest.approx(0.5 * out.intensity[0, 0])
    assert out.heat_per_area is not None and out.heat_per_area[0, 2] == out.heat_per_area[0, 0]


def test_output_does_not_depend_on_fire_state() -> None:
    """The simulator caches rule output; it must not read state, timers, intensity or arrival times."""
    g = slope_ramp((4, 4), 30.0, fuel_model=161, slope=0.2)
    wx, stencil = Weather(wind_speed=4.0, wind_direction=90.0), get_stencil(16)
    before = RULE.rate_of_spread(g, wx, stencil)
    g.state[:] = int(CellState.BURNING)
    g.burn_timer[:] = 50.0
    g.intensity[:] = 1000.0
    g.arrival_time[:] = 10.0
    after = RULE.rate_of_spread(g, wx, stencil)
    np.testing.assert_array_equal(before.ros, after.ros)


def test_grid_cells_match_single_cells() -> None:
    """Mixed fuels, slopes, aspects and canopy on one grid: each cell gets its own answer."""
    rng = np.random.default_rng(1)
    g = Grid.empty((5, 6), 30.0)
    g.fuel_model[:] = rng.choice([1, 102, 145, 161, 183], size=g.shape)
    g.slope[:] = rng.uniform(0, 0.8, size=g.shape)
    g.aspect[:] = rng.uniform(0, 359, size=g.shape)
    g.canopy_cover[:] = rng.uniform(0, 0.9, size=g.shape)
    wx, stencil = Weather(wind_speed=7.0, wind_direction=30.0), get_stencil(16)
    whole = RULE.rate_of_spread(g, wx, stencil)
    for r, c in np.ndindex(g.shape):
        one = RULE.rate_of_spread(
            cell(int(g.fuel_model[r, c]), float(g.slope[r, c]), float(g.aspect[r, c]), float(g.canopy_cover[r, c])),
            wx,
            stencil,
        )
        np.testing.assert_allclose(whole.ros[:, r, c], one.ros[:, 0, 0], rtol=1e-12)


def test_wind_limit_is_an_option_off_by_default() -> None:
    """Open decision (docs/plan.md section 13): off in ``configs/rothermel.yaml``; the argument overrides it."""
    assert load_rothermel_params().apply_wind_limit is False
    g = cell(fuel_model=101)  # GR1: low reaction intensity, so the limit binds in a strong wind
    wx = Weather(wind_speed=15.0, moisture_1h=0.09, moisture_10h=0.10, moisture_100h=0.11)
    free = RULE.fire(g, wx)
    capped = RothermelRule(apply_wind_limit=True).fire(g, wx)
    assert not free.wind_limited[0, 0] and capped.wind_limited[0, 0]
    assert capped.ros_head[0, 0] < free.ros_head[0, 0]
    assert capped.length_to_breadth[0, 0] < free.length_to_breadth[0, 0]


def test_parameters_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canopy_height"):
        RothermelRule(canopy_height=0.0)
    with pytest.raises(ValueError, match="crown_ratio"):
        RothermelRule(crown_ratio=1.5)
    text = (configs_dir() / "rothermel.yaml").read_text(encoding="utf-8").replace("value: 8.0", "value: 0.5")
    bad = tmp_path / "rothermel.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="lb_max"):
        RothermelRule(params_path=bad)


# ----------------------------------------------------------------------------- BehavePlus engine

BEHAVE_CASES = [
    # (fuel model, 10 m wind m/s, wind from deg, slope deg, aspect deg, canopy cover)
    (102, 5.0, 270.0, 0.0, 0.0, 0.0),
    (102, 8.0, 270.0, 20.0, 180.0, 0.0),
    (145, 3.0, 0.0, 30.0, 90.0, 0.0),
    (183, 6.0, 45.0, 15.0, 300.0, 0.6),
    (161, 12.0, 200.0, 10.0, 0.0, 0.3),
    (122, 0.0, 0.0, 25.0, 45.0, 0.0),
    (101, 15.0, 90.0, 5.0, 270.0, 0.0),
]


@pytest.mark.parametrize(("fm", "u10", "wind_from", "slope_deg", "aspect", "cc"), BEHAVE_CASES)
def test_agrees_with_the_behaveplus_engine(
    fm: int, u10: float, wind_from: float, slope_deg: float, aspect: float, cc: float
) -> None:
    """Head rate, head direction and the rate in eight directions against BehavePlus (via ``pyrothermel``).

    10 m wind input with BehavePlus's own 10 m and wind adjustment conversions, wind and slope in
    any directions, sheltered and open fuel, wind limit on as in BehavePlus. Measured agreement
    better than 1e-7; the tolerance is the plan's 2 %.
    """
    pyro = pytest.importorskip("pyrothermel")
    ch_h = 20.1168 / 3600.0
    rule = RothermelRule(apply_wind_limit=True)
    g = cell(fm, math.tan(math.radians(slope_deg)), aspect, cc)
    fire = rule.fire(g, Weather(wind_speed=u10, wind_direction=wind_from))
    run = pyro.PyrothermelRun(
        pyro.FuelModel.from_existing(fm, "us_standard"),
        pyro.MoistureScenario(0.06, 0.07, 0.08, 0.60, 0.90),
        u10 / MPH,
        units_preset="us_standard",
        wind_input_mode="ten_meter",
        wind_direction=wind_from,
        slope=slope_deg,
        aspect=aspect,
        canopy_cover=cc,
        canopy_height=rule.canopy_height / FT,
        canopy_ratio=rule.crown_ratio,
    )
    head = run.run_surface_fire_in_direction_of_max_spread()
    assert fire.ros_head[0, 0] == pytest.approx(head["spread_rate"] * ch_h, rel=0.02)
    assert ((fire.head_bearing[0, 0] - head["direction"] + 180) % 360) - 180 == pytest.approx(0.0, abs=0.1)
    bearings = np.arange(0.0, 360.0, 45.0)
    mine = directional_rates(fire.ros_head[0, 0], fire.head_bearing[0, 0], fire.length_to_breadth[0, 0], bearings)
    for b, r in zip(bearings, mine, strict=True):
        ref = run.run_surface_fire_in_direction_of_interest(float(b))["spread_rate"] * ch_h
        assert float(r) == pytest.approx(ref, rel=0.02), b
