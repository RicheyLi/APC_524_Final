"""Tests for front-normal fireline intensity (WP 1.12, docs/plan.md section 2.2).

A spread rule reports the intensity of a head fire. Byram's intensity is heat per unit area
times the rate at which the front advances normal to itself, so on the analytic ellipse the
flank burns at ``sqrt((1 - e) / (1 + e))`` of the head intensity and the back at
``(1 - e) / (1 + e)``. The simulator estimates the front-normal rate of each cell at ignition
from a least-squares plane through the arrival times around it.

Three kinds of test:

* the estimator on arrival-time fields with a known answer (a plane, a cone);
* the engine in the noise-free limit against the *lattice's own* front. With constant rates the
  burn is the polygon spanned by the stencil directions scaled by their rates
  (docs/lattice_study.md), and each of its sides advances at a known normal speed, which the
  estimator must reproduce to 1 %;
* the engine with the default kernel against the *analytic ellipse*, which is what the model
  is meant to represent. Here the error is the lattice's shape error, not the estimator's, and
  the bounds are set from the measurement (see :data:`ELLIPSE_BOUNDS`).

Set-up shared by the fire tests: ``R_head = 0.2`` m/s, ``I_head = 1000`` kW/m, so a rule that
gives heat per unit area gives ``I_head / R_head = 5000`` kJ/m^2. The grid is sized from the
analytic ellipse so that the back, the slowest part, travels at least 8 cells.
"""

from __future__ import annotations

import functools
import importlib.util
import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from wildfire_sim.config import ScenarioConfig
from wildfire_sim.ensemble import ScaledRateRule
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import (
    FlameLengthModel,
    RunResult,
    flame_length_from_intensity,
    front_normal_rate,
    load_flame_length_model,
    run,
)
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.types import STENCIL_8, STENCIL_16, CellState, FloatArray, Stencil, get_stencil

DX = 30.0  # m
ROS = 0.2  # head rate of spread, m/s
I_HEAD = 1000.0  # head fireline intensity, kW/m
HEAT = I_HEAD / ROS  # kJ/m^2
DETERMINISTIC = 1e6  # arrival_shape with thresholds of 1 within 0.1 %; its kernel correction is 1
BACK_CELLS = 8  # the back of every test fire travels at least this far
MIN_HEAD_CELLS = 60

_BURNING = int(CellState.BURNING)


@pytest.fixture(params=["numpy", "numba"])
def engine(request: pytest.FixtureRequest) -> str:
    if request.param == "numba" and importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    return request.param


class EllipticalRule:
    """``R(theta) = R_head (1 - e) / (1 - e cos theta)`` with a uniform head intensity ``I_HEAD``.

    ``theta`` is the angle between a stencil direction and the head bearing (the rule of
    ``tests/test_physics.py``). ``heat`` selects whether the rule also gives heat per unit area.
    """

    name = "elliptical"

    def __init__(self, lb: float, head_bearing_deg: float, *, heat: bool = True, intensity: float = I_HEAD) -> None:
        self.lb, self.head, self.heat, self.intensity = lb, head_bearing_deg, heat, intensity

    def rates(self, stencil: Stencil) -> FloatArray:
        """``(K,)`` rate per stencil direction, m/s."""
        e = eccentricity(self.lb)
        theta = np.radians(np.asarray(stencil.bearings_deg, dtype=np.float64) - self.head)
        return ROS * (1.0 - e) / (1.0 - e * np.cos(theta))

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        h, w = grid.shape
        mask = grid.burnable_mask().astype(np.float64)
        return SpreadOutput(
            ros=self.rates(stencil)[:, None, None] * mask[None],
            intensity=np.full((h, w), self.intensity),
            flame_length=np.zeros((h, w)),
            heat_per_area=np.full((h, w), self.intensity / ROS) if self.heat else None,
        )


def eccentricity(lb: float) -> float:
    return math.sqrt(lb * lb - 1.0) / lb


def back_ratio(lb: float) -> float:
    """Back rate over head rate, ``(1 - e) / (1 + e)``; the flank ratio is its square root."""
    e = eccentricity(lb)
    return (1.0 - e) / (1.0 + e)


@functools.cache
def fire(
    lb: float, head_deg: float, *, arrival_shape: float = 200.0, stencil: int = 16, seed: int = 0, heat: bool = True
) -> tuple[RunResult, tuple[int, int]]:
    """Run an elliptical fire on a grid sized to hold it; returns the result and the ignition cell."""
    ratio = back_ratio(lb)
    head_cells = max(MIN_HEAD_CELLS, BACK_CELLS / ratio)
    a, b, c = head_cells * (1 + ratio) / 2, head_cells * math.sqrt(ratio), head_cells * (1 - ratio) / 2
    phi = math.radians(head_deg)
    east, north = math.sin(phi), math.cos(phi)
    half_e, half_n = math.hypot(a * east, b * north), math.hypot(a * north, b * east)
    margin = 6
    east_lo, east_hi = c * east - half_e - margin, c * east + half_e + margin
    north_lo, north_hi = c * north - half_n - margin, c * north + half_n + margin
    shape = (math.ceil(north_hi - north_lo) + 1, math.ceil(east_hi - east_lo) + 1)
    ignition = (round(north_hi), round(-east_lo))
    dt = 120.0
    cfg = ScenarioConfig.model_validate(
        {
            "name": "ellipse",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": list(shape), "dx": DX},
            "spread": {"rule": "constant", "params": {"ros": ROS}, "arrival_shape": arrival_shape},
            "ignition": {"cells": [list(ignition)]},
            "dt_s": dt,
            "duration_s": math.floor(head_cells * DX / ROS / dt) * dt,
            "stencil": stencil,
        }
    )
    result = run(cfg, seed=seed, rule=EllipticalRule(lb, head_deg, heat=heat), record_events=False)
    assert result.provenance["reached_edge_s"] is None, "the test grid is too small"
    return result, ignition


def ray_mean(
    result: RunResult, ignition: tuple[int, int], bearing_deg: float, *, min_cells: float, half_width: float = 1.0
) -> float:
    """Mean ``front_intensity / I_HEAD`` over burned cells along a ray from the ignition.

    Cells whose centre lies within ``half_width`` cells of the ray and at least ``min_cells``
    cells from the ignition. The arrival-time field of a fire with constant rates scales with
    distance from the ignition, so the front crosses every cell of a ray at the same angle.
    """
    rows, cols = np.nonzero(result.burned_mask)
    east, north = (cols - ignition[1]).astype(np.float64), -(rows - ignition[0]).astype(np.float64)
    phi = math.radians(bearing_deg)
    along = east * math.sin(phi) + north * math.cos(phi)
    across = east * math.cos(phi) - north * math.sin(phi)
    chosen = (np.abs(across) <= half_width) & (along >= min_cells)
    assert chosen.sum() >= 5, f"only {chosen.sum()} cells on the ray at {bearing_deg} deg"
    return float(result.front_intensity[rows[chosen], cols[chosen]].mean() / I_HEAD)


def flank_bearings(lb: float, head_deg: float) -> tuple[float, float]:
    """Bearings from the ignition (the rear focus) to the two widest points of the ellipse, degrees."""
    ratio = back_ratio(lb)
    offset = math.degrees(math.atan2(math.sqrt(ratio), (1 - ratio) / 2))
    return head_deg + offset, head_deg - offset


# ----------------------------------------------------------------------------- the estimator


@pytest.mark.parametrize("stencil", [STENCIL_8, STENCIL_16])
@pytest.mark.parametrize("bearing", [0.0, 22.5, 45.0, 77.0, 135.0, 200.0, 291.0])
def test_plane_wave_gives_its_speed_exactly(stencil: Stencil, bearing: float) -> None:
    """A straight front moving at 0.37 m/s on any bearing: every interior cell reads 0.37 m/s."""
    speed = 0.37
    rows, cols = np.mgrid[0:25, 0:31]
    phi = math.radians(bearing)
    distance = (cols * math.sin(phi) - rows * math.cos(phi)) * DX  # m along the bearing (north is -row)
    arrival = (distance - distance.min()) / speed
    rate = front_normal_rate(arrival, stencil, DX)
    np.testing.assert_allclose(rate[3:-3, 3:-3], speed, rtol=1e-9)
    assert rate.shape == arrival.shape


@pytest.mark.parametrize("stencil", [STENCIL_8, STENCIL_16])
def test_round_front_is_within_four_percent_ten_cells_out(stencil: Stencil) -> None:
    """A circular front has curvature the one-sided plane cannot follow; the error falls as 1 / r.

    Measured on the exact cone ``T = r / v``, per cell, from 10 cells out: -1.5 % to +3.8 % with
    8 neighbours and -2.9 % to +2.1 % with 16 (half that from 20 cells out). The mean over all
    cells is within 0.15 % of ``v`` for both. Tolerances: 4 % per cell, 0.5 % on the mean.
    """
    speed = 0.2
    rows, cols = np.mgrid[0:81, 0:81]
    radius = np.hypot(rows - 40, cols - 40) * DX
    rate = front_normal_rate(radius / speed, stencil, DX)
    far = (radius >= 10 * DX) & (radius <= 38 * DX)
    np.testing.assert_allclose(rate[far], speed, rtol=0.04)
    assert rate[far].mean() == pytest.approx(speed, rel=0.005)
    assert np.isnan(rate[40, 40])  # the origin has no earlier neighbour


def test_estimator_is_undetermined_without_two_non_collinear_neighbours() -> None:
    arrival = np.full((7, 7), np.nan)
    assert np.isnan(front_normal_rate(arrival, STENCIL_8, DX)).all()
    arrival[3, 3] = 0.0
    arrival[3, 4] = 100.0  # one earlier neighbour
    arrival[3, 5] = 200.0  # two, collinear with the cell (16 neighbours do not reach (3, 3) from here)
    for stencil in (STENCIL_8, STENCIL_16):
        assert np.isnan(front_normal_rate(arrival, stencil, DX)).all()
    arrival[2, 4] = 150.0  # (3, 5) now has (3, 4) and (2, 4), and (2, 4) has (3, 3) and (3, 4): not collinear
    rate = front_normal_rate(arrival, STENCIL_8, DX)
    assert np.isfinite(rate[3, 5]) and rate[3, 5] > 0
    assert np.isfinite(rate[2, 4]) and rate[2, 4] > 0
    assert np.isnan(rate[3, 3]) and np.isnan(rate[3, 4]) and np.isnan(rate[0, 0])


def test_zero_gradient_gives_an_infinite_rate() -> None:
    """Neighbours that burned at the same time as the cell: the front arrived everywhere at once."""
    arrival = np.full((5, 5), np.nan)
    arrival[1:4, 1:4] = 50.0
    assert np.isinf(front_normal_rate(arrival, STENCIL_8, DX)[2, 2])


def test_only_neighbours_that_burned_no_later_are_used() -> None:
    """Later arrivals around a cell must not change its estimate."""
    rows, cols = np.mgrid[0:15, 0:15]
    arrival = cols * DX / 0.25
    base = front_normal_rate(arrival, STENCIL_16, DX)
    changed = arrival.copy()
    changed[:, 9:] += 5000.0  # everything east of column 8 burns much later
    rate = front_normal_rate(changed, STENCIL_16, DX)
    np.testing.assert_array_equal(rate[:, :9], base[:, :9])
    assert rows.shape == rate.shape


# ----------------------------------------------------------------------------- noise-free limit: the lattice polygon


def polygon_sides(rates: FloatArray, stencil: Stencil) -> list[tuple[float, float, float]]:
    """Sides of the lattice burn polygon: ``(mid bearing deg, angular width deg, normal speed m/s)``.

    The polygon is the convex hull of the stencil directions scaled by their rates. A side
    between two hull vertices moves outward at its distance from the origin per unit time.
    """
    bearings = np.radians(np.asarray(stencil.bearings_deg, dtype=np.float64))
    points = np.column_stack([rates * np.sin(bearings), rates * np.cos(bearings)])  # east, north
    hull = ConvexHull(points)
    simplices = np.asarray(hull.simplices, dtype=np.int64)  # (n_sides, 2) vertex indices
    offsets = np.asarray(hull.equations, dtype=np.float64)[:, 2]  # normal . x + offset = 0 on a side
    sides = []
    for (i, j), offset in zip(simplices.tolist(), offsets.tolist(), strict=True):
        b0 = math.degrees(math.atan2(points[i, 0], points[i, 1]))
        b1 = math.degrees(math.atan2(points[j, 0], points[j, 1]))
        width = (b1 - b0 + 180.0) % 360.0 - 180.0
        sides.append(((b0 + width / 2) % 360.0, abs(width), -offset))
    return sides


@pytest.mark.parametrize("head_deg", [90.0, 22.5])
@pytest.mark.parametrize("lb", [1.5, 3.0])
def test_noise_free_intensity_is_the_normal_speed_of_each_polygon_side(lb: float, head_deg: float) -> None:
    """Without kernel noise every side of the lattice polygon reads ``heat * its normal speed``.

    For each side at least 10 degrees wide, the mean intensity over cells along the ray through
    its middle, from 15 cells out, against the exact value. Tolerance 1 %: thresholds differ
    from 1 by 0.1 % at this shape and float32 storage adds less.
    """
    stencil = get_stencil(16)
    result, ignition = fire(lb, head_deg, arrival_shape=DETERMINISTIC)
    sides = [s for s in polygon_sides(EllipticalRule(lb, head_deg).rates(stencil), stencil) if s[1] >= 10.0]
    assert len(sides) >= 6
    rows, cols = np.nonzero(result.burned_mask)
    east, north = cols - ignition[1], -(rows - ignition[0])
    checked = 0
    for bearing, _, normal_speed in sides:
        phi = math.radians(bearing)
        along = east * math.sin(phi) + north * math.cos(phi)
        across = east * math.cos(phi) - north * math.sin(phi)
        if np.count_nonzero((np.abs(across) <= 0.75) & (along >= 15.0)) < 10:
            continue  # the fire did not get far enough this way to have cells clear of the origin
        measured = ray_mean(result, ignition, bearing, min_cells=15.0, half_width=0.75)
        assert measured == pytest.approx(normal_speed / ROS, rel=0.01), (bearing, measured, normal_speed / ROS)
        checked += 1
    assert checked >= 2


# ----------------------------------------------------------------------------- default kernel: the analytic ellipse

# Bounds on (mean simulated intensity) / (analytic intensity) for head, flank and back, set
# from the measurement: default kernel (16 neighbours, shape 200), seeds 0-4, heat per unit
# area given. Measured ranges over the five seeds:
#
#   LB   head bearing   head          flank         back
#   1    any            0.98 - 0.99   0.98 - 0.99   0.98 - 0.99
#   1.5  90 (axis)      0.96 - 0.98   0.97 - 1.08   1.00 - 1.06
#   1.5  22.5           0.88 - 0.98   1.02 - 1.17   1.00 - 1.09
#   3    90 (axis)      0.77 - 0.98   1.27 - 1.37   1.05 - 1.08
#   3    22.5           0.38 - 0.41   1.41 - 1.45   1.04 - 1.11
#
# The estimator is not what is off (it is exact on the lattice polygon, see the test above).
# The lattice burns a polygon inside the ellipse: its head is a corner on a stencil direction
# with slow sides either way, and its flank is a side that is too fast. At LB 1.5 the polygon is
# close to the ellipse and every part is within 20 %. At LB 3 the flank is 27-45 % too hot and,
# when the head lies between two stencil directions (22.5 degrees), the cells on the head
# bearing sit on a slow side and read 0.4 of the head intensity. That is the lattice limit of
# docs/lattice_study.md, to be removed by the lattice-fitted rates and length-to-breadth cap of
# WP 2.1d; test_plan_target_at_lb_3 records it as a strict expected failure. The bounds here
# are regression guards with 0.03 of slack on the measured range.
ELLIPSE_BOUNDS: dict[tuple[float, float], dict[str, tuple[float, float]]] = {
    (1.5, 90.0): {"head": (0.93, 1.0), "flank": (0.94, 1.11), "back": (0.97, 1.09)},
    (1.5, 22.5): {"head": (0.85, 1.0), "flank": (0.99, 1.20), "back": (0.97, 1.12)},
    (3.0, 90.0): {"head": (0.74, 1.0), "flank": (1.24, 1.40), "back": (1.02, 1.11)},
    (3.0, 22.5): {"head": (0.35, 0.44), "flank": (1.38, 1.48), "back": (1.01, 1.14)},
}
PLAN_TOLERANCE = 0.20  # what LB 1.5 meets on every part, and the target LB 3 misses


def ellipse_ratios(lb: float, head_deg: float, *, heat: bool = True, seed: int = 0) -> dict[str, float]:
    """Simulated over analytic intensity for the head, the flank (mean of both sides) and the back."""
    result, ignition = fire(lb, head_deg, heat=heat, seed=seed)
    left, right = flank_bearings(lb, head_deg)
    ratio = back_ratio(lb)
    flank = (ray_mean(result, ignition, left, min_cells=10) + ray_mean(result, ignition, right, min_cells=10)) / 2
    return {
        "head": ray_mean(result, ignition, head_deg, min_cells=10),
        "flank": flank / math.sqrt(ratio),
        "back": ray_mean(result, ignition, head_deg + 180.0, min_cells=3) / ratio,
    }


@pytest.mark.parametrize(("lb", "head_deg"), list(ELLIPSE_BOUNDS))
def test_head_flank_and_back_intensity_against_the_ellipse(lb: float, head_deg: float) -> None:
    """WP 1.12 acceptance: mean intensity on the head, flank and back cells of an elliptical fire.

    Against ``I_head``, ``I_head sqrt((1 - e) / (1 + e))`` and ``I_head (1 - e) / (1 + e)``, within
    the measured bounds of :data:`ELLIPSE_BOUNDS` (seed 0; the comment there has the evidence
    and explains the LB 3 values). Head cells are the cells within one cell of the head ray, at
    least 10 cells out; flank cells the same on the rays to the two widest points of the
    ellipse; back cells on the opposite ray, at least 3 cells out (the back travels 8 cells).
    """
    ratios = ellipse_ratios(lb, head_deg)
    for part, (low, high) in ELLIPSE_BOUNDS[(lb, head_deg)].items():
        assert low <= ratios[part] <= high, (part, ratios[part])
    if lb == 1.5:
        for part, value in ratios.items():
            assert value == pytest.approx(1.0, abs=PLAN_TOLERANCE), part


@pytest.mark.parametrize("head_deg", [90.0, 22.5])
@pytest.mark.xfail(strict=True, reason="lattice limit at LB 3 (docs/lattice_study.md); WP 2.1d fits the rates")
def test_plan_target_at_lb_3(head_deg: float) -> None:
    """Head, flank and back within 20 % of the analytic ellipse at LB 3: not met on this lattice."""
    for part, value in ellipse_ratios(3.0, head_deg).items():
        assert value == pytest.approx(1.0, abs=PLAN_TOLERANCE), part


@pytest.mark.parametrize("head_deg", [90.0, 22.5])
def test_a_round_fire_has_the_head_intensity_everywhere(head_deg: float) -> None:
    """LB 1: every direction is a head. Mean within 3 % of ``I_head`` on any ray and over the whole fire.

    Measured: 0.98 to 0.99 on every ray (the cap at ``I_head`` trims the upward half of the
    noise, and the round front's curvature costs about 1 %); 99 % of cells beyond 10 cells
    from the ignition are within 10 %.
    """
    result, ignition = fire(1.0, head_deg)
    for bearing in (head_deg, head_deg + 90.0, head_deg + 180.0, head_deg + 270.0, head_deg + 33.0):
        assert ray_mean(result, ignition, bearing, min_cells=10) == pytest.approx(1.0, abs=0.03)
    rows, cols = np.nonzero(result.burned_mask)
    far = np.hypot(rows - ignition[0], cols - ignition[1]) >= 10
    values = result.front_intensity[rows[far], cols[far]] / I_HEAD
    assert values.mean() == pytest.approx(1.0, abs=0.03)
    assert np.mean(np.abs(values - 1.0) <= 0.10) >= 0.99
    assert values.max() <= 1.0


@pytest.mark.parametrize(("lb", "head_deg"), [(1.5, 90.0), (1.5, 22.5), (3.0, 90.0)])
def test_without_heat_per_area_the_head_intensity_is_scaled(lb: float, head_deg: float) -> None:
    """``heat_per_area`` None: ``I = I_head * R_n / max_k ros``, capped at ``I_head``.

    Same cells, so the two forms differ only by the factor ``R_head / max_k ros``, which is 1
    when the head lies on a stencil direction and a little above 1 otherwise.
    """
    stencil = get_stencil(16)
    with_heat, _ = fire(lb, head_deg, heat=True)
    scaled, _ = fire(lb, head_deg, heat=False)
    np.testing.assert_array_equal(with_heat.arrival_time, scaled.arrival_time)
    factor = ROS / EllipticalRule(lb, head_deg).rates(stencil).max()
    assert factor >= 1.0
    burned = with_heat.burned_mask
    expected = np.minimum(with_heat.front_intensity[burned] * factor, I_HEAD)
    capped = with_heat.front_intensity[burned] >= I_HEAD
    np.testing.assert_allclose(scaled.front_intensity[burned][~capped], expected[~capped], rtol=1e-9)
    assert np.all(scaled.front_intensity[burned][capped] == I_HEAD)


# ----------------------------------------------------------------------------- engine behaviour


def small_config(engine: str = "numpy", **overrides: object) -> ScenarioConfig:
    data: dict[str, object] = {
        "name": "small",
        "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [41, 61], "dx": DX},
        "spread": {"rule": "constant", "params": {"ros": ROS}},
        "ignition": {"cells": [[20, 12]]},
        "dt_s": 30.0,
        "duration_s": 3600.0,
        "engine": engine,
    }
    data.update(overrides)
    return ScenarioConfig.model_validate(data)


def test_front_intensity_bookkeeping(engine: str) -> None:
    """NaN where unburned, capped at the head value, head value at the ignition, shown while flaming."""
    result = run(small_config(engine), rule=EllipticalRule(1.5, 90.0), seed=1)
    front = result.front_intensity
    assert front.shape == result.grid.shape and front.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(front), ~result.burned_mask)
    assert front[20, 12] == I_HEAD
    assert np.nanmax(front) <= I_HEAD and np.nanmin(front) > 0
    assert np.nanmin(front) < 0.3 * I_HEAD  # the back

    burning = result.grid.state == _BURNING
    flaming = burning & (result.grid.burn_timer > 0)
    assert flaming.sum() > 10
    np.testing.assert_allclose(result.grid.intensity[flaming], front[flaming], rtol=1e-6)
    assert not result.grid.intensity[~flaming].any()


def test_intensity_does_not_change_arrival_times(engine: str) -> None:
    """Rules that differ only in intensity and heat give bit-identical fires."""
    cfg = small_config(engine)
    hot = run(cfg, rule=EllipticalRule(1.5, 22.5, heat=True), seed=2)
    scaled = run(cfg, rule=EllipticalRule(1.5, 22.5, heat=False, intensity=250.0), seed=2)
    cold = run(cfg, rule=EllipticalRule(1.5, 22.5, heat=False, intensity=0.0), seed=2)
    for other in (scaled, cold):
        np.testing.assert_array_equal(hot.arrival_time, other.arrival_time)
        np.testing.assert_array_equal(hot.grid.state, other.grid.state)
        np.testing.assert_array_equal(hot.grid.burn_timer, other.grid.burn_timer)
    assert not cold.front_intensity[cold.burned_mask].any()
    assert not cold.grid.intensity.any()


def test_front_intensity_does_not_depend_on_dt(engine: str) -> None:
    """Arrival times do not depend on dt, and the intensity is a function of arrival times."""
    rule = EllipticalRule(1.5, 22.5)
    coarse = run(small_config(engine, dt_s=120.0), rule=rule, seed=3)
    fine = run(small_config(engine, dt_s=15.0), rule=rule, seed=3)
    np.testing.assert_array_equal(coarse.burned_mask, fine.burned_mask)
    np.testing.assert_allclose(coarse.front_intensity, fine.front_intensity, rtol=1e-7, equal_nan=True)


def test_engines_agree_on_front_intensity() -> None:
    pytest.importorskip("numba")
    for lb, head in ((1.5, 22.5), (3.0, 90.0), (1.0, 0.0)):
        for heat in (True, False):
            rule = EllipticalRule(lb, head, heat=heat)
            ref = run(small_config("numpy", stencil=16), rule=rule, seed=5)
            fast = run(small_config("numba", stencil=16), rule=rule, seed=5)
            np.testing.assert_allclose(fast.front_intensity, ref.front_intensity, rtol=1e-9, equal_nan=True)
            np.testing.assert_allclose(fast.grid.intensity, ref.grid.intensity, rtol=1e-6, atol=1e-6)


def test_front_intensity_matches_the_estimator_on_the_final_arrival_times(engine: str) -> None:
    """The engine's value is ``heat * front_normal_rate(arrival)``, capped, wherever the plane is determined."""
    result = run(small_config(engine), rule=EllipticalRule(1.5, 90.0), seed=1)
    rate = front_normal_rate(result.arrival_time, get_stencil(16), DX)
    determined = np.isfinite(rate)
    assert determined.sum() > 300
    expected = np.minimum(HEAT * rate[determined], I_HEAD)
    np.testing.assert_allclose(result.front_intensity[determined], expected, rtol=1e-9)


class CorridorRule:
    """Uniform rate, half as fast in columns 10-14; heat per unit area ``HEAT``; head intensity 5000 kW/m."""

    name = "corridor"

    def __init__(self, heat: bool) -> None:
        self.heat = heat

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        h, w = grid.shape
        cell = np.where(grid.burnable_mask(), ROS, 0.0)
        cell[:, 10:15] /= 2.0
        return SpreadOutput(
            ros=np.broadcast_to(cell, (stencil.size, h, w)).copy(),
            intensity=np.full((h, w), 5000.0),
            flame_length=np.zeros((h, w)),
            heat_per_area=np.full((h, w), HEAT) if self.heat else None,
        )


@pytest.mark.parametrize("stencil", [8, 16])
def test_fallback_is_the_rate_of_the_igniting_link(engine: str, stencil: int) -> None:
    """In a corridor one cell wide every neighbour is collinear, so the igniting link's rate is used.

    The link rate is the harmonic mean of the two cells' rates (0.2 m/s in the open, 0.1 m/s
    inside the slow strip, 2 * 0.2 * 0.1 / 0.3 across its ends), without the front-speed
    correction. With heat per unit area the intensity is ``HEAT`` times that; without it, the
    head intensity times that over the cell's own rate, capped at the head intensity.
    """
    cfg = ScenarioConfig.model_validate(
        {
            "name": "corridor",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [1, 25], "dx": DX},
            "spread": {"rule": "constant", "params": {"ros": ROS}},
            "ignition": {"cells": [[0, 0]]},
            "duration_s": 14400.0,
            "stencil": stencil,
            "engine": engine,
        }
    )
    mixed = 2 * 0.2 * 0.1 / 0.3
    link = np.full(25, ROS)
    link[11:15] = ROS / 2
    link[10] = link[15] = mixed

    with_heat = run(cfg, rule=CorridorRule(heat=True)).front_intensity[0]
    assert with_heat[0] == 5000.0  # the initial ignition gets the head value
    np.testing.assert_allclose(with_heat[1:], HEAT * link[1:], rtol=1e-12)

    scaled = run(cfg, rule=CorridorRule(heat=False)).front_intensity[0]
    own = np.full(25, ROS)
    own[10:15] = ROS / 2
    np.testing.assert_allclose(scaled[1:], np.minimum(5000.0 * link[1:] / own[1:], 5000.0), rtol=1e-12)
    assert scaled[10] == 5000.0  # entering the strip: link rate above the cell's own, capped
    assert scaled[15] == pytest.approx(5000.0 * mixed / ROS)


def test_rules_without_intensity_still_give_zero(engine: str) -> None:
    result = run(small_config(engine, duration_s=900.0))
    assert not result.grid.intensity.any()
    assert not result.front_intensity[result.burned_mask].any()
    assert np.isnan(result.front_intensity[~result.burned_mask]).all()


def test_weather_change_does_not_rewrite_the_intensity_of_cells_already_burned() -> None:
    """Intensity is fixed at ignition: a later, windier hour does not change cells behind the front."""

    class WindScaled:
        name = "wind_scaled"

        def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
            h, w = grid.shape
            factor = 1.0 + weather.wind_speed
            cell = np.where(grid.burnable_mask(), ROS * factor, 0.0)
            return SpreadOutput(
                ros=np.broadcast_to(cell, (stencil.size, h, w)).copy(),
                intensity=np.full((h, w), I_HEAD * factor),
                flame_length=np.zeros((h, w)),
                heat_per_area=np.full((h, w), HEAT),
            )

    weather = {"series": [{"time_s": 0.0, "wind_speed": 0.0}, {"time_s": 1800.0, "wind_speed": 1.0}]}
    calm = run(small_config(duration_s=1800.0), rule=WindScaled(), seed=4)
    windy = run(small_config(duration_s=3000.0, weather=weather), rule=WindScaled(), seed=4)
    early = calm.burned_mask
    np.testing.assert_array_equal(windy.front_intensity[early], calm.front_intensity[early])
    late = windy.burned_mask & (windy.arrival_time > 2400.0)
    assert late.sum() > 20
    assert windy.front_intensity[late].mean() == pytest.approx(2 * I_HEAD, rel=0.05)
    assert np.nanmax(calm.front_intensity) <= I_HEAD


def test_ensemble_rate_multiplier_leaves_intensities_alone() -> None:
    """``ScaledRateRule`` doubles every rate: the fire is the same one, twice as fast, equally intense.

    The wrapper stands for uncertainty in the rate, not in the energy release. It drops
    ``heat_per_area``, so the intensity is the head value times ``R_n / max_k ros``, in which
    the multiplier cancels.
    """
    inner = EllipticalRule(1.5, 22.5, heat=True)
    plain = run(small_config(duration_s=3600.0), rule=EllipticalRule(1.5, 22.5, heat=False), seed=6)
    doubled = run(small_config(duration_s=1800.0, dt_s=15.0), rule=ScaledRateRule(inner, 2.0), seed=6)
    np.testing.assert_array_equal(doubled.burned_mask, plain.burned_mask)
    np.testing.assert_allclose(doubled.arrival_time, plain.arrival_time / 2.0, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(doubled.front_intensity, plain.front_intensity, rtol=1e-6, equal_nan=True)


# ----------------------------------------------------------------------------- flame length (Byram 1959)


def test_shipped_flame_length_model() -> None:
    model = load_flame_length_model()
    assert isinstance(model, FlameLengthModel)
    assert model.coefficient.value == 0.0775 and model.exponent.value == 0.46
    assert "Byram" in model.coefficient.source and "Byram" in model.exponent.source


def test_flame_length_coefficient_is_byrams_in_si_units() -> None:
    """Byram: L = 0.45 I^0.46 in ft and Btu/(ft s). 1 Btu/(ft s) = 1.05506 kJ / 0.3048 m."""
    converted = 0.3048 * 0.45 / (1.05506 / 0.3048) ** 0.46
    assert load_flame_length_model().coefficient.value == pytest.approx(converted, rel=2e-3)


def test_flame_length_from_intensity() -> None:
    assert float(flame_length_from_intensity(0.0)) == 0.0
    assert float(flame_length_from_intensity(1000.0)) == pytest.approx(0.0775 * 1000.0**0.46)
    assert float(flame_length_from_intensity(1000.0)) == pytest.approx(1.86, abs=0.01)
    # 346 kW/m is 100 Btu/(ft s), Byram's 0.45 * 100^0.46 = 3.74 ft = 1.14 m: the 4 ft hand-crew limit
    assert float(flame_length_from_intensity(346.0)) == pytest.approx(1.14, abs=0.01)
    grid = np.array([[0.0, 10.0], [100.0, 10000.0]], dtype=np.float32)
    lengths = flame_length_from_intensity(grid)
    assert lengths.shape == (2, 2) and lengths.dtype == np.float64
    assert np.all(np.diff(lengths.ravel()) > 0)
    custom = FlameLengthModel.model_validate(
        {"coefficient": {"value": 2.0, "source": "x"}, "exponent": {"value": 0.5, "source": "x"}}
    )
    assert float(flame_length_from_intensity(16.0, custom)) == 8.0


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_flame_length_rejects_bad_intensity(bad: float) -> None:
    with pytest.raises(ValueError, match="intensity"):
        flame_length_from_intensity(np.array([1.0, bad]))


@pytest.mark.parametrize("field", ["coefficient", "exponent"])
def test_flame_length_model_rejects_non_positive_constants(field: str) -> None:
    data = {"coefficient": {"value": 0.0775, "source": "x"}, "exponent": {"value": 0.46, "source": "x"}}
    data[field] = {"value": 0.0, "source": "x"}
    with pytest.raises(ValueError, match="flame length"):
        FlameLengthModel.model_validate(data)


def test_flank_flames_are_shorter_than_head_flames() -> None:
    """The point of WP 1.12 for crews: on one fire the flank is workable where the head is not.

    LB 1.5 with a 2000 kW/m head: head flames about 2.5 m, above the 1.2 m (4 ft) hand-crew
    limit; flank flames, from 0.38 to 0.45 of the head intensity, well under the head's.
    """
    result, ignition = fire(1.5, 90.0)
    scale = 2000.0 / I_HEAD
    left, _ = flank_bearings(1.5, 90.0)
    head = float(flame_length_from_intensity(scale * I_HEAD * ray_mean(result, ignition, 90.0, min_cells=10)))
    flank = float(flame_length_from_intensity(scale * I_HEAD * ray_mean(result, ignition, left, min_cells=10)))
    assert head == pytest.approx(2.5, abs=0.1)
    assert flank < 0.7 * head
