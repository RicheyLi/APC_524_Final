"""Tests for the spread contract: ignition probability kernel and neighbour combination."""

import numpy as np
import pytest

from wildfire_sim import spread
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread.base import (
    SpreadOutput,
    SpreadRule,
    ignition_probability,
    max_stable_dt,
    neighbour_ignition_probability,
)
from wildfire_sim.types import STENCIL_8, STENCIL_16, Stencil


def test_ignition_probability_formula() -> None:
    ros = np.full((8, 3, 3), 0.5)  # m/s
    p = ignition_probability(ros, dt=30.0, dx=30.0, stencil=STENCIL_8)
    assert p.shape == (8, 3, 3)
    expected_ortho = 1 - np.exp(-0.5 * 30 / 30)
    expected_diag = 1 - np.exp(-0.5 * 30 / (30 * np.sqrt(2)))
    assert p[0, 1, 1] == pytest.approx(expected_ortho)
    assert p[1, 1, 1] == pytest.approx(expected_diag)
    assert p[0, 1, 1] > p[1, 1, 1]
    # zero ros -> zero probability
    assert ignition_probability(np.zeros((8, 2, 2)), 30, 30, STENCIL_8).max() == 0.0


def test_ignition_probability_mean_arrival_bias() -> None:
    """Mean arrival time of the per-step geometric draw is ``(d/R) * x / (1 - exp(-x))``, ``x = R dt / d``.

    So the mean is exactly ``d / R`` only as ``dt -> 0``; the relative bias is ~x/2 for small x
    (5% at x = 0.1, 18% at x = 1/3). This is why ``max_stable_dt`` keeps ``p`` small and why the
    dt-halving physics test (WP 1.4) compares distributions rather than demanding equality.
    """
    ros = np.full((8, 1, 1), 1.0)
    d = 30.0
    for dt in (0.5, 1.0, 5.0, 10.0):
        p = ignition_probability(ros, dt, d, STENCIL_8)[0, 0, 0]
        x = 1.0 * dt / d
        expected = d / 1.0 * x / (1 - np.exp(-x))
        assert dt / p == pytest.approx(expected, rel=1e-9)
    p_small = ignition_probability(ros, 0.5, d, STENCIL_8)[0, 0, 0]
    assert 0.5 / p_small == pytest.approx(d, rel=0.01)


def test_ignition_probability_rejects_bad_steps() -> None:
    with pytest.raises(ValueError):
        ignition_probability(np.zeros((8, 1, 1)), 0.0, 30.0, STENCIL_8)


def test_neighbour_combination_single_source() -> None:
    h = w = 5
    p = np.full((8, h, w), 0.3)
    burning = np.zeros((h, w), bool)
    burning[2, 2] = True
    out = neighbour_ignition_probability(p, burning, STENCIL_8)
    assert out.shape == (h, w)
    # the 8 neighbours of the centre each get exactly 0.3, everything else 0
    assert out[2, 2] == 0.0
    nbrs = out[1:4, 1:4].copy()
    nbrs[1, 1] = 0.3
    assert np.allclose(nbrs, 0.3)
    out[1:4, 1:4] = 0.0
    assert out.max() == 0.0


def test_neighbour_combination_two_sources() -> None:
    h = w = 3
    p = np.full((8, h, w), 0.5)
    burning = np.zeros((h, w), bool)
    burning[1, 0] = True  # west neighbour of centre
    burning[1, 2] = True  # east neighbour of centre
    out = neighbour_ignition_probability(p, burning, STENCIL_8)
    assert out[1, 1] == pytest.approx(1 - 0.5 * 0.5)


def test_neighbour_combination_respects_edges() -> None:
    h = w = 3
    p = np.ones((8, h, w))
    burning = np.zeros((h, w), bool)
    burning[0, 0] = True
    out = neighbour_ignition_probability(p, burning, STENCIL_8)
    assert out[0, 1] == 1.0 and out[1, 0] == 1.0 and out[1, 1] == 1.0
    assert out[2, 2] == 0.0


def test_neighbour_combination_16_stencil() -> None:
    h = w = 5
    p = np.full((16, h, w), 0.1)
    burning = np.zeros((h, w), bool)
    burning[2, 2] = True
    out = neighbour_ignition_probability(p, burning, STENCIL_16)
    assert np.count_nonzero(out) == 16
    assert out[0, 1] == pytest.approx(0.1)  # knight move (-2, -1)


def test_neighbour_combination_shape_check() -> None:
    with pytest.raises(ValueError):
        neighbour_ignition_probability(np.zeros((8, 2, 2)), np.zeros((3, 3), bool), STENCIL_8)


def test_max_stable_dt() -> None:
    assert max_stable_dt(1.0, 30.0) == pytest.approx(-np.log(0.5) * 30)
    assert max_stable_dt(0.0, 30.0) == float("inf")
    with pytest.raises(ValueError):
        max_stable_dt(1.0, 30.0, p_max=1.0)


def test_spread_output_validate(small_grid: Grid) -> None:
    h, w = small_grid.shape
    good = SpreadOutput(ros=np.zeros((8, h, w)), intensity=np.zeros((h, w)), flame_length=np.zeros((h, w)))
    good.validate(STENCIL_8, (h, w))
    with pytest.raises(ValueError):
        SpreadOutput(ros=np.zeros((16, h, w)), intensity=np.zeros((h, w)), flame_length=np.zeros((h, w))).validate(
            STENCIL_8, (h, w)
        )
    with pytest.raises(ValueError):
        SpreadOutput(ros=np.full((8, h, w), -1.0), intensity=np.zeros((h, w)), flame_length=np.zeros((h, w))).validate(
            STENCIL_8, (h, w)
        )


class _DummyRule:
    name = "dummy_for_test"

    def __init__(self, ros: float = 0.1) -> None:
        self.ros = ros

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        h, w = grid.shape
        ros = np.where(grid.burnable_mask(), self.ros, 0.0)
        return SpreadOutput(
            ros=np.broadcast_to(ros, (stencil.size, h, w)).copy(),
            intensity=np.zeros((h, w)),
            flame_length=np.zeros((h, w)),
        )


def test_registry_roundtrip(small_grid: Grid, calm_weather: Weather) -> None:
    name = "dummy_for_test"
    if name not in spread.available_rules():
        spread.register(name)(_DummyRule)
    rule = spread.make_spread_rule(name, ros=0.25)
    assert isinstance(rule, SpreadRule)
    out = rule.rate_of_spread(small_grid, calm_weather, STENCIL_8)
    out.validate(STENCIL_8, small_grid.shape)
    assert out.ros.max() == 0.25
    with pytest.raises(ValueError):
        spread.register(name)(_DummyRule)
    with pytest.raises(KeyError):
        spread.make_spread_rule("no_such_rule")
