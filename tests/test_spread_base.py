"""Tests for the spread contract: ignition probability kernel and neighbour combination."""

import numpy as np
import pytest
from scipy import stats

from wildfire_sim import spread
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread.base import (
    SpreadOutput,
    SpreadRule,
    draw_crossing_thresholds,
    ignition_probability,
    link_rate_of_spread,
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


# ----------------------------------------------------------------------------- gamma arrival kernel


def test_ignition_probability_shape_one_ignores_age() -> None:
    """shape = 1 is the memoryless kernel: the source cell's age must not matter."""
    ros = np.full((8, 3, 3), 0.5)
    age = np.full((3, 3), 400.0)
    base = ignition_probability(ros, 30.0, 30.0, STENCIL_8)
    np.testing.assert_allclose(ignition_probability(ros, 30.0, 30.0, STENCIL_8, age=age, shape=1.0), base)


def test_ignition_probability_shape_needs_valid_age() -> None:
    ros = np.full((8, 3, 3), 0.5)
    with pytest.raises(ValueError, match="age"):
        ignition_probability(ros, 30.0, 30.0, STENCIL_8, shape=10.0)
    with pytest.raises(ValueError, match="shape"):
        ignition_probability(ros, 30.0, 30.0, STENCIL_8, age=np.zeros((3, 3)), shape=0.5)
    with pytest.raises(ValueError, match="age"):
        ignition_probability(ros, 30.0, 30.0, STENCIL_8, age=np.zeros((2, 2)), shape=10.0)
    with pytest.raises(ValueError, match="age"):
        ignition_probability(ros, 30.0, 30.0, STENCIL_8, age=np.full((3, 3), -1.0), shape=10.0)


def test_ignition_probability_gamma_hazard_rises_with_age() -> None:
    """For shape > 1 a freshly ignited cell rarely ignites a neighbour; an old one almost surely does."""
    ros = np.full((8, 1, 1), 0.2)  # mean orthogonal arrival = 30 / 0.2 = 150 s
    young = ignition_probability(ros, 15.0, 30.0, STENCIL_8, age=np.zeros((1, 1)), shape=20.0)
    mid = ignition_probability(ros, 15.0, 30.0, STENCIL_8, age=np.full((1, 1), 150.0), shape=20.0)
    old = ignition_probability(ros, 15.0, 30.0, STENCIL_8, age=np.full((1, 1), 3000.0), shape=20.0)
    assert young[0, 0, 0] < 1e-6
    assert young[0, 0, 0] < mid[0, 0, 0] < old[0, 0, 0] <= 1.0
    assert young.min() >= 0.0 and old.max() <= 1.0
    # zero ros never ignites, whatever the age
    zero = ignition_probability(np.zeros((8, 1, 1)), 15.0, 30.0, STENCIL_8, age=np.full((1, 1), 1e6), shape=20.0)
    assert zero.max() == 0.0


@pytest.mark.parametrize("shape", [1.0, 5.0, 20.0])
@pytest.mark.parametrize("dt", [30.0, 7.5])
def test_ignition_probability_mean_arrival_is_d_over_r(shape: float, dt: float) -> None:
    """Stepping the kernel reproduces mean arrival time d / R for any time step (to within dt).

    Survival to the end of step n is the product of (1 - p) over the steps so far. Arrivals are
    recorded at step ends, so the mean discrete arrival time must lie in [d/R, d/R + dt].
    """
    ros_value, dx = 0.2, 30.0
    ros = np.full((8, 1, 1), ros_value)
    survive, mean, t = 1.0, 0.0, 0.0
    while t < 3000.0:
        p = ignition_probability(ros, dt, dx, STENCIL_8, age=np.full((1, 1), t), shape=shape)[0, 0, 0]
        t += dt
        mean += t * survive * p
        survive *= 1.0 - p
    assert survive < 1e-6
    assert dx / ros_value <= mean <= dx / ros_value + dt


def test_ignition_probability_gamma_is_dt_consistent() -> None:
    """Two half steps leave the same survival probability as one full step."""
    ros = np.full((8, 1, 1), 0.2)
    age = np.full((1, 1), 100.0)
    full = ignition_probability(ros, 30.0, 30.0, STENCIL_8, age=age, shape=10.0)
    half1 = ignition_probability(ros, 15.0, 30.0, STENCIL_8, age=age, shape=10.0)
    half2 = ignition_probability(ros, 15.0, 30.0, STENCIL_8, age=age + 15.0, shape=10.0)
    np.testing.assert_allclose(1 - full, (1 - half1) * (1 - half2), rtol=1e-9)


# ----------------------------------------------------------------------------- registry


def test_make_spread_rule_imports_rule_module_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rule whose module has not been imported yet is found as ``wildfire_sim.spread.<name>``."""
    name = "_lazy_rule"

    def fake_import(modname: str) -> object:
        if modname == f"wildfire_sim.spread.{name}":
            spread.register(name)(_DummyRule)
            return object()
        raise ModuleNotFoundError(name=modname)

    monkeypatch.setattr(spread.importlib, "import_module", fake_import)
    try:
        rule = spread.make_spread_rule(name, ros=2.0)
        assert isinstance(rule, _DummyRule)
        with pytest.raises(KeyError, match="unknown spread rule"):
            spread.make_spread_rule("_no_such_rule")
    finally:
        spread._REGISTRY.pop(name, None)


# ----------------------------------------------------------------------------- link rates


def test_link_rate_is_unchanged_in_uniform_fuel_and_zero_off_grid() -> None:
    ros = np.full((8, 4, 5), 0.3)
    link = link_rate_of_spread(ros, STENCIL_8)
    assert link.shape == (8, 4, 5)
    np.testing.assert_allclose(link[:, 1:-1, 1:-1], 0.3)
    # direction 0 is north: the top row has no northern neighbour
    assert np.all(link[0, 0, :] == 0.0)
    # direction 2 is east: the last column has no eastern neighbour
    assert np.all(link[2, :, -1] == 0.0)


def test_link_rate_is_harmonic_mean_of_source_and_target() -> None:
    """Fire crosses half of each cell, so travel time is d/2/R_source + d/2/R_target."""
    ros = np.full((8, 3, 3), 0.4)
    ros[:, 1, 2] = 0.1  # slow cell east of the centre
    link = link_rate_of_spread(ros, STENCIL_8)
    expected = 2 * 0.4 * 0.1 / (0.4 + 0.1)
    assert link[2, 1, 1] == pytest.approx(expected)  # centre -> east
    assert link[6, 1, 2] == pytest.approx(expected)  # slow cell -> west, same link reversed
    assert link[0, 1, 1] == pytest.approx(0.4)  # centre -> north is unaffected


def test_link_rate_is_zero_into_a_cell_that_cannot_spread() -> None:
    """A fully retarded or unburnable target (ros 0) must not be reachable at the source's rate."""
    ros = np.full((8, 3, 3), 0.4)
    ros[:, 1, 2] = 0.0
    link = link_rate_of_spread(ros, STENCIL_8)
    assert link[2, 1, 1] == 0.0
    assert np.all(link[:, 1, 2] == 0.0)
    assert np.all(np.isfinite(link))


def test_link_rate_rejects_wrong_stencil() -> None:
    with pytest.raises(ValueError, match="stencil"):
        link_rate_of_spread(np.zeros((8, 3, 3)), STENCIL_16)


# ----------------------------------------------------------------------------- crossing thresholds


def test_crossing_thresholds_shape_dtype_and_determinism() -> None:
    a = draw_crossing_thresholds(np.random.default_rng(7), 20.0, STENCIL_16, (5, 6))
    b = draw_crossing_thresholds(np.random.default_rng(7), 20.0, STENCIL_16, (5, 6))
    assert a.shape == (16, 5, 6)
    assert a.dtype == np.float32
    assert np.all(a > 0)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="shape"):
        draw_crossing_thresholds(np.random.default_rng(7), 0.5, STENCIL_8, (5, 6))


@pytest.mark.parametrize("shape", [1.0, 5.0, 20.0])
def test_crossing_thresholds_match_the_hazard_kernel(shape: float) -> None:
    """Thresholds are the same crossing-time law as ignition_probability: unit-mean gamma(k).

    A link completes when its accumulated progress (integral of R dt / d) reaches its threshold,
    so P(not yet complete at progress x) must equal the kernel's survival Q(k, k x).
    """
    thresholds = draw_crossing_thresholds(np.random.default_rng(0), shape, STENCIL_8, (60, 60)).ravel()
    assert thresholds.mean() == pytest.approx(1.0, abs=0.02)
    assert thresholds.var() == pytest.approx(1.0 / shape, rel=0.05)
    result = stats.kstest(thresholds, stats.gamma(a=shape, scale=1.0 / shape).cdf)
    assert result.pvalue > 0.001
    # survival after one step from age 0 with ros 0.2, dt 60, dx 30: progress x = 0.4
    p = ignition_probability(np.full((8, 1, 1), 0.2), 60.0, 30.0, STENCIL_8, age=np.zeros((1, 1)), shape=shape)
    assert np.mean(thresholds <= 0.4) == pytest.approx(p[0, 0, 0], abs=0.01)


def test_knight_link_rate_includes_the_cells_it_crosses() -> None:
    """A knight link's segment spends a quarter of its length in each of four cells.

    From (r, c) to (r + 1, c + 2) it crosses (r, c + 1) and (r + 1, c + 1) as well as the two end
    cells, so the travel time is d/4 * (1/R_a + 1/R_b + 1/R_c + 1/R_d). Using the end cells
    alone let fire tunnel through a strip of slow fuel.
    """
    k = [tuple(o) for o in STENCIL_16.offsets.tolist()].index((1, 2))
    ros = np.full((16, 5, 6), 0.4)
    np.testing.assert_allclose(link_rate_of_spread(ros, STENCIL_16)[k, 1, 1], 0.4)  # uniform fuel: unchanged
    ros[:, 1, 2] = 0.1  # first crossed cell (r, c + 1) is slow
    link = link_rate_of_spread(ros, STENCIL_16)
    assert link[k, 1, 1] == pytest.approx(4.0 / (1 / 0.4 + 1 / 0.1 + 1 / 0.4 + 1 / 0.4))
    ros[:, 1, 2] = 0.4
    ros[:, 2, 2] = 0.0  # second crossed cell (r + 1, c + 1) cannot carry fire
    link = link_rate_of_spread(ros, STENCIL_16)
    assert link[k, 1, 1] == 0.0
    # the vertical knight move (2, 1) from (1, 1) crosses (2, 1) and (2, 2)
    k2 = [tuple(o) for o in STENCIL_16.offsets.tolist()].index((2, 1))
    assert link[k2, 1, 1] == 0.0
    # Moore links are untouched by cells they do not cross
    k_e = [tuple(o) for o in STENCIL_16.offsets.tolist()].index((0, 1))
    assert link[k_e, 1, 0] == pytest.approx(0.4)
    assert np.all(np.isfinite(link))
