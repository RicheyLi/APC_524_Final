"""Tests for wildfire_sim.metrics (WP 2.8).

Acceptance (docs/plan.md WP 2.8): hand-computed cases; intervals cover a known mean at the
stated rate in a simulation test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wildfire_sim.config import ScenarioConfig, load_costs
from wildfire_sim.grid import Grid
from wildfire_sim.metrics import (
    ResourceUse,
    area_ratio,
    arrival_interval_error,
    asset_loss,
    bootstrap_ci,
    burned_area_ha,
    consensus_mask,
    footprint_arrival,
    interval_error,
    jaccard,
    paired_difference_ci,
    sorensen,
    suppression_cost,
)
from wildfire_sim.types import CellState

NAN = float("nan")


# ----------------------------------------------------------------------------- overlap


def test_overlap_by_hand() -> None:
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[0:2, 0:2] = True  # 4 cells
    b[1:3, 1:3] = True  # 4 cells, 1 shared
    assert jaccard(a, b) == pytest.approx(1 / 7)
    assert sorensen(a, b) == pytest.approx(2 / 8)
    assert jaccard(a, a) == 1.0 and sorensen(a, a) == 1.0
    assert jaccard(a, ~a) == 0.0
    empty = np.zeros((4, 4), dtype=bool)
    assert jaccard(empty, empty) == 1.0 and sorensen(empty, empty) == 1.0
    assert area_ratio(a, b) == 1.0 and area_ratio(a, a | b) == pytest.approx(4 / 7)
    assert area_ratio(30.0, 20.0) == 1.5
    with pytest.raises(ValueError, match="observed area is 0"):
        area_ratio(a, empty)
    with pytest.raises(ValueError, match="shape"):
        jaccard(a, np.zeros((3, 3), dtype=bool))
    p = np.array([[0.0, 0.49], [0.5, 1.0]])
    assert consensus_mask(p).tolist() == [[False, False], [True, True]]


def test_burned_area(flat_scenario: ScenarioConfig) -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :2] = True
    assert burned_area_ha(mask, dx=30.0) == pytest.approx(10 * 900 / 1e4)
    with pytest.raises(ValueError):
        burned_area_ha(mask)
    from wildfire_sim.sim import run

    result = run(flat_scenario, seed=0)
    assert burned_area_ha(result) == pytest.approx(result.burned_mask.sum() * result.grid.dx**2 / 1e4)


# ----------------------------------------------------------------------------- arrival intervals


def test_interval_error_by_hand() -> None:
    # interval (100, 200]
    assert interval_error(150.0, 100.0, 200.0, 1000.0) == 0.0
    assert interval_error(200.0, 100.0, 200.0, 1000.0) == 0.0  # the detecting pass itself counts
    assert interval_error(100.0, 100.0, 200.0, 1000.0) == 0.0  # arrival exactly at t_clear: clear then
    assert interval_error(40.0, 100.0, 200.0, 1000.0) == 60.0  # too early
    assert interval_error(260.0, 100.0, 200.0, 1000.0) == 60.0  # too late
    # never reached by the window end: arrival > 1000, so at least 800 s late
    assert interval_error(NAN, 100.0, 200.0, 1000.0) == 800.0
    # window ends inside the interval: arrival may still be on time
    assert interval_error(NAN, 100.0, 200.0, 150.0) == 0.0
    with pytest.raises(ValueError):
        interval_error(1.0, 200.0, 100.0, 1000.0)


def test_arrival_interval_error_uses_the_earliest_cell_of_the_footprint() -> None:
    arrival = np.full((40, 40), NAN)
    arrival[20, 20] = 500.0
    arrival[25, 25] = 150.0  # 5 cells away: inside a 375 m footprint (6 cells each way at 30 m)
    arrival[20, 30] = 50.0  # 10 cells away: outside
    assert footprint_arrival(arrival, 20, 20, 6) == 150.0
    dets = [(100.0, 200.0, 20, 20), (100.0, 200.0, 2, 2), (300.0, 400.0, 20, 30)]
    errors = arrival_interval_error(arrival, dets, 375.0, 1000.0, dx=30.0)
    # first: 150 inside (100, 200]; second: nothing burned nearby -> 1000 - 200; third: 50 is 250 early
    assert errors.tolist() == [0.0, 800.0, 250.0]
    assert arrival_interval_error(arrival, [], 375.0, 1000.0, dx=30.0).shape == (0,)
    with pytest.raises(ValueError):
        arrival_interval_error(arrival, dets, 0.0, 1000.0, dx=30.0)


def test_footprint_at_the_grid_edge() -> None:
    arrival = np.full((5, 5), NAN)
    arrival[0, 0] = 7.0
    assert footprint_arrival(arrival, 1, 1, 6) == 7.0
    assert math.isnan(footprint_arrival(np.full((5, 5), NAN), 2, 2, 1))


# ----------------------------------------------------------------------------- losses and costs


def test_asset_loss_counts_burning_and_burned_cells() -> None:
    grid = Grid.empty((3, 3), 30.0)
    grid.asset_value[:] = 100.0
    grid.state[0, 0] = int(CellState.BURNING)
    grid.state[1, 1] = int(CellState.BURNED)
    grid.state[2, 2] = int(CellState.UNBURNABLE)
    assert asset_loss(grid) == 200.0


def test_suppression_cost_by_hand() -> None:
    costs = load_costs("configs/costs.yaml")
    fleet = [
        ResourceUse("crew", count=2, hours=10.0),  # 2 x 10 x 1500
        ResourceUse("aircraft", count=1, hours=2.0, retardant_l=10_000.0),  # 2 x 8000 + 10000 x 0.8
        ResourceUse("engine", count=1, hours=4.0, water_l=5000.0, hourly_cost_usd=100.0),  # own rate
    ]
    expected = 2 * 10 * 1500 + 2 * 8000 + 10_000 * 0.8 + 4 * 100 + 5000 * 0.001
    assert suppression_cost(fleet, costs) == pytest.approx(expected)
    assert suppression_cost([], costs) == 0.0
    with pytest.raises(ValueError):
        ResourceUse("crew", count=-1, hours=1.0)


# ----------------------------------------------------------------------------- intervals


def test_bootstrap_ci_by_hand() -> None:
    mean, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], seed=1)
    assert mean == 3.0 and 1.0 < lo < 3.0 < hi < 5.0
    assert bootstrap_ci([1.0, 2.0, 3.0], seed=7) == bootstrap_ci([1.0, 2.0, 3.0], seed=7)  # seeded
    assert bootstrap_ci([4.0, 4.0, 4.0]) == (4.0, 4.0, 4.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], level=1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, NAN])
    d, lo, hi = paired_difference_ci([10.0, 11.0, 12.0, 13.0], [9.0, 10.0, 11.0, 12.0])
    assert (d, lo, hi) == (1.0, 1.0, 1.0)  # every pair differs by exactly 1
    with pytest.raises(ValueError):
        paired_difference_ci([1.0, 2.0], [1.0])


@pytest.mark.slow
def test_intervals_cover_a_known_mean_at_the_stated_rate() -> None:
    """Acceptance: 95 % intervals from samples of 50 cover the true mean about 95 % of the time.

    600 simulated samples from a skewed (gamma) distribution like burned areas; the coverage
    must lie within 3 binomial standard errors (about 2.7 points) of 95 %. The percentile
    bootstrap is known to undercover slightly at small n, so the band is not tightened further.
    """
    rng = np.random.default_rng(2024)
    shape, scale = 2.0, 3.0  # mean 6
    true_mean = shape * scale
    n_rep, n = 600, 50
    hits = paired_hits = 0
    for i in range(n_rep):
        x = rng.gamma(shape, scale, n)
        _, lo, hi = bootstrap_ci(x, 0.95, n_boot=2000, seed=i)
        hits += lo <= true_mean <= hi
        # paired: b = a - 1 + noise, so the true mean difference is 1
        y = x - 1.0 + rng.normal(0.0, 0.5, n)
        _, lo, hi = paired_difference_ci(x, y, 0.95, n_boot=2000, seed=i)
        paired_hits += lo <= 1.0 <= hi
    se = math.sqrt(0.95 * 0.05 / n_rep)
    assert abs(hits / n_rep - 0.95) < 3 * se
    assert abs(paired_hits / n_rep - 0.95) < 3 * se
