"""Tests for Monte Carlo ensembles (WP 1.6).

Grids are small (41 x 41) and runs short so that the spawn-pool start-up dominates. No test
asserts an absolute burned area or front speed: the kernel correction table rescales every rate,
so runs are compared with each other instead.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import stats

from wildfire_sim import sim
from wildfire_sim.config import EnsembleConfig, ScenarioConfig
from wildfire_sim.data import build_grid
from wildfire_sim.ensemble import (
    PERTURBATION_KEYS,
    EnsembleResult,
    ScaledRateRule,
    derive_seeds,
    draw_factors,
    jitter_ignition,
    mean_ci,
    perturb_config,
    perturbation_rng,
    run_ensemble,
    running_mean,
    runs_for_precision,
)
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.types import STENCIL_8, CellState

NEUTRAL = {
    "wind_speed_factor": 1.0,
    "wind_direction_offset_deg": 0.0,
    "moisture_factor": 1.0,
    "ros_multiplier": 1.0,
}


def make_config(
    *,
    shape: tuple[int, int] = (41, 41),
    generator: str = "flat_uniform",
    grid_params: dict[str, Any] | None = None,
    rule: str = "constant",
    rule_params: dict[str, Any] | None = None,
    duration: float = 900.0,
    cells: list[tuple[int, int]] | None = None,
    weather: list[dict[str, float]] | None = None,
    ensemble: dict[str, Any] | None = None,
) -> ScenarioConfig:
    data: dict[str, Any] = {
        "name": "ens",
        "grid": {
            "kind": "synthetic",
            "generator": generator,
            "shape": list(shape),
            "dx": 30.0,
            "params": grid_params or {},
        },
        "spread": {"rule": rule, "params": rule_params if rule_params is not None else {"ros": 0.2}},
        "ignition": {"cells": [list(c) for c in (cells or [])], "center": cells is None},
        "dt_s": 30.0,
        "duration_s": duration,
    }
    if weather is not None:
        data["weather"] = {"series": weather}
    if ensemble is not None:
        data["ensemble"] = ensemble
    return ScenarioConfig.model_validate(data)


PERTURBED = {
    "wind_speed_sigma_frac": 0.3,
    "wind_direction_sigma_deg": 20.0,
    "moisture_sigma_frac": 0.2,
    "ros_multiplier_sigma_frac": 0.3,
}


def assert_same(a: EnsembleResult, b: EnsembleResult) -> None:
    np.testing.assert_array_equal(a.arrival_times, b.arrival_times)
    np.testing.assert_array_equal(a.burned_area_m2, b.burned_area_m2)
    np.testing.assert_array_equal(a.burn_probability, b.burn_probability)
    assert a.seeds == b.seeds
    assert a.perturbations == b.perturbations


# ----------------------------------------------------------------------------- seeds, shapes


def test_result_shapes_and_dtypes() -> None:
    cfg = make_config()
    res = run_ensemble(cfg, n_runs=3, base_seed=1)
    assert res.arrival_times.shape == (3, 41, 41)
    assert res.arrival_times.dtype == np.float32
    assert res.burned_area_m2.shape == (3,)
    assert res.burned_area_m2.dtype == np.float64
    assert res.burn_probability.shape == (41, 41)
    assert res.burn_probability.dtype == np.float64
    assert len(res.seeds) == 3 and len(res.perturbations) == 3
    assert all(isinstance(s, int) and s >= 0 for s in res.seeds)
    assert res.elapsed_s >= 0
    # burned area = burned cells x dx^2
    burned = (~np.isnan(res.arrival_times)).sum(axis=(1, 2))
    np.testing.assert_array_equal(res.burned_area_m2, burned * 30.0 * 30.0)


def test_parallel_equals_serial_bit_for_bit() -> None:
    cfg = make_config(ensemble={**PERTURBED, "ignition_jitter_cells": 2})
    serial = run_ensemble(cfg, n_runs=4, base_seed=7, n_workers=1)
    parallel = run_ensemble(cfg, n_runs=4, base_seed=7, n_workers=2)
    assert_same(serial, parallel)


def test_run_i_does_not_depend_on_n_runs() -> None:
    cfg = make_config(ensemble=PERTURBED)
    four = run_ensemble(cfg, n_runs=4, base_seed=3)
    eight = run_ensemble(cfg, n_runs=8, base_seed=3)
    assert eight.seeds[:4] == four.seeds
    assert eight.perturbations[:4] == four.perturbations
    np.testing.assert_array_equal(eight.arrival_times[:4], four.arrival_times)
    assert derive_seeds(3, 8)[:4] == derive_seeds(3, 4) == four.seeds


def test_different_base_seeds_differ() -> None:
    cfg = make_config()
    a = run_ensemble(cfg, n_runs=2, base_seed=0)
    b = run_ensemble(cfg, n_runs=2, base_seed=1)
    assert set(a.seeds).isdisjoint(b.seeds)
    assert not np.array_equal(a.arrival_times, b.arrival_times, equal_nan=True)


def test_seeds_within_a_run_differ_and_are_documented_derivation() -> None:
    seeds = derive_seeds(11, 5)
    assert len(set(seeds)) == 5
    children = np.random.SeedSequence(11).spawn(5)
    assert seeds == tuple(int(c.generate_state(1, dtype=np.uint64)[0] >> 1) for c in children)


def test_defaults_come_from_config() -> None:
    cfg = make_config(ensemble={"n_runs": 2, "base_seed": 5})
    res = run_ensemble(cfg)
    assert res.seeds == derive_seeds(5, 2)


@pytest.mark.parametrize(
    "kwargs,match",
    [({"n_runs": 0}, "n_runs"), ({"n_workers": 0}, "n_workers"), ({"base_seed": -1}, "base_seed")],
)
def test_run_ensemble_validates_inputs(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_ensemble(make_config(), **kwargs)


# ----------------------------------------------------------------------------- perturbation off


def test_perturbation_off_equals_plain_seeded_runs() -> None:
    cfg = make_config()
    res = run_ensemble(cfg, n_runs=3, base_seed=2)
    for i, seed in enumerate(res.seeds):
        plain = sim.run(cfg, seed=seed)
        np.testing.assert_array_equal(res.arrival_times[i], plain.arrival_time.astype(np.float32))
        assert res.burned_area_m2[i] == plain.grid.burned_area_m2()
    for p in res.perturbations:
        assert {k: p[k] for k in NEUTRAL} == NEUTRAL
        assert all(v == 0.0 for k, v in p.items() if k not in NEUTRAL)


# ----------------------------------------------------------------------------- perturbation on


def test_perturbations_recorded_and_differ_between_runs() -> None:
    cfg = make_config(ensemble=PERTURBED)
    res = run_ensemble(cfg, n_runs=4, base_seed=0)
    for key in PERTURBATION_KEYS:
        values = [p[key] for p in res.perturbations]
        assert len(set(values)) == 4, key
    for p in res.perturbations:
        assert p["wind_speed_factor"] > 0 and p["moisture_factor"] > 0 and p["ros_multiplier"] > 0


def test_draw_factors_spread_matches_sigmas() -> None:
    ens = EnsembleConfig(**PERTURBED)
    rng = np.random.default_rng(np.random.SeedSequence(99))
    draws = [draw_factors(rng, ens) for _ in range(4000)]
    for key, sigma in [
        ("wind_speed_factor", 0.3),
        ("moisture_factor", 0.2),
        ("ros_multiplier", 0.3),
    ]:
        logs = np.log([d[key] for d in draws])
        assert abs(np.median(logs)) < 0.03, key  # median factor 1
        assert np.std(logs) == pytest.approx(sigma, rel=0.06), key
    offsets = np.array([d["wind_direction_offset_deg"] for d in draws])
    assert abs(offsets.mean()) < 1.5
    assert np.std(offsets) == pytest.approx(20.0, rel=0.06)


def test_draw_factors_neutral_with_zero_sigmas() -> None:
    rng = np.random.default_rng(0)
    assert draw_factors(rng, EnsembleConfig()) == NEUTRAL


def test_wind_perturbation_does_not_change_rate_multiplier_draws() -> None:
    only_ros = make_config(ensemble={"ros_multiplier_sigma_frac": 0.3})
    with_wind = make_config(ensemble={"ros_multiplier_sigma_frac": 0.3, "wind_speed_sigma_frac": 0.5})
    a = run_ensemble(only_ros, n_runs=3, base_seed=4)
    b = run_ensemble(with_wind, n_runs=3, base_seed=4)
    assert [p["ros_multiplier"] for p in a.perturbations] == [p["ros_multiplier"] for p in b.perturbations]
    assert all(p["wind_speed_factor"] == 1.0 for p in a.perturbations)
    assert any(p["wind_speed_factor"] != 1.0 for p in b.perturbations)


def test_wind_direction_stays_in_range() -> None:
    cfg = make_config(
        rule="alexandridis",
        rule_params={"ros_base": 0.2},
        weather=[{"time_s": 0, "wind_speed": 4.0, "wind_direction": 355.0}, {"time_s": 300, "wind_direction": 2.0}],
        ensemble={"wind_direction_sigma_deg": 90.0, "wind_speed_sigma_frac": 0.5},
    )
    res = run_ensemble(cfg, n_runs=4, base_seed=0)  # Weather/WeatherPoint reject directions outside [0, 360)
    assert len(res.seeds) == 4
    ens = cfg.ensemble
    for seed in derive_seeds(0, 200):
        factors = draw_factors(perturbation_rng(seed), ens)
        perturbed = perturb_config(cfg, factors)
        for p in perturbed.weather.series:
            assert 0.0 <= p.wind_direction < 360.0


def test_perturb_config_scales_weather_and_keeps_the_rest() -> None:
    cfg = make_config(weather=[{"time_s": 0, "wind_speed": 4.0, "wind_direction": 350.0, "moisture_1h": 0.05}])
    factors = {
        "wind_speed_factor": 1.5,
        "wind_direction_offset_deg": 20.0,
        "moisture_factor": 2.0,
        "ros_multiplier": 3.0,
    }
    out = perturb_config(cfg, factors)
    p0, q0 = cfg.weather.series[0], out.weather.series[0]
    assert q0.wind_speed == pytest.approx(6.0)
    assert q0.wind_direction == pytest.approx(10.0)
    assert q0.moisture_1h == pytest.approx(0.10)
    assert q0.moisture_live_woody == pytest.approx(2 * p0.moisture_live_woody)
    assert q0.temperature_c == p0.temperature_c and q0.relative_humidity == p0.relative_humidity
    assert out.grid == cfg.grid and out.spread == cfg.spread and out.ignition == cfg.ignition


def test_burned_area_varies_more_with_perturbation() -> None:
    off = run_ensemble(make_config(), n_runs=8, base_seed=0)
    on = run_ensemble(make_config(ensemble={"ros_multiplier_sigma_frac": 0.4}), n_runs=8, base_seed=0)
    assert np.std(on.burned_area_m2) > 2 * np.std(off.burned_area_m2)


# ----------------------------------------------------------------------------- rate multiplier


def test_scaled_rate_rule_scales_ros_only() -> None:
    grid = build_grid(make_config().grid)
    inner = make_spread_rule("alexandridis", ros_base=0.2)
    weather = Weather(wind_speed=5.0, wind_direction=270.0)
    base = inner.rate_of_spread(grid, weather, STENCIL_8)
    scaled = ScaledRateRule(inner, 2.5).rate_of_spread(grid, weather, STENCIL_8)
    scaled.validate(STENCIL_8, grid.shape)
    np.testing.assert_allclose(scaled.ros, 2.5 * base.ros)
    np.testing.assert_array_equal(scaled.intensity, base.intensity)
    np.testing.assert_array_equal(scaled.flame_length, base.flame_length)
    assert ScaledRateRule(inner, 2.5).name == inner.name


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_scaled_rate_rule_rejects_bad_multiplier(bad: float) -> None:
    with pytest.raises(ValueError, match="multiplier"):
        ScaledRateRule(make_spread_rule("constant", ros=0.2), bad)


def test_multiplier_two_roughly_doubles_front_distance() -> None:
    cfg = make_config(shape=(61, 61), duration=1200.0)
    inner = make_spread_rule("constant", ros=0.2)
    slow = sim.run(cfg, seed=5, rule=inner)
    fast = sim.run(cfg, seed=5, rule=ScaledRateRule(inner, 2.0))
    # Same thresholds, rates doubled: every arrival time halves, so the front covers twice the
    # distance in the same time.
    both = ~np.isnan(slow.arrival_time) & ~np.isnan(fast.arrival_time) & (slow.arrival_time > 0)
    assert both.sum() > 20
    ratio = fast.arrival_time[both] / slow.arrival_time[both]
    np.testing.assert_allclose(ratio, 0.5, rtol=1e-6)
    centre = np.array([30, 30])

    def reach(arrival: np.ndarray, t: float) -> float:
        rows, cols = np.nonzero(arrival <= t)
        return float(np.hypot(rows - centre[0], cols - centre[1]).max())

    t = cfg.duration_s / 2
    assert reach(fast.arrival_time, t) == pytest.approx(reach(slow.arrival_time, 2 * t), abs=1e-9)
    assert reach(fast.arrival_time, t) == pytest.approx(2 * reach(slow.arrival_time, t), rel=0.35)


class _NegativeRule:
    name = "negative"

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Any) -> SpreadOutput:
        h, w = grid.shape
        return SpreadOutput(np.full((stencil.size, h, w), 0.1), np.zeros((h, w)), np.zeros((h, w)))


def test_scaled_rate_rule_output_validates() -> None:
    grid = build_grid(make_config().grid)
    out = ScaledRateRule(_NegativeRule(), 3.0).rate_of_spread(grid, Weather(), STENCIL_8)
    out.validate(STENCIL_8, grid.shape)
    np.testing.assert_allclose(out.ros, 0.3)


# ----------------------------------------------------------------------------- ignition jitter


def _town_grid() -> Grid:
    cfg = make_config(generator="with_town", grid_params={"size": 7})
    grid = build_grid(cfg.grid)
    grid.apply_unburnable()
    return grid


def test_jitter_stays_inside_grid_and_on_fuel() -> None:
    grid = _town_grid()
    cells = [(20, 16), (0, 0), (40, 40), (16, 20)]  # next to the town and in the corners
    rng = np.random.default_rng(np.random.SeedSequence(3))
    moved = 0
    for _ in range(500):
        out, offsets = jitter_ignition(rng, cells, grid, 3)
        assert len(out) == len(cells) and offsets.shape == (len(cells), 2)
        for (r, c), (r0, c0), (dr, dc) in zip(out, cells, offsets.tolist(), strict=True):
            assert 0 <= r < 41 and 0 <= c < 41
            assert grid.state[r, c] == int(CellState.FUEL)
            assert abs(r - r0) <= 3 and abs(c - c0) <= 3
            assert (r - r0, c - c0) == (dr, dc)
            moved += (r, c) != (r0, c0)
    assert moved > 0


def test_jitter_zero_is_identity() -> None:
    grid = _town_grid()
    out, offsets = jitter_ignition(np.random.default_rng(0), [(5, 5)], grid, 0)
    assert out == [(5, 5)]
    assert not offsets.any()


def test_ensemble_with_jitter_lights_fuel_only() -> None:
    cfg = make_config(
        generator="with_town",
        grid_params={"size": 7},
        cells=[(20, 16)],
        duration=60.0,
        ensemble={"ignition_jitter_cells": 3},
    )
    grid = _town_grid()
    res = run_ensemble(cfg, n_runs=6, base_seed=0)
    starts = set()
    for i, p in enumerate(res.perturbations):
        lit = np.argwhere(res.arrival_times[i] == 0.0)
        assert len(lit) == 1
        r, c = lit[0]
        assert grid.state[r, c] == int(CellState.FUEL)
        assert (r - 20, c - 16) == (p["ignition_drow_0"], p["ignition_dcol_0"])
        starts.add((int(r), int(c)))
    assert len(starts) > 1


# ----------------------------------------------------------------------------- burn probability


def test_burn_probability_matches_arrivals_and_is_zero_on_water() -> None:
    cfg = make_config(
        generator="with_river",
        grid_params={"col": 25, "width": 2},
        cells=[(20, 18)],
        duration=1800.0,
        ensemble={"ros_multiplier_sigma_frac": 0.3},
    )
    res = run_ensemble(cfg, n_runs=4, base_seed=0)
    p = res.burn_probability
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_array_equal(p, np.mean(~np.isnan(res.arrival_times), axis=0))
    grid = build_grid(cfg.grid)
    assert np.all(p[~grid.burnable_mask()] == 0.0)
    assert p.max() == 1.0


# ----------------------------------------------------------------------------- statistics


def test_running_mean() -> None:
    np.testing.assert_allclose(running_mean(np.array([1.0, 2.0, 3.0, 4.0])), [1.0, 1.5, 2.0, 2.5])
    with pytest.raises(ValueError):
        running_mean(np.array([]))


def test_mean_ci_basic() -> None:
    m, lo, hi = mean_ci(np.array([1.0, 2.0, 3.0]), level=0.95)
    assert m == pytest.approx(2.0)
    # t(0.975, 2) = 4.302653, s = 1
    assert hi - m == pytest.approx(4.302653 / math.sqrt(3), rel=1e-5)
    assert m - lo == pytest.approx(hi - m)
    assert mean_ci(np.array([5.0, 5.0])) == (5.0, 5.0, 5.0)


def test_mean_ci_coverage(rng: np.random.Generator) -> None:
    hits = 0
    trials = 2000
    for _ in range(trials):
        _, lo, hi = mean_ci(rng.normal(3.0, 2.0, size=10), level=0.9)
        hits += lo <= 3.0 <= hi
    assert hits / trials == pytest.approx(0.9, abs=0.02)


def test_mean_ci_narrower_with_more_data(rng: np.random.Generator) -> None:
    sample = rng.normal(0.0, 1.0, size=400)
    _, lo_small, hi_small = mean_ci(sample[:20])
    _, lo_big, hi_big = mean_ci(sample)
    assert hi_big - lo_big < 0.5 * (hi_small - lo_small)


@pytest.mark.parametrize(
    "values,level", [(np.array([1.0]), 0.95), (np.array([1.0, 2.0]), 0.0), (np.array([1.0, 2.0]), 1.0)]
)
def test_mean_ci_validates(values: np.ndarray, level: float) -> None:
    with pytest.raises(ValueError):
        mean_ci(values, level)


def test_runs_for_precision(rng: np.random.Generator) -> None:
    values = rng.normal(10.0, 1.0, size=50)
    n = runs_for_precision(values, rel_half_width=0.01, level=0.95)
    m, s = abs(float(np.mean(values))), float(np.std(values, ddof=1))

    def half(k: int) -> float:
        return float(stats.t.ppf(0.975, k - 1)) * s / math.sqrt(k)

    assert half(n) <= 0.01 * m < half(n - 1)
    assert 300 < n < 500  # ~ (1.96 * 1 / 0.1)^2 = 384
    assert runs_for_precision(values, rel_half_width=2.0) == 2  # already precise at n = 2
    assert runs_for_precision(np.array([3.0, 3.0, 3.0]), 0.01) == 2


@pytest.mark.parametrize(
    "values,rel", [(np.array([1.0, 2.0]), 0.0), (np.array([1.0]), 0.1), (np.array([-1.0, 1.0]), 0.1)]
)
def test_runs_for_precision_validates(values: np.ndarray, rel: float) -> None:
    with pytest.raises(ValueError):
        runs_for_precision(values, rel)
