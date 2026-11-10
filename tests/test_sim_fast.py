"""Tests for the compiled fire engine (WP 1.7, ``wildfire_sim.sim_fast``).

The NumPy engine in :mod:`wildfire_sim.sim` is the reference. For the same scenario and seed
the Numba engine must burn the same cells in the same final states, with arrival times equal
to ``ARRIVAL_RTOL`` relative (the two engines evaluate the same formulas, but the in-step
cascade visits cells in a different order, so bit equality is not promised). The float32 grid
layers (``burn_timer``, ``intensity``, ``arrival_time``) are compared to one float32 rounding
step on top of that (``LAYER_RTOL``).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("numba")

from wildfire_sim import sim, sim_fast  # noqa: E402
from wildfire_sim.config import ScenarioConfig, load_scenario  # noqa: E402
from wildfire_sim.data.synthetic import flat_uniform  # noqa: E402
from wildfire_sim.ensemble import run_ensemble  # noqa: E402
from wildfire_sim.grid import Grid, Weather  # noqa: E402
from wildfire_sim.sim import RunResult  # noqa: E402
from wildfire_sim.spread.base import SpreadOutput  # noqa: E402
from wildfire_sim.types import CellState, Stencil  # noqa: E402

SCENARIOS = Path(__file__).resolve().parent.parent / "configs" / "scenarios"
ARRIVAL_RTOL = 1e-9  # relative, on float64 arrival times (s)
LAYER_RTOL = 1e-6  # relative, on float32 layers: one float32 rounding step (~6e-8) plus ARRIVAL_RTOL
LAYER_ATOL = 1e-6  # s or kW/m, for values that are 0 in one engine and a rounding residue in the other


# ----------------------------------------------------------------------------- helpers


def make_config(
    *,
    shape: tuple[int, int] = (41, 41),
    generator: str = "flat_uniform",
    grid_params: dict[str, Any] | None = None,
    spread: dict[str, Any] | None = None,
    dt: float = 30.0,
    duration: float = 3600.0,
    stencil: int = 8,
    ignition: dict[str, Any] | None = None,
    weather: list[dict[str, float]] | None = None,
    engine: str = "numpy",
) -> ScenarioConfig:
    """Scenario on a synthetic 30 m grid, constant rule at 0.2 m/s unless ``spread`` says otherwise."""
    grid: dict[str, Any] = {"kind": "synthetic", "generator": generator, "shape": list(shape), "dx": 30.0}
    if grid_params:
        grid["params"] = grid_params
    data: dict[str, Any] = {
        "name": "fast",
        "grid": grid,
        "spread": spread or {"rule": "constant", "params": {"ros": 0.2}},
        "ignition": ignition or {"center": True},
        "dt_s": dt,
        "duration_s": duration,
        "stencil": stencil,
        "engine": engine,
    }
    if weather is not None:
        data["weather"] = {"series": weather}
    return ScenarioConfig.model_validate(data)


def numba_config(config: ScenarioConfig) -> ScenarioConfig:
    return config.model_copy(update={"engine": "numba"})


def assert_same_run(ref: RunResult, fast: RunResult) -> None:
    """The parity contract between the NumPy (``ref``) and Numba (``fast``) engines."""
    np.testing.assert_array_equal(ref.burned_mask, fast.burned_mask)
    np.testing.assert_allclose(fast.arrival_time, ref.arrival_time, rtol=ARRIVAL_RTOL, atol=0.0, equal_nan=True)
    np.testing.assert_array_equal(ref.grid.state, fast.grid.state)
    for layer in ("burn_timer", "intensity", "arrival_time"):
        np.testing.assert_allclose(
            getattr(fast.grid, layer),
            getattr(ref.grid, layer),
            rtol=LAYER_RTOL,
            atol=LAYER_ATOL,
            equal_nan=True,
            err_msg=layer,
        )
    assert len(ref.events) == len(fast.events)
    ref_by_cell = {(kind, r, c): t for t, kind, r, c in ref.events}
    fast_by_cell = {(kind, r, c): t for t, kind, r, c in fast.events}
    assert ref_by_cell.keys() == fast_by_cell.keys()
    for key, t in ref_by_cell.items():
        assert fast_by_cell[key] == pytest.approx(t, rel=ARRIVAL_RTOL, abs=0.0)
    assert [e[0] for e in fast.events] == sorted(e[0] for e in fast.events)
    assert ref.provenance["n_steps"] == fast.provenance["n_steps"]


def run_both(config: ScenarioConfig, seed: int, **kwargs: Any) -> tuple[RunResult, RunResult]:
    ref = sim.run(config, seed=seed, **kwargs)
    fast = sim.run(numba_config(config), seed=seed, **kwargs)
    assert ref.provenance["engine"] == "numpy"
    assert fast.provenance["engine"] == "numba"
    return ref, fast


class PatchyRule:
    """Anisotropic rule on a patchy landscape: ``ros_k = base (1 + canopy) (1 - retardant) (1 + a cos)``.

    ``a cos`` favours spread toward the east; zero on cells that cannot burn. Deterministic.
    """

    name = "patchy"

    def __init__(self, base: float = 0.15, anisotropy: float = 0.6) -> None:
        self.base, self.anisotropy = base, anisotropy

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        cell = self.base * (1.0 + grid.canopy_cover.astype(np.float64)) * (1.0 - grid.retardant.astype(np.float64))
        cell = np.where(grid.burnable_mask(), cell, 0.0)
        angle = np.arctan2(stencil.offsets[:, 0], stencil.offsets[:, 1]).astype(np.float64)  # 0 = east
        factor = 1.0 + self.anisotropy * np.cos(angle) * (1.0 + 0.1 * weather.wind_speed)
        ros = cell[None, :, :] * np.clip(factor, 0.05, None)[:, None, None]
        intensity = 100.0 * cell.astype(np.float64)
        return SpreadOutput(ros=ros, intensity=intensity, flame_length=np.zeros_like(intensity))


def patchy_grid(shape: tuple[int, int], seed: int) -> Grid:
    """Flat grid with random unburnable cells (fuel model 0), canopy and retardant patches."""
    rng = np.random.default_rng(seed)
    grid = flat_uniform(shape, 30.0)
    grid.fuel_model[rng.random(shape) < 0.12] = 0
    grid.canopy_cover[:] = rng.random(shape).astype(np.float32)
    grid.retardant[rng.random(shape) < 0.15] = np.float32(0.6)
    h, w = shape
    grid.fuel_model[h // 2, w // 2] = 1
    grid.retardant[h // 2, w // 2] = 0.0
    return grid


# ----------------------------------------------------------------------------- parity


@pytest.mark.parametrize("stencil", [8, 16])
@pytest.mark.parametrize("seed", [0, 3, 11])
def test_parity_flat_no_wind(stencil: int, seed: int) -> None:
    cfg = load_scenario(SCENARIOS / "flat_no_wind.yaml").model_copy(update={"stencil": stencil, "duration_s": 3600.0})
    ref, fast = run_both(cfg, seed)
    assert ref.burned_mask.sum() > 100
    assert_same_run(ref, fast)


@pytest.mark.parametrize("stencil", [8, 16])
def test_parity_steady_wind_on_slope(stencil: int) -> None:
    cfg = make_config(
        generator="slope_ramp",
        grid_params={"slope": 0.3, "aspect_deg": 200.0},
        spread={"rule": "alexandridis", "params": {"ros_base": 0.2}},
        weather=[{"time_s": 0.0, "wind_speed": 5.0, "wind_direction": 250.0}],
        stencil=stencil,
    )
    for seed in (1, 2):
        ref, fast = run_both(cfg, seed)
        assert ref.burned_mask.sum() > 50
        assert_same_run(ref, fast)


@pytest.mark.parametrize("stencil", [8, 16])
def test_parity_river_barrier(stencil: int) -> None:
    cfg = load_scenario(SCENARIOS / "river_barrier.yaml").model_copy(update={"stencil": stencil})
    ref, fast = run_both(cfg, 5)
    assert not fast.burned_mask[:, 60:].any()
    assert_same_run(ref, fast)


@pytest.mark.parametrize("stencil", [8, 16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parity_patchy_landscape_with_retardant(stencil: int, seed: int) -> None:
    grid = patchy_grid((45, 45), seed + 100)
    cfg = make_config(shape=(45, 45), stencil=stencil, duration=5400.0)
    ref, fast = run_both(cfg, seed, rule=PatchyRule(), grid=grid)
    assert ref.burned_mask.sum() > 30
    assert (ref.grid.state == int(CellState.UNBURNABLE)).any()
    assert_same_run(ref, fast)


@pytest.mark.parametrize("stencil", [8, 16])
@pytest.mark.parametrize(("dt", "ros"), [(5.0, 0.2), (600.0, 0.5)])
def test_parity_small_and_large_dt(stencil: int, dt: float, ros: float) -> None:
    """dt = 600 s at 0.5 m/s crosses about ten 30 m cells per step: a long in-step cascade."""
    cfg = make_config(spread={"rule": "constant", "params": {"ros": ros}}, dt=dt, duration=3000.0, stencil=stencil)
    ref, fast = run_both(cfg, 4)
    assert_same_run(ref, fast)


@pytest.mark.parametrize("stencil", [8, 16])
def test_parity_delayed_ignition(stencil: int) -> None:
    cfg = make_config(ignition={"center": True, "time_s": 95.0}, dt=20.0, duration=1800.0, stencil=stencil)
    ref, fast = run_both(cfg, 6)
    assert fast.arrival_time[20, 20] == 95.0
    assert fast.events[0] == (95.0, "ignition", 20, 20)
    assert_same_run(ref, fast)


@pytest.mark.parametrize("stencil", [8, 16])
def test_parity_weather_change_mid_run(stencil: int) -> None:
    cfg = make_config(
        spread={"rule": "alexandridis", "params": {"ros_base": 0.2}},
        weather=[
            {"time_s": 0.0, "wind_speed": 2.0, "wind_direction": 270.0},
            {"time_s": 1500.0, "wind_speed": 7.0, "wind_direction": 180.0},
        ],
        duration=3600.0,
        stencil=stencil,
    )
    ref, fast = run_both(cfg, 7)
    assert_same_run(ref, fast)


@pytest.mark.parametrize("stencil", [8, 16])
def test_parity_hook_builds_fireline_and_extinguishes(stencil: int) -> None:
    wall = 28

    def build_line(grid: Grid, t: float, dt: float, rng: np.random.Generator) -> bool:
        if t == 600.0:
            grid.fireline[:, wall] = True
            grid.apply_unburnable()
            return True
        if t == 900.0:  # put out the burning cells in the top rows
            top = grid.state[:10] == int(CellState.BURNING)
            grid.state[:10][top] = int(CellState.BURNED)
        return False

    cfg = make_config(ignition={"cells": [[20, 3]]}, duration=6000.0, stencil=stencil)
    ref, fast = run_both(cfg, 8, hooks=[build_line])
    assert not fast.burned_mask[:, wall:].any()
    assert_same_run(ref, fast)


@pytest.mark.parametrize("stencil", [8, 16])
def test_parity_max_source_age(stencil: int) -> None:
    cfg = make_config(stencil=stencil, duration=3600.0)
    for age in (200.0, 400.0):
        ref, fast = run_both(cfg, 9, max_source_age_s=age)
        assert_same_run(ref, fast)


def test_parity_without_events() -> None:
    cfg = make_config(duration=1800.0)
    ref, fast = run_both(cfg, 10, record_events=False)
    assert fast.events == [] == ref.events
    assert_same_run(ref, fast)
    with_events = sim.run(numba_config(cfg), seed=10)
    np.testing.assert_array_equal(with_events.arrival_time, fast.arrival_time)
    assert len(with_events.events) == int(with_events.burned_mask.sum())


def test_numba_engine_is_deterministic_and_does_not_mutate_passed_grid() -> None:
    grid = patchy_grid((31, 31), 1)
    before = grid.copy()
    cfg = make_config(shape=(31, 31), engine="numba")
    a = sim.run(cfg, seed=2, rule=PatchyRule(), grid=grid)
    b = sim.run(cfg, seed=2, rule=PatchyRule(), grid=grid)
    assert grid == before
    np.testing.assert_array_equal(a.arrival_time, b.arrival_time)
    assert a.grid == b.grid
    assert a.events == b.events


# ----------------------------------------------------------------------------- selection and provenance


def test_engine_from_yaml_is_honoured_and_recorded(tmp_path: Path) -> None:
    text = (SCENARIOS / "flat_no_wind.yaml").read_text(encoding="utf-8") + "engine: numba\n"
    path = tmp_path / "fast.yaml"
    path.write_text(text, encoding="utf-8")
    cfg = load_scenario(path).model_copy(update={"duration_s": 600.0})
    assert cfg.engine == "numba"
    result = sim.run(cfg, seed=1)
    assert result.provenance["engine"] == "numba"
    assert result.provenance["config"]["engine"] == "numba"
    assert "numba" in result.provenance["versions"]
    assert sim.model_provenance(cfg)["engine"] == "numba"


def test_default_engine_is_numpy() -> None:
    cfg = load_scenario(SCENARIOS / "flat_no_wind.yaml").model_copy(update={"duration_s": 300.0})
    assert cfg.engine == "numpy"
    result = sim.run(cfg)
    assert result.provenance["engine"] == "numpy"
    assert "numba" not in result.provenance["versions"]


def test_missing_numba_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "numba", None)  # makes ``import numba`` raise ImportError
    monkeypatch.setattr(sim_fast, "_KERNELS", None)
    cfg = make_config(duration=300.0)
    with pytest.raises(ImportError, match=r"uv sync --extra fast"):
        sim.run(numba_config(cfg))
    assert sim.run(cfg).burned_mask.sum() >= 1  # the NumPy engine does not need numba


def test_package_and_cli_import_without_numba() -> None:
    code = (
        "import sys; sys.modules['numba'] = None\n"
        "import wildfire_sim, wildfire_sim.sim, wildfire_sim.cli, wildfire_sim.ensemble\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# ----------------------------------------------------------------------------- ensemble


@pytest.mark.parametrize("n_workers", [1, 2])
def test_ensemble_with_numba_engine_matches_numpy(n_workers: int) -> None:
    data = make_config(
        spread={"rule": "alexandridis", "params": {"ros_base": 0.2}},
        weather=[{"time_s": 0.0, "wind_speed": 4.0, "wind_direction": 270.0}],
        duration=1800.0,
    ).model_dump()
    data["ensemble"] = {"wind_speed_sigma_frac": 0.2, "wind_direction_sigma_deg": 10.0}
    cfg = ScenarioConfig.model_validate(data)
    ref = run_ensemble(cfg, n_runs=3, base_seed=4, n_workers=1)
    fast = run_ensemble(numba_config(cfg), n_runs=3, base_seed=4, n_workers=n_workers)
    assert fast.seeds == ref.seeds
    assert fast.perturbations == ref.perturbations
    np.testing.assert_array_equal(fast.burn_probability, ref.burn_probability)
    np.testing.assert_array_equal(fast.burned_area_m2, ref.burned_area_m2)
    np.testing.assert_allclose(fast.arrival_times, ref.arrival_times, rtol=LAYER_RTOL, atol=0.0, equal_nan=True)


# ----------------------------------------------------------------------------- speed


@pytest.mark.slow
def test_numba_is_not_slower_than_numpy_after_warmup() -> None:
    """A sanity check, not the benchmark (scripts/bench.py): generous margin, best of two runs."""
    cfg = make_config(shape=(151, 151), duration=10800.0)
    fast_cfg = numba_config(cfg)
    sim.run(fast_cfg, seed=0, record_events=False)  # JIT compile / cache load

    def best(config: ScenarioConfig) -> float:
        times = []
        for _ in range(2):
            start = time.perf_counter()
            sim.run(config, seed=0, record_events=False)
            times.append(time.perf_counter() - start)
        return min(times)

    assert best(fast_cfg) < best(cfg)
