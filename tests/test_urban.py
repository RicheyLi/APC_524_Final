"""Tests for burnable urban cells (WP 1.13, docs/plan.md section 2.6).

Urban cells burn like wildland cells, scaled: they carry a surrogate fuel model, the simulator
multiplies their rate of spread by ``ros_factor`` after the spread rule has run and their
residence time by ``residence_factor`` (``configs/urban.yaml``: TL3, 0.5 and 30, all three
team assumptions).

The crossing-time tests use the set-up of physics test (g) in ``tests/test_physics.py``: a
planar front (line ignition along column 0) meets a strip five cells wide, here of urban cells
in a landscape of the surrogate fuel. With harmonic-mean link rates, crossing the strip takes
``w dx / (f R)`` where plain fuel takes ``w dx / R``, so the front arrives beyond it
``(1 / f - 1) w dx / R`` late.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import numpy as np
import pytest

from wildfire_sim import sim
from wildfire_sim.config import ScenarioConfig
from wildfire_sim.data.synthetic import flat_uniform, with_town
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import load_residence_times, run
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.types import CellState, LandUse, Stencil, get_stencil
from wildfire_sim.urban import UrbanModel, load_urban_model, residence_scale, ros_scale, urban_mask

DX, ROS = 30.0, 0.2
SURROGATE = 183  # TL3, configs/urban.yaml
STRIP_C0, STRIP_WIDTH = 20, 5  # urban strip in columns [20, 25)
DETERMINISTIC = 1e6

_FUEL, _BURNING, _BURNED, _UNBURNABLE = (
    int(CellState.FUEL),
    int(CellState.BURNING),
    int(CellState.BURNED),
    int(CellState.UNBURNABLE),
)


@pytest.fixture(params=["numpy", "numba"])
def engine(request: pytest.FixtureRequest) -> str:
    if request.param == "numba" and importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    return request.param


def urban_model(ros_factor: float = 0.5, residence_factor: float = 30.0, fuel_model: float = SURROGATE) -> UrbanModel:
    return UrbanModel.model_validate(
        {
            "fuel_model": {"value": fuel_model, "source": "test"},
            "ros_factor": {"value": ros_factor, "source": "test"},
            "residence_factor": {"value": residence_factor, "source": "test"},
        }
    )


def use_urban(monkeypatch: pytest.MonkeyPatch, **kwargs: float) -> None:
    """Make every run use the given urban factors instead of ``configs/urban.yaml``."""
    model = urban_model(**kwargs)
    monkeypatch.setattr(sim, "load_urban_model", lambda path=None: model)


def strip_grid(urban: bool, shape: tuple[int, int] = (21, 60)) -> Grid:
    """Surrogate fuel everywhere; with ``urban``, columns [20, 25) are urban cells of the same fuel."""
    grid = flat_uniform(shape, DX, fuel_model=SURROGATE)
    if urban:
        grid.land_use[:, STRIP_C0 : STRIP_C0 + STRIP_WIDTH] = int(LandUse.URBAN)
    return grid


def strip_config(
    engine: str = "numpy",
    *,
    stencil: int = 16,
    arrival_shape: float = 200.0,
    duration: float = 16000.0,
    spread: dict[str, Any] | None = None,
    shape: tuple[int, int] = (21, 60),
) -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "name": "urban_strip",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": list(shape), "dx": DX},
            "spread": {**(spread or {"rule": "constant", "params": {"ros": ROS}}), "arrival_shape": arrival_shape},
            "ignition": {"cells": [[r, 0] for r in range(shape[0])]},
            "duration_s": duration,
            "stencil": stencil,
            "engine": engine,
        }
    )


def strip_delay(cfg: ScenarioConfig, seeds: range) -> float:
    """Mean delay (s) at column 40 of the run with the urban strip against the same seed without it."""
    delays = []
    for seed in seeds:
        town = run(cfg, seed=seed, grid=strip_grid(True), record_events=False)
        plain = run(cfg, seed=seed, grid=strip_grid(False), record_events=False)
        assert not np.isnan(town.arrival_time[:, 40]).any()
        delays.append(float(np.mean(town.arrival_time[:, 40] - plain.arrival_time[:, 40])))
    return float(np.mean(delays))


# ----------------------------------------------------------------------------- the contracts


def test_urban_land_use_is_burnable_and_water_and_roads_are_not() -> None:
    assert LandUse.URBAN.burnable and LandUse.WILDLAND.burnable
    assert not LandUse.WATER.burnable and not LandUse.ROAD.burnable


def test_burnable_mask_follows_land_use() -> None:
    grid = Grid.empty((2, 4), DX)
    grid.fuel_model[:] = SURROGATE
    grid.land_use[0] = [int(LandUse.WILDLAND), int(LandUse.URBAN), int(LandUse.WATER), int(LandUse.ROAD)]
    grid.land_use[1] = int(LandUse.URBAN)
    grid.fuel_model[1, 0] = 0  # urban without a fuel model
    grid.fireline[1, 1] = True  # a line built through a town
    np.testing.assert_array_equal(grid.burnable_mask(), [[True, True, False, False], [False, False, True, True]])
    grid.apply_unburnable()
    np.testing.assert_array_equal(grid.state == _FUEL, grid.burnable_mask())
    np.testing.assert_array_equal(urban_mask(grid), [[False, True, False, False], [True, True, True, True]])


def test_shipped_urban_model_is_the_plans_placeholder() -> None:
    """docs/plan.md section 13: TL3, ros_factor 0.5, residence_factor 30, each flagged as an assumption."""
    model = load_urban_model()
    assert model.surrogate_fuel_model == SURROGATE
    assert model.ros_factor.value == 0.5
    assert model.residence_factor.value == 30.0
    assert model.burnable
    for cited in (model.fuel_model, model.ros_factor, model.residence_factor):
        assert "PLACEHOLDER" in cited.source and "team assumption" in cited.source


@pytest.mark.parametrize(
    "bad",
    [
        {"ros_factor": -0.1},
        {"ros_factor": 1.5},
        {"ros_factor": float("nan")},
        {"residence_factor": 0.5},
        {"residence_factor": float("inf")},
        {"fuel_model": 0},
        {"fuel_model": 183.5},
    ],
)
def test_urban_model_rejects_values_out_of_range(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="urban"):
        urban_model(**bad)


def test_scale_fields() -> None:
    grid = strip_grid(True, (3, 30))
    model = urban_model(0.25, 12.0)
    ros, residence = ros_scale(grid, model), residence_scale(grid, model)
    assert ros.shape == residence.shape == (3, 30)
    assert np.all(ros[:, 20:25] == 0.25) and np.all(residence[:, 20:25] == 12.0)
    assert np.all(np.delete(ros, np.s_[20:25], axis=1) == 1.0)
    assert np.all(np.delete(residence, np.s_[20:25], axis=1) == 1.0)
    assert not urban_model(0.0).burnable


def test_with_town_writes_the_surrogate_fuel_model() -> None:
    grid = with_town((21, 21), size=5)
    town = urban_mask(grid)
    assert town.sum() == 25
    assert np.all(grid.fuel_model[town] == SURROGATE) and np.all(grid.fuel_model[~town] == 1)
    assert np.all(grid.state == _FUEL)
    assert grid.metadata["source"] == "synthetic:with_town"


# ----------------------------------------------------------------------------- acceptance: a town burns


def town_config(engine: str, duration: float = 30000.0) -> ScenarioConfig:
    """31 x 41 cells of short grass with a 7 x 7 town in the middle; ignition at the west edge."""
    return ScenarioConfig.model_validate(
        {
            "name": "town",
            "grid": {
                "kind": "synthetic",
                "generator": "with_town",
                "shape": [31, 41],
                "dx": DX,
                "params": {"size": 7, "value_per_cell": 400_000.0},
            },
            "spread": {"rule": "constant", "params": {"ros": ROS}},
            "ignition": {"cells": [[15, 2]]},
            "duration_s": duration,
            "engine": engine,
        }
    )


def test_fire_burns_through_a_town(engine: str) -> None:
    """WP 1.13 acceptance: fire crosses the town of ``with_town`` and every town cell it reaches is BURNED.

    The whole grid burns, the town included, and the run ends by itself once the last urban
    cell has stopped flaming. Every town cell burns later than the same cell would as grass
    (same seed, so the same link thresholds), and along the ignition row the front moves
    through the town's west half at half speed.
    """
    cfg = town_config(engine)
    result = run(cfg)
    town = urban_mask(result.grid)
    assert town.sum() == 49
    assert result.burned_mask.all()
    assert np.all(result.grid.state == _BURNED)
    assert result.provenance["n_steps"] < cfg.n_steps  # burned out before the time limit
    assert float(result.grid.asset_value[result.burned_mask].sum()) == 49 * 400_000.0

    arrival = result.arrival_time
    grass = run(cfg, grid=flat_uniform((31, 41), DX)).arrival_time
    assert np.all(arrival[town] > grass[town])
    assert np.all(arrival[:, :16] == grass[:, :16])  # nothing changes upwind of the town
    # along the ignition row, each town cell west of the core burns after its western neighbour
    assert np.all(np.diff(arrival[15, 16:21]) > 0)
    # half speed inside: 3 cells from the town's west edge to its core take about twice 3 dx / R
    assert arrival[15, 20] - arrival[15, 17] == pytest.approx(2 * 3 * DX / ROS, rel=0.10)
    assert grass[15, 20] - grass[15, 17] == pytest.approx(3 * DX / ROS, rel=0.10)


def test_town_cells_are_burned_as_the_fire_passes(engine: str) -> None:
    """Mid-run: every town cell the fire has reached is BURNING or BURNED, and the rest are still FUEL."""
    result = run(town_config(engine, duration=3600.0))
    town = urban_mask(result.grid)
    reached = town & result.burned_mask
    assert 0 < reached.sum() < town.sum()
    assert np.all(np.isin(result.grid.state[reached], [_BURNING, _BURNED]))
    assert np.all(result.grid.state[town & ~result.burned_mask] == _FUEL)


# ----------------------------------------------------------------------------- acceptance: crossing time


@pytest.mark.parametrize("stencil", [8, 16])
def test_urban_strip_takes_one_over_ros_factor_times_as_long_to_cross(stencil: int) -> None:
    """WP 1.13 acceptance: with ``ros_factor = 0.5`` the strip takes twice as long, within 10 %.

    Plain surrogate fuel is crossed in ``w dx / R = 750 s``, the urban strip in ``1500 s``, so
    the front beyond it is 750 s late. Tolerance 10 % on the delay, as in physics test (g);
    default kernel (shape 200), 5 seeds. Measured: -3.1 % (8 neighbours), +1.0 % (16).
    """
    plain_s = STRIP_WIDTH * DX / ROS
    delay = strip_delay(strip_config(stencil=stencil), range(5))
    assert delay == pytest.approx(plain_s * (1 / 0.5 - 1), rel=0.10)
    assert (delay + plain_s) / plain_s == pytest.approx(1 / 0.5, rel=0.10)


@pytest.mark.parametrize("ros_factor", [0.5, 0.25, 1.0])
@pytest.mark.parametrize("stencil", [8, 16])
def test_urban_strip_delay_is_exact_without_noise(
    engine: str, stencil: int, ros_factor: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic limit: the delay is ``(1 / f - 1) w dx / R`` within 0.1 %, for any factor.

    ``f = 1`` gives no delay at all: an urban cell then spreads exactly like its surrogate fuel.
    """
    use_urban(monkeypatch, ros_factor=ros_factor)
    delay = strip_delay(strip_config(engine, stencil=stencil, arrival_shape=DETERMINISTIC), range(1))
    predicted = (1 / ros_factor - 1) * STRIP_WIDTH * DX / ROS
    assert delay == pytest.approx(predicted, rel=1e-3, abs=1e-6)


def test_every_rule_is_scaled_the_same_way() -> None:
    """The factor is applied after the spread rule: the Alexandridis rule gets the same relative delay.

    Deterministic limit, no wind, flat: the rule's own eastward rate ``R`` for the surrogate
    fuel gives the predicted delay ``(1 / f - 1) w dx / R``.
    """
    spread = {"rule": "alexandridis", "params": {"ros_base": 0.2}}
    cfg = strip_config(arrival_shape=DETERMINISTIC, spread=spread, duration=40000.0)
    rule = make_spread_rule("alexandridis", ros_base=0.2)
    out = rule.rate_of_spread(strip_grid(False), Weather(), get_stencil(16))
    east = int(np.argmin(np.abs(get_stencil(16).bearings_deg - 90.0)))
    rate = float(out.ros[east, 10, 30])
    assert 0 < rate < ROS  # timber litter under the Alexandridis factors: slower than the base rate
    assert strip_delay(cfg, range(1)) == pytest.approx((1 / 0.5 - 1) * STRIP_WIDTH * DX / rate, rel=1e-3)


def test_rule_output_is_not_modified() -> None:
    """A rule may return the same arrays every call; the urban scaling must work on a copy."""

    class Cached:
        name = "cached"

        def __init__(self) -> None:
            self.out: SpreadOutput | None = None

        def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
            if self.out is None:
                h, w = grid.shape
                cell = np.where(grid.burnable_mask(), ROS, 0.0)
                self.out = SpreadOutput(
                    ros=np.broadcast_to(cell, (stencil.size, h, w)).copy(),
                    intensity=np.zeros((h, w)),
                    flame_length=np.zeros((h, w)),
                )
            return self.out

    rule = Cached()
    run(strip_config(duration=3000.0), grid=strip_grid(True), rule=rule)
    assert rule.out is not None and np.all(rule.out.ros == ROS)


# ----------------------------------------------------------------------------- acceptance: flaming time


def test_urban_cells_flame_residence_factor_times_as_long(engine: str) -> None:
    """WP 1.13 acceptance: flaming time = crossing time + ``residence_factor`` x the fuel's residence time.

    For an urban cell of the surrogate fuel: ``dx / (0.5 R) = 300 s`` to cross plus ``30 x`` the
    residence time of TL3 (placeholder 120 s) = 3900 s; for the same fuel as wildland,
    ``150 + 120 = 270 s``. Checked on cells still flaming at the end of the run, where
    ``burn_timer + (T - arrival)`` is the flaming time given at ignition (float32 layer: 1e-5).
    """
    residence = float(load_residence_times().lookup(np.array([SURROGATE]))[0])
    cfg = strip_config(engine, duration=5400.0, shape=(9, 60))
    result = run(cfg, grid=strip_grid(True, (9, 60)))
    town = urban_mask(result.grid)
    flaming = (result.grid.state == _BURNING) & (result.grid.burn_timer > 0)
    given = result.grid.burn_timer + (cfg.duration_s - result.arrival_time)

    assert (flaming & town).sum() >= 20 and (flaming & ~town).sum() >= 5
    np.testing.assert_allclose(given[flaming & town], DX / (0.5 * ROS) + 30.0 * residence, rtol=1e-5)
    np.testing.assert_allclose(given[flaming & ~town & (np.arange(60) > 26)], DX / ROS + residence, rtol=1e-5)
    assert (30.0 * residence) / residence == 30.0


def test_urban_cells_outlast_the_wildland_around_them(engine: str) -> None:
    """At 6000 s the whole town is still flaming (it lit between 2250 s and 4200 s and flames for
    3900 s) and the grass the fire came through has long burned out."""
    result = run(town_config(engine, duration=6000.0))
    town = urban_mask(result.grid)
    burning = result.grid.state == _BURNING
    assert burning[town].all() and np.all(result.grid.burn_timer[town] > 0)
    west = np.zeros_like(town)
    west[:, :10] = True
    assert not burning[west].any()


# ----------------------------------------------------------------------------- acceptance: ros_factor 0


def test_with_ros_factor_zero_the_town_holds(engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """WP 1.13 acceptance: ``ros_factor = 0`` is the behaviour before: no urban cell burns.

    Urban cells are UNBURNABLE for the run, the fire burns everything else, and the run ends
    without waiting on the town (its neighbours are not dormant sources).
    """
    use_urban(monkeypatch, ros_factor=0.0)
    cfg = town_config(engine)
    result = run(cfg)
    town = urban_mask(result.grid)
    assert not result.burned_mask[town].any()
    assert np.all(result.grid.state[town] == _UNBURNABLE)
    assert result.burned_mask[~town].all()
    assert not np.any(result.grid.state == _BURNING)
    assert result.provenance["final_time_s"] < 9000.0
    assert float(result.grid.asset_value[result.burned_mask].sum()) == 0.0


def test_with_ros_factor_zero_an_ignition_in_town_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    use_urban(monkeypatch, ros_factor=0.0)
    cfg = town_config("numpy")
    cfg = cfg.model_copy(update={"ignition": cfg.ignition.model_copy(update={"cells": [(15, 20)]})})
    with pytest.raises(ValueError, match="not burnable fuel"):
        run(cfg)


def test_fire_can_start_in_town() -> None:
    cfg = town_config("numpy", duration=1800.0)
    cfg = cfg.model_copy(update={"ignition": cfg.ignition.model_copy(update={"cells": [(15, 20)]})})
    result = run(cfg)
    assert result.burned_mask[15, 20] and result.burned_mask.sum() > 5


# ----------------------------------------------------------------------------- what must not change


def test_runs_without_urban_cells_do_not_depend_on_the_urban_factors(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No urban cell on the grid: bit-identical results whatever ``configs/urban.yaml`` says."""
    cfg = strip_config(engine, duration=3000.0)
    base = run(cfg, grid=strip_grid(False), seed=2)
    use_urban(monkeypatch, ros_factor=0.1, residence_factor=99.0)
    other = run(cfg, grid=strip_grid(False), seed=2)
    np.testing.assert_array_equal(base.arrival_time, other.arrival_time)
    np.testing.assert_array_equal(base.grid.burn_timer, other.grid.burn_timer)
    np.testing.assert_array_equal(base.grid.state, other.grid.state)


def test_water_and_roads_still_never_burn(engine: str) -> None:
    grid = flat_uniform((15, 31), DX)
    grid.land_use[:, 10] = int(LandUse.ROAD)
    grid.land_use[:, 11] = int(LandUse.ROAD)
    grid.land_use[:, 20:22] = int(LandUse.WATER)
    cfg = ScenarioConfig.model_validate(
        {
            "name": "barriers",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [15, 31]},
            "spread": {"rule": "constant", "params": {"ros": ROS}},
            "ignition": {"cells": [[7, 2]]},
            "duration_s": 9000.0,
            "engine": engine,
        }
    )
    result = run(cfg, grid=grid)
    assert result.burned_mask[:, :10].all()
    assert not result.burned_mask[:, 10:].any()


def test_engines_agree_on_a_town() -> None:
    pytest.importorskip("numba")
    ref, fast = run(town_config("numpy", 6000.0), seed=3), run(town_config("numba", 6000.0), seed=3)
    np.testing.assert_array_equal(ref.grid.state, fast.grid.state)
    np.testing.assert_allclose(fast.arrival_time, ref.arrival_time, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(fast.grid.burn_timer, ref.grid.burn_timer, rtol=1e-6, atol=1e-6)


def test_urban_parameters_are_in_the_provenance() -> None:
    result = run(town_config("numpy", duration=300.0))
    urban = result.provenance["urban"]
    assert urban["fuel_model"]["value"] == SURROGATE and urban["ros_factor"]["value"] == 0.5
    assert urban["residence_factor"]["value"] == 30.0
    assert sim.model_provenance(town_config("numpy"))["urban"] == urban
    json.dumps(result.provenance, allow_nan=False)
