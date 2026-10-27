"""Shared fixtures. All tests are offline and deterministic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wildfire_sim.config import ScenarioConfig, load_scenario
from wildfire_sim.data.synthetic import flat_uniform
from wildfire_sim.grid import Grid, Weather

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-network", action="store_true", default=False, help="run tests marked 'network'")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def small_grid() -> Grid:
    """21 x 21 flat uniform grid at 30 m, fuel model 1."""
    return flat_uniform((21, 21), 30.0, fuel_model=1)


@pytest.fixture
def calm_weather() -> Weather:
    return Weather()


@pytest.fixture
def windy_weather() -> Weather:
    """5 m/s from the west (blowing toward east, 90 deg)."""
    return Weather(wind_speed=5.0, wind_direction=270.0)


@pytest.fixture
def flat_scenario() -> ScenarioConfig:
    return load_scenario(CONFIGS / "scenarios" / "flat_no_wind.yaml")


@pytest.fixture
def fixture_grid_path() -> Path:
    return FIXTURES / "flat_21x21.npz"
