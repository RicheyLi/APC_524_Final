"""Tests for wildfire_sim.config: schema validation and YAML round-trips."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from wildfire_sim.config import (
    CostConfig,
    ScenarioConfig,
    SyntheticGridConfig,
    configs_dir,
    dump_yaml,
    load_costs,
    load_scenario,
    load_yaml,
)
from wildfire_sim.grid import WeatherSeries

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("name", ["flat_no_wind", "steady_wind", "river_barrier"])
def test_shipped_scenarios_validate(name: str) -> None:
    cfg = load_scenario(CONFIGS / "scenarios" / f"{name}.yaml")
    assert cfg.name == name
    assert cfg.n_steps == int(cfg.duration_s // cfg.dt_s)


def test_shipped_costs_validate() -> None:
    costs = load_costs(CONFIGS / "costs.yaml")
    assert costs.crew_usd_per_h.value > 0
    assert costs.crew_usd_per_h.source


def test_roundtrip_through_yaml(flat_scenario: ScenarioConfig, tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    dump_yaml(flat_scenario, p)
    again = load_scenario(p)
    assert again == flat_scenario


def test_minimal_scenario_defaults() -> None:
    cfg = ScenarioConfig.model_validate(
        {"name": "m", "grid": {"kind": "synthetic", "generator": "flat_uniform"}, "ignition": {"center": True}}
    )
    assert isinstance(cfg.grid, SyntheticGridConfig)
    assert cfg.spread.rule == "constant"
    assert cfg.stencil == 8
    assert cfg.suppression.policy == "none"
    assert len(cfg.weather.series) == 1
    series = cfg.weather.to_series()
    assert isinstance(series, WeatherSeries)
    assert series.at(0).wind_speed == 0.0


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"dt_s": 0}, "dt_s"),
        ({"dt_s": 100, "duration_s": 50}, "duration_s"),
        ({"stencil": 12}, "stencil"),
        ({"ignition": {}}, "ignition"),
        ({"ignition": {"center": True, "time_s": 1e9}}, "ignition.time_s"),
        ({"weather": {"series": [{"time_s": 10}]}}, "time_s = 0"),
        ({"weather": {"series": [{"time_s": 0}, {"time_s": 0}]}}, "increasing"),
        ({"grid": {"kind": "synthetic", "generator": "volcano"}}, "generator"),
        ({"grid": {"kind": "cached"}}, "path"),
        ({"bogus_key": 1}, "bogus_key"),
    ],
)
def test_invalid_scenarios_rejected(patch: dict, match: str) -> None:
    base = {"name": "m", "grid": {"kind": "synthetic", "generator": "flat_uniform"}, "ignition": {"center": True}}
    base.update(patch)
    with pytest.raises(ValidationError, match=match):
        ScenarioConfig.model_validate(base)


def test_cost_config_requires_source() -> None:
    with pytest.raises(ValidationError, match="source"):
        CostConfig.model_validate({k: {"value": 1.0} for k in CostConfig.model_fields})


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p, ScenarioConfig)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValidationError):
        load_yaml(empty, ScenarioConfig)


@pytest.mark.parametrize(
    ("dt", "duration", "expected"),
    [(30.0, 7200.0, 240), (0.1, 3600.0, 36000), (0.3, 0.9, 3), (7.0, 20.0, 2)],
)
def test_n_steps_is_robust_to_float_rounding(dt: float, duration: float, expected: int) -> None:
    cfg = ScenarioConfig.model_validate(
        {
            "name": "t",
            "grid": {"kind": "synthetic", "generator": "flat_uniform"},
            "ignition": {"center": True},
            "dt_s": dt,
            "duration_s": duration,
        }
    )
    assert cfg.n_steps == expected


def test_ignition_cells_must_lie_inside_a_synthetic_grid() -> None:
    raw: dict = {
        "name": "t",
        "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [11, 21]},
        "ignition": {"cells": [[10, 20]]},
    }
    ScenarioConfig.model_validate(raw)
    raw["ignition"] = {"cells": [[11, 0]]}
    with pytest.raises(ValidationError, match="outside the grid"):
        ScenarioConfig.model_validate(raw)


def test_arrival_shape_default_and_bounds() -> None:
    raw: dict = {"name": "t", "grid": {"kind": "synthetic", "generator": "flat_uniform"}, "ignition": {"center": True}}
    assert ScenarioConfig.model_validate(raw).spread.arrival_shape == 20.0
    raw["spread"] = {"arrival_shape": 0.5}
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(raw)


def test_costs_cover_every_resource_type() -> None:
    """Each ResourceConfig type has an hourly rate in CostConfig."""
    costs = load_costs(CONFIGS / "costs.yaml")
    for kind in ("crew", "dozer", "engine", "aircraft"):
        assert getattr(costs, f"{kind}_usd_per_h").value >= 0


def test_load_yaml_reports_malformed_yaml_as_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_scenario(bad)


def test_configs_dir_defaults_to_repo_configs_and_honours_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WILDFIRE_SIM_CONFIGS", raising=False)
    assert configs_dir() == CONFIGS
    assert (configs_dir() / "costs.yaml").is_file()
    monkeypatch.setenv("WILDFIRE_SIM_CONFIGS", str(tmp_path))
    assert configs_dir() == tmp_path
    monkeypatch.setenv("WILDFIRE_SIM_CONFIGS", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="WILDFIRE_SIM_CONFIGS"):
        configs_dir()
