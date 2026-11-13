"""Tests for wildfire_sim.config: schema validation and YAML round-trips."""

from datetime import UTC, datetime, timedelta
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
    assert cfg.stencil == 16
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
    assert ScenarioConfig.model_validate(raw).spread.arrival_shape == 200.0
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


# ----------------------------------------------------------------------------- ensemble section (WP 1.6)

_MINIMAL = {"name": "m", "grid": {"kind": "synthetic", "generator": "flat_uniform"}, "ignition": {"center": True}}


@pytest.mark.parametrize("name", ["flat_no_wind", "steady_wind", "river_barrier"])
def test_shipped_scenarios_get_a_neutral_ensemble_section(name: str) -> None:
    from wildfire_sim.config import EnsembleConfig

    cfg = load_scenario(CONFIGS / "scenarios" / f"{name}.yaml")
    assert cfg.ensemble == EnsembleConfig()


def test_ensemble_defaults_are_neutral() -> None:
    cfg = ScenarioConfig.model_validate(_MINIMAL)
    ens = cfg.ensemble
    assert ens.wind_speed_sigma_frac == 0.0
    assert ens.wind_direction_sigma_deg == 0.0
    assert ens.moisture_sigma_frac == 0.0
    assert ens.ros_multiplier_sigma_frac == 0.0
    assert ens.ignition_jitter_cells == 0
    assert ens.n_runs == 50 and ens.base_seed == 0
    assert not ens.perturbs


def test_ensemble_section_roundtrips_through_yaml(tmp_path: Path) -> None:
    p = tmp_path / "e.yaml"
    p.write_text(
        "name: e\n"
        "grid: {kind: synthetic, generator: flat_uniform}\n"
        "ignition: {center: true}\n"
        "ensemble:\n"
        "  wind_speed_sigma_frac: 0.2\n"
        "  wind_direction_sigma_deg: 15\n"
        "  moisture_sigma_frac: 0.1\n"
        "  ros_multiplier_sigma_frac: 0.25\n"
        "  ignition_jitter_cells: 2\n"
        "  n_runs: 100\n"
        "  base_seed: 7\n",
        encoding="utf-8",
    )
    cfg = load_scenario(p)
    assert cfg.ensemble.wind_direction_sigma_deg == 15.0
    assert cfg.ensemble.ignition_jitter_cells == 2
    assert cfg.ensemble.perturbs
    out = tmp_path / "again.yaml"
    dump_yaml(cfg, out)
    assert load_scenario(out) == cfg


@pytest.mark.parametrize(
    "ensemble",
    [
        {"wind_speed_sigma_frac": -0.1},
        {"wind_direction_sigma_deg": -1},
        {"moisture_sigma_frac": -0.5},
        {"ros_multiplier_sigma_frac": -0.01},
        {"ignition_jitter_cells": -1},
        {"n_runs": 0},
        {"base_seed": -3},
        {"wind_sped_sigma_frac": 0.1},
    ],
)
def test_invalid_ensemble_sections_rejected(ensemble: dict) -> None:
    with pytest.raises(ValidationError, match="ensemble"):
        ScenarioConfig.model_validate({**_MINIMAL, "ensemble": ensemble})


# ----------------------------------------------------------------------------- engine (WP 1.7)


@pytest.mark.parametrize("name", ["flat_no_wind", "steady_wind", "river_barrier"])
def test_engine_defaults_to_numpy(name: str) -> None:
    assert load_scenario(CONFIGS / "scenarios" / f"{name}.yaml").engine == "numpy"
    assert ScenarioConfig.model_validate(_MINIMAL).engine == "numpy"


def test_engine_numba_accepted_and_roundtrips(tmp_path: Path) -> None:
    cfg = ScenarioConfig.model_validate({**_MINIMAL, "engine": "numba"})
    assert cfg.engine == "numba"
    p = tmp_path / "n.yaml"
    dump_yaml(cfg, p)
    assert "engine: numba" in p.read_text(encoding="utf-8")
    assert load_scenario(p) == cfg


@pytest.mark.parametrize("engine", ["cython", "NUMBA", "", 1])
def test_invalid_engine_rejected(engine: object) -> None:
    with pytest.raises(ValidationError, match="engine"):
        ScenarioConfig.model_validate({**_MINIMAL, "engine": engine})


# ----------------------------------------------------------------------------- wall clock (WP 1.14)


def _minimal(**extra: object) -> dict:
    return {
        "name": "m",
        "grid": {"kind": "synthetic", "generator": "flat_uniform"},
        "ignition": {"center": True},
        **extra,
    }


@pytest.mark.parametrize("name", ["flat_no_wind", "steady_wind", "river_barrier"])
def test_shipped_scenarios_have_no_start_time(name: str) -> None:
    cfg = load_scenario(CONFIGS / "scenarios" / f"{name}.yaml")
    assert cfg.start_time_utc is None
    assert cfg.time_at(0.0) is None
    assert cfg.time_at(3600.0) is None


def test_time_at_is_start_plus_simulated_seconds() -> None:
    cfg = ScenarioConfig.model_validate(_minimal(start_time_utc="2016-08-13T23:00:00Z"))
    start = datetime(2016, 8, 13, 23, 0, 0, tzinfo=UTC)
    assert cfg.start_time_utc == start
    assert cfg.time_at(0) == start
    assert cfg.time_at(3600) == start + timedelta(hours=1)
    assert cfg.time_at(3600) == datetime(2016, 8, 14, 0, 0, 0, tzinfo=UTC)
    assert cfg.time_at(0.5) == start + timedelta(milliseconds=500)
    at = cfg.time_at(86400.0)
    assert at is not None and at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("naive", ["2016-08-13T23:00:00", "2016-08-13 23:00", datetime(2016, 8, 13, 23, 0, 0)])
def test_start_time_without_a_timezone_is_rejected(naive: object) -> None:
    with pytest.raises(ValidationError, match="start_time_utc"):
        ScenarioConfig.model_validate(_minimal(start_time_utc=naive))


def test_start_time_with_an_offset_is_stored_in_utc() -> None:
    """16:00 at UTC-7 (Pacific daylight time) is 23:00 UTC: the same instant, normalised."""
    cfg = ScenarioConfig.model_validate(_minimal(start_time_utc="2016-08-13T16:00:00-07:00"))
    assert cfg.start_time_utc == datetime(2016, 8, 13, 23, 0, 0, tzinfo=UTC)
    assert cfg.start_time_utc is not None and cfg.start_time_utc.utcoffset() == timedelta(0)
    assert cfg.model_dump(mode="json")["start_time_utc"] == "2016-08-13T23:00:00Z"


def test_start_time_roundtrips_through_yaml(tmp_path: Path) -> None:
    for start in ("2016-08-13T23:00:00Z", None):
        cfg = ScenarioConfig.model_validate(_minimal(start_time_utc=start))
        path = tmp_path / "s.yaml"
        dump_yaml(cfg, path)
        assert load_scenario(path) == cfg


def test_start_time_as_a_yaml_timestamp(tmp_path: Path) -> None:
    """An unquoted ISO timestamp in YAML (parsed by PyYAML into a datetime) is accepted with an offset only."""
    path = tmp_path / "s.yaml"
    body = "name: m\ngrid: {kind: synthetic, generator: flat_uniform}\nignition: {center: true}\n"
    path.write_text(body + "start_time_utc: 2016-08-13T23:00:00+00:00\n")
    assert load_scenario(path).start_time_utc == datetime(2016, 8, 13, 23, 0, 0, tzinfo=UTC)
    path.write_text(body + "start_time_utc: 2016-08-13T23:00:00\n")
    with pytest.raises(ValueError, match="start_time_utc"):
        load_scenario(path)
