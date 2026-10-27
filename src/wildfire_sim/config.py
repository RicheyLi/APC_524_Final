"""Pydantic schemas for YAML-driven runs.

A run is fully determined by a :class:`ScenarioConfig` plus a seed. Physical and cost
constants live in YAML under ``configs/`` as :class:`CitedValue` entries so every number
carries its source (CONTRIBUTING.md rule 3).

This module is a frozen contract (see CONTRIBUTING.md). Adding optional fields with defaults is
fine; renaming or removing fields is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildfire_sim.grid import Weather, WeatherSeries
from wildfire_sim.types import StencilSize


class _Strict(BaseModel):
    """Base with unknown keys rejected, so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- cited values


class CitedValue(_Strict):
    """A numeric parameter together with where it came from."""

    value: float
    source: Annotated[str, Field(min_length=1, description="Citation or 'PLACEHOLDER - replace in WP x.y'")]
    units: str = ""
    note: str = ""


# ----------------------------------------------------------------------------- grid sources


class SyntheticGridConfig(_Strict):
    """Build the grid with a generator from :mod:`wildfire_sim.data.synthetic`."""

    kind: Literal["synthetic"] = "synthetic"
    generator: Literal["flat_uniform", "slope_ramp", "with_river", "with_town"]
    shape: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] = (101, 101)
    dx: Annotated[float, Field(gt=0, description="cell size, m")] = 30.0
    fuel_model: Annotated[int, Field(ge=1, description="Scott & Burgan fuel model number")] = 1
    params: dict[str, Any] = Field(default_factory=dict, description="generator-specific keyword arguments")


class CachedGridConfig(_Strict):
    """Load a grid saved by :meth:`wildfire_sim.grid.Grid.save` (or built by the data pipeline)."""

    kind: Literal["cached"] = "cached"
    path: Path


GridConfig = Annotated[SyntheticGridConfig | CachedGridConfig, Field(discriminator="kind")]


# ----------------------------------------------------------------------------- weather


class WeatherPoint(_Strict):
    """One breakpoint of a piecewise-constant weather series. Units as in :class:`Weather`."""

    time_s: Annotated[float, Field(ge=0)] = 0.0
    wind_speed: Annotated[float, Field(ge=0)] = 0.0
    wind_direction: Annotated[float, Field(ge=0, lt=360)] = 0.0
    moisture_1h: Annotated[float, Field(ge=0)] = 0.06
    moisture_10h: Annotated[float, Field(ge=0)] = 0.07
    moisture_100h: Annotated[float, Field(ge=0)] = 0.08
    moisture_live_herb: Annotated[float, Field(ge=0)] = 0.60
    moisture_live_woody: Annotated[float, Field(ge=0)] = 0.90
    temperature_c: float = 25.0
    relative_humidity: Annotated[float, Field(ge=0, le=1)] = 0.30

    def to_weather(self) -> Weather:
        return Weather(**self.model_dump(exclude={"time_s"}))


class WeatherConfig(_Strict):
    series: Annotated[list[WeatherPoint], Field(min_length=1)] = Field(default_factory=lambda: [WeatherPoint()])

    @model_validator(mode="after")
    def _check_series(self) -> WeatherConfig:
        if self.series[0].time_s != 0:
            raise ValueError("first weather point must be at time_s = 0")
        times = [p.time_s for p in self.series]
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("weather time_s must be strictly increasing")
        return self

    def to_series(self) -> WeatherSeries:
        return WeatherSeries(
            times_s=tuple(p.time_s for p in self.series),
            values=tuple(p.to_weather() for p in self.series),
        )


# ----------------------------------------------------------------------------- spread / ignition


class SpreadConfig(_Strict):
    rule: Annotated[str, Field(min_length=1, description="registered spread rule name")] = "constant"
    params: dict[str, Any] = Field(default_factory=dict, description="keyword arguments for the rule")


class IgnitionConfig(_Strict):
    """Where and when the fire starts. ``cells`` are ``(row, col)``; ``center`` ignites the grid centre."""

    cells: list[tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]]] = Field(default_factory=list)
    center: bool = False
    time_s: Annotated[float, Field(ge=0)] = 0.0

    @model_validator(mode="after")
    def _at_least_one(self) -> IgnitionConfig:
        if not self.cells and not self.center:
            raise ValueError("ignition needs at least one cell or center = true")
        return self


# ----------------------------------------------------------------------------- suppression / cost


class ResourceConfig(_Strict):
    """One suppression resource type. Parameter semantics are defined in WP 4.2."""

    type: Literal["crew", "dozer", "engine", "aircraft"]
    count: Annotated[int, Field(ge=0)] = 0
    dispatch_delay_s: Annotated[float, Field(ge=0)] = 0.0
    work_rate: Annotated[float, Field(ge=0, description="m/h of line, or L per drop")] = 0.0
    hourly_cost_usd: Annotated[float, Field(ge=0)] = 0.0
    max_flame_length_m: Annotated[float, Field(ge=0, description="NWCG safety limit; 0 = unlimited")] = 0.0
    source: str = ""


class SuppressionConfig(_Strict):
    policy: Annotated[str, Field(min_length=1)] = "none"
    params: dict[str, Any] = Field(default_factory=dict, description="policy parameters tuned by the optimizer")
    resources: list[ResourceConfig] = Field(default_factory=list)
    decision_interval_s: Annotated[float, Field(gt=0)] = 1800.0


class CostConfig(_Strict):
    """Unit costs for the objective in docs/revised_plan.md (Strategy search). All cited."""

    crew_usd_per_h: CitedValue
    dozer_usd_per_h: CitedValue
    aircraft_usd_per_h: CitedValue
    water_usd_per_l: CitedValue
    retardant_usd_per_l: CitedValue
    usd_per_building: CitedValue
    usd_per_ha_wildland: CitedValue


# ----------------------------------------------------------------------------- scenario


class ScenarioConfig(_Strict):
    """Everything needed to reproduce one simulation run, apart from the seed."""

    name: Annotated[str, Field(min_length=1)]
    description: str = ""
    grid: GridConfig
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    spread: SpreadConfig = Field(default_factory=SpreadConfig)
    ignition: IgnitionConfig
    suppression: SuppressionConfig = Field(default_factory=SuppressionConfig)
    dt_s: Annotated[float, Field(gt=0, description="time step, s")] = 30.0
    duration_s: Annotated[float, Field(gt=0, description="simulated length of the run, s")] = 3600.0
    stencil: StencilSize = 8
    seed: Annotated[int, Field(ge=0, description="default seed; the CLI/ensemble may override")] = 0

    @model_validator(mode="after")
    def _check_times(self) -> ScenarioConfig:
        if self.duration_s < self.dt_s:
            raise ValueError("duration_s must be >= dt_s")
        if self.ignition.time_s > self.duration_s:
            raise ValueError("ignition.time_s must be <= duration_s")
        return self

    @property
    def n_steps(self) -> int:
        return int(self.duration_s // self.dt_s)


# ----------------------------------------------------------------------------- loading


def load_yaml[T: BaseModel](path: str | Path, model: type[T]) -> T:
    """Parse ``path`` as YAML and validate it as ``model``."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return model.model_validate(raw)


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load a :class:`ScenarioConfig` from YAML."""
    return load_yaml(path, ScenarioConfig)


def load_costs(path: str | Path) -> CostConfig:
    """Load a :class:`CostConfig` from YAML."""
    return load_yaml(path, CostConfig)


def dump_yaml(model: BaseModel, path: str | Path) -> None:
    """Write ``model`` to ``path`` as YAML (round-trips through :func:`load_yaml`)."""
    data = model.model_dump(mode="json")
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
