"""Pydantic schemas for YAML-driven runs.

A run is fully determined by a :class:`ScenarioConfig` plus a seed. Physical and cost
constants live in YAML under ``configs/`` as :class:`CitedValue` entries so every number
carries its source (CONTRIBUTING.md rule 3).

This module is a frozen contract (see CONTRIBUTING.md). Adding optional fields with defaults is
fine; renaming or removing fields is not.
"""

from __future__ import annotations

import os
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
    arrival_shape: Annotated[
        float,
        Field(
            ge=1,
            description=(
                "gamma shape k of the cell-to-cell crossing time, passed to "
                "spread.base.draw_crossing_thresholds. 200 (default) is nearly deterministic, so the "
                "front-speed correction is close to 1 and elongated fires keep their shape; 1 is "
                "memoryless (front runs several times too fast)"
            ),
        ),
    ] = 200.0


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
    hourly_cost_usd: Annotated[float, Field(ge=0, description="0 = use the CostConfig rate for this type")] = 0.0
    max_flame_length_m: Annotated[float, Field(ge=0, description="NWCG safety limit; 0 = unlimited")] = 0.0
    source: str = ""


class SuppressionConfig(_Strict):
    policy: Annotated[str, Field(min_length=1)] = "none"
    params: dict[str, Any] = Field(default_factory=dict, description="policy parameters tuned by the optimizer")
    resources: list[ResourceConfig] = Field(default_factory=list)
    decision_interval_s: Annotated[float, Field(gt=0)] = 1800.0


class CostConfig(_Strict):
    """Unit costs for the objective in docs/plan.md (Strategy search). All cited.

    One hourly rate per :class:`ResourceConfig` type. A resource whose own
    ``hourly_cost_usd`` is 0 is charged at the rate given here.
    """

    crew_usd_per_h: CitedValue
    dozer_usd_per_h: CitedValue
    engine_usd_per_h: CitedValue
    aircraft_usd_per_h: CitedValue
    water_usd_per_l: CitedValue
    retardant_usd_per_l: CitedValue
    usd_per_building: CitedValue
    usd_per_ha_wildland: CitedValue


# ----------------------------------------------------------------------------- ensemble


class EnsembleConfig(_Strict):
    """Monte Carlo ensemble settings (WP 1.6, docs/plan.md section 6).

    Each run of an ensemble draws one value of every factor below and applies it to the whole
    run (every point of the weather series), so a run is "a bit windier", not noisy in time.
    The defaults perturb nothing, so a scenario without an ``ensemble`` section behaves as
    before. How the draws are made is documented in :mod:`wildfire_sim.ensemble`.
    """

    wind_speed_sigma_frac: Annotated[
        float,
        Field(ge=0, description="sigma of ln(factor) for a log-normal wind-speed factor (median 1), dimensionless"),
    ] = 0.0
    wind_direction_sigma_deg: Annotated[
        float,
        Field(ge=0, description="standard deviation of a normal offset added to wind direction, degrees"),
    ] = 0.0
    moisture_sigma_frac: Annotated[
        float,
        Field(
            ge=0,
            description="sigma of ln(factor) for a log-normal factor (median 1) on all dead and live fuel "
            "moistures together, dimensionless",
        ),
    ] = 0.0
    ros_multiplier_sigma_frac: Annotated[
        float,
        Field(
            ge=0, description="sigma of ln(factor) for a log-normal rate-of-spread multiplier (median 1), dimensionless"
        ),
    ] = 0.0
    ignition_jitter_cells: Annotated[
        int,
        Field(
            ge=0,
            description="each ignition cell moves by a uniform integer offset in [-j, j] cells in row and column; "
            "kept inside the grid and on burnable fuel",
        ),
    ] = 0
    n_runs: Annotated[int, Field(ge=1, description="default number of runs, for the CLI")] = 50
    base_seed: Annotated[int, Field(ge=0, description="default base seed of the ensemble, for the CLI")] = 0

    @property
    def perturbs(self) -> bool:
        """True if any input perturbation is switched on."""
        return (
            self.wind_speed_sigma_frac > 0
            or self.wind_direction_sigma_deg > 0
            or self.moisture_sigma_frac > 0
            or self.ros_multiplier_sigma_frac > 0
            or self.ignition_jitter_cells > 0
        )


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
    stencil: StencilSize = 16
    seed: Annotated[int, Field(ge=0, description="default seed; the CLI/ensemble may override")] = 0
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    engine: Literal["numpy", "numba"] = Field(
        default="numpy",
        description=(
            "per-step fire engine: 'numpy' (reference, always available) or 'numba' (compiled, same "
            "results to floating-point rounding; needs the 'fast' extra: uv sync --extra fast)"
        ),
    )

    @model_validator(mode="after")
    def _check_times(self) -> ScenarioConfig:
        if self.duration_s < self.dt_s:
            raise ValueError("duration_s must be >= dt_s")
        if self.ignition.time_s > self.duration_s:
            raise ValueError("ignition.time_s must be <= duration_s")
        return self

    @model_validator(mode="after")
    def _check_ignition_inside_grid(self) -> ScenarioConfig:
        # Only synthetic grids declare their shape here; cached grids are checked by the simulator.
        if isinstance(self.grid, SyntheticGridConfig):
            h, w = self.grid.shape
            for row, col in self.ignition.cells:
                if row >= h or col >= w:
                    raise ValueError(f"ignition cell ({row}, {col}) is outside the grid of shape ({h}, {w})")
        return self

    @property
    def n_steps(self) -> int:
        """Number of whole time steps in the run: ``floor(duration_s / dt_s)``.

        Tolerates floating-point rounding, so ``duration_s = 3600`` with ``dt_s = 0.1`` gives
        36000 steps, not 35999.
        """
        ratio = self.duration_s / self.dt_s
        nearest = round(ratio)
        if abs(ratio - nearest) <= 1e-9 * max(1.0, abs(ratio)):
            return int(nearest)
        return int(ratio)


# ----------------------------------------------------------------------------- loading


def configs_dir() -> Path:
    """Directory holding the cited parameter tables (``alexandridis.yaml``, ``costs.yaml``, ...).

    The environment variable ``WILDFIRE_SIM_CONFIGS`` overrides the default, which is the
    ``configs/`` directory of the repository checkout this package is installed from.

    Raises:
        FileNotFoundError: The directory does not exist.
    """
    override = os.environ.get("WILDFIRE_SIM_CONFIGS")
    path = Path(override) if override else Path(__file__).resolve().parents[2] / "configs"
    if not path.is_dir():
        raise FileNotFoundError(
            f"configs directory {path} not found; run from a repository checkout or set WILDFIRE_SIM_CONFIGS"
        )
    return path


def load_yaml[T: BaseModel](path: str | Path, model: type[T]) -> T:
    """Parse ``path`` as YAML and validate it as ``model``.

    Relative paths inside the file (for example ``grid.path``) are left as written and resolve
    against the current working directory, so run commands from the repository root.

    Raises:
        ValueError: The file is not valid YAML, is not a mapping, or fails validation
            (``pydantic.ValidationError`` is a ``ValueError``).
    """
    with Path(path).open(encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: not valid YAML: {exc}") from exc
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
