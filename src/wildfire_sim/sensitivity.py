"""Morris screening of the simulator's inputs (WP 3.2, docs/plan.md section 7).

Which inputs drive the burned area and the head arrival time, and therefore where data
quality matters? This module answers it with the elementary-effects method of Morris (1991),
in the form of Campolongo, Cariboni & Saltelli (2007), implemented here in NumPy.
``scripts/sensitivity.py`` runs the study of ``configs/sensitivity.yaml`` and writes the tables
and figures of ``docs/sensitivity.md``.

**The method.** Each of the ``k`` inputs is mapped to the unit interval, which is cut into a
grid of ``p`` levels ``{0, 1/(p-1), ..., 1}`` (``p`` even). A *trajectory* is ``k + 1`` points:
a random base point, then ``k`` steps, each of which changes one input, every input exactly
once, by ``+-delta`` with ``delta = p / (2 (p - 1))`` (Morris 1991; 2/3 for ``p = 4``). The
step along input ``i`` gives one *elementary effect*

.. math::

    EE_i = \\frac{y(x + \\Delta e_i) - y(x)}{\\Delta}

and ``r`` independent trajectories give ``r`` effects per input (``r (k + 1)`` model runs).
Their statistics are ``mu`` (the mean), ``mu*`` (the mean absolute value, which does not let
effects of opposite sign cancel; Campolongo et al. 2007) and ``sigma`` (the standard deviation,
``ddof = 1``): a large ``mu*`` marks an influential input, a large ``sigma`` one whose effect
depends on where in the input space it is taken (a nonlinearity or an interaction). Inputs
are ranked by ``mu*``. Points are built from the orientation matrix of Morris (1991)
(:func:`morris_design`), in integer level units so the grid is exact.

**Discrete inputs.** An input with ``L`` listed levels (the stencil, ``L = 2``; the kernel
shape) takes level ``floor(j L / (p - 1))`` (capped at ``L - 1``) at grid index ``j``: the unit
interval is cut into ``L`` equal bins. With ``L <= p``
every level is used, and since ``delta > 1/2`` is wider than a bin, every Morris step moves the
input to another level, so each of its elementary effects is a real change of level divided by
``delta``, on the same footing as the continuous inputs. A two-level input always changes
between its two values.

**Inputs and outputs** (:class:`SensitivityStudy`, ``configs/sensitivity.yaml``). An input is
applied to a whole run: a factor or offset on every point of the weather series (wind speed
and direction through :func:`wildfire_sim.ensemble.perturb_config`), a value for a fuel
moisture, a rate multiplier (:class:`wildfire_sim.ensemble.ScaledRateRule`), the kernel shape,
the stencil, a spread-rule keyword argument, a scale on the break widths of the grid, or a value
in one of the cited parameter files of ``configs/`` (through a temporary copy selected with
``WILDFIRE_SIM_CONFIGS``, the override :func:`wildfire_sim.config.configs_dir` documents). Each
run gives (:func:`run_outputs`):

* ``burned_area_ha``: cells ignited by ``window_s`` after the ignition, ha;
* ``head_arrival_s``: time after the ignition at which the fire first ignites a cell whose
  centre is at least ``head_distance_m`` from the ignition cell's centre, s; the first cell to
  get that far is reached by the fastest part of the front, its head. NaN if no cell gets that
  far within the run, and an elementary effect touching such a run is left out of the
  statistics and counted (``MorrisIndices.n_valid``), never filled in.

``head_distance_m`` must be shorter than the distance from the ignition to the grid's edge band
(:func:`wildfire_sim.sim.edge_band_mask` for the 16-neighbour stencil), so the head arrival
time is never affected by the grid edge. The burned area is: a run whose fire reaches the edge
band within the window (``provenance["reached_edge_s"]``) has a lower bound for its area, and
is flagged (``RunRecord.edge_in_window``).

**Random numbers.** The trajectories come from a ``numpy.random.Generator`` passed in by the
caller. Every run of one screening uses the same run seed: common random numbers, so two runs
that differ in one input see the same kernel thresholds and breach draws wherever the input
leaves them alone (changing the stencil or the kernel shape changes the thresholds themselves).
An input the model does not read therefore has elementary effects of exactly 0.

References:
    Campolongo, F., Cariboni, J. & Saltelli, A. (2007). An effective screening design for
    sensitivity analysis of large models. Environ. Model. Softw. 22, 1509-1518.
    Morris, M. D. (1991). Factorial sampling plans for preliminary computational experiments.
    Technometrics 33, 161-174.
    Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M.
    & Tarantola, S. (2008). Global Sensitivity Analysis: The Primer. Wiley, chapter 3.
"""

from __future__ import annotations

import importlib.util
import math
import multiprocessing
import os
import shutil
import tempfile
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildfire_sim.config import ScenarioConfig, WeatherConfig, configs_dir, load_yaml
from wildfire_sim.data import build_grid
from wildfire_sim.ensemble import ScaledRateRule, perturb_config, perturbed_wind_field
from wildfire_sim.grid import Grid
from wildfire_sim.sim import RunResult, edge_band_mask, run
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.base import SpreadRule
from wildfire_sim.types import FloatArray, IntArray, get_stencil

OUTPUTS: tuple[str, ...] = ("burned_area_ha", "head_arrival_s")
"""Names of the screened outputs, in the order of :attr:`ScreeningResult.outputs` columns."""

M2_PER_HA = 10_000.0


# ----------------------------------------------------------------------------- the method


def morris_step(levels: int) -> float:
    """Morris step ``delta = p / (2 (p - 1))`` on the unit interval for ``p = levels`` (even, >= 2)."""
    _check_levels(levels)
    return levels / (2.0 * (levels - 1))


def _check_levels(levels: int) -> None:
    if levels < 2 or levels % 2:
        raise ValueError(f"the number of levels must be even and >= 2, got {levels}")


def morris_design(n_inputs: int, n_trajectories: int, levels: int, rng: np.random.Generator) -> IntArray:
    """Morris (1991) trajectories as grid indices.

    Each trajectory is ``B* = (J x* + (delta / 2) [(2 B - J) D* + J]) P*`` in unit-interval
    coordinates, with ``B`` the ``(k + 1, k)`` strictly lower-triangular matrix of ones, ``J`` a
    matrix of ones, ``x*`` a random base point on the levels that leave room for one step up,
    ``D*`` a random diagonal of +-1 and ``P*`` a random permutation. In grid indices (``x = j /
    (p - 1)``) a step is ``p / 2`` indices. Per trajectory the generator draws, in this order,
    ``x*`` (``rng.integers(0, p // 2, k)``), the signs (``rng.choice([-1, 1], k)``) and the
    permutation (``rng.permutation(k)``).

    Args:
        n_inputs: Number of inputs ``k`` (>= 1).
        n_trajectories: Number of trajectories ``r`` (>= 1).
        levels: Number of grid levels ``p`` (even, >= 2).
        rng: Source of all randomness.

    Returns:
        ``(r, k + 1, k)`` int64 grid indices in ``[0, p - 1]``; divide by ``p - 1`` for the point
        in the unit cube. Consecutive points of a trajectory differ in exactly one input, by
        ``+-p / 2``, and each input changes exactly once per trajectory.
    """
    _check_levels(levels)
    if n_inputs < 1 or n_trajectories < 1:
        raise ValueError(f"need at least one input and one trajectory, got {n_inputs} and {n_trajectories}")
    k, half = n_inputs, levels // 2
    lower = np.tril(np.ones((k + 1, k), dtype=np.int64), -1)  # B
    out = np.empty((n_trajectories, k + 1, k), dtype=np.int64)
    for t in range(n_trajectories):
        base = rng.integers(0, half, size=k)  # x*, so that x* + p/2 <= p - 1
        signs = rng.choice(np.array([-1, 1]), size=k)  # diagonal of D*
        perm = rng.permutation(k)  # P*
        up = ((2 * lower - 1) * signs[None, :] + 1) // 2  # ((2B - J) D* + J) / 2, entries 0 or 1
        out[t] = (base[None, :] + half * up)[:, perm]
    return out


def elementary_effects(design: IntArray, levels: int, outputs: FloatArray) -> FloatArray:
    """Elementary effects of every input along every trajectory.

    Args:
        design: ``(r, k + 1, k)`` grid indices from :func:`morris_design`.
        levels: Number of grid levels ``p`` the design was built with.
        outputs: ``(r, k + 1)`` model output at each point (any units); NaN for an undefined
            output.

    Returns:
        ``(r, k)`` float64, ``EE[t, i] = (y(x + delta e_i) - y(x)) / delta`` in output units per
        unit of the normalised input; NaN where either run of the step has an undefined output.

    Raises:
        ValueError: Shapes disagree, or a step does not change exactly one input by ``p / 2``.
    """
    design = np.asarray(design)
    y = np.asarray(outputs, dtype=np.float64)
    r, n_points, k = design.shape
    if n_points != k + 1 or y.shape != (r, k + 1):
        raise ValueError(f"design {design.shape} and outputs {y.shape} do not match")
    delta, half = morris_step(levels), levels // 2
    ee = np.full((r, k), np.nan)
    for t in range(r):
        for j in range(k):
            diff = design[t, j + 1] - design[t, j]
            moved = np.flatnonzero(diff)
            if moved.size != 1 or abs(int(diff[moved[0]])) != half:
                raise ValueError(f"step {j} of trajectory {t} does not change exactly one input by p / 2")
            i = int(moved[0])
            ee[t, i] = np.sign(diff[i]) * (y[t, j + 1] - y[t, j]) / delta
    return ee


@dataclass(frozen=True)
class MorrisIndices:
    """Morris statistics of one output, one entry per input (all ``(k,)``).

    Attributes:
        mu: Mean elementary effect, output units per unit of the normalised input.
        mu_star: Mean absolute elementary effect (Campolongo et al. 2007), same units.
        sigma: Standard deviation of the elementary effects (``ddof = 1``), same units; NaN
            with fewer than two valid effects.
        n_valid: Number of finite elementary effects behind each statistic (``r`` when every
            run had a defined output).
    """

    mu: FloatArray
    mu_star: FloatArray
    sigma: FloatArray
    n_valid: IntArray

    @property
    def rank(self) -> IntArray:
        """Rank by ``mu*``, 1 for the largest; equal values share the best rank, NaN ranks last."""
        key = np.where(np.isnan(self.mu_star), -np.inf, self.mu_star)
        return 1 + (key[None, :] > key[:, None]).sum(axis=1)


def morris_indices(design: IntArray, levels: int, outputs: FloatArray) -> MorrisIndices:
    """``mu``, ``mu*``, ``sigma`` and the count of valid effects per input (see :func:`elementary_effects`)."""
    ee = elementary_effects(design, levels, outputs)
    valid = np.isfinite(ee)
    n = valid.sum(axis=0)
    mu = np.full(ee.shape[1], np.nan)
    mu_star = np.full(ee.shape[1], np.nan)
    sigma = np.full(ee.shape[1], np.nan)
    for i in range(ee.shape[1]):
        e = ee[valid[:, i], i]
        if e.size:
            mu[i] = e.mean()
            mu_star[i] = np.abs(e).mean()
        if e.size >= 2:
            sigma[i] = e.std(ddof=1)
    return MorrisIndices(mu=mu, mu_star=mu_star, sigma=sigma, n_valid=n.astype(np.int64))


# ----------------------------------------------------------------------------- the study schema


InputTarget = Literal[
    "wind_speed_factor",
    "wind_direction_offset_deg",
    "weather_scale",
    "weather_shift",
    "weather_set",
    "ros_multiplier",
    "arrival_shape",
    "stencil",
    "spread_param",
    "break_width_scale",
    "parameter_file",
]

_WEATHER_FIELDS = (
    "moisture_1h",
    "moisture_10h",
    "moisture_100h",
    "moisture_live_herb",
    "moisture_live_woody",
    "temperature_c",
    "relative_humidity",
)


class SensitivityInput(BaseModel):
    """One screened input: what it changes and over which range (``configs/sensitivity.yaml``).

    Exactly one of ``range`` (continuous; ``scale`` linear or log) and ``levels`` (discrete) is
    given. The value at the normalised position ``x`` in [0, 1] is ``lo + x (hi - lo)``,
    ``lo (hi / lo) ** x`` on a log scale, or the level of bin ``floor(x L)`` (module docstring).

    Targets (each applied to the whole run):

    * ``wind_speed_factor``: factor on the 10 m wind speed of every weather point (and of every
      cell of the scenario's wind field, if it has one).
    * ``wind_direction_offset_deg``: degrees added to every wind direction (wrapped), the wind
      field's included.
    * ``weather_scale`` / ``weather_shift`` / ``weather_set``: factor on, offset added to, or
      value given to the weather ``fields`` of every point (not wind; moistures as fractions,
      temperature in deg C).
    * ``ros_multiplier``: rate-of-spread factor through :class:`~wildfire_sim.ensemble.ScaledRateRule`.
    * ``arrival_shape``: ``spread.arrival_shape``. ``stencil``: 8 or 16.
    * ``spread_param``: keyword argument ``param`` of the spread rule.
    * ``break_width_scale``: factor on every ``break_width_m > 0`` of the grid.
    * ``parameter_file``: the value at dotted ``key`` of the parameter file ``file`` in ``configs/``.
    """

    model_config = ConfigDict(extra="forbid")

    target: InputTarget
    range: tuple[float, float] | None = None
    scale: Literal["linear", "log"] = "linear"
    levels: list[bool | int | float] | None = None
    fields: list[str] = Field(default_factory=list)
    param: str = ""
    file: str = ""
    key: str = ""
    units: Annotated[str, Field(min_length=1)]
    source: Annotated[str, Field(min_length=1)]
    note: str = ""
    inert: bool = Field(
        default=False,
        description="a deliberately inert control the model does not read: its effects must be exactly 0",
    )

    @model_validator(mode="after")
    def _check(self) -> SensitivityInput:
        if (self.range is None) == (self.levels is None):
            raise ValueError("give exactly one of range and levels")
        if self.range is not None:
            lo, hi = self.range
            if not lo < hi:
                raise ValueError(f"range must be increasing, got {self.range}")
            if self.scale == "log" and lo <= 0:
                raise ValueError("a log range must be positive")
        elif self.levels is not None and len(self.levels) < 2:
            raise ValueError("a discrete input needs at least two levels")
        weather = self.target in ("weather_scale", "weather_shift", "weather_set")
        if weather != bool(self.fields) or any(f not in _WEATHER_FIELDS for f in self.fields):
            raise ValueError(f"weather targets need fields from {_WEATHER_FIELDS}; other targets none")
        if (self.target == "spread_param") != bool(self.param):
            raise ValueError("param is given exactly for target spread_param")
        if (self.target == "parameter_file") != bool(self.file and self.key):
            raise ValueError("file and key are given exactly for target parameter_file")
        if self.target == "stencil" and (self.levels is None or set(self.levels) - {8, 16}):
            raise ValueError("the stencil takes levels from {8, 16}")
        return self

    def value(self, index: int, levels: int) -> Any:
        """Value at grid index ``index`` of a ``levels``-level Morris grid (units as ``units``)."""
        x = index / (levels - 1)
        if self.levels is not None:
            n = len(self.levels)
            return self.levels[min(index * n // (levels - 1), n - 1)]
        assert self.range is not None
        lo, hi = self.range
        if self.scale == "log":
            return lo * (hi / lo) ** x
        return lo + x * (hi - lo)


class SensitivityScenario(BaseModel):
    """One landscape and weather on which the inputs are screened.

    Attributes:
        description: What the scenario is, for the doc.
        scenario: The base run. Exactly one ignition cell (or ``center``). ``duration_s`` (s) is
            the length of every run, long enough for slow fires to reach ``head_distance_m``.
        weather_file: A ``WeatherConfig`` YAML replacing ``scenario.weather``; relative paths
            here and in ``scenario.grid.path`` are relative to the study file.
        window_s: Burned area is measured this long after the ignition, s.
        head_distance_m: Distance from the ignition cell for the head arrival time, m.
        inputs: Names of the screened inputs (keys of :attr:`SensitivityStudy.inputs`).
    """

    model_config = ConfigDict(extra="forbid")

    description: Annotated[str, Field(min_length=1)]
    scenario: ScenarioConfig
    weather_file: Path | None = None
    window_s: Annotated[float, Field(gt=0)]
    head_distance_m: Annotated[float, Field(gt=0)]
    inputs: Annotated[list[str], Field(min_length=1)]

    @model_validator(mode="after")
    def _check(self) -> SensitivityScenario:
        ign = self.scenario.ignition
        if len(ign.cells) + int(ign.center) != 1:
            raise ValueError("a sensitivity scenario needs exactly one ignition cell")
        if self.window_s > self.scenario.duration_s - ign.time_s:
            raise ValueError("window_s must end within the run (duration_s - ignition.time_s)")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("duplicate input names")
        return self


class SensitivityStudy(BaseModel):
    """Schema of ``configs/sensitivity.yaml``: the Morris design, the inputs and the scenarios.

    Attributes:
        levels: Number of grid levels ``p`` (even).
        trajectories: Number of trajectories ``r`` per scenario and run seed.
        design_seed: Seed of the generator that draws the trajectories.
        run_seeds: Simulator seeds; the screening is repeated once per seed (common random
            numbers within each).
        inputs: Input name -> definition.
        scenarios: Scenario name -> definition.
    """

    model_config = ConfigDict(extra="forbid")

    levels: Annotated[int, Field(ge=2)] = 4
    trajectories: Annotated[int, Field(ge=2)]
    design_seed: Annotated[int, Field(ge=0)]
    run_seeds: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1)]
    inputs: dict[str, SensitivityInput]
    scenarios: dict[str, SensitivityScenario]

    @model_validator(mode="after")
    def _check(self) -> SensitivityStudy:
        _check_levels(self.levels)
        for name, spec in self.inputs.items():
            if spec.levels is not None and len(spec.levels) > self.levels:
                raise ValueError(f"input {name!r} has more levels than the {self.levels}-level grid can reach")
        for sname, scen in self.scenarios.items():
            unknown = [i for i in scen.inputs if i not in self.inputs]
            if unknown:
                raise ValueError(f"scenario {sname!r} names unknown inputs {unknown}")
        return self


def load_study(path: str | Path) -> SensitivityStudy:
    """Load a :class:`SensitivityStudy`, resolving ``weather_file`` and cached grid paths.

    Relative paths are taken relative to the directory of ``path``. A ``weather_file`` replaces
    the scenario's weather series, and a scenario's grid path and wind field path are made absolute.
    """
    path = Path(path)
    study = load_yaml(path, SensitivityStudy)
    for scen in study.scenarios.values():
        data = scen.scenario.model_dump()
        if data["grid"]["kind"] == "cached":
            data["grid"]["path"] = str((path.parent / data["grid"]["path"]).resolve())
        if scen.weather_file is not None:
            scen.weather_file = (path.parent / scen.weather_file).resolve()
            data["weather"] = load_yaml(scen.weather_file, WeatherConfig).model_dump()
        if data["weather"].get("wind_field"):
            data["weather"]["wind_field"] = str((path.parent / data["weather"]["wind_field"]).resolve())
        scen.scenario = ScenarioConfig.model_validate(data)
    return study


# ----------------------------------------------------------------------------- one run


def available_engine() -> Literal["numpy", "numba"]:
    """``"numba"`` when numba is installed, else ``"numpy"`` (the two agree to rounding)."""
    return "numba" if importlib.util.find_spec("numba") is not None else "numpy"


def ignition_cell(config: ScenarioConfig, shape: tuple[int, int]) -> tuple[int, int]:
    """The single ignition cell ``(row, col)`` of a sensitivity scenario."""
    if config.ignition.cells:
        row, col = config.ignition.cells[0]
        return int(row), int(col)
    return shape[0] // 2, shape[1] // 2


def distance_from(cell: tuple[int, int], shape: tuple[int, int], dx: float) -> FloatArray:
    """``(H, W)`` distance, m, from the centre of ``cell`` to every cell centre (cell size ``dx`` m)."""
    rows, cols = np.indices(shape)
    return np.hypot(rows - cell[0], cols - cell[1]) * dx


def edge_distance_m(cell: tuple[int, int], shape: tuple[int, int], dx: float) -> float:
    """Distance, m, from ``cell`` to the nearest cell of the 16-neighbour stencil's edge band.

    The edge band of the 8-neighbour stencil lies inside it, so a fire is at least this far
    from ``cell`` before it reaches the edge band with either stencil.
    """
    band = edge_band_mask(shape, get_stencil(16).max_offset)
    return float(distance_from(cell, shape, dx)[band].min())


@dataclass(frozen=True)
class RunRecord:
    """Outputs of one run of a screening.

    Attributes:
        burned_area_ha: Area ignited by the end of the window, ha.
        head_arrival_s: Time after the ignition at which the fire first reached the head
            distance, s; NaN if it did not within the run.
        reached_edge_s: Time after the ignition at which the fire reached the grid's edge band,
            s, or NaN (``provenance["reached_edge_s"]``).
        edge_in_window: The edge was reached within the window: ``burned_area_ha`` is a lower
            bound.
        lb_capped_cell_hours: Burning cell-hours under the lattice cap (WP 2.1d), whole run.
        breaches: Breaks breached in the whole run.
    """

    burned_area_ha: float
    head_arrival_s: float
    reached_edge_s: float
    edge_in_window: bool
    lb_capped_cell_hours: float
    breaches: int


def run_outputs(result: RunResult, cell: tuple[int, int], window_s: float, head_distance_m: float) -> RunRecord:
    """Screened outputs of a finished run (module docstring).

    Args:
        result: The run.
        cell: Its ignition cell ``(row, col)``.
        window_s: Burned area is counted this long after the ignition, s.
        head_distance_m: Distance of the head arrival time, m.
    """
    t_ign = float(result.provenance["config"]["ignition"]["time_s"])
    arrival = result.arrival_time - t_ign  # s after the ignition, NaN where unburned
    dx = result.grid.dx
    with np.errstate(invalid="ignore"):
        burned = np.count_nonzero(arrival <= window_s)
    far = distance_from(cell, arrival.shape, dx) >= head_distance_m
    far_arrival = arrival[far & ~np.isnan(arrival)]
    edge = result.provenance["reached_edge_s"]
    edge_s = math.nan if edge is None else float(edge) - t_ign
    return RunRecord(
        burned_area_ha=float(burned * dx * dx / M2_PER_HA),
        head_arrival_s=float(far_arrival.min()) if far_arrival.size else math.nan,
        reached_edge_s=edge_s,
        edge_in_window=bool(edge_s <= window_s),  # NaN compares False
        lb_capped_cell_hours=float(result.provenance["lb_capped_cell_hours"]),
        breaches=int(result.provenance["breaks"].get("breaches", 0)),
    )


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    *path, last = key.split(".")
    node = data
    for part in path:
        node = node[part]
    if last not in node:
        raise KeyError(f"no key {key!r} in the parameter file")
    node[last] = value


@contextmanager
def parameter_files(edits: dict[str, dict[str, Any]]) -> Generator[None]:
    """Run with edited copies of parameter files of ``configs/`` (nothing to do if ``edits`` is empty).

    Copies the whole ``configs_dir()`` into a temporary directory, writes each ``file -> {dotted
    key: value}`` edit into its copy and points ``WILDFIRE_SIM_CONFIGS`` at it for the duration;
    the variable is restored afterwards. Every loader that reads ``configs_dir()`` (holdover,
    urban, breach, ... tables) then sees the edited values.
    """
    if not edits:
        yield
        return
    old = os.environ.get("WILDFIRE_SIM_CONFIGS")
    with tempfile.TemporaryDirectory(prefix="wildfire_sensitivity_") as tmp:
        copy = Path(tmp) / "configs"
        shutil.copytree(configs_dir(), copy)
        for name, values in edits.items():
            target = copy / name
            data = yaml.safe_load(target.read_text(encoding="utf-8"))
            for key, value in values.items():
                _set_dotted(data, key, value)
            target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        os.environ["WILDFIRE_SIM_CONFIGS"] = str(copy)
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("WILDFIRE_SIM_CONFIGS", None)
            else:
                os.environ["WILDFIRE_SIM_CONFIGS"] = old


def run_point(
    base: ScenarioConfig,
    inputs: dict[str, SensitivityInput],
    values: dict[str, Any],
    seed: int,
    window_s: float,
    head_distance_m: float,
) -> RunRecord:
    """One run of ``base`` with ``values`` (input name -> value) applied, and its outputs.

    The configured engine is replaced by :func:`available_engine`. With a ``ros_multiplier``
    input the rule is always wrapped in :class:`~wildfire_sim.ensemble.ScaledRateRule`, so every
    run of the design computes intensity the same way.

    Args:
        base: Scenario before the inputs are applied.
        inputs: Definitions of the inputs named in ``values``.
        values: Input name -> value (units of the input).
        seed: Simulator seed.
        window_s: Window of the burned area, s after the ignition.
        head_distance_m: Distance of the head arrival time, m.
    """
    factors = {"wind_speed_factor": 1.0, "wind_direction_offset_deg": 0.0, "moisture_factor": 1.0}
    scale: dict[str, float] = {}
    shift: dict[str, float] = {}
    setto: dict[str, float] = {}
    edits: dict[str, dict[str, Any]] = {}
    multiplier: float | None = None
    width_scale: float | None = None
    data = base.model_dump()
    data["engine"] = available_engine()
    for name, value in values.items():
        spec = inputs[name]
        match spec.target:
            case "wind_speed_factor" | "wind_direction_offset_deg":
                factors[spec.target] = float(value)
            case "weather_scale":
                scale.update(dict.fromkeys(spec.fields, float(value)))
            case "weather_shift":
                shift.update(dict.fromkeys(spec.fields, float(value)))
            case "weather_set":
                setto.update(dict.fromkeys(spec.fields, float(value)))
            case "ros_multiplier":
                multiplier = float(value)
            case "arrival_shape":
                data["spread"]["arrival_shape"] = float(value)
            case "stencil":
                data["stencil"] = int(value)
            case "spread_param":
                data["spread"]["params"][spec.param] = value
            case "break_width_scale":
                width_scale = float(value)
            case "parameter_file":
                edits.setdefault(spec.file, {})[spec.key] = value

    data["weather"] = perturb_config(base, factors).weather.model_dump()  # wind speed and direction
    for point in data["weather"]["series"]:
        for field, f in scale.items():
            point[field] *= f
        for field, d in shift.items():
            point[field] += d
        point.update(setto)
    config = ScenarioConfig.model_validate(data)

    grid: Grid = build_grid(config.grid)
    if width_scale is not None:
        grid.break_width_m = grid.break_width_m * width_scale
    with parameter_files(edits):
        rule: SpreadRule = make_spread_rule(config.spread.rule, **config.spread.params)
        if multiplier is not None:
            rule = ScaledRateRule(rule, multiplier)
        wind = perturbed_wind_field(config, factors)
        result = run(config, seed=seed, rule=rule, grid=grid, record_events=False, wind_field=wind)
    return run_outputs(result, ignition_cell(config, grid.shape), window_s, head_distance_m)


def _run_task(task: tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], int, float, float]) -> RunRecord:
    """:func:`run_point` from plain data, for a ``spawn`` worker pool."""
    base, inputs, values, seed, window_s, head_distance_m = task
    specs = {name: SensitivityInput.model_validate(spec) for name, spec in inputs.items()}
    return run_point(ScenarioConfig.model_validate(base), specs, values, seed, window_s, head_distance_m)


# ----------------------------------------------------------------------------- a screening


@dataclass(frozen=True)
class ScreeningResult:
    """One Morris screening of one scenario with one run seed.

    Attributes:
        scenario: Scenario name.
        input_names: The ``k`` inputs, in design column order.
        levels: Number of grid levels ``p``.
        design: ``(r, k + 1, k)`` grid indices (:func:`morris_design`).
        values: ``(r, k + 1)`` nested lists of input name -> value applied in each run.
        records: ``(r, k + 1)`` nested lists of :class:`RunRecord`.
        indices: Output name (:data:`OUTPUTS`) -> :class:`MorrisIndices`.
        run_seed: Simulator seed of every run.
        elapsed_s: Wall-clock time of the screening, s.
    """

    scenario: str
    input_names: tuple[str, ...]
    levels: int
    design: IntArray
    values: list[list[dict[str, Any]]]
    records: list[list[RunRecord]]
    indices: dict[str, MorrisIndices]
    run_seed: int
    elapsed_s: float

    def output(self, name: str) -> FloatArray:
        """``(r, k + 1)`` values of output ``name`` (ha or s)."""
        return np.array([[getattr(rec, name) for rec in row] for row in self.records], dtype=np.float64)

    @property
    def n_edge_in_window(self) -> int:
        """Runs whose fire reached the grid's edge band within the window (area is a lower bound)."""
        return sum(rec.edge_in_window for row in self.records for rec in row)

    def edge_effects(self) -> IntArray:
        """``(k,)`` burned-area elementary effects per input with a run that reached the edge in the window.

        Such an effect compares at least one lower bound. The head arrival time is never
        affected (module docstring).
        """
        counts = np.zeros(len(self.input_names), dtype=np.int64)
        for t, traj in enumerate(self.design):
            for j in range(len(self.input_names)):
                if self.records[t][j].edge_in_window or self.records[t][j + 1].edge_in_window:
                    counts[int(np.flatnonzero(traj[j + 1] - traj[j])[0])] += 1
        return counts


def run_screening(
    study: SensitivityStudy,
    scenario: str,
    run_seed: int,
    *,
    trajectories: int | None = None,
    n_workers: int = 1,
) -> ScreeningResult:
    """Morris screening of one scenario of ``study`` with simulator seed ``run_seed``.

    The trajectories are drawn from ``numpy.random.default_rng(study.design_seed)``, so every
    run seed and both scenarios with the same number of inputs share a design.

    Args:
        study: The study.
        scenario: Name of one of ``study.scenarios``.
        run_seed: Simulator seed of every run (>= 0).
        trajectories: Number of trajectories; None uses ``study.trajectories``.
        n_workers: Processes (``spawn`` pool for more than 1); results do not depend on it.

    Raises:
        ValueError: ``head_distance_m`` is not shorter than the distance from the ignition to
            the grid's edge band, or ``n_workers < 1``.
    """
    start = time.perf_counter()
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    scen = study.scenarios[scenario]
    r = study.trajectories if trajectories is None else int(trajectories)
    base = scen.scenario
    grid = build_grid(base.grid)
    limit = edge_distance_m(ignition_cell(base, grid.shape), grid.shape, grid.dx)
    if not scen.head_distance_m < limit:
        raise ValueError(f"head_distance_m {scen.head_distance_m} m must be < {limit:.0f} m, the edge band's distance")

    names = tuple(scen.inputs)
    design = morris_design(len(names), r, study.levels, np.random.default_rng(study.design_seed))
    values = [
        [{n: study.inputs[n].value(int(j), study.levels) for n, j in zip(names, point, strict=True)} for point in traj]
        for traj in design
    ]
    specs = {n: study.inputs[n].model_dump(mode="json") for n in names}
    base_data = base.model_dump(mode="json")
    tasks = [(base_data, specs, v, run_seed, scen.window_s, scen.head_distance_m) for traj in values for v in traj]
    if n_workers == 1:
        flat = [_run_task(t) for t in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(processes=min(n_workers, len(tasks))) as pool:
            flat = pool.map(_run_task, tasks, chunksize=1)
    n_points = len(names) + 1
    records = [flat[t * n_points : (t + 1) * n_points] for t in range(r)]
    indices = {
        out: morris_indices(design, study.levels, np.array([[getattr(rec, out) for rec in row] for row in records]))
        for out in OUTPUTS
    }
    return ScreeningResult(
        scenario=scenario,
        input_names=names,
        levels=study.levels,
        design=design,
        values=values,
        records=records,
        indices=indices,
        run_seed=run_seed,
        elapsed_s=time.perf_counter() - start,
    )


def summary_rows(result: ScreeningResult) -> list[dict[str, Any]]:
    """One row per (output, input): ``mu``, ``mu_star``, ``sigma`` (output units), ``n_valid``, ``rank``
    and ``n_edge`` (elementary effects with a run that reached the grid edge within the window;
    always 0 for the head arrival time)."""
    rows: list[dict[str, Any]] = []
    edge = result.edge_effects()
    for out in OUTPUTS:
        ind = result.indices[out]
        rank = ind.rank
        for i, name in enumerate(result.input_names):
            rows.append(
                {
                    "output": out,
                    "input": name,
                    "mu": float(ind.mu[i]),
                    "mu_star": float(ind.mu_star[i]),
                    "sigma": float(ind.sigma[i]),
                    "n_valid": int(ind.n_valid[i]),
                    "rank": int(rank[i]),
                    "n_edge": int(edge[i]) if out == "burned_area_ha" else 0,
                }
            )
    return rows


def flat_records(result: ScreeningResult) -> Sequence[tuple[int, int, dict[str, Any], RunRecord]]:
    """``(trajectory, point, values, record)`` for every run, in design order."""
    return [
        (t, j, result.values[t][j], result.records[t][j])
        for t in range(len(result.records))
        for j in range(len(result.records[t]))
    ]


__all__ = [
    "OUTPUTS",
    "MorrisIndices",
    "RunRecord",
    "ScreeningResult",
    "SensitivityInput",
    "SensitivityScenario",
    "SensitivityStudy",
    "available_engine",
    "distance_from",
    "edge_distance_m",
    "elementary_effects",
    "flat_records",
    "ignition_cell",
    "load_study",
    "morris_design",
    "morris_indices",
    "morris_step",
    "parameter_files",
    "run_outputs",
    "run_point",
    "run_screening",
    "summary_rows",
]
