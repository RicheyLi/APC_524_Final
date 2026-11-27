"""Calibration of a spread rule against an observed fire (WP 3.1, docs/plan.md section 7).

A few parameters of the spread rule are tuned so that unsuppressed runs reproduce the early
growth of a real fire (fire A), before containment lines held:

* ``"rothermel"``: ``ros_multiplier``, the rate-of-spread adjustment factor (FARSITE's
  adjustment factors, Finney 1998), and ``lb_scale``, which scales the elongation of the fire
  ellipse, ``LB' = 1 + lb_scale (LB - 1)`` (:class:`~wildfire_sim.spread.rothermel.RothermelRule`);
* ``"alexandridis"``: its base rate ``ros_base`` (m/s) and wind direction coefficient ``c2``
  (s/m) (:class:`~wildfire_sim.spread.alexandridis.AlexandridisRule`).

**Objective.** For a parameter set the scenario is run once per fixed seed over the
early-growth window, from the ignition at ``t_0`` to the window end ``t_1``. Per seed,

.. math::

    f = w_o\\,(1 - J) + w_a\\,\\frac{\\overline{\\varepsilon}}{t_1 - t_0}

with ``J`` the Jaccard overlap of the simulated and the observed burned masks at ``t_1``
(:func:`wildfire_sim.metrics.jaccard`) and ``eps`` the mean arrival-interval error of the
detections in the window (:func:`wildfire_sim.metrics.arrival_interval_error`, the 375 m VIIRS
footprint), divided by the window length so that both terms are dimensionless and lie in
[0, 1]. The objective is ``f`` averaged over the seeds. The seeds are the same for every trial
(common random numbers), so two parameter sets are compared on the same link thresholds.

**Edge rule.** A run that reaches the grid edge (``provenance["reached_edge_s"]`` set, WP 1.11)
is a lower bound on its fire, so its score is meaningless: a trial in which any run reaches the
edge is rejected, not scored (an Optuna trial in state ``PRUNED``). It can never be the best
trial, and the number of rejected trials is reported. If every trial is rejected the study
fails instead of writing parameters.

**Search.** An Optuna study (multivariate TPE sampler, seeded, so a study reproduces; Akiba et
al. 2019) over the bounds of
``configs/calibration.yaml``; a parameter marked ``log`` (the rate multiplier) is sampled on a
log scale. Afterwards the objective is evaluated on a grid around the optimum
(:func:`objective_surface`), so it is visible whether the data identify the parameters or
whether they trade off against each other (section 7).

**Targets.** :func:`observed_target` builds the observed burned mask (the daily perimeters
posted by the window end) and the VIIRS detections of a built region. :func:`synthetic_target`
builds a synthetic "observed" fire from a run with known parameters, with detections sampled
from its arrival times at synthetic overpass times by the rule of
:mod:`wildfire_sim.data.validation`; recovering those parameters is the test of the method
(WP 3.1 acceptance).

Settings, weights and bounds are read from ``configs/calibration.yaml``
(:func:`load_calibration_settings`); the result is written to ``configs/calibrated.yaml``
(:func:`write_calibrated`, :func:`load_calibrated`). ``optuna`` is an optional extra
(``uv sync --extra opt``), imported only by :func:`run_study`.

References:
    Finney, M. A. (1998). FARSITE: Fire Area Simulator - model development and evaluation.
    USDA For. Serv. Res. Pap. RMRS-RP-4.
    Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: a next-generation
    hyperparameter optimization framework. Proc. KDD 2019, 2623-2631.
"""

from __future__ import annotations

import datetime as dt
import importlib
import importlib.util
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildfire_sim import __version__
from wildfire_sim.config import CitedValue, ScenarioConfig, WeatherConfig, configs_dir, dump_yaml, load_yaml
from wildfire_sim.data import build_grid
from wildfire_sim.data.cache import load_regions
from wildfire_sim.data.landfire import DEFAULT_CACHE
from wildfire_sim.data.validation import Detection, FireObservations, fetch_observations
from wildfire_sim.ensemble import derive_seeds
from wildfire_sim.grid import Grid
from wildfire_sim.metrics import arrival_interval_error, footprint_arrival, jaccard
from wildfire_sim.sim import RunResult, model_provenance, run
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.types import BoolArray, FloatArray

# ----------------------------------------------------------------------------- settings


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParameterBound(_Strict):
    """Search range of one rule parameter (a keyword argument of the rule)."""

    low: Annotated[float, Field(gt=0)]
    high: Annotated[float, Field(gt=0)]
    log: bool = False
    units: str = ""
    source: Annotated[str, Field(min_length=1)]
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> ParameterBound:
        if not self.low < self.high:
            raise ValueError("low must be < high")
        return self


class CalibrationSettings(_Strict):
    """Contents of ``configs/calibration.yaml`` (every value cited, CONTRIBUTING rule 3)."""

    weight_overlap: CitedValue
    weight_arrival: CitedValue
    footprint_m: CitedValue
    window_hours: CitedValue
    n_trials: CitedValue
    n_seeds: CitedValue
    base_seed: CitedValue
    sampler_seed: CitedValue
    surface_steps: Annotated[list[float], Field(min_length=1)]
    surface_steps_source: Annotated[str, Field(min_length=1)]
    overpass_interval_s: CitedValue
    truth_seed: CitedValue
    parameters: dict[str, dict[str, ParameterBound]]

    @model_validator(mode="after")
    def _check(self) -> CalibrationSettings:
        if self.weight_overlap.value < 0 or self.weight_arrival.value < 0:
            raise ValueError("weights must be >= 0")
        if self.weight_overlap.value + self.weight_arrival.value <= 0:
            raise ValueError("at least one weight must be > 0")
        for name in ("footprint_m", "window_hours", "overpass_interval_s"):
            if not getattr(self, name).value > 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("n_trials", "n_seeds", "base_seed", "sampler_seed", "truth_seed"):
            value = getattr(self, name).value
            if value != int(value) or value < (1 if name in ("n_trials", "n_seeds") else 0):
                raise ValueError(f"{name} must be a whole number (>= 1 for counts, >= 0 for seeds)")
        if any(not s > -1 for s in self.surface_steps):
            raise ValueError("surface_steps are relative changes and must be > -1")
        if not self.parameters:
            raise ValueError("parameters must name at least one rule")
        return self

    def space(self, rule: str) -> dict[str, ParameterBound]:
        """Bounds of the calibrated parameters of ``rule``.

        Raises:
            KeyError: ``rule`` has no calibration parameters in the file.
        """
        if rule not in self.parameters:
            raise KeyError(f"no calibration parameters for rule {rule!r}; available: {sorted(self.parameters)}")
        return self.parameters[rule]

    @property
    def weights(self) -> tuple[float, float]:
        """``(w_overlap, w_arrival)``, dimensionless."""
        return float(self.weight_overlap.value), float(self.weight_arrival.value)

    def seeds(self, n_seeds: int | None = None) -> tuple[int, ...]:
        """The fixed run seeds: :func:`~wildfire_sim.ensemble.derive_seeds` of ``base_seed``."""
        return derive_seeds(int(self.base_seed.value), int(self.n_seeds.value) if n_seeds is None else n_seeds)


def load_calibration_settings(path: str | Path | None = None) -> CalibrationSettings:
    """Read ``configs/calibration.yaml`` (or ``path``)."""
    return load_yaml(configs_dir() / "calibration.yaml" if path is None else path, CalibrationSettings)


# ----------------------------------------------------------------------------- targets


@dataclass(frozen=True)
class Target:
    """What a calibration run is compared with.

    Attributes:
        observed: ``(H, W)`` bool, cells observed burned at the window end.
        detections: First detections in the window, ``(t_clear, t_detect, row, col)``, s.
        ignition_s: Ignition time ``t_0``, s of simulated time.
        window_end_s: Window end ``t_1``, s of simulated time.
        footprint_m: Side of a detection's footprint, m.
        description: What the target is (for the provenance record).
    """

    observed: BoolArray
    detections: tuple[Detection, ...]
    ignition_s: float
    window_end_s: float
    footprint_m: float
    description: str

    def __post_init__(self) -> None:
        if not self.window_end_s > self.ignition_s:
            raise ValueError("the window must end after the ignition")
        if not self.footprint_m > 0:
            raise ValueError("footprint_m must be > 0 m")
        if not np.any(self.observed):
            raise ValueError("the observed burned mask is empty")
        late = [d for d in self.detections if d.t_detect > self.window_end_s]
        if late:
            raise ValueError(f"{len(late)} detections fall after the window end")

    @property
    def window_s(self) -> float:
        """Window length ``t_1 - t_0``, s."""
        return self.window_end_s - self.ignition_s

    def terms(self, result: RunResult) -> tuple[float, float]:
        """``(1 - J, mean arrival error / window length)`` of one run, both dimensionless in [0, 1].

        The arrival term is 0 when there are no detections.
        """
        overlap = 1.0 - jaccard(result.burned_mask, self.observed)
        if not self.detections:
            return overlap, 0.0
        err = arrival_interval_error(
            result.arrival_time, self.detections, self.footprint_m, self.window_end_s, dx=result.grid.dx
        )
        return overlap, float(np.mean(err)) / self.window_s


def synthetic_detections(
    arrival: FloatArray,
    overpass_s: Sequence[float] | FloatArray,
    ignition_s: float,
    window_end_s: float,
    footprint_m: float,
    dx: float,
) -> list[Detection]:
    """First detections of a fire with known arrival times, seen at synthetic overpasses.

    Pixel centres are laid out every ``round(footprint_m / dx)`` cells in row and column, starting
    half a footprint from the grid corner. A pixel sees the fire once any cell of its footprint
    has burned (:func:`wildfire_sim.metrics.footprint_arrival`, the footprint the arrival error
    searches). Its detection follows :mod:`wildfire_sim.data.validation`: ``t_detect`` is the
    first overpass at or after that arrival, ``t_clear`` the overpass before it, or the
    ignition if that came later. Pixels not seen by ``window_end_s`` give no detection.

    Args:
        arrival: ``(H, W)`` arrival times, s (NaN where never burned).
        overpass_s: Overpass times, s, increasing.
        ignition_s: Ignition time, s.
        window_end_s: End of the window, s.
        footprint_m: Side of a pixel's footprint, m.
        dx: Cell size, m.

    Returns:
        Detections ordered by detection time, then row and column.
    """
    times = np.asarray(overpass_s, dtype=np.float64)
    if times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("overpass_s must be strictly increasing")
    if footprint_m <= 0 or dx <= 0:
        raise ValueError("footprint_m and dx must be > 0")
    step = max(1, round(footprint_m / dx))
    half = int(footprint_m / 2 // dx)
    h, w = arrival.shape
    out: list[Detection] = []
    for r, c in itertools.product(range(step // 2, h, step), range(step // 2, w, step)):
        t_fire = footprint_arrival(arrival, r, c, half)
        if not math.isfinite(t_fire):
            continue
        k = int(np.searchsorted(times, t_fire, side="left"))  # first overpass at or after the arrival
        if k == times.size or times[k] > window_end_s or times[k] < ignition_s:
            continue
        t_clear = max(float(times[k - 1]), ignition_s) if k > 0 else ignition_s
        out.append(Detection(t_clear, float(times[k]), r, c))
    return sorted(out, key=lambda d: (d.t_detect, d.row, d.col))


def synthetic_target(
    config: ScenarioConfig,
    truth: Mapping[str, float],
    *,
    seed: int,
    overpass_interval_s: float,
    footprint_m: float,
    grid: Grid | None = None,
) -> tuple[Target, RunResult]:
    """A synthetic "observed" fire: one run of ``config`` with the parameters ``truth``.

    The window is ``config.ignition.time_s`` to ``config.duration_s``. The observed mask is the
    run's burned mask at the window end; the detections are sampled from its arrival times at
    overpasses every ``overpass_interval_s`` after the ignition (:func:`synthetic_detections`).

    Args:
        config: Scenario of the fire.
        truth: Rule parameters of the truth run (merged into ``config.spread.params``).
        seed: Seed of the truth run; use one that is not among the calibration seeds.
        overpass_interval_s: Time between synthetic overpasses, s.
        footprint_m: Side of a detection's footprint, m.
        grid: Starting grid; None builds it from ``config.grid``.

    Returns:
        The target and the truth run.

    Raises:
        ValueError: The truth run reached the grid edge, so it is not a whole fire.
    """
    if not overpass_interval_s > 0:
        raise ValueError("overpass_interval_s must be > 0 s")
    result = run(with_params(config, truth), seed=seed, grid=grid, record_events=False)
    if result.provenance["reached_edge_s"] is not None:
        raise ValueError("the synthetic observed fire reached the grid edge: enlarge the grid or shorten the window")
    t0, t1 = config.ignition.time_s, config.duration_s
    overpasses = np.arange(t0 + overpass_interval_s, t1 + 1e-9, overpass_interval_s)
    detections = synthetic_detections(result.arrival_time, overpasses, t0, t1, footprint_m, result.grid.dx)
    text = ", ".join(f"{k}={v:g}" for k, v in truth.items())
    target = Target(
        result.burned_mask, tuple(detections), t0, t1, footprint_m, f"synthetic fire on {config.name} ({text})"
    )
    return target, result


@dataclass(frozen=True)
class RealFire:
    """A validation fire set up on its own grid (``data/cache/<region>``).

    Attributes:
        name: Region name (``configs/regions.yaml``).
        config: Scenario: the cached grid and weather, the rule, ignited at the recorded cell and
            time, ``duration_s`` the window end, ``start_time_utc`` the grid's weather start.
        grid: The region's grid.
        observations: Ignition, perimeters and VIIRS overpasses of the fire.
    """

    name: str
    config: ScenarioConfig
    grid: Grid
    observations: FireObservations

    @property
    def ignition_s(self) -> float:
        """Ignition time, s of simulated time."""
        return self.config.ignition.time_s

    @property
    def end_s(self) -> float:
        """End of the run, s of simulated time."""
        return self.config.duration_s

    @property
    def until(self) -> dt.datetime:
        """UTC instant of the end of the run."""
        assert self.config.start_time_utc is not None
        return self.config.start_time_utc + dt.timedelta(seconds=self.end_s)


def real_fire(
    name: str,
    hours: float,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
    dt_s: float = 60.0,
    rule: str = "rothermel",
    engine: Literal["numpy", "numba"] | None = None,
    viirs: bool = True,
) -> RealFire:
    """Set up a built region's fire for ``hours`` after its recorded ignition.

    Listing the VIIRS overpasses queries NASA's public catalogue, so with ``viirs`` this needs an
    internet connection even when every overpass is cached.

    Args:
        name: Region name.
        hours: Hours simulated after the ignition, h.
        cache_root: Directory of the built regions.
        dt_s: Time step, s.
        rule: Spread rule name.
        engine: None uses ``"numba"`` when numba is installed, else ``"numpy"``.
        viirs: Also load the VIIRS overpasses (False: ignition and perimeters only, offline
            once cached).

    Raises:
        FileNotFoundError: The region is not built; the message says how to build it.
    """
    region = load_regions().regions[name]
    root = Path(cache_root)
    grid_path = root / name / "grid.npz"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"{name} is not built in {root}; build it first with\n"
            f"  uv run python scripts/download_data.py {name} --observations"
        )
    grid = Grid.load(grid_path)
    weather = load_yaml(root / name / "weather.yaml", WeatherConfig)
    obs = fetch_observations(name, region, cache_root=root, viirs=viirs)
    ignition_s = obs.seconds(grid, obs.ignition.time_utc)
    if engine is None:
        engine = "numba" if importlib.util.find_spec("numba") else "numpy"
    config = ScenarioConfig.model_validate(
        {
            "name": name,
            "grid": {"kind": "cached", "path": str(grid_path)},
            "weather": weather.model_dump(),
            "spread": {"rule": rule},
            "ignition": {"cells": [list(obs.ignition_cell(grid))], "time_s": ignition_s},
            "dt_s": dt_s,
            "duration_s": ignition_s + hours * 3600.0,
            "start_time_utc": region.weather_start_utc.isoformat(),
            "engine": engine,
        }
    )
    return RealFire(name, config, grid, obs)


def observed_target(fire: RealFire, footprint_m: float) -> Target:
    """The observed growth of a real fire over its window (docs/plan.md section 7).

    The observed mask is the cells inside a daily perimeter posted by the window end; the
    detections are the VIIRS first detections up to the window end.

    Raises:
        ValueError: The fire has no daily perimeters (the final MTBS perimeter includes
            suppression and the whole fire, so it cannot score early growth), or none was posted
            by the window end.
    """
    obs = fire.observations
    if not obs.daily:
        raise ValueError(
            f"{fire.name} has no daily perimeters: calibration scores early growth, and the final "
            "perimeter includes suppression (docs/plan.md section 7)"
        )
    observed = obs.observed_burn_mask(fire.grid, fire.until)
    if not observed.any():
        raise ValueError(f"no daily perimeter of {fire.name} was posted by {fire.until:%Y-%m-%d %H:%M} UTC")
    detections = tuple(d for d in obs.detections(fire.grid) if d.t_detect <= fire.end_s)
    hours = (fire.end_s - fire.ignition_s) / 3600.0
    return Target(observed, detections, fire.ignition_s, fire.end_s, footprint_m, f"region {fire.name}, {hours:g} h")


# ----------------------------------------------------------------------------- objective


def with_params(config: ScenarioConfig, params: Mapping[str, float]) -> ScenarioConfig:
    """``config`` with ``params`` merged into ``spread.params`` (re-validated)."""
    data = config.model_dump()
    data["spread"]["params"] = {**data["spread"]["params"], **{k: float(v) for k, v in params.items()}}
    return ScenarioConfig.model_validate(data)


@dataclass(frozen=True)
class Evaluation:
    """The objective of one parameter set.

    Attributes:
        params: Rule parameters.
        objective: ``w_o (1 - J) + w_a E`` averaged over the seeds; None if rejected.
        overlap: ``1 - J`` per seed (dimensionless), for the seeds run.
        arrival: Normalized arrival-interval error ``E`` per seed (dimensionless).
        reached_edge_s: Time, s, at which the first run to reach the grid edge did so; None if
            no run did. The remaining seeds are not run once one has.
    """

    params: dict[str, float]
    objective: float | None
    overlap: tuple[float, ...]
    arrival: tuple[float, ...]
    reached_edge_s: float | None

    @property
    def rejected(self) -> bool:
        """True if a run reached the grid edge (the edge rule)."""
        return self.reached_edge_s is not None


def evaluate(
    config: ScenarioConfig,
    params: Mapping[str, float],
    target: Target,
    seeds: Sequence[int],
    weights: tuple[float, float],
    *,
    grid: Grid | None = None,
) -> Evaluation:
    """Run ``config`` with ``params`` once per seed and score it against ``target`` (module docstring).

    Args:
        config: Scenario; its ignition time and ``duration_s`` must be the target's window.
        params: Rule parameters merged into ``config.spread.params``.
        target: The observed fire.
        seeds: Run seeds, the same for every parameter set.
        weights: ``(w_overlap, w_arrival)``, dimensionless.
        grid: Starting grid; None builds it from ``config.grid`` (pass one to build it once).
    """
    if not seeds:
        raise ValueError("need at least one seed")
    if config.ignition.time_s != target.ignition_s or config.duration_s != target.window_end_s:
        raise ValueError("the scenario's ignition time and duration must be the target's window")
    scenario = with_params(config, params)
    rule = make_spread_rule(scenario.spread.rule, **scenario.spread.params)
    g = build_grid(scenario.grid) if grid is None else grid
    overlap: list[float] = []
    arrival: list[float] = []
    for seed in seeds:
        result = run(scenario, seed=seed, rule=rule, grid=g, record_events=False)
        edge = result.provenance["reached_edge_s"]
        if edge is not None:
            return Evaluation(dict(params), None, tuple(overlap), tuple(arrival), float(edge))
        o, a = target.terms(result)
        overlap.append(o)
        arrival.append(a)
    w_o, w_a = weights
    objective = float(np.mean([w_o * o + w_a * a for o, a in zip(overlap, arrival, strict=True)]))
    return Evaluation(dict(params), objective, tuple(overlap), tuple(arrival), None)


def objective_surface(
    config: ScenarioConfig,
    target: Target,
    center: Mapping[str, float],
    space: Mapping[str, ParameterBound],
    steps: Sequence[float],
    seeds: Sequence[int],
    weights: tuple[float, float],
    *,
    grid: Grid | None = None,
) -> list[Evaluation]:
    """The objective on the grid ``center * (1 + step)`` of every parameter, clipped to its bounds.

    Every combination of the parameters' values is evaluated (``len(steps) ** n`` points before
    duplicates from clipping are removed), on the same seeds as the study.
    """
    axes = []
    for name in space:
        b = space[name]
        axes.append(sorted({min(max(center[name] * (1.0 + s), b.low), b.high) for s in steps}))
    g = build_grid(config.grid) if grid is None else grid
    return [
        evaluate(config, dict(zip(space, values, strict=True)), target, seeds, weights, grid=g)
        for values in itertools.product(*axes)
    ]


# ----------------------------------------------------------------------------- the study


@dataclass(frozen=True)
class StudyResult:
    """Outcome of :func:`run_study`.

    Attributes:
        best: Evaluation of the best scored trial.
        trials: Every trial in order; rejected ones have ``objective`` None.
        seeds: The fixed run seeds.
        sampler_seed: Seed of the TPE sampler.
    """

    best: Evaluation
    trials: tuple[Evaluation, ...]
    seeds: tuple[int, ...]
    sampler_seed: int

    @property
    def n_rejected(self) -> int:
        """Trials rejected by the edge rule."""
        return sum(t.rejected for t in self.trials)


def run_study(
    config: ScenarioConfig,
    target: Target,
    space: Mapping[str, ParameterBound],
    *,
    n_trials: int,
    seeds: Sequence[int],
    weights: tuple[float, float],
    sampler_seed: int,
    grid: Grid | None = None,
    storage: str | None = None,
    study_name: str | None = None,
    progress: Callable[[int, Evaluation], None] | None = None,
) -> StudyResult:
    """Minimize the objective over ``space`` with an Optuna study (TPE sampler).

    Args:
        config: Scenario; ignition time and ``duration_s`` are the target's window.
        target: The observed fire.
        space: Bounds of the calibrated parameters (rule keyword arguments).
        n_trials: Trials to run (>= 1).
        seeds: Fixed run seeds.
        weights: ``(w_overlap, w_arrival)``, dimensionless.
        sampler_seed: Seed of the TPE sampler; with a fresh study the trials reproduce.
        grid: Starting grid; None builds it once from ``config.grid``.
        storage: Optuna storage URL (e.g. ``sqlite:///runs/calibration.db``) that keeps a record
            of the study; None keeps it in memory only.
        study_name: Name of the study in ``storage``; it must not exist there yet.
        progress: Called after every trial with its number and evaluation.

    Raises:
        ImportError: optuna is not installed.
        RuntimeError: Every trial was rejected by the edge rule.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    try:
        optuna = importlib.import_module("optuna")
    except ImportError as exc:
        raise ImportError("calibration needs optuna: install the opt extra (uv sync --extra opt)") from exc
    g = build_grid(config.grid) if grid is None else grid
    evaluations: dict[int, Evaluation] = {}

    def objective(trial: Any) -> float:
        params = {name: trial.suggest_float(name, b.low, b.high, log=b.log) for name, b in space.items()}
        ev = evaluate(config, params, target, seeds, weights, grid=g)
        evaluations[trial.number] = ev
        if progress is not None:
            progress(trial.number, ev)
        if ev.objective is None:
            trial.set_user_attr("reached_edge_s", ev.reached_edge_s)
            raise optuna.TrialPruned("a run reached the grid edge")
        trial.set_user_attr("overlap", float(np.mean(ev.overlap)))
        trial.set_user_attr("arrival", float(np.mean(ev.arrival)))
        return ev.objective

    study = optuna.create_study(
        direction="minimize",
        # multivariate: the parameters are modelled jointly, as they can trade off; the default
        # since optuna 5, explicit so optuna 3.x and 4.x run the same search
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True),
        storage=storage,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials)
    trials = [evaluations[n] for n in sorted(evaluations)]
    scored = [t for t in trials if t.objective is not None]
    if not scored:
        raise RuntimeError(
            f"all {len(trials)} trials were rejected: in each, a run reached the grid edge. The grid is too "
            "small for the window or the bounds allow too fast a fire; no parameters were written"
        )
    best = min(scored, key=lambda t: t.objective if t.objective is not None else math.inf)
    return StudyResult(best, tuple(trials), tuple(int(s) for s in seeds), sampler_seed)


# ----------------------------------------------------------------------------- configs/calibrated.yaml


class SurfacePoint(_Strict):
    """One point of the objective surface around the optimum (None: rejected by the edge rule)."""

    params: dict[str, float]
    objective: float | None
    overlap: float | None
    arrival: float | None


class CalibrationProvenance(_Strict):
    """How a calibrated parameter set was produced."""

    target: str
    rule: str
    window_hours: float
    ignition_s: float
    observed_cells: int
    n_detections: int
    seeds: list[int]
    sampler_seed: int
    n_trials: int
    n_rejected: int
    objective: float
    overlap: float
    arrival: float
    weights: dict[str, float]
    bounds: dict[str, ParameterBound]
    surface: list[SurfacePoint]
    git_commit: str | None
    package_version: str
    created_utc: dt.datetime
    command: str


class CalibratedParams(_Strict):
    """Contents of ``configs/calibrated.yaml``: rule parameters and how they were obtained.

    ``status`` is ``"placeholder"`` (neutral values, produced by no study), ``"calibrated"``
    (a study on a real fire) or ``"synthetic"`` (a study on a synthetic observed fire, a test of
    the method). Only a placeholder may lack ``provenance``.
    """

    status: Literal["placeholder", "calibrated", "synthetic"]
    source: Annotated[str, Field(min_length=1)]
    rule: Annotated[str, Field(min_length=1)]
    params: dict[str, float]
    provenance: CalibrationProvenance | None = None

    @model_validator(mode="after")
    def _check(self) -> CalibratedParams:
        if (self.status == "placeholder") != (self.provenance is None):
            raise ValueError("a placeholder has no provenance, and a study result must have one")
        return self

    def apply(self, config: ScenarioConfig) -> ScenarioConfig:
        """``config`` with these parameters in ``spread.params``.

        Raises:
            ValueError: ``config`` uses another spread rule.
        """
        if config.spread.rule != self.rule:
            raise ValueError(f"these parameters are for rule {self.rule!r}, the scenario uses {config.spread.rule!r}")
        return with_params(config, self.params)


def load_calibrated(path: str | Path | None = None) -> CalibratedParams:
    """Read ``configs/calibrated.yaml`` (or ``path``)."""
    return load_yaml(configs_dir() / "calibrated.yaml" if path is None else path, CalibratedParams)


def calibrated_params(
    config: ScenarioConfig,
    target: Target,
    study: StudyResult,
    surface: Sequence[Evaluation],
    space: Mapping[str, ParameterBound],
    weights: tuple[float, float],
    *,
    status: Literal["calibrated", "synthetic"],
    command: str,
) -> CalibratedParams:
    """The record of a finished study, ready for :func:`write_calibrated`."""
    best = study.best
    assert best.objective is not None
    points = [
        SurfacePoint(
            params=p.params,
            objective=p.objective,
            overlap=None if p.rejected else float(np.mean(p.overlap)),
            arrival=None if p.rejected else float(np.mean(p.arrival)),
        )
        for p in surface
    ]
    code = model_provenance(config)
    provenance = CalibrationProvenance(
        target=target.description,
        rule=config.spread.rule,
        window_hours=target.window_s / 3600.0,
        ignition_s=target.ignition_s,
        observed_cells=int(np.count_nonzero(target.observed)),
        n_detections=len(target.detections),
        seeds=list(study.seeds),
        sampler_seed=study.sampler_seed,
        n_trials=len(study.trials),
        n_rejected=study.n_rejected,
        objective=best.objective,
        overlap=float(np.mean(best.overlap)),
        arrival=float(np.mean(best.arrival)),
        weights={"overlap": weights[0], "arrival": weights[1]},
        bounds=dict(space),
        surface=points,
        git_commit=code["git_commit"],
        package_version=__version__,
        created_utc=dt.datetime.now(dt.UTC),
        command=command,
    )
    source = (
        f"WP 3.1 study on {target.description}: {len(study.trials)} trials x {len(study.seeds)} seeds"
        if status == "calibrated"
        else f"SYNTHETIC - a test of the method, not a calibration of the model: {target.description}"
    )
    return CalibratedParams(
        status=status, source=source, rule=config.spread.rule, params=dict(best.params), provenance=provenance
    )


def write_calibrated(record: CalibratedParams, path: str | Path) -> None:
    """Write ``record`` as YAML (round-trips through :func:`load_calibrated`)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(record, path)
