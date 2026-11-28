"""Validation of the calibrated model against an observed fire (WP 3.3, docs/plan.md section 7).

The model is calibrated on fire A (WP 3.1) and run, without retuning, on fire B. A validation
of one fire is an ensemble of unsuppressed runs from the recorded ignition to the end of a
comparison window, scored against what was observed (:class:`Observed`):

* **Ensemble** (docs/plan.md section 6). Run ``i`` uses the seed
  ``derive_seeds(base_seed, n)[i]``: its link thresholds come from that seed and its input
  perturbation (wind speed and direction, fuel moisture, a rate multiplier) from the seed's
  perturbation stream, drawn and applied exactly as :func:`wildfire_sim.ensemble.run_ensemble`
  does (the member here is bit-for-bit the ensemble's; tested). The ignition is not perturbed.
  The members are run here, not through ``run_ensemble``, because validation also needs each
  run's cell-hours under the lattice cap and its breaches, which ``EnsembleResult`` does not
  keep, and because it needs no ``(n, H, W)`` stack of arrival times: each run is scored as it
  arrives and only its scores are kept.
* **Edge rule** (section 2). A run that reaches the grid edge is a lower bound on its fire. It
  is counted and listed, and left out of every metric; burned area is also given over all runs,
  labelled as a lower bound. Milestone M2 requires that no reported run reaches the edge.
* **Metrics** (section 7, all from :mod:`wildfire_sim.metrics`), over the runs that stayed
  inside the grid:

  - Jaccard and Sorensen overlap of the consensus mask (cells burned in at least
    ``threshold`` of the runs) with the observed burned mask at the window end (the daily
    perimeters posted by then), and the mean of each run's overlap with a percentile
    bootstrap interval across runs;
  - the same against the final (MTBS) perimeter, reported separately because a real fire's
    final perimeter includes suppression, which the runs do not have (the suppression
    confound);
  - per VIIRS detection in the window, the arrival-interval error
    (:func:`~wildfire_sim.metrics.arrival_interval_error`); per run the fraction of detections
    inside their interval and the median and mean error, each averaged over runs with an
    interval;
  - per day with an observed daily perimeter, the simulated over observed burned area;
  - the burned area (Student-t interval, :func:`~wildfire_sim.ensemble.mean_ci`) and the number
    of runs that would bring its interval under a stated fraction of the mean
    (:func:`~wildfire_sim.ensemble.runs_for_precision`);
  - the burning cell-hours under the lattice cap (``lb_capped_cell_hours``, WP 2.1d) and the
    breaks breached.

The settings are read from ``configs/validation.yaml`` (:func:`load_validation_settings`).
``scripts/validate.py`` runs a validation and writes its tables and figures.

References:
    Efron, B. & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall,
    chapter 13 (percentile intervals).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import multiprocessing
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from wildfire_sim import sim
from wildfire_sim.calibration import CalibratedParams, RealFire, Target
from wildfire_sim.config import CitedValue, ScenarioConfig, configs_dir, load_yaml
from wildfire_sim.data.validation import Detection
from wildfire_sim.ensemble import (
    ScaledRateRule,
    derive_seeds,
    draw_factors,
    mean_ci,
    perturb_config,
    perturbation_rng,
    perturbed_wind_field,
)
from wildfire_sim.ensemble import runs_for_precision as _runs_for_precision
from wildfire_sim.metrics import (
    M2_PER_HA,
    arrival_interval_error,
    bootstrap_ci,
    consensus_mask,
    footprint_arrival,
    interval_error,
    jaccard,
    sorensen,
)
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.spread.base import SpreadRule
from wildfire_sim.types import BoolArray, FloatArray, IntArray

DAY_S = 86400.0
"""Length of a day, s."""

ParamStatus = Literal["placeholder", "calibrated", "synthetic"]


# ----------------------------------------------------------------------------- settings


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnsembleSigmas(_Strict):
    """Input uncertainty of every run (the perturbation fields of :class:`~wildfire_sim.config.EnsembleConfig`)."""

    wind_speed_sigma_frac: CitedValue
    wind_direction_sigma_deg: CitedValue
    moisture_sigma_frac: CitedValue
    ros_multiplier_sigma_frac: CitedValue

    def values(self) -> dict[str, float]:
        """Field name -> sigma (units of the field)."""
        return {name: float(getattr(self, name).value) for name in type(self).model_fields}


class ValidationSettings(_Strict):
    """Contents of ``configs/validation.yaml`` (every value cited, CONTRIBUTING rule 3)."""

    n_runs: CitedValue
    base_seed: CitedValue
    consensus_threshold: CitedValue
    ci_level: CitedValue
    n_boot: CitedValue
    bootstrap_seed: CitedValue
    rel_half_width: CitedValue
    dt_s: CitedValue
    window_hours: dict[str, CitedValue]
    ensemble: EnsembleSigmas

    @model_validator(mode="after")
    def _check(self) -> ValidationSettings:
        for name in ("n_runs", "base_seed", "n_boot", "bootstrap_seed"):
            value = getattr(self, name).value
            if value != int(value) or value < (1 if name == "n_runs" else 0):
                raise ValueError(f"{name} must be a whole number (>= 1 for counts, >= 0 for seeds)")
        if self.n_boot.value < 100:
            raise ValueError("n_boot must be at least 100")
        if not 0 < self.consensus_threshold.value <= 1:
            raise ValueError("consensus_threshold must lie in (0, 1]")
        if not 0 < self.ci_level.value < 1:
            raise ValueError("ci_level must lie in (0, 1)")
        if not self.rel_half_width.value > 0 or not self.dt_s.value > 0:
            raise ValueError("rel_half_width and dt_s must be > 0")
        if any(not w.value > 0 for w in self.window_hours.values()):
            raise ValueError("window_hours must be > 0 h")
        if any(v < 0 for v in self.ensemble.values().values()):
            raise ValueError("ensemble sigmas must be >= 0")
        return self

    def seeds(self, n_runs: int | None = None) -> tuple[int, ...]:
        """The run seeds: :func:`~wildfire_sim.ensemble.derive_seeds` of ``base_seed``."""
        return derive_seeds(int(self.base_seed.value), int(self.n_runs.value) if n_runs is None else n_runs)

    def with_ensemble(self, config: ScenarioConfig, n_runs: int | None = None) -> ScenarioConfig:
        """``config`` with these sigmas, the run count and the base seed in its ``ensemble`` section."""
        data = config.model_dump()
        data["ensemble"] = {
            **self.ensemble.values(),
            "ignition_jitter_cells": 0,
            "n_runs": int(self.n_runs.value) if n_runs is None else n_runs,
            "base_seed": int(self.base_seed.value),
        }
        return ScenarioConfig.model_validate(data)


def load_validation_settings(path: str | Path | None = None) -> ValidationSettings:
    """Read ``configs/validation.yaml`` (or ``path``)."""
    return load_yaml(configs_dir() / "validation.yaml" if path is None else path, ValidationSettings)


# ----------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class ParameterRecord:
    """Which parameters a validation ran with, for its outputs.

    Attributes:
        status: ``status`` of the calibrated file: ``"calibrated"`` (a study on a real fire),
            ``"placeholder"`` (neutral values, produced by no study) or ``"synthetic"`` (a study
            on a synthetic fire).
        rule: Spread rule.
        params: Rule parameters applied, as in the file.
        source: The file's ``source`` line.
        path: The calibrated file.
        sha256: Hex SHA-256 of the file's bytes.
        calibrated_on: The study's target (``provenance.target``), None for a placeholder.
    """

    status: ParamStatus
    rule: str
    params: dict[str, float]
    source: str
    path: str
    sha256: str
    calibrated_on: str | None

    @classmethod
    def read(cls, record: CalibratedParams, path: str | Path) -> ParameterRecord:
        """The record of ``record``, loaded from ``path``."""
        return cls(
            status=record.status,
            rule=record.rule,
            params=dict(record.params),
            source=record.source,
            path=str(path),
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            calibrated_on=None if record.provenance is None else record.provenance.target,
        )

    def warning(self) -> str | None:
        """A warning to print and stamp on every figure, or None for calibrated parameters."""
        if self.status == "placeholder":
            return "PARAMETERS NOT CALIBRATED (placeholder, produced by no study)"
        if self.status == "synthetic":
            return "PARAMETERS FROM A SYNTHETIC CALIBRATION (a test of the method, not of the model)"
        return None


def check_not_tuned_on(record: CalibratedParams, region: str) -> None:
    """Refuse parameters calibrated on ``region`` itself (fire B is tested without retuning).

    Raises:
        ValueError: The study's target is ``region``.
    """
    if record.provenance is not None and record.provenance.target.startswith(f"region {region},"):
        raise ValueError(
            f"these parameters were calibrated on {region} itself ({record.provenance.target!r}); "
            "the test fire must be run with fire A's parameters, unchanged (docs/plan.md section 7)"
        )


def check_seeds_disjoint(seeds: Sequence[int], used: Iterable[int]) -> None:
    """Refuse run seeds that the calibration used (its run seeds, its truth seed).

    Raises:
        ValueError: A seed is shared.
    """
    shared = sorted(set(seeds) & set(used))
    if shared:
        raise ValueError(f"{len(shared)} validation seeds were used by the calibration: {shared[:3]}; change base_seed")


# ----------------------------------------------------------------------------- what was observed


@dataclass(frozen=True)
class Observed:
    """What a validation ensemble is compared with.

    Attributes:
        description: What the observations are (for the outputs).
        synthetic: True for a synthetic observed fire (a test of the pipeline, not a validation).
        ignition_s: Ignition time, s of simulated time.
        window_end_s: End of the comparison window, s.
        footprint_m: Side of a detection's footprint, m.
        ignition_cell: ``(row, col)`` of the ignition.
        dx: Cell size of the grid the masks and cells are on, m.
        detections: First detections with ``t_detect`` in the window, ``(t_clear, t_detect, row,
            col)``, s.
        window_mask: ``(H, W)`` bool, cells observed burned by the window end (inside a daily
            perimeter posted by then); None when the fire has no daily perimeter by then.
        final_mask: ``(H, W)`` bool, cells inside the final perimeter.
        final_label: What the final perimeter is (for a real fire: it includes suppression).
        daily: ``(t_s, (H, W) bool)`` per day end in the window: cells observed burned by
            ``t_s``; only days with a non-empty observed mask.
        notes: Caveats carried into the outputs.
    """

    description: str
    synthetic: bool
    ignition_s: float
    window_end_s: float
    footprint_m: float
    ignition_cell: tuple[int, int]
    dx: float
    detections: tuple[Detection, ...]
    window_mask: BoolArray | None
    final_mask: BoolArray
    final_label: str
    daily: tuple[tuple[float, BoolArray], ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.window_end_s > self.ignition_s:
            raise ValueError("the window must end after the ignition")
        if not np.any(self.final_mask):
            raise ValueError("the final perimeter is empty")
        if any(not self.ignition_s <= d.t_detect <= self.window_end_s for d in self.detections):
            raise ValueError("every detection must fall in the window")

    @property
    def window_hours(self) -> float:
        """Window length, h."""
        return (self.window_end_s - self.ignition_s) / 3600.0


def day_ends(ignition_s: float, window_end_s: float) -> list[float]:
    """Ends of the days of a window, s: every 24 h after the ignition before the window end, then the window end."""
    if not window_end_s > ignition_s:
        raise ValueError("the window must end after the ignition")
    n_full = math.ceil((window_end_s - ignition_s) / DAY_S) - 1
    return [ignition_s + k * DAY_S for k in range(1, n_full + 1)] + [window_end_s]


def observed_fire(fire: RealFire, footprint_m: float) -> Observed:
    """The observations of a real fire over its window (``fire.end_s`` is the window end).

    The window mask and the daily masks are the cells inside a daily perimeter posted by then
    (:meth:`~wildfire_sim.data.validation.FireObservations.observed_burn_mask`); a fire without
    daily perimeters (fire B: NIFC's archive starts in 2022) has neither, and is compared with
    its final MTBS perimeter, which includes suppression, and its VIIRS detections.
    """
    obs, grid = fire.observations, fire.grid
    start = fire.config.start_time_utc
    assert start is not None  # real_fire sets it
    notes: list[str] = []
    window_mask: BoolArray | None = None
    daily: list[tuple[float, BoolArray]] = []
    if not obs.daily:
        notes.append(
            f"{fire.name} has no daily perimeters (NIFC's archive starts in 2022): no overlap at the window end "
            "and no daily area ratio; the overlap is scored against the final perimeter only, which includes "
            "suppression"
        )
    else:
        for t in day_ends(fire.ignition_s, fire.end_s):
            mask = obs.observed_burn_mask(grid, start + dt.timedelta(seconds=t))
            if mask.any():
                daily.append((t, mask))
        if daily and daily[-1][0] == fire.end_s:
            window_mask = daily[-1][1]
        else:
            notes.append(
                f"no daily perimeter of {fire.name} was posted by the window end: no overlap at the window end"
            )
    detections = tuple(d for d in obs.detections(grid) if d.t_detect <= fire.end_s)
    return Observed(
        description=f"region {fire.name}, {(fire.end_s - fire.ignition_s) / 3600.0:g} h after the recorded ignition",
        synthetic=False,
        ignition_s=fire.ignition_s,
        window_end_s=fire.end_s,
        footprint_m=footprint_m,
        ignition_cell=obs.ignition_cell(grid),
        dx=grid.dx,
        detections=detections,
        window_mask=window_mask,
        final_mask=obs.observed_burn_mask(grid),
        final_label="final MTBS perimeter (includes suppression)",
        daily=tuple(daily),
        notes=tuple(notes),
    )


def synthetic_observed(
    target: Target, truth_arrival: FloatArray, ignition_cell: tuple[int, int], dx: float
) -> Observed:
    """A synthetic observed fire (:func:`wildfire_sim.calibration.synthetic_target`) as an :class:`Observed`.

    The truth run is unsuppressed and ends at the window end, so its final mask is its window
    mask; its daily masks are its cells burned by each day end.

    Args:
        target: The synthetic target.
        truth_arrival: ``(H, W)`` arrival times of the truth run, s (NaN where unburned).
        ignition_cell: ``(row, col)`` of its ignition.
        dx: Cell size, m.
    """
    with np.errstate(invalid="ignore"):
        daily = tuple((t, truth_arrival <= t) for t in day_ends(target.ignition_s, target.window_end_s))
    return Observed(
        description=f"SYNTHETIC observed fire: {target.description}",
        synthetic=True,
        ignition_s=target.ignition_s,
        window_end_s=target.window_end_s,
        footprint_m=target.footprint_m,
        ignition_cell=ignition_cell,
        dx=dx,
        detections=target.detections,
        window_mask=target.observed,
        final_mask=target.observed,
        final_label="synthetic final mask (the truth run at the window end; no suppression)",
        daily=tuple((t, m) for t, m in daily if m.any()),
        notes=("SYNTHETIC observed fire: a test of the pipeline, not a validation result",),
    )


# ----------------------------------------------------------------------------- one run


@dataclass(frozen=True)
class Member:
    """One ensemble run, before scoring.

    Attributes:
        seed: Run seed.
        perturbation: Factors applied (keys of :data:`wildfire_sim.ensemble.PERTURBATION_KEYS`).
        arrival: ``(H, W)`` float64 arrival times, s (NaN where unburned).
        reached_edge_s: Time the fire reached the grid's edge band, s; NaN if it did not.
        lb_capped_cell_hours: Burning cell-hours under the lattice cap (WP 2.1d), cell-h.
        breaches: Breaks breached.
    """

    seed: int
    perturbation: dict[str, float]
    arrival: FloatArray
    reached_edge_s: float
    lb_capped_cell_hours: float
    breaches: int


def run_member(config: ScenarioConfig, seed: int) -> Member:
    """Run ``seed`` of the ensemble of ``config`` (perturbations from ``config.ensemble``).

    The draws and their application are those of :func:`wildfire_sim.ensemble.run_ensemble`
    (module docstring of :mod:`wildfire_sim.ensemble`); the rate factor wraps the rule so that
    the cell-hours under the lattice cap are still counted.

    Raises:
        ValueError: ``config.ensemble.ignition_jitter_cells`` is not 0 (the ignition is not
            perturbed here).
    """
    if config.ensemble.ignition_jitter_cells:
        raise ValueError("validation does not perturb the ignition: ignition_jitter_cells must be 0")
    factors = draw_factors(perturbation_rng(seed), config.ensemble)
    rule: SpreadRule = make_spread_rule(config.spread.rule, **config.spread.params)
    if factors["ros_multiplier"] != 1.0:
        rule = ScaledRateRule(rule, factors["ros_multiplier"])
    wind = perturbed_wind_field(config, factors)
    result = sim.run(perturb_config(config, factors), seed=seed, rule=rule, record_events=False, wind_field=wind)
    edge = result.provenance["reached_edge_s"]
    return Member(
        seed=seed,
        perturbation=factors,
        arrival=result.arrival_time,
        reached_edge_s=math.nan if edge is None else float(edge),
        lb_capped_cell_hours=float(result.provenance["lb_capped_cell_hours"]),
        breaches=int(result.provenance["breaks"]["breaches"]),
    )


def _run_member_task(task: tuple[dict[str, Any], int]) -> Member:
    """:func:`run_member` from plain data, for a ``spawn`` worker pool."""
    data, seed = task
    return run_member(ScenarioConfig.model_validate(data), seed)


@dataclass(frozen=True)
class RunScore:
    """The scores of one run against the observations (units in the field names: s, h, ha).

    Overlaps are NaN where there is no observed mask, detection values are per detection in the
    order of :attr:`Observed.detections`, and ``area_ratio`` is per entry of
    :attr:`Observed.daily`.
    """

    seed: int
    perturbation: dict[str, float]
    reached_edge_s: float
    burned_area_ha: float
    lb_capped_cell_hours: float
    breaches: int
    jaccard_window: float
    sorensen_window: float
    jaccard_final: float
    sorensen_final: float
    detection_arrival_s: tuple[float, ...]
    detection_error_s: tuple[float, ...]
    area_ratio: tuple[float, ...]

    @property
    def reached_edge(self) -> bool:
        """The run reached the grid edge: a lower bound on its fire."""
        return not math.isnan(self.reached_edge_s)

    @property
    def inside_fraction(self) -> float:
        """Fraction of detections whose simulated arrival lies in its interval (NaN without detections)."""
        err = np.asarray(self.detection_error_s)
        return float(np.mean(err == 0.0)) if err.size else math.nan

    @property
    def median_error_h(self) -> float:
        """Median arrival-interval error, h (NaN without detections)."""
        return float(np.median(self.detection_error_s)) / 3600.0 if self.detection_error_s else math.nan

    @property
    def mean_error_h(self) -> float:
        """Mean arrival-interval error, h (NaN without detections)."""
        return float(np.mean(self.detection_error_s)) / 3600.0 if self.detection_error_s else math.nan


def score_run(member: Member, observed: Observed) -> RunScore:
    """Score one run against ``observed`` (module docstring); arrivals after the window end count as unburned."""
    dx = observed.dx
    arrival = np.where(member.arrival <= observed.window_end_s, member.arrival, np.nan)
    burned = ~np.isnan(arrival)
    half = int(observed.footprint_m / 2 // dx)
    dets = observed.detections
    errors = arrival_interval_error(arrival, dets, observed.footprint_m, observed.window_end_s, dx=dx) if dets else []
    window = observed.window_mask
    return RunScore(
        seed=member.seed,
        perturbation=dict(member.perturbation),
        reached_edge_s=member.reached_edge_s,
        burned_area_ha=float(np.count_nonzero(burned)) * dx * dx / M2_PER_HA,
        lb_capped_cell_hours=member.lb_capped_cell_hours,
        breaches=member.breaches,
        jaccard_window=math.nan if window is None else jaccard(burned, window),
        sorensen_window=math.nan if window is None else sorensen(burned, window),
        jaccard_final=jaccard(burned, observed.final_mask),
        sorensen_final=sorensen(burned, observed.final_mask),
        detection_arrival_s=tuple(footprint_arrival(arrival, d.row, d.col, half) for d in dets),
        detection_error_s=tuple(float(e) for e in errors),
        area_ratio=tuple(
            float(np.count_nonzero(arrival <= t)) / float(np.count_nonzero(mask)) for t, mask in observed.daily
        ),
    )


# ----------------------------------------------------------------------------- the ensemble


@dataclass(frozen=True)
class ValidationRuns:
    """Outcome of :func:`run_validation`.

    Attributes:
        runs: Every run's scores, in seed order.
        burn_count_inside: ``(H, W)`` int, number of runs that stayed inside the grid and burned
            the cell by the window end.
        elapsed_s: Wall-clock time, s.
    """

    runs: tuple[RunScore, ...]
    burn_count_inside: IntArray
    elapsed_s: float

    @property
    def inside(self) -> list[RunScore]:
        """The runs that stayed inside the grid (the only ones scored)."""
        return [r for r in self.runs if not r.reached_edge]

    @property
    def edge(self) -> list[tuple[int, RunScore]]:
        """``(run index, run)`` of every run that reached the grid edge."""
        return [(i, r) for i, r in enumerate(self.runs) if r.reached_edge]

    @property
    def burn_probability(self) -> FloatArray | None:
        """``(H, W)`` fraction of the runs inside the grid that burned each cell; None if none stayed inside."""
        n = len(self.inside)
        return None if n == 0 else self.burn_count_inside / n


def run_validation(
    config: ScenarioConfig, observed: Observed, *, n_runs: int, base_seed: int, n_workers: int = 1
) -> ValidationRuns:
    """Run and score the ensemble of ``config`` against ``observed``.

    Run ``i`` uses ``derive_seeds(base_seed, n_runs)[i]``; the result does not depend on
    ``n_workers`` (``spawn`` pool for more than 1).

    Args:
        config: Scenario with the calibrated parameters and the ``ensemble`` sigmas; its ignition
            time and ``duration_s`` are the window.
        observed: The observations.
        n_runs: Number of runs (>= 1).
        base_seed: Ensemble seed (>= 0).
        n_workers: Processes (>= 1).
    """
    start = time.perf_counter()
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if config.ignition.time_s != observed.ignition_s or config.duration_s != observed.window_end_s:
        raise ValueError("the scenario's ignition time and duration must be the observed window")
    data = config.model_dump(mode="json")
    tasks = [(data, seed) for seed in derive_seeds(base_seed, n_runs)]
    count = np.zeros(observed.final_mask.shape, dtype=np.int64)
    runs: list[RunScore] = []

    def take(members: Iterable[Member]) -> None:
        for member in members:
            score = score_run(member, observed)
            if not score.reached_edge:
                count[member.arrival <= observed.window_end_s] += 1
            runs.append(score)

    if n_workers == 1:
        take(_run_member_task(t) for t in tasks)
    else:
        with multiprocessing.get_context("spawn").Pool(processes=min(n_workers, n_runs)) as pool:
            take(pool.imap(_run_member_task, tasks, chunksize=1))
    return ValidationRuns(tuple(runs), count, time.perf_counter() - start)


# ----------------------------------------------------------------------------- metrics table


@dataclass(frozen=True)
class MetricRow:
    """One row of the metrics table.

    Attributes:
        metric: What is measured (e.g. ``"jaccard"``).
        comparison: What it is compared with (e.g. the final perimeter).
        statistic: How the runs are combined (consensus mask, mean over runs, count).
        value: The value, in ``units``.
        lower: Lower end of its interval (NaN: none).
        upper: Upper end of its interval (NaN: none).
        n_runs: Number of runs it is computed from.
        units: Units of ``value``.
        note: Caveat.
    """

    metric: str
    comparison: str
    statistic: str
    value: float
    lower: float
    upper: float
    n_runs: int
    units: str
    note: str = ""


def _bootstrap(values: Sequence[float], level: float, n_boot: int, seed: int) -> tuple[float, float, float]:
    """``bootstrap_ci`` of ``values``; the interval is NaN with fewer than 2 values."""
    if len(values) >= 2:
        return bootstrap_ci(values, level, n_boot=n_boot, seed=seed)
    return (float(values[0]) if values else math.nan), math.nan, math.nan


def metric_rows(
    result: ValidationRuns,
    observed: Observed,
    settings: ValidationSettings,
    *,
    rule_caps: bool,
) -> list[MetricRow]:
    """The metrics table of a validation (module docstring), over the runs inside the grid.

    Args:
        result: The scored runs.
        observed: The observations they were scored against.
        settings: Consensus threshold, interval level, bootstrap and precision settings.
        rule_caps: Whether the spread rule caps the length-to-breadth ratio (it has an
            ``lb_capped`` mask); if not, the cell-hours under the cap are not applicable.

    Returns:
        The rows; empty-sample metrics are left out, and the edge count is always the first row.
    """
    level = float(settings.ci_level.value)
    n_boot, seed = int(settings.n_boot.value), int(settings.bootstrap_seed.value)
    inside, n_all = result.inside, len(result.runs)
    n = len(inside)

    def boot(name: str, comparison: str, values: Sequence[float], units: str, note: str = "") -> MetricRow:
        value, lo, hi = _bootstrap(values, level, n_boot, seed)
        return MetricRow(name, comparison, "mean over runs, bootstrap interval", value, lo, hi, n, units, note)

    edge_note = "lower bounds on their fires, left out of every other row; milestone M2 requires 0"
    rows = [
        MetricRow(
            "runs_reaching_edge", "grid edge", "count", len(result.edge), math.nan, math.nan, n_all, "runs", edge_note
        )
    ]
    prob = result.burn_probability
    if prob is None:
        return rows
    threshold = float(settings.consensus_threshold.value)
    consensus = consensus_mask(prob, threshold)
    consensus_stat = f"consensus mask (burned in >= {threshold:g} of the runs)"
    comparisons: list[tuple[str, BoolArray, str, str]] = []
    if observed.window_mask is not None:
        comparisons.append(("window", observed.window_mask, "jaccard_window", "sorensen_window"))
    comparisons.append(("final", observed.final_mask, "jaccard_final", "sorensen_final"))
    for which, mask, j_field, s_field in comparisons:
        label = (
            f"observed burned by the window end ({observed.window_hours:g} h)"
            if which == "window"
            else observed.final_label
        )
        note = "includes suppression, which the runs do not have" if which == "final" and not observed.synthetic else ""
        for metric, fn, field in (("jaccard", jaccard, j_field), ("sorensen", sorensen, s_field)):
            rows.append(
                MetricRow(
                    metric, label, consensus_stat, fn(consensus, mask), math.nan, math.nan, n, "dimensionless", note
                )
            )
            rows.append(boot(metric, label, [getattr(r, field) for r in inside], "dimensionless", note))
    if observed.detections:
        dets = f"{len(observed.detections)} VIIRS first detections in the window"
        rows.append(boot("arrival_inside_fraction", dets, [r.inside_fraction for r in inside], "fraction"))
        rows.append(boot("arrival_error_median", dets, [r.median_error_h for r in inside], "h"))
        rows.append(boot("arrival_error_mean", dets, [r.mean_error_h for r in inside], "h"))
    for k, (t, mask) in enumerate(observed.daily):
        hours = (t - observed.ignition_s) / 3600.0
        obs_ha = float(np.count_nonzero(mask)) * observed.dx**2 / M2_PER_HA
        rows.append(
            boot(
                "area_ratio",
                f"observed burned {hours:g} h after the ignition ({obs_ha:.4g} ha)",
                [r.area_ratio[k] for r in inside],
                "dimensionless",
                "simulated over observed burned area",
            )
        )
    areas = [r.burned_area_ha for r in inside]
    if n >= 2:
        mean, lo, hi = mean_ci(np.asarray(areas), level)
        rows.append(MetricRow("burned_area", "runs inside the grid", "mean, Student-t interval", mean, lo, hi, n, "ha"))
    else:
        rows.append(MetricRow("burned_area", "runs inside the grid", "mean", areas[0], math.nan, math.nan, n, "ha"))
    if result.edge:
        all_mean = float(np.mean([r.burned_area_ha for r in result.runs]))
        note = f"includes {len(result.edge)} runs cut short at the grid edge: a lower bound"
        rows.append(
            MetricRow("burned_area", "all runs", "mean (lower bound)", all_mean, math.nan, math.nan, n_all, "ha", note)
        )
    if n >= 2 and float(np.mean(areas)) > 0:
        rel = float(settings.rel_half_width.value)
        needed = _runs_for_precision(np.asarray(areas), rel, level)
        rows.append(
            MetricRow(
                "runs_for_precision",
                f"burned area, half-width <= {rel:g} of the mean",
                "count",
                needed,
                math.nan,
                math.nan,
                n,
                "runs",
                "enough" if needed <= n else "more runs needed",
            )
        )
    if rule_caps:
        rows.append(
            boot("lb_capped_cell_hours", "lattice cap (WP 2.1d)", [r.lb_capped_cell_hours for r in inside], "cell-h")
        )
        capped = [float(r.lb_capped_cell_hours > 0) for r in inside]
        rows.append(boot("runs_capped_fraction", "lattice cap (WP 2.1d)", capped, "fraction"))
    else:
        rows.append(
            MetricRow(
                "lb_capped_cell_hours",
                "lattice cap (WP 2.1d)",
                "not applicable",
                math.nan,
                math.nan,
                math.nan,
                n,
                "cell-h",
                "the spread rule does not cap the length-to-breadth ratio",
            )
        )
    rows.append(boot("breaches", "breaks (WP 2.9)", [float(r.breaches) for r in inside], "breaks"))
    return rows


@dataclass(frozen=True)
class DetectionSummary:
    """The simulated arrival at one detection across the runs inside the grid.

    The quantiles count a run that never reached the footprint by the window end as arriving
    after it (``inf``), so a quantile is ``inf`` when more than that fraction of runs did not
    reach it. Times are s of simulated time.

    Attributes:
        detection: The detection.
        q05, median, q95: 5 %, 50 % and 95 % quantiles of the simulated arrival, s (``inf``: later
            than the window end).
        reached_fraction: Fraction of runs whose fire reached the footprint in the window.
        median_error_s: Arrival-interval error of the median arrival, s.
    """

    detection: Detection
    q05: float
    median: float
    q95: float
    reached_fraction: float
    median_error_s: float


def detection_summaries(result: ValidationRuns, observed: Observed) -> list[DetectionSummary]:
    """Per detection, the spread of the simulated arrival over the runs inside the grid (empty if none)."""
    inside = result.inside
    if not inside or not observed.detections:
        return []
    arrivals = np.array([r.detection_arrival_s for r in inside], dtype=np.float64)  # (runs, detections)
    later = np.where(np.isnan(arrivals), np.inf, arrivals)
    q = np.quantile(later, [0.05, 0.5, 0.95], axis=0, method="inverted_cdf")  # no interpolation with inf
    out = []
    for k, d in enumerate(observed.detections):
        median = float(q[1, k])
        err = interval_error(
            median if math.isfinite(median) else math.nan, d.t_clear, d.t_detect, observed.window_end_s
        )
        out.append(
            DetectionSummary(d, float(q[0, k]), median, float(q[2, k]), float(np.mean(~np.isnan(arrivals[:, k]))), err)
        )
    return out
