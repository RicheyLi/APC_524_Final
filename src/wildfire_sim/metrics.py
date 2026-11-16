"""Scores for comparing simulated and observed fires, and for strategies (WP 2.8).

The validation metrics of docs/plan.md section 7 and the objective terms of the strategy search:

* **Burned area and overlap.** :func:`burned_area_ha`; :func:`jaccard` (intersection over
  union) and :func:`sorensen` (``2|A and B| / (|A| + |B|)``) between a simulated and an observed
  burned mask; for an ensemble the simulated mask is :func:`consensus_mask`, the cells burned in
  at least half the runs (section 7); :func:`area_ratio`, simulated over observed area.
* **Arrival-time error against an interval** (:func:`arrival_interval_error`). A VIIRS detection
  says the fire arrived after ``t_clear`` and by ``t_detect`` (WP 2.7). The simulated arrival
  of a detection is the earliest arrival among the cells of its footprint (a 375 m pixel covers
  about 12 x 12 cells of 30 m). The error is 0 when it falls in ``(t_clear, t_detect]`` and the
  distance to the nearer end otherwise. If no cell of the footprint burned by the end of the
  comparison window, the arrival is only known to be later than the window end, and the error
  is the smallest one consistent with that: ``max(0, window_end - t_detect)``. Scoring the
  detection time as the arrival would bias a calibrated spread rate low (section 7).
* **Losses and costs.** :func:`asset_loss`, the asset value of every burned cell (urban cells
  included, WP 1.13); :func:`suppression_cost`, resource hours at their hourly rates plus water
  and retardant (``configs/costs.yaml``).
* **Uncertainty.** :func:`bootstrap_ci`, a percentile bootstrap interval for a mean, and
  :func:`paired_difference_ci`, the same for the mean difference of paired values (two
  strategies run on the same seeds). Both are seeded, so results reproduce. For a plain
  Student-t interval see :func:`wildfire_sim.ensemble.mean_ci`.

References:
    Efron, B. & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall,
    chapter 13 (percentile intervals).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from wildfire_sim.config import CostConfig
from wildfire_sim.ensemble import EnsembleResult
from wildfire_sim.grid import Grid
from wildfire_sim.sim import RunResult
from wildfire_sim.types import BoolArray, FloatArray

M2_PER_HA = 10_000.0


# ----------------------------------------------------------------------------- area and overlap


def burned_area_ha(result: RunResult | EnsembleResult | BoolArray, dx: float | None = None) -> float | FloatArray:
    """Burned area, ha.

    Args:
        result: A run (its burned cells), an ensemble (one value per run) or a burned mask.
        dx: Cell size, m; needed (only) for a mask.

    Returns:
        A float for a run or a mask, an ``(n,)`` array for an ensemble.
    """
    if isinstance(result, RunResult):
        return float(result.burned_mask.sum()) * result.grid.dx**2 / M2_PER_HA
    if isinstance(result, EnsembleResult):
        return np.asarray(result.burned_area_m2, dtype=np.float64) / M2_PER_HA
    if dx is None or dx <= 0:
        raise ValueError("a burned mask needs the cell size dx > 0")
    return float(np.asarray(result, dtype=bool).sum()) * dx * dx / M2_PER_HA


def consensus_mask(burn_probability: FloatArray, threshold: float = 0.5) -> BoolArray:
    """Cells burned in at least ``threshold`` of the runs (section 7 uses half)."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must lie in (0, 1]")
    return np.asarray(burn_probability) >= threshold


def _masks(sim_mask: BoolArray, obs_mask: BoolArray) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(sim_mask, dtype=bool), np.asarray(obs_mask, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"masks differ in shape: {a.shape} and {b.shape}")
    return a, b


def jaccard(sim_mask: BoolArray, obs_mask: BoolArray) -> float:
    """``|A and B| / |A or B|``; 1.0 when both masks are empty (they agree)."""
    a, b = _masks(sim_mask, obs_mask)
    union = int((a | b).sum())
    return 1.0 if union == 0 else int((a & b).sum()) / union


def sorensen(sim_mask: BoolArray, obs_mask: BoolArray) -> float:
    """``2 |A and B| / (|A| + |B|)`` (Sorensen-Dice); 1.0 when both masks are empty."""
    a, b = _masks(sim_mask, obs_mask)
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2 * int((a & b).sum()) / total


def area_ratio(sim: BoolArray | float, obs: BoolArray | float) -> float:
    """Simulated over observed burned area, from two masks or two areas in the same units.

    Raises:
        ValueError: The observed area is 0.
    """
    s = float(np.asarray(sim, dtype=bool).sum()) if np.ndim(sim) else float(sim)
    o = float(np.asarray(obs, dtype=bool).sum()) if np.ndim(obs) else float(obs)
    if o <= 0:
        raise ValueError("the observed area is 0")
    return s / o


# ----------------------------------------------------------------------------- arrival times


def footprint_arrival(arrival: FloatArray, row: int, col: int, half_cells: int) -> float:
    """Earliest finite arrival among the cells within ``half_cells`` rows and columns of a cell (NaN if none)."""
    h, w = arrival.shape
    r0, r1 = max(0, row - half_cells), min(h, row + half_cells + 1)
    c0, c1 = max(0, col - half_cells), min(w, col + half_cells + 1)
    window = np.asarray(arrival[r0:r1, c0:c1], dtype=np.float64)
    finite = window[np.isfinite(window)]
    return float(finite.min()) if finite.size else float("nan")


def interval_error(t_sim: float, t_clear: float, t_detect: float, window_end_s: float) -> float:
    """Error of one simulated arrival against the interval ``(t_clear, t_detect]``, s.

    ``t_sim`` NaN means not reached by ``window_end_s``: the arrival is somewhere after it, and
    the error is the smallest consistent one, ``max(0, window_end_s - t_detect)``.
    """
    if t_detect < t_clear:
        raise ValueError("t_detect must not precede t_clear")
    if not np.isfinite(t_sim) or t_sim > window_end_s:
        return max(0.0, window_end_s - t_detect)
    if t_sim <= t_clear:
        return t_clear - t_sim
    if t_sim > t_detect:
        return t_sim - t_detect
    return 0.0


def arrival_interval_error(
    arrival: FloatArray,
    detections: Sequence[tuple[float, float, int, int]],
    footprint_m: float,
    window_end_s: float,
    *,
    dx: float,
) -> FloatArray:
    """Per detection, the error of the simulated arrival against its interval, s.

    Args:
        arrival: ``(H, W)`` simulated arrival times, s (NaN where never burned), e.g.
            ``RunResult.arrival_time``.
        detections: ``(t_clear, t_detect, row, col)`` per detection, s (WP 2.7), on the same
            clock as ``arrival``.
        footprint_m: Side of a detection's footprint, m (375 for VIIRS I-band); the cells within
            ``footprint_m / 2`` of the detection's cell in row and column are searched.
        window_end_s: End of the comparison window, s.
        dx: Cell size, m.

    Returns:
        ``(n,)`` float64 errors, s, in the order of ``detections``.
    """
    if footprint_m <= 0 or dx <= 0:
        raise ValueError("footprint_m and dx must be > 0")
    arr = np.asarray(arrival, dtype=np.float64)
    half = int(footprint_m / 2 // dx)
    out = np.empty(len(detections))
    for i, (t_clear, t_detect, row, col) in enumerate(detections):
        t_sim = footprint_arrival(arr, int(row), int(col), half)
        out[i] = interval_error(t_sim, float(t_clear), float(t_detect), window_end_s)
    return out


# ----------------------------------------------------------------------------- losses and costs


def asset_loss(grid: Grid) -> float:
    """Asset value of every burning or burned cell, USD (urban cells included)."""
    return float(grid.asset_value[grid.burned_mask()].astype(np.float64).sum())


ResourceType = Literal["crew", "dozer", "engine", "aircraft"]


@dataclass(frozen=True)
class ResourceUse:
    """What one kind of resource did in a run (filled by the suppression model, Phase 4).

    Attributes:
        type: Resource type, as in ``ResourceConfig``.
        count: Number of units.
        hours: Hours each unit was committed (dispatch to release), h.
        water_l: Water dropped in total, L.
        retardant_l: Retardant dropped in total, L.
        hourly_cost_usd: Rate per unit-hour, USD; 0 uses the ``CostConfig`` rate for the type
            (the convention of ``ResourceConfig``).
    """

    type: ResourceType
    count: int
    hours: float
    water_l: float = 0.0
    retardant_l: float = 0.0
    hourly_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.count < 0 or self.hours < 0 or self.water_l < 0 or self.retardant_l < 0 or self.hourly_cost_usd < 0:
            raise ValueError("resource use must not be negative")


def suppression_cost(fleet: Sequence[ResourceUse], costs: CostConfig) -> float:
    """``sum(count x hours x rate) + water x price + retardant x price``, USD."""
    rates = {
        "crew": costs.crew_usd_per_h.value,
        "dozer": costs.dozer_usd_per_h.value,
        "engine": costs.engine_usd_per_h.value,
        "aircraft": costs.aircraft_usd_per_h.value,
    }
    total = 0.0
    for use in fleet:
        rate = use.hourly_cost_usd if use.hourly_cost_usd > 0 else rates[use.type]
        total += use.count * use.hours * rate
        total += use.water_l * costs.water_usd_per_l.value + use.retardant_l * costs.retardant_usd_per_l.value
    return total


# ----------------------------------------------------------------------------- intervals


def _check(level: float, n_boot: int) -> None:
    if not 0 < level < 1:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")


def bootstrap_ci(
    values: npt.ArrayLike, level: float = 0.95, *, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and its percentile bootstrap interval (Efron & Tibshirani 1993, ch. 13).

    Args:
        values: ``(n,)`` finite values, n >= 2.
        level: Confidence level in (0, 1).
        n_boot: Number of resamples.
        seed: Seed of the resampling, so the interval reproduces.

    Returns:
        ``(mean, lower, upper)``.
    """
    _check(level, n_boot)
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("need at least 2 finite values")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(x.mean()), float(lo), float(hi)


def paired_difference_ci(
    a: npt.ArrayLike, b: npt.ArrayLike, level: float = 0.95, *, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``a - b`` over paired values and its percentile bootstrap interval.

    Pairs (the same seed under two strategies) are resampled together, so the interval reflects
    the spread of the differences, not of each strategy's values.

    Returns:
        ``(mean difference, lower, upper)``.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("a and b must have one value per pair")
    return bootstrap_ci(x - y, level, n_boot=n_boot, seed=seed)
