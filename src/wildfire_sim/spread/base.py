"""The spread-rule contract and the shared ignition-probability kernel.

Every spread rule (constant, Alexandridis, Rothermel) implements :class:`SpreadRule`: given the
grid and current weather it returns a directional rate of spread for every cell. The
simulator then turns rates into per-step ignition probabilities with
:func:`ignition_probability` (gamma-distributed crossing times with mean ``d / R``) and
combines contributions from burning neighbours with
:func:`neighbour_ignition_probability`. Rules never draw random numbers.

This module is a frozen contract (see CONTRIBUTING.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import gammaincc

from wildfire_sim.grid import Grid, Weather
from wildfire_sim.types import BoolArray, FloatArray, Stencil


@dataclass(frozen=True)
class SpreadOutput:
    """Rate-of-spread field produced by a :class:`SpreadRule` for one time step.

    Attributes:
        ros: ``(K, H, W)`` float array, m/s. ``ros[k, r, c]`` is the rate at which fire in cell
            ``(r, c)`` spreads toward its neighbour in stencil direction ``k``. Must be >= 0 and
            exactly 0 wherever the source cell cannot burn.
        intensity: ``(H, W)`` Byram fireline intensity at the head, kW/m. Used for line
            breaching and crew safety. May be all zeros for rules that do not model it.
        flame_length: ``(H, W)`` head flame length, m. May be all zeros.
    """

    ros: FloatArray
    intensity: FloatArray
    flame_length: FloatArray

    def validate(self, stencil: Stencil, shape: tuple[int, int]) -> None:
        k = stencil.size
        if self.ros.shape != (k, *shape):
            raise ValueError(f"ros has shape {self.ros.shape}, expected {(k, *shape)}")
        if self.intensity.shape != shape or self.flame_length.shape != shape:
            raise ValueError("intensity and flame_length must have the grid shape")
        if np.any(self.ros < 0) or not np.all(np.isfinite(self.ros)):
            raise ValueError("ros must be finite and >= 0")


@runtime_checkable
class SpreadRule(Protocol):
    """Interface every spread rule implements.

    Implementations are constructed from their entry in ``ScenarioConfig.spread.params`` and
    must be pure: same inputs, same output, no randomness.
    """

    name: str
    """Registry key, e.g. ``"constant"``, ``"alexandridis"``, ``"rothermel"``."""

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        """Compute directional rate of spread for every cell.

        Args:
            grid: Current grid. Rules read static layers, ``retardant`` and ``state``; they
                must not modify the grid.
            weather: Weather in force this step.
            stencil: Neighbour directions; ``ros`` has one slab per direction.

        Returns:
            A :class:`SpreadOutput` whose ``ros`` already includes the retardant multiplier
            ``(1 - grid.retardant)``.
        """
        ...


def ignition_probability(
    ros: FloatArray,
    dt: float,
    dx: float,
    stencil: Stencil,
    *,
    age: FloatArray | None = None,
    shape: float = 1.0,
) -> FloatArray:
    """Per-step probability that a burning cell ignites each neighbour.

    The time fire takes to cross from cell ``i`` to neighbour ``j`` is a gamma-distributed
    random variable with mean ``d / R`` and shape parameter ``shape`` (coefficient of variation
    ``1 / sqrt(shape)``). The probability returned for a step is the conditional probability
    that the arrival falls in ``[age, age + dt)`` given that it has not happened by ``age``:

    .. math:: p_{ij} = 1 - \\frac{S(a_i + \\Delta t)}{S(a_i)}, \\qquad
              S(t) = Q\\!\\left(k,\\; k\\, R_{ij}\\, t / d_{ij}\\right)

    where ``Q`` is the regularized upper incomplete gamma function, ``k`` is ``shape`` and
    ``a_i`` is the time since cell ``i`` ignited. Because the survival probabilities of
    successive steps multiply to ``S``, the arrival-time distribution does not depend on
    ``dt`` (arrivals are recorded at step ends, which adds at most ``dt`` to the mean).

    With ``shape = 1`` the arrival is exponential and this reduces to the memoryless form
    ``p = 1 - exp(-R dt / d)``, for which ``age`` is not needed. **Do not use ``shape = 1`` for
    quantitative runs:** on a lattice the front follows the fastest of many random paths, so a
    memoryless kernel spreads 3 to 4.5 times faster than ``R`` and the speed-up grows as ``dt``
    shrinks. With ``shape = 20`` the measured front speed is within about 10 % of ``R`` on the
    8-neighbour stencil (see the kernel section of docs/plan.md).

    Args:
        ros: ``(K, H, W)`` rate of spread, m/s (from :class:`SpreadOutput`). Taken as constant
            over the life of the source cell; the simulator passes the current value.
        dt: Time step, s.
        dx: Cell size, m.
        stencil: Provides the ``(K,)`` distance factors so ``d = dx * distances[k]``.
        age: ``(H, W)`` time since each source cell ignited, s, >= 0. Values for cells that
            are not burning are ignored by :func:`neighbour_ignition_probability`. Required
            when ``shape != 1``.
        shape: Gamma shape parameter ``k`` (dimensionless), >= 1.

    Returns:
        ``(K, H, W)`` float64 array of probabilities in [0, 1].
    """
    if dt <= 0 or dx <= 0:
        raise ValueError("dt and dx must be positive")
    if shape < 1:
        raise ValueError("shape must be >= 1")
    d = (dx * stencil.distances).astype(np.float64)[:, None, None]
    rate = np.asarray(ros, dtype=np.float64) / d  # 1/s; mean arrival time is 1 / rate
    if shape == 1:
        return 1.0 - np.exp(-rate * dt)
    if age is None:
        raise ValueError("age is required when shape != 1")
    a = np.asarray(age, dtype=np.float64)
    if a.shape != rate.shape[1:]:
        raise ValueError(f"age has shape {a.shape}, expected {rate.shape[1:]}")
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError("age must be finite and >= 0")
    survive_now = gammaincc(shape, shape * rate * a[None])
    survive_next = gammaincc(shape, shape * rate * (a[None] + dt))
    # Where survival has underflowed to zero the arrival is overdue: ignite with certainty.
    ratio = np.divide(survive_next, survive_now, out=np.zeros_like(survive_now), where=survive_now > 0)
    return np.clip(1.0 - ratio, 0.0, 1.0)


def neighbour_ignition_probability(p: FloatArray, burning: BoolArray, stencil: Stencil) -> FloatArray:
    """Probability that each cell ignites this step given which neighbours are burning.

    Combines independent contributions from every burning neighbour:

    .. math:: P_j = 1 - \\prod_{i \\in \\text{burning nbrs}} (1 - p_{ij})

    ``p[k, r, c]`` is the chance that burning cell ``(r, c)`` ignites the cell at
    ``(r, c) + offsets[k]``; this function shifts each slab so contributions land on the
    target cell. Cells beyond the grid edge receive nothing.

    Args:
        p: ``(K, H, W)`` from :func:`ignition_probability`.
        burning: ``(H, W)`` bool mask of currently burning cells.
        stencil: Stencil used to build ``p``.

    Returns:
        ``(H, W)`` float64 array. The caller masks out non-FUEL targets.
    """
    k, h, w = p.shape
    if burning.shape != (h, w) or k != stencil.size:
        raise ValueError("p, burning and stencil shapes are inconsistent")
    survive = np.ones((h, w), dtype=np.float64)
    for i in range(k):
        dr, dc = int(stencil.offsets[i, 0]), int(stencil.offsets[i, 1])
        src_contrib = np.where(burning, p[i], 0.0)
        # target[r+dr, c+dc] <- src[r, c]
        tr0, tr1 = max(dr, 0), h + min(dr, 0)
        tc0, tc1 = max(dc, 0), w + min(dc, 0)
        sr0, sr1 = max(-dr, 0), h + min(-dr, 0)
        sc0, sc1 = max(-dc, 0), w + min(-dc, 0)
        survive[tr0:tr1, tc0:tc1] *= 1.0 - src_contrib[sr0:sr1, sc0:sc1]
    return 1.0 - survive


def max_stable_dt(ros_max: float, dx: float, p_max: float = 0.5) -> float:
    """Largest time step keeping the per-step ignition probability below ``p_max``.

    From the memoryless form ``p = 1 - exp(-R dt / dx)``: ``dt = -ln(1 - p_max) * dx / R``.
    With the default ``p_max = 0.5`` and 1 m/s grass fire on 30 m cells this gives about 21 s.
    The same bound keeps ``dt`` well under the mean crossing time ``dx / R`` for the gamma
    kernel (``shape > 1``), so the step-end rounding of arrival times stays small.

    Args:
        ros_max: Fastest rate of spread expected in the domain, m/s.
        dx: Cell size, m.
        p_max: Upper bound on per-step probability along the orthogonal direction.
    """
    if ros_max <= 0:
        return float("inf")
    if not 0 < p_max < 1:
        raise ValueError("p_max must lie in (0, 1)")
    return float(-np.log1p(-p_max) * dx / ros_max)
