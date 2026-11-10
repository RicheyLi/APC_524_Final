"""Constant-rate spread rule: the simplest baseline (WP 1.1).

Fire spreads at the same rate in every stencil direction from every burnable cell,
independent of fuel type, terrain and weather. Combined with the stochastic ignition kernel
this gives a Drossel-Schwabl-style percolation baseline against which the physical rules
(Alexandridis, Rothermel) are compared. Only the retardant multiplier ``(1 - retardant)``
modifies the rate.
"""

from __future__ import annotations

import math

import numpy as np

from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread import register
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.types import Stencil


@register("constant")
class ConstantRule:
    """Same rate of spread in every direction for every burnable cell.

    Args:
        ros: Rate of spread, m/s. Must be finite and >= 0. Its value comes from
            ``ScenarioConfig.spread.params`` (it is a scenario parameter, not a physical
            constant).

    Raises:
        ValueError: ``ros`` is negative, NaN or infinite.
    """

    name: str = "constant"

    def __init__(self, ros: float) -> None:
        ros = float(ros)
        if not math.isfinite(ros) or ros < 0:
            raise ValueError(f"ros must be finite and >= 0 m/s, got {ros!r}")
        self.ros = ros

    def __repr__(self) -> str:
        return f"ConstantRule(ros={self.ros!r})"

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        """Return a uniform directional rate of spread; weather is ignored.

        Args:
            grid: Current grid with layers of shape ``(H, W)``. Read only: ``burnable_mask()``
                and ``retardant`` (fraction in [0, 1]) are used, nothing is modified.
            weather: Ignored by this rule (accepted to satisfy the protocol).
            stencil: Neighbour stencil with ``K`` directions.

        Returns:
            :class:`SpreadOutput` with ``ros`` a ``(K, H, W)`` float64 array in m/s equal to
            ``self.ros * (1 - grid.retardant)`` on burnable cells and exactly 0 elsewhere (the
            same value in all ``K`` slabs), and ``intensity`` (kW/m) and ``flame_length`` (m)
            as ``(H, W)`` float64 zero arrays.
        """
        h, w = grid.shape
        retardant = np.clip(np.asarray(grid.retardant, dtype=np.float64), 0.0, 1.0)
        cell_ros = np.where(grid.burnable_mask(), self.ros * (1.0 - retardant), 0.0)
        ros = np.broadcast_to(cell_ros, (stencil.size, h, w)).copy()
        return SpreadOutput(
            ros=ros,
            intensity=np.zeros((h, w), dtype=np.float64),
            flame_length=np.zeros((h, w), dtype=np.float64),
        )
