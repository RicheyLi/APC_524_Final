"""Shared enumerations, neighbour stencils and array type aliases.

Conventions used throughout wildfire-sim
----------------------------------------
* Grids are 2-D NumPy arrays indexed ``[row, col]``. Row 0 is the **north** edge and rows
  increase southward; column 0 is the **west** edge and columns increase eastward.
* Compass bearings are degrees clockwise from north (north = 0, east = 90).
* Wind direction follows the meteorological convention: the bearing the wind blows **from**.
* Aspect is the bearing a slope faces (the downhill direction).
* Slope is stored as a rise/run fraction (``tan`` of the slope angle), which is what
  Rothermel's slope factor uses.
* Time is in seconds, length in metres, rate of spread in m/s, fireline intensity in kW/m.

This module is a frozen contract (see CONTRIBUTING.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


class CellState(IntEnum):
    """Fire state of one cell. Stored as ``int8`` in :attr:`wildfire_sim.grid.Grid.state`."""

    UNBURNABLE = 0
    """Cannot ignite: water, bare rock, a wide road, or a completed fireline."""
    FUEL = 1
    """Unburned fuel that can ignite."""
    BURNING = 2
    """Actively burning; :attr:`wildfire_sim.grid.Grid.burn_timer` counts down to burn-out."""
    BURNED = 3
    """Burned out. Cannot re-ignite."""


class LandUse(IntEnum):
    """Static land-use class of a cell. Stored as ``int8`` in :attr:`wildfire_sim.grid.Grid.land_use`."""

    WILDLAND = 0
    URBAN = 1
    WATER = 2
    ROAD = 3

    @property
    def burnable(self) -> bool:
        """Whether cells of this class can carry fire at all.

        Wildland and urban cells can (urban cells burn by a scaled surrogate fuel, see
        :mod:`wildfire_sim.urban`); water and roads cannot.
        """
        return self in (LandUse.WILDLAND, LandUse.URBAN)


StencilSize = Literal[8, 16]


@dataclass(frozen=True)
class Stencil:
    """A neighbourhood stencil: the set of directions fire can spread cell-to-cell.

    Attributes:
        offsets: ``(K, 2)`` int array of ``(d_row, d_col)`` for each of the ``K`` directions.
        distances: ``(K,)`` float array of centre-to-centre distance in units of the cell size
            ``dx`` (1 for orthogonal, ``sqrt(2)`` for diagonal, ``sqrt(5)`` for knight moves).
        bearings_deg: ``(K,)`` float array, compass bearing from the source cell toward the
            neighbour, degrees clockwise from north.
    """

    offsets: IntArray
    distances: FloatArray
    bearings_deg: FloatArray

    def __post_init__(self) -> None:
        k = self.offsets.shape[0]
        if self.offsets.shape != (k, 2) or self.distances.shape != (k,) or self.bearings_deg.shape != (k,):
            raise ValueError("Stencil arrays have inconsistent shapes")
        self.offsets.setflags(write=False)
        self.distances.setflags(write=False)
        self.bearings_deg.setflags(write=False)

    @property
    def size(self) -> int:
        """Number of directions ``K``."""
        return int(self.offsets.shape[0])

    @property
    def max_offset(self) -> int:
        """Largest absolute row/col offset; the halo width the simulator must pad by."""
        return int(np.abs(self.offsets).max())


def _make_stencil(offsets: list[tuple[int, int]]) -> Stencil:
    off = np.array(offsets, dtype=np.int64)
    dist = np.sqrt((off.astype(np.float64) ** 2).sum(axis=1))
    # bearing: north is -row, east is +col.
    bearing = (np.degrees(np.arctan2(off[:, 1], -off[:, 0])) + 360.0) % 360.0
    return Stencil(offsets=off, distances=dist, bearings_deg=bearing)


STENCIL_8: Stencil = _make_stencil(
    [
        (-1, 0),  # N
        (-1, 1),  # NE
        (0, 1),  # E
        (1, 1),  # SE
        (1, 0),  # S
        (1, -1),  # SW
        (0, -1),  # W
        (-1, -1),  # NW
    ]
)
"""Moore neighbourhood, ordered clockwise from north."""

STENCIL_16: Stencil = _make_stencil(
    [
        (-1, 0),
        (-2, 1),
        (-1, 1),
        (-1, 2),
        (0, 1),
        (1, 2),
        (1, 1),
        (2, 1),
        (1, 0),
        (2, -1),
        (1, -1),
        (1, -2),
        (0, -1),
        (-1, -2),
        (-1, -1),
        (-2, -1),
    ]
)
"""Moore neighbourhood plus the eight knight-move directions, ordered clockwise from north.

Reduces the octagonal bias of the 8-neighbour CA (see docs/plan.md, "Why the design looks like this", item 4).
"""


def get_stencil(size: StencilSize) -> Stencil:
    """Return the standard stencil with ``size`` directions (8 or 16)."""
    if size == 8:
        return STENCIL_8
    if size == 16:
        return STENCIL_16
    raise ValueError(f"Unsupported stencil size {size!r}; expected 8 or 16")
