"""Synthetic landscapes for tests, physics checks and demos.

Every generator returns a validated :class:`~wildfire_sim.grid.Grid` and needs no network.
Keep these simple and deterministic; they are the fixtures the physics tests rely on.
"""

from __future__ import annotations

import numpy as np

from wildfire_sim.grid import Grid
from wildfire_sim.types import LandUse
from wildfire_sim.urban import load_urban_model


def flat_uniform(shape: tuple[int, int] = (101, 101), dx: float = 30.0, fuel_model: int = 1) -> Grid:
    """Flat terrain, one fuel model everywhere, no canopy, no assets.

    Used by the no-wind circularity and the dt-halving tests.

    Args:
        shape: ``(H, W)``.
        dx: Cell size, m.
        fuel_model: Scott & Burgan fuel model number applied to every cell (>= 1).
    """
    if fuel_model < 1:
        raise ValueError("fuel_model must be >= 1 for a burnable landscape")
    grid = Grid.empty(shape, dx)
    grid.fuel_model[:] = fuel_model
    grid.metadata["source"] = "synthetic:flat_uniform"
    grid.apply_unburnable()
    grid.validate()
    return grid


def slope_ramp(
    shape: tuple[int, int] = (101, 101),
    dx: float = 30.0,
    fuel_model: int = 1,
    slope: float = 0.3,
    aspect_deg: float = 180.0,
) -> Grid:
    """Uniform fuel on a plane tilted so it faces ``aspect_deg`` with rise/run ``slope``.

    Elevation increases in the direction opposite to the aspect (uphill = ``aspect + 180``),
    and the ``slope``/``aspect`` layers are set consistently with the elevation field, so
    terrain code (WP 2.3) can be checked against it.

    Args:
        shape: ``(H, W)``.
        dx: Cell size, m.
        fuel_model: Fuel model number for every cell.
        slope: Rise/run fraction, >= 0.
        aspect_deg: Downhill bearing, degrees clockwise from north.
    """
    if slope < 0:
        raise ValueError("slope must be >= 0")
    grid = flat_uniform(shape, dx, fuel_model)
    h, w = shape
    rows, cols = np.mgrid[0:h, 0:w]
    # unit vector pointing uphill (bearing aspect + 180): north is -row, east is +col
    uphill = np.radians((aspect_deg + 180.0) % 360.0)
    up_col = np.sin(uphill)
    up_row = -np.cos(uphill)
    dist_uphill = (rows * up_row + cols * up_col) * dx
    grid.elevation[:] = (dist_uphill - dist_uphill.min()) * slope
    grid.slope[:] = slope
    grid.aspect[:] = aspect_deg % 360.0
    grid.metadata["source"] = "synthetic:slope_ramp"
    grid.validate()
    return grid


def with_river(
    shape: tuple[int, int] = (101, 101),
    dx: float = 30.0,
    fuel_model: int = 1,
    col: int | None = None,
    width: int = 2,
) -> Grid:
    """Flat uniform fuel cut by a north-south water strip ``width`` cells wide starting at ``col``.

    Used to test that water never burns and that a complete natural barrier stops fire.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    grid = flat_uniform(shape, dx, fuel_model)
    c0 = shape[1] // 2 if col is None else col
    if not 0 <= c0 < shape[1]:
        raise ValueError("col is outside the grid")
    grid.land_use[:, c0 : c0 + width] = int(LandUse.WATER)
    grid.fuel_model[:, c0 : c0 + width] = 0
    grid.metadata["source"] = "synthetic:with_river"
    grid.apply_unburnable()
    grid.validate()
    return grid


def with_town(
    shape: tuple[int, int] = (101, 101),
    dx: float = 30.0,
    fuel_model: int = 1,
    center: tuple[int, int] | None = None,
    size: int = 7,
    value_per_cell: float = 500_000.0,
    urban_fuel_model: int | None = None,
) -> Grid:
    """Flat uniform fuel with a square urban block of ``size`` cells carrying asset value.

    Urban cells burn (docs/plan.md section 2.6): each carries a surrogate fuel model, and the
    simulator slows its spread and lengthens its flaming with the factors of
    ``configs/urban.yaml``. An urban cell's ``asset_value`` is lost when the cell burns (the
    loss rule itself is defined in WP 2.8 ``metrics``).

    Args:
        shape: ``(H, W)``.
        dx: Cell size, m.
        fuel_model: Fuel model number of the wildland cells.
        center: ``(row, col)`` of the block centre; defaults to the grid centre.
        size: Side length in cells.
        value_per_cell: USD per urban cell.
        urban_fuel_model: Fuel model number written into the urban cells (>= 1); None uses the
            surrogate fuel model of ``configs/urban.yaml``.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    grid = flat_uniform(shape, dx, fuel_model)
    r, c = (shape[0] // 2, shape[1] // 2) if center is None else center
    half = size // 2
    r0, r1 = max(r - half, 0), min(r - half + size, shape[0])
    c0, c1 = max(c - half, 0), min(c - half + size, shape[1])
    if r0 >= r1 or c0 >= c1:
        raise ValueError("town lies outside the grid")
    surrogate = load_urban_model().surrogate_fuel_model if urban_fuel_model is None else int(urban_fuel_model)
    if surrogate < 1:
        raise ValueError("urban_fuel_model must be >= 1")
    grid.land_use[r0:r1, c0:c1] = int(LandUse.URBAN)
    grid.fuel_model[r0:r1, c0:c1] = surrogate
    grid.asset_value[r0:r1, c0:c1] = value_per_cell
    grid.metadata["source"] = "synthetic:with_town"
    grid.apply_unburnable()
    grid.validate()
    return grid


GENERATORS = {
    "flat_uniform": flat_uniform,
    "slope_ramp": slope_ramp,
    "with_river": with_river,
    "with_town": with_town,
}
"""Name -> generator, matching ``SyntheticGridConfig.generator``."""
