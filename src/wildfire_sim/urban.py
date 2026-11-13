"""Urban cells that burn (WP 1.13, docs/plan.md section 2.6).

An urban cell follows the same logic as a wildland cell, scaled to be harder to burn. It
carries a *surrogate* timber fuel model; whichever spread rule is in use computes its rate of
spread from that fuel, the simulator multiplies the rate by ``ros_factor`` (below 1) and the
fuel's flame residence time by ``residence_factor`` (above 1). Because the scaling is applied
after the spread rule, every rule is scaled the same way.

The surrogate model and both factors live in ``configs/urban.yaml``. They are team
assumptions, not measurements: this is a stand-in for structure-to-structure fire spread, not
a model of it, and the three values are part of the sensitivity analysis (WP 3.2).

Water and roads stay unburnable. With ``ros_factor = 0`` urban cells are unburnable too, the
behaviour before this work package.

This module is imported both by the simulator (the factors) and by the data generators (the
surrogate fuel model they write into urban cells).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from wildfire_sim.config import CitedValue, configs_dir, load_yaml
from wildfire_sim.grid import Grid
from wildfire_sim.types import BoolArray, FloatArray, LandUse


class UrbanModel(BaseModel):
    """How urban cells burn (``configs/urban.yaml``).

    Attributes:
        fuel_model: Surrogate fuel model number (Scott & Burgan 2005) written into urban cells;
            an integer >= 1.
        ros_factor: Dimensionless factor in [0, 1] on the rate of spread of an urban cell in
            every direction. 0 makes urban cells unburnable.
        residence_factor: Dimensionless factor >= 1 on the flame residence time of an urban cell.
    """

    model_config = ConfigDict(extra="forbid")

    fuel_model: CitedValue
    ros_factor: CitedValue
    residence_factor: CitedValue

    @model_validator(mode="after")
    def _check(self) -> UrbanModel:
        fm = self.fuel_model.value
        if not (np.isfinite(fm) and fm >= 1 and fm == int(fm)):
            raise ValueError(f"urban fuel_model must be an integer >= 1, got {fm}")
        if not 0.0 <= self.ros_factor.value <= 1.0:
            raise ValueError(f"urban ros_factor must lie in [0, 1], got {self.ros_factor.value}")
        if not (np.isfinite(self.residence_factor.value) and self.residence_factor.value >= 1.0):
            raise ValueError(f"urban residence_factor must be finite and >= 1, got {self.residence_factor.value}")
        return self

    @property
    def surrogate_fuel_model(self) -> int:
        """The surrogate fuel model number as an int."""
        return int(self.fuel_model.value)

    @property
    def burnable(self) -> bool:
        """Whether urban cells can burn at all (``ros_factor > 0``)."""
        return self.ros_factor.value > 0.0


def load_urban_model(path: str | Path | None = None) -> UrbanModel:
    """Load the urban-cell parameters; default ``configs_dir() / "urban.yaml"``."""
    return load_yaml(configs_dir() / "urban.yaml" if path is None else path, UrbanModel)


def urban_mask(grid: Grid) -> BoolArray:
    """``(H, W)`` bool, True where ``grid.land_use`` is URBAN."""
    return np.asarray(grid.land_use == int(LandUse.URBAN))


def ros_scale(grid: Grid, model: UrbanModel) -> FloatArray:
    """``(H, W)`` float64 factor on the rate of spread: ``ros_factor`` on urban cells, 1 elsewhere."""
    return np.where(urban_mask(grid), model.ros_factor.value, 1.0)


def residence_scale(grid: Grid, model: UrbanModel) -> FloatArray:
    """``(H, W)`` float64 factor on the residence time: ``residence_factor`` on urban cells, 1 elsewhere."""
    return np.where(urban_mask(grid), model.residence_factor.value, 1.0)


__all__ = ["UrbanModel", "load_urban_model", "residence_scale", "ros_scale", "urban_mask"]
