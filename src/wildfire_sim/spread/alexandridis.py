"""Alexandridis et al. (2008) spread rule, expressed as a directional rate of spread.

The paper's cellular automaton multiplies a base ignition probability ``p_h`` by vegetation,
density, wind and slope factors. Here the base probability is replaced by a base rate
``ros_base`` (m/s) so the result plugs into the gamma crossing-time kernel of
:mod:`wildfire_sim.spread.base`:

.. math:: R_k = R_0 (1 + p_{veg})(1 + p_{den})\\, p_w(\\theta_k)\\, p_s(k)\\, (1 - \\text{retardant})

* :math:`p_w = \\exp(c_1 V)\\exp(c_2 V(\\cos\\theta_k - 1))`, with ``V`` the 10 m wind speed (m/s)
  and :math:`\\theta_k` the angle between stencil direction ``k`` and the wind-to bearing.
* :math:`p_s = \\exp(a\\,\\theta_s)`, with :math:`\\theta_s` the slope angle (degrees) from the cell
  to its neighbour in direction ``k``, positive uphill. Toward neighbours outside the grid
  :math:`\\theta_s = 0`.

Every constant lives in ``configs/alexandridis.yaml`` (schema :class:`AlexandridisParams`).

Reference: Alexandridis, A., Vakalis, D., Siettos, C. I. & Bafas, G. V. (2008). A cellular
automata model for forest fire spread prediction: The case of the wildfire that swept through
Spetses Island in 1990. Applied Mathematics and Computation 204, 191-201.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildfire_sim.config import CitedValue, configs_dir, load_yaml
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.spread import register
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.types import FloatArray, Stencil

VegetationClass = Literal["agricultural", "thickets", "forest"]

_MAX_LOG_FACTOR = 50.0
"""Bound on ``|log(p_w p_s)|``. Physical inputs stay far inside it (5 m/s wind and a 45 deg
slope give about 3.7); it only guards against overflow to ``inf`` for absurd inputs."""


# ----------------------------------------------------------------------------- schema


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VegetationFactors(_Strict):
    """Vegetation-type factor ``p_veg`` per class (dimensionless, each > -1)."""

    agricultural: CitedValue
    thickets: CitedValue
    forest: CitedValue


class DensityFactors(_Strict):
    """Vegetation-density factor ``p_den`` per class (dimensionless, each > -1)."""

    sparse: CitedValue
    normal: CitedValue
    dense: CitedValue


class FuelRange(_Strict):
    """Inclusive range of fuel model numbers assigned to one vegetation class."""

    first: Annotated[int, Field(ge=1)]
    last: Annotated[int, Field(ge=1)]
    vegetation: VegetationClass
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> FuelRange:
        if self.last < self.first:
            raise ValueError(f"fuel range {self.first}-{self.last}: last must be >= first")
        return self


class VegetationMap(_Strict):
    """Mapping from ``Grid.fuel_model`` numbers to the paper's vegetation types (cited assumption)."""

    source: Annotated[str, Field(min_length=1)]
    note: str = ""
    ranges: Annotated[list[FuelRange], Field(min_length=1)]

    @model_validator(mode="after")
    def _no_overlap(self) -> VegetationMap:
        spans = sorted((r.first, r.last) for r in self.ranges)
        for (_, prev_last), (first, last) in itertools.pairwise(spans):
            if first <= prev_last:
                raise ValueError(f"fuel model ranges overlap at {first}-{last}")
        return self


class DensityMap(_Strict):
    """Canopy-cover thresholds (fraction in [0, 1]) separating sparse / normal / dense."""

    sparse_below: CitedValue
    dense_at_or_above: CitedValue

    @model_validator(mode="after")
    def _ordered(self) -> DensityMap:
        lo, hi = self.sparse_below.value, self.dense_at_or_above.value
        if not 0 <= lo <= hi <= 1:
            raise ValueError("density thresholds need 0 <= sparse_below <= dense_at_or_above <= 1")
        return self


class AlexandridisParams(_Strict):
    """Schema of ``configs/alexandridis.yaml``. Units: ``ros_base`` m/s, ``a`` 1/deg, ``c1``/``c2`` s/m."""

    ros_base: CitedValue
    a: CitedValue
    c1: CitedValue
    c2: CitedValue
    p_veg: VegetationFactors
    p_den: DensityFactors
    vegetation_map: VegetationMap
    density_map: DensityMap

    @model_validator(mode="after")
    def _factors_keep_rate_non_negative(self) -> AlexandridisParams:
        if self.ros_base.value < 0:
            raise ValueError("ros_base must be >= 0")
        for group in (self.p_veg, self.p_den):
            for name, cited in group:
                if cited.value <= -1:
                    raise ValueError(f"factor {name!r} must be > -1 so (1 + p) stays positive")
        return self

    def vegetation_class(self, fuel_model: int) -> VegetationClass | None:
        """Vegetation class of one fuel model number, or ``None`` if no range covers it."""
        for r in self.vegetation_map.ranges:
            if r.first <= fuel_model <= r.last:
                return r.vegetation
        return None


# ----------------------------------------------------------------------------- rule


@register("alexandridis")
class AlexandridisRule:
    """Vegetation, density, wind and slope factors on a base rate (Alexandridis et al. 2008).

    Args:
        ros_base: Base rate of spread, m/s, >= 0. ``None`` uses the YAML value.
        c1: Wind coefficient, s/m. ``None`` uses the YAML value.
        c2: Wind direction coefficient, s/m. ``None`` uses the YAML value.
        a: Slope coefficient, 1/deg. ``None`` uses the YAML value.
        params_path: Parameter file; defaults to ``configs_dir() / "alexandridis.yaml"``.
            Read once, here.

    Raises:
        ValueError: The parameter file is invalid or ``ros_base`` is negative.
    """

    name = "alexandridis"

    def __init__(
        self,
        ros_base: float | None = None,
        c1: float | None = None,
        c2: float | None = None,
        a: float | None = None,
        params_path: str | Path | None = None,
    ) -> None:
        path = Path(params_path) if params_path is not None else configs_dir() / "alexandridis.yaml"
        self.params: AlexandridisParams = load_yaml(path, AlexandridisParams)
        p = self.params
        self._ros_base = float(p.ros_base.value if ros_base is None else ros_base)
        self._c1 = float(p.c1.value if c1 is None else c1)
        self._c2 = float(p.c2.value if c2 is None else c2)
        self._a = float(p.a.value if a is None else a)
        if not self._ros_base >= 0:
            raise ValueError(f"ros_base must be >= 0, got {self._ros_base!r}")

        # Lookup table fuel model number -> 1 + p_veg; NaN marks numbers with no class.
        top = max(r.last for r in p.vegetation_map.ranges)
        self._veg_lut = np.full(top + 1, np.nan, dtype=np.float64)
        for r in p.vegetation_map.ranges:
            self._veg_lut[r.first : r.last + 1] = 1.0 + getattr(p.p_veg, r.vegetation).value

    def _vegetation_multiplier(self, fuel_model: np.ndarray, burnable: np.ndarray) -> FloatArray:
        """``(H, W)`` array of ``1 + p_veg``; 0 where not burnable. Raises on unmapped fuel models."""
        fm = fuel_model.astype(np.int64)
        in_lut = (fm >= 0) & (fm < self._veg_lut.size)
        mult = np.full(fm.shape, np.nan, dtype=np.float64)
        mult[in_lut] = self._veg_lut[fm[in_lut]]
        unmapped = burnable & np.isnan(mult)
        if unmapped.any():
            codes = sorted({int(v) for v in fm[unmapped]})
            raise ValueError(f"fuel model(s) {codes} have no vegetation class in alexandridis.yaml")
        return np.where(burnable, mult, 0.0)

    def _density_multiplier(self, canopy_cover: np.ndarray) -> FloatArray:
        """``(H, W)`` array of ``1 + p_den`` from canopy cover (fraction)."""
        dm, pd = self.params.density_map, self.params.p_den
        cc = canopy_cover.astype(np.float64)
        p = np.where(
            cc < dm.sparse_below.value,
            pd.sparse.value,
            np.where(cc >= dm.dense_at_or_above.value, pd.dense.value, pd.normal.value),
        )
        return 1.0 + p

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        """Directional rate of spread for every cell.

        Args:
            grid: Reads ``fuel_model``, ``canopy_cover``, ``elevation`` (m), ``retardant``
                (fraction), ``dx`` (m) and the burnable mask. Not modified.
            weather: Uses ``wind_speed`` (m/s) and ``wind_to_bearing`` (deg), or the per-cell
                speed and "to" bearing of its ``wind_field`` when present (WP 3.4).
            stencil: ``K`` neighbour directions.

        Returns:
            :class:`SpreadOutput` with ``ros`` of shape ``(K, H, W)`` in m/s (0 for
            non-burnable cells, already multiplied by ``1 - retardant``) and all-zero
            ``(H, W)`` ``intensity`` and ``flame_length``.

        Raises:
            ValueError: A burnable cell has a fuel model with no vegetation class.
        """
        h, w = grid.shape
        burnable = grid.burnable_mask()
        retardant = np.clip(grid.retardant.astype(np.float64), 0.0, 1.0)
        base = (
            self._ros_base
            * self._vegetation_multiplier(grid.fuel_model, burnable)
            * self._density_multiplier(grid.canopy_cover)
            * (1.0 - retardant)
        )

        bearings = stencil.bearings_deg.astype(np.float64)
        if weather.wind_field is None:
            v = float(weather.wind_speed)
            cos_theta = np.cos(np.radians(bearings - weather.wind_to_bearing))
            log_wind = self._c1 * v + self._c2 * v * (cos_theta - 1.0)  # (K,)
        else:
            if weather.wind_field.shape != (h, w):
                raise ValueError(f"wind field shape {weather.wind_field.shape} differs from the grid's {(h, w)}")
            v_cell = weather.wind_field.speed
            cos_cell = np.cos(np.radians(bearings[:, None, None] - weather.wind_field.to_bearing[None]))
            log_wind = self._c1 * v_cell + self._c2 * v_cell * (cos_cell - 1.0)  # (K, H, W)

        elev = grid.elevation.astype(np.float64)
        ros = np.zeros((stencil.size, h, w), dtype=np.float64)
        for k in range(stencil.size):
            dr, dc = int(stencil.offsets[k, 0]), int(stencil.offsets[k, 1])
            # Neighbour elevation; cells whose neighbour is off-grid keep their own (theta_s = 0).
            neighbour = elev.copy()
            sr0, sr1 = max(-dr, 0), h + min(-dr, 0)
            sc0, sc1 = max(-dc, 0), w + min(-dc, 0)
            if sr0 < sr1 and sc0 < sc1:
                neighbour[sr0:sr1, sc0:sc1] = elev[sr0 + dr : sr1 + dr, sc0 + dc : sc1 + dc]
            run = grid.dx * float(stencil.distances[k])
            theta_s = np.degrees(np.arctan((neighbour - elev) / run))
            log_factor = np.clip(log_wind[k] + self._a * theta_s, -_MAX_LOG_FACTOR, _MAX_LOG_FACTOR)
            ros[k] = base * np.exp(log_factor)

        zeros = np.zeros((h, w), dtype=np.float64)
        return SpreadOutput(ros=ros, intensity=zeros, flame_length=zeros.copy())
