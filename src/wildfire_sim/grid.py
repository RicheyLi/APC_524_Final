"""The simulation grid and the weather inputs that drive it.

:class:`Grid` holds every per-cell layer as a 2-D NumPy array of a fixed dtype. It is a plain
container: it does no physics. Spread rules read from it, the simulator writes state back.

This module is a frozen contract (see CONTRIBUTING.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_sim.types import BoolArray, CellState, FloatArray, IntArray, LandUse

# Layer name -> (dtype, default fill). Order here is the canonical order in save files.
LAYER_SPECS: dict[str, tuple[type, float | int | bool]] = {
    # static
    "fuel_model": (np.int16, 0),
    "canopy_cover": (np.float32, 0.0),
    "elevation": (np.float32, 0.0),
    "slope": (np.float32, 0.0),
    "aspect": (np.float32, 0.0),
    "land_use": (np.int8, int(LandUse.WILDLAND)),
    "asset_value": (np.float32, 0.0),
    # suppression
    "fireline": (np.bool_, False),
    "retardant": (np.float32, 0.0),
    # dynamic state
    "state": (np.int8, int(CellState.FUEL)),
    "burn_timer": (np.float32, 0.0),
    "intensity": (np.float32, 0.0),
    "arrival_time": (np.float32, np.nan),
}

_SAVE_FORMAT_VERSION = 1


@dataclass
class Grid:
    """All per-cell layers of a study area on a regular square grid.

    Every layer has shape ``(H, W)``. Use :meth:`empty` to build a grid with default values and
    :meth:`validate` after editing layers by hand.

    Attributes:
        dx: Cell size in metres (30 m for LANDFIRE).
        fuel_model: Scott & Burgan (2005) fuel model number (1-204); 0 = non-burnable. ``int16``.
        canopy_cover: Canopy cover fraction in [0, 1]. ``float32``.
        elevation: Elevation above sea level, m. ``float32``.
        slope: Rise/run fraction (tan of slope angle), >= 0. ``float32``.
        aspect: Downhill bearing, degrees clockwise from north in [0, 360). ``float32``.
        land_use: :class:`~wildfire_sim.types.LandUse` code. ``int8``.
        asset_value: Value at risk in the cell, USD. ``float32``.
        fireline: True where a completed control line makes the cell unburnable. ``bool``.
        retardant: Retardant strength in [0, 1]; the spread rate is multiplied by
            ``1 - retardant``. Decays over time. ``float32``.
        state: :class:`~wildfire_sim.types.CellState` code. ``int8``.
        burn_timer: Seconds of residence time remaining for burning cells. ``float32``.
        intensity: Byram fireline intensity of the most recent step, kW/m. ``float32``.
        arrival_time: Simulated time at which the cell ignited, s; NaN if never. ``float32``.
    """

    dx: float
    fuel_model: IntArray
    canopy_cover: FloatArray
    elevation: FloatArray
    slope: FloatArray
    aspect: FloatArray
    land_use: IntArray
    asset_value: FloatArray
    fireline: BoolArray
    retardant: FloatArray
    state: IntArray
    burn_timer: FloatArray
    intensity: FloatArray
    arrival_time: FloatArray
    metadata: dict[str, str] = field(default_factory=dict)
    """Free-form provenance (data source, bbox, CRS, download date). Strings only."""

    # ------------------------------------------------------------------ construction

    @classmethod
    def empty(cls, shape: tuple[int, int], dx: float) -> Grid:
        """Create a grid of ``shape`` with every layer at its default value.

        Defaults: fuel model 0, wildland land use, state FUEL, no fireline, no retardant.
        Callers must set ``fuel_model`` to a burnable value for fire to spread.
        """
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"shape must be (H, W) with H, W >= 1, got {shape!r}")
        if dx <= 0:
            raise ValueError(f"dx must be positive, got {dx!r}")
        layers: dict[str, Any] = {
            name: np.full(shape, fill, dtype=dtype) for name, (dtype, fill) in LAYER_SPECS.items()
        }
        return cls(dx=float(dx), **layers)

    # ------------------------------------------------------------------ properties

    @property
    def shape(self) -> tuple[int, int]:
        """``(H, W)`` of every layer."""
        return (int(self.state.shape[0]), int(self.state.shape[1]))

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(LAYER_SPECS)

    # ------------------------------------------------------------------ validation

    def validate(self) -> None:
        """Raise ``ValueError`` if any layer has the wrong shape or dtype, or values are out of range."""
        shape = self.shape
        for name, (dtype, _) in LAYER_SPECS.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"layer {name!r} has shape {arr.shape}, expected {shape}")
            if arr.dtype != np.dtype(dtype):
                raise ValueError(f"layer {name!r} has dtype {arr.dtype}, expected {np.dtype(dtype)}")
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        for name in ("canopy_cover", "elevation", "slope", "aspect", "asset_value", "retardant"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"layer {name!r} must be finite everywhere")
        if np.any(self.asset_value < 0):
            raise ValueError("asset_value must be >= 0")
        if not (np.all(self.canopy_cover >= 0) and np.all(self.canopy_cover <= 1)):
            raise ValueError("canopy_cover must lie in [0, 1]")
        if np.any(self.slope < 0):
            raise ValueError("slope must be >= 0")
        if not (np.all(self.aspect >= 0) and np.all(self.aspect < 360)):
            raise ValueError("aspect must lie in [0, 360)")
        if not (np.all(self.retardant >= 0) and np.all(self.retardant <= 1)):
            raise ValueError("retardant must lie in [0, 1]")
        valid_states = np.array([int(s) for s in CellState], dtype=np.int8)
        if not np.isin(self.state, valid_states).all():
            raise ValueError("state contains values outside CellState")
        valid_lu = np.array([int(s) for s in LandUse], dtype=np.int8)
        if not np.isin(self.land_use, valid_lu).all():
            raise ValueError("land_use contains values outside LandUse")

    # ------------------------------------------------------------------ derived masks

    def burnable_mask(self) -> BoolArray:
        """Cells that can ever carry fire: wildland, burnable fuel model, no fireline."""
        wildland = self.land_use == int(LandUse.WILDLAND)
        has_fuel = self.fuel_model > 0
        return wildland & has_fuel & ~self.fireline

    def apply_unburnable(self) -> None:
        """Set ``state`` to UNBURNABLE wherever :meth:`burnable_mask` is False and the cell is FUEL.

        Idempotent. Call after editing static layers or adding a fireline. Cells that are
        already BURNING or BURNED keep their state: a line built under a fire does not unburn it.
        """
        self.state[~self.burnable_mask() & (self.state == int(CellState.FUEL))] = int(CellState.UNBURNABLE)

    def burning_mask(self) -> BoolArray:
        return self.state == int(CellState.BURNING)

    def burned_mask(self) -> BoolArray:
        """Cells burning now or already burned out."""
        return (self.state == int(CellState.BURNING)) | (self.state == int(CellState.BURNED))

    def burned_area_m2(self) -> float:
        """Area of cells that have burned (burning or burned out), m^2."""
        return float(self.burned_mask().sum()) * self.dx * self.dx

    # ------------------------------------------------------------------ copy / io

    def copy(self) -> Grid:
        """Deep copy of all layers."""
        kwargs: dict[str, Any] = {name: getattr(self, name).copy() for name in LAYER_SPECS}
        return Grid(dx=self.dx, metadata=dict(self.metadata), **kwargs)

    def save(self, path: str | Path) -> None:
        """Write all layers to a compressed ``.npz`` file."""
        meta_keys = list(self.metadata)
        meta_vals = [self.metadata[k] for k in meta_keys]
        np.savez_compressed(
            Path(path),
            _format_version=np.int64(_SAVE_FORMAT_VERSION),
            dx=np.float64(self.dx),
            _meta_keys=np.array(meta_keys, dtype=str),
            _meta_vals=np.array(meta_vals, dtype=str),
            **{name: getattr(self, name) for name in LAYER_SPECS},
        )

    @classmethod
    def load(cls, path: str | Path) -> Grid:
        """Read a grid written by :meth:`save`. Validates on load."""
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["_format_version"])
            if version != _SAVE_FORMAT_VERSION:
                raise ValueError(f"unsupported grid file version {version}")
            layers: dict[str, Any] = {name: np.asarray(data[name]) for name in LAYER_SPECS}
            keys = [str(k) for k in data["_meta_keys"]]
            vals = [str(v) for v in data["_meta_vals"]]
            grid = cls(dx=float(data["dx"]), metadata=dict(zip(keys, vals, strict=True)), **layers)
        grid.validate()
        return grid

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        if self.dx != other.dx or self.metadata != other.metadata:
            return False
        return all(
            np.array_equal(getattr(self, f.name), getattr(other, f.name), equal_nan=True)
            for f in fields(self)
            if f.name in LAYER_SPECS
        )


@dataclass(frozen=True)
class Weather:
    """Spatially uniform weather for one instant.

    Weather sources (gridMET ~4 km) are coarser than the 10 km study areas, so a single
    value per time applies to the whole grid.

    Attributes:
        wind_speed: Open 10 m wind speed, m/s. Spread rules apply their own midflame adjustment.
        wind_direction: Bearing the wind blows **from**, degrees clockwise from north.
        moisture_1h: Dead fuel moisture, 1-hour class, fraction of dry weight.
        moisture_10h: Dead fuel moisture, 10-hour class, fraction.
        moisture_100h: Dead fuel moisture, 100-hour class, fraction.
        moisture_live_herb: Live herbaceous fuel moisture, fraction.
        moisture_live_woody: Live woody fuel moisture, fraction.
        temperature_c: Air temperature, deg C (informational; enters via moisture).
        relative_humidity: Fraction in [0, 1] (informational; enters via moisture).
    """

    wind_speed: float = 0.0
    wind_direction: float = 0.0
    moisture_1h: float = 0.06
    moisture_10h: float = 0.07
    moisture_100h: float = 0.08
    moisture_live_herb: float = 0.60
    moisture_live_woody: float = 0.90
    temperature_c: float = 25.0
    relative_humidity: float = 0.30

    def __post_init__(self) -> None:
        if self.wind_speed < 0:
            raise ValueError("wind_speed must be >= 0")
        if not 0 <= self.wind_direction < 360:
            raise ValueError("wind_direction must lie in [0, 360)")
        for name in ("moisture_1h", "moisture_10h", "moisture_100h", "moisture_live_herb", "moisture_live_woody"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0 <= self.relative_humidity <= 1:
            raise ValueError("relative_humidity must lie in [0, 1]")

    @property
    def wind_to_bearing(self) -> float:
        """Bearing the wind blows **toward** (direction of head-fire spread), degrees."""
        return (self.wind_direction + 180.0) % 360.0


@dataclass(frozen=True)
class WeatherSeries:
    """Piecewise-constant weather in time.

    ``times_s[i]`` is the simulated time (s) from which ``values[i]`` applies. The first entry
    must be at ``t = 0``; before any later breakpoint the previous value holds.
    """

    times_s: tuple[float, ...]
    values: tuple[Weather, ...]

    def __post_init__(self) -> None:
        if len(self.times_s) == 0 or len(self.times_s) != len(self.values):
            raise ValueError("times_s and values must be non-empty and the same length")
        if self.times_s[0] != 0:
            raise ValueError("the first weather breakpoint must be at t = 0 s")
        if any(b <= a for a, b in zip(self.times_s, self.times_s[1:], strict=False)):
            raise ValueError("times_s must be strictly increasing")

    @classmethod
    def constant(cls, weather: Weather) -> WeatherSeries:
        return cls(times_s=(0.0,), values=(weather,))

    def at(self, t: float) -> Weather:
        """Weather in force at simulated time ``t`` (s)."""
        if t < 0:
            raise ValueError("t must be >= 0")
        idx = int(np.searchsorted(np.asarray(self.times_s), t, side="right")) - 1
        return self.values[idx]
