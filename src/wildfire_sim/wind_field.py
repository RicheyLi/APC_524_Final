"""Spatially varying wind from the HRRR's native grid (WP 3.4).

A region's ``wind_field.npz`` (written by the region build, :mod:`wildfire_sim.data.cache`) holds
the hourly HRRR 10 m wind on the analysis nodes of a small window around the region, already
rotated to earth-relative components, and every grid cell's fractional position in that window.
:class:`WindFieldSeries` turns it into a :class:`~wildfire_sim.grid.WindField` for the grid at
any simulated time, with numpy only (no geospatial dependency at run time):

* **Space.** Bilinear interpolation of the east and north components ``u`` and ``v`` (not of
  speed and direction, which do not interpolate: two opposite winds average to calm, not to a
  wind at right angles) between the four nodes around each cell, in the HRRR grid's index
  space, as :func:`wildfire_sim.data.hrrr.sample_bilinear` does at a point. There is no
  downscaling: the field is as smooth as the 3 km analysis.
* **Time.** Piecewise constant, like :class:`~wildfire_sim.grid.WeatherSeries`: hour ``k``
  applies from ``times_s[k]`` until the next one; the series uses the same times as the
  region's ``weather.yaml``. Each hour's field is computed once and reused.
* **Speed and direction** from the interpolated components with the convention of
  :func:`wildfire_sim.data.hrrr.wind_from`: ``speed = hypot(u, v)`` (m/s) and the bearing the wind
  blows from, ``atan2(-u, -v)`` in degrees clockwise from north (0 where calm).

File format (``_format_version`` 1), all arrays C-ordered:

* ``times_s`` ``(T,)`` float64: simulated seconds from the region's ``weather_start_utc``;
  ``times_s[0] = 0``, strictly increasing.
* ``u``, ``v`` ``(T, ny, nx)`` float32: earth-relative east and north 10 m wind, m/s, on the
  window's nodes (row 0 is the window's first row in the HRRR raster as GDAL reads it).
* ``cell_i``, ``cell_j`` ``(H, W)`` float32: each grid cell centre's fractional (row, column)
  in the window, node centres at integers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_sim.grid import WindField
from wildfire_sim.types import FloatArray

WIND_FIELD_FORMAT_VERSION = 1
"""Version of the ``wind_field.npz`` layout described in the module docstring."""

WIND_FIELD_FILE = "wind_field.npz"
"""Name of the wind field file in a region directory, beside ``weather.yaml``."""

_ARRAYS = ("times_s", "u", "v", "cell_i", "cell_j")


def save_wind_field(
    path: str | Path,
    times_s: FloatArray,
    u: FloatArray,
    v: FloatArray,
    cell_i: FloatArray,
    cell_j: FloatArray,
) -> None:
    """Write a wind field file (format in the module docstring).

    Args:
        path: Output ``.npz`` file.
        times_s: ``(T,)`` simulated times, s.
        u, v: ``(T, ny, nx)`` earth-relative 10 m wind components, m/s.
        cell_i, cell_j: ``(H, W)`` fractional window row and column of each cell.

    Raises:
        ValueError: The arrays are inconsistent (checked by :class:`WindFieldSeries`).
    """
    series = WindFieldSeries(
        np.asarray(times_s, dtype=np.float64),
        np.asarray(u, dtype=np.float32),
        np.asarray(v, dtype=np.float32),
        np.asarray(cell_i, dtype=np.float32),
        np.asarray(cell_j, dtype=np.float32),
    )
    np.savez_compressed(
        Path(path),
        _format_version=np.int64(WIND_FIELD_FORMAT_VERSION),
        **{name: getattr(series, name) for name in _ARRAYS},
    )


def wind_from_components(u: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Speed (m/s) and "from" bearing (degrees in [0, 360)) of east/north components, element-wise.

    The array form of :func:`wildfire_sim.data.hrrr.wind_from`; the bearing is 0 where calm.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    speed = np.hypot(u, v)
    direction = np.where(speed > 0, np.degrees(np.arctan2(-u, -v)) % 360.0, 0.0)
    return speed, np.where(direction >= 360.0, 0.0, direction)


@dataclass(frozen=True, eq=False)
class WindFieldSeries:
    """Hourly HRRR wind on a node window and the grid cells' positions in it.

    Attributes:
        times_s: ``(T,)`` float64, simulated time, s, from which each hour applies.
        u, v: ``(T, ny, nx)`` float32, earth-relative east and north 10 m wind, m/s.
        cell_i, cell_j: ``(H, W)`` float32, fractional window row and column of each cell.
        path: File the series was read from (None if built in memory).
        sha256: SHA-256 of that file.
        speed_factor: Factor applied to every cell's speed (ensemble perturbation), dimensionless.
        direction_offset_deg: Degrees added to every cell's "from" bearing (ensemble perturbation).
    """

    times_s: FloatArray
    u: FloatArray
    v: FloatArray
    cell_i: FloatArray
    cell_j: FloatArray
    path: Path | None = None
    sha256: str | None = None
    speed_factor: float = 1.0
    direction_offset_deg: float = 0.0
    _cache: dict[int, WindField] = field(init=False, repr=False)
    _weights: tuple[Any, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t = np.asarray(self.times_s, dtype=np.float64)
        if t.ndim != 1 or t.size == 0:
            raise ValueError("times_s must be a non-empty 1-D array")
        if t[0] != 0:
            raise ValueError("the first wind field time must be 0 s")
        if np.any(np.diff(t) <= 0):
            raise ValueError("wind field times_s must be strictly increasing")
        u, v = np.asarray(self.u), np.asarray(self.v)
        if u.ndim != 3 or u.shape != v.shape or u.shape[0] != t.size:
            raise ValueError(f"u {u.shape} and v {v.shape} must both be (T, ny, nx) with T = {t.size}")
        ny, nx = u.shape[1:]
        if ny < 2 or nx < 2:
            raise ValueError(f"the node window must be at least 2 x 2, got {ny} x {nx}")
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            raise ValueError("wind field u and v must be finite")
        ci, cj = np.asarray(self.cell_i), np.asarray(self.cell_j)
        if ci.ndim != 2 or ci.shape != cj.shape:
            raise ValueError(f"cell_i {ci.shape} and cell_j {cj.shape} must be equal (H, W) shapes")
        if not (np.all(np.isfinite(ci)) and np.all(np.isfinite(cj))):
            raise ValueError("cell_i and cell_j must be finite")
        if ci.min() < 0 or ci.max() > ny - 1 or cj.min() < 0 or cj.max() > nx - 1:
            raise ValueError(f"cells lie outside the {ny} x {nx} node window")
        if not (math.isfinite(self.speed_factor) and self.speed_factor > 0):
            raise ValueError(f"speed_factor must be finite and > 0, got {self.speed_factor!r}")
        if not math.isfinite(self.direction_offset_deg):
            raise ValueError("direction_offset_deg must be finite")
        object.__setattr__(self, "times_s", t)

        fi64, fj64 = ci.astype(np.float64), cj.astype(np.float64)
        i0 = np.minimum(np.floor(fi64).astype(np.intp), ny - 2)
        j0 = np.minimum(np.floor(fj64).astype(np.intp), nx - 2)
        object.__setattr__(self, "_weights", (i0, j0, fi64 - i0, fj64 - j0))
        object.__setattr__(self, "_cache", {})

    # ------------------------------------------------------------------ properties

    @property
    def shape(self) -> tuple[int, int]:
        """``(H, W)`` of the grid the field is for."""
        return (int(self.cell_i.shape[0]), int(self.cell_i.shape[1]))

    @property
    def window_shape(self) -> tuple[int, int]:
        """``(ny, nx)`` of the HRRR node window."""
        return (int(self.u.shape[1]), int(self.u.shape[2]))

    @property
    def n_times(self) -> int:
        return int(self.times_s.size)

    # ------------------------------------------------------------------ evaluation

    def index_at(self, t: float) -> int:
        """Hour in force at simulated time ``t`` (s): the rule of :meth:`WeatherSeries.at`."""
        if t < 0:
            raise ValueError("t must be >= 0")
        return int(np.searchsorted(self.times_s, t, side="right")) - 1

    def components(self, k: int) -> tuple[FloatArray, FloatArray]:
        """``(H, W)`` float64 east and north wind of hour ``k`` at the cells, m/s (bilinear, unperturbed)."""
        i0, j0, fi, fj = self._weights

        def interp(a: FloatArray) -> FloatArray:
            # nested linear interpolation in the form a + f (b - a): exact where the nodes are equal
            a = a.astype(np.float64)
            top = a[i0, j0] + fj * (a[i0, j0 + 1] - a[i0, j0])
            bottom = a[i0 + 1, j0] + fj * (a[i0 + 1, j0 + 1] - a[i0 + 1, j0])
            return top + fi * (bottom - top)

        return interp(self.u[k]), interp(self.v[k])

    def at(self, t: float, shape: tuple[int, int] | None = None) -> WindField:
        """The field in force at simulated time ``t`` (s), perturbed by the series' factors.

        Args:
            t: Simulated time, s, >= 0.
            shape: When given, the grid shape the field must have.

        Raises:
            ValueError: ``t < 0`` or ``shape`` differs from the field's.
        """
        if shape is not None and tuple(shape) != self.shape:
            raise ValueError(f"the wind field is for a {self.shape} grid, but the grid is {tuple(shape)}")
        k = self.index_at(t)
        cached = self._cache.get(k)
        if cached is None:
            speed, direction = wind_from_components(*self.components(k))
            if self.speed_factor != 1.0:
                speed = speed * self.speed_factor
            if self.direction_offset_deg != 0.0:
                direction = (direction + self.direction_offset_deg) % 360.0
                direction = np.where(direction >= 360.0, 0.0, direction)  # a tiny negative value rounds to 360
            cached = WindField(speed, direction)
            self._cache[k] = cached
        return cached

    def perturbed(self, speed_factor: float = 1.0, direction_offset_deg: float = 0.0) -> WindFieldSeries:
        """This series with every cell's speed times ``speed_factor`` and ``direction_offset_deg``
        (degrees) added to every cell's "from" bearing, on top of any perturbation it already has."""
        return WindFieldSeries(
            self.times_s,
            self.u,
            self.v,
            self.cell_i,
            self.cell_j,
            self.path,
            self.sha256,
            self.speed_factor * float(speed_factor),
            self.direction_offset_deg + float(direction_offset_deg),
        )

    def provenance(self) -> dict[str, Any]:
        """JSON-serializable record: file, SHA-256, sizes and perturbation."""
        return {
            "path": None if self.path is None else str(self.path),
            "sha256": self.sha256,
            "n_times": self.n_times,
            "ny": self.window_shape[0],
            "nx": self.window_shape[1],
            "grid_shape": list(self.shape),
            "speed_factor": self.speed_factor,
            "direction_offset_deg": self.direction_offset_deg,
        }


def load_wind_field(path: str | Path) -> WindFieldSeries:
    """Read a wind field file written by :func:`save_wind_field` (validated).

    Raises:
        FileNotFoundError: No such file; the message says how to build it.
        ValueError: Unsupported format version or inconsistent arrays.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"wind field file {p} not found; build it for a region with\n"
            "  uv run --extra geo python scripts/download_data.py <region> --wind-field\n"
            "or remove weather.wind_field from the scenario to use the single-point wind"
        )
    raw = p.read_bytes()
    with np.load(p, allow_pickle=False) as data:
        version = int(data["_format_version"]) if "_format_version" in data.files else -1
        if version != WIND_FIELD_FORMAT_VERSION:
            raise ValueError(f"{p}: unsupported wind field file version {version}")
        missing = [name for name in _ARRAYS if name not in data.files]
        if missing:
            raise ValueError(f"{p}: wind field file lacks {missing}")
        arrays = {name: np.asarray(data[name]) for name in _ARRAYS}
    return WindFieldSeries(
        arrays["times_s"],
        arrays["u"],
        arrays["v"],
        arrays["cell_i"],
        arrays["cell_j"],
        path=p,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def wind_field_for(config: Any) -> WindFieldSeries | None:
    """The wind field of a scenario (``config.weather.wind_field``), or None when it has none."""
    path = config.weather.wind_field
    return None if path is None else load_wind_field(path)
