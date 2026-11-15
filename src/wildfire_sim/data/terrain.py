"""Elevation, slope and aspect for a grid (docs/plan.md section 3, WP 2.3).

Elevation comes from the USGS 3D Elevation Program (3DEP) through ``py3dep``: its 10 m DEM, the
finest that covers the whole conterminous US, is fetched for the bounding box in EPSG:5070
(CONUS Albers equal-area, the projection LANDFIRE uses), cached, and resampled bilinearly onto
the 30 m LANDFIRE grid (:func:`reproject_to`). Slope and aspect are then computed on that grid
(:func:`slope_aspect`), so they describe the terrain at the scale the fire model sees.

**Slope and aspect.** With ``z`` the elevation (m) on a north-up grid, ``np.gradient`` gives the
change per row and per column; with cell sizes ``dx`` (east) and ``dy`` (north) in metres,

.. math::

    \\frac{\\partial z}{\\partial x} = \\frac{1}{dx}\\frac{\\partial z}{\\partial c}, \\qquad
    \\frac{\\partial z}{\\partial y} = -\\frac{1}{dy}\\frac{\\partial z}{\\partial r}

(rows increase southward). Slope is ``|grad z|`` (rise over run, ``Grid.slope``) and aspect is
the bearing of ``-grad z``, the downhill direction, in degrees clockwise from north
(``Grid.aspect``; 0 on flat cells). ``np.gradient`` uses central differences inside and
one-sided differences on the edge, so a plane is recovered exactly everywhere. In EPSG:5070 the
grid's metres are true lengths to within the equal-area projection's scale error (under about
1 % across the conterminous US), which is the error of slope too.

**Grid metadata.** :func:`grid_metadata` gives the strings ``Grid.metadata`` carries for a real
grid: ``crs`` (WKT), ``transform`` (the six affine coefficients ``a b c d e f`` of the
north-up transform, rasterio order), ``centroid_lon`` and ``centroid_lat`` (degrees, WGS84, the
centre of the grid).

The download needs the ``geo`` extra (``py3dep``, ``rasterio``); :func:`slope_aspect` needs only
NumPy.

References:
    U.S. Geological Survey, 3D Elevation Program, https://www.usgs.gov/3d-elevation-program.
    Chegini, T., Li, H.-Y. & Leung, L. R. (2021). HyRiver: Hydroclimate data retriever. Journal of
    Open Source Software 6(66), 3175 (py3dep).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_sim.data.landfire import DEFAULT_CACHE, BBox, optional_import
from wildfire_sim.types import FloatArray

DEM_CRS = "EPSG:5070"
"""CRS the DEM is requested in: CONUS Albers equal-area, as LANDFIRE."""

DEM_RESOLUTION_M = 10
"""3DEP source resolution, m: the 10 m DEM covers the conterminous US."""


# ----------------------------------------------------------------------------- slope and aspect


def slope_aspect(elevation: FloatArray, dx: float, dy: float | None = None) -> tuple[FloatArray, FloatArray]:
    """Slope (rise over run) and aspect (downhill bearing, degrees) of a north-up elevation grid.

    Args:
        elevation: ``(H, W)`` elevation, m, row 0 at the north edge. Must be finite.
        dx: Cell width (east-west), m, > 0.
        dy: Cell height (north-south), m, > 0; None means ``dx``.

    Returns:
        ``(slope, aspect)``: float64 ``(H, W)`` arrays; slope >= 0, aspect in [0, 360) with 0 on
        flat cells.

    Raises:
        ValueError: Non-finite elevation, a grid smaller than 2 x 2, or a non-positive cell size.
    """
    z = np.asarray(elevation, dtype=np.float64)
    dy = dx if dy is None else dy
    if dx <= 0 or dy <= 0:
        raise ValueError("cell sizes must be > 0")
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("elevation must be a 2-D grid of at least 2 x 2 cells")
    if not np.all(np.isfinite(z)):
        raise ValueError("elevation must be finite (fill nodata first)")
    dz_drow, dz_dcol = np.gradient(z)
    east = dz_dcol / dx  # dz/dx, x eastward
    north = -dz_drow / dy  # dz/dy, y northward (rows run south)
    slope = np.hypot(east, north)
    flat = slope <= 1e-12
    aspect = np.degrees(np.arctan2(-east, -north)) % 360.0  # bearing of -grad z, the way down
    aspect = np.where(flat | (aspect >= 360.0), 0.0, aspect)
    return slope, aspect


# ----------------------------------------------------------------------------- the 3DEP DEM


@dataclass(frozen=True)
class DemFile:
    """A cached DEM.

    Attributes:
        path: GeoTIFF, elevation in m, in :data:`DEM_CRS`.
        manifest: Contents of the ``manifest.json`` next to it.
    """

    path: Path
    manifest: dict[str, Any]


DemSource = Callable[[BBox, int, Path], None]
"""Writes a DEM GeoTIFF for a bounding box at a resolution (m) to a path."""


def py3dep_source(bbox: BBox, resolution_m: int, path: Path) -> None:
    """:data:`DemSource` using ``py3dep.get_dem``; the result is in EPSG:5070."""
    py3dep = optional_import("py3dep")

    dem = py3dep.get_dem((bbox.west, bbox.south, bbox.east, bbox.north), resolution_m, crs=4326)
    dem.rio.to_raster(path, driver="GTiff", compress="deflate")


def dem_cache_key(bbox: BBox, resolution_m: int) -> str:
    """16 hex digits naming the cache directory of a (bounding box, resolution) DEM."""
    canon = json.dumps({"aoi": bbox.aoi(), "dataset": "3DEP", "resolution_m": resolution_m}, sort_keys=True)
    return "dem-" + hashlib.sha256(canon.encode()).hexdigest()[:16]


def fetch_dem(
    bbox: BBox,
    *,
    resolution_m: int = DEM_RESOLUTION_M,
    cache_root: str | Path = DEFAULT_CACHE,
    source: DemSource | None = None,
) -> DemFile:
    """The 3DEP DEM of a bounding box, from the cache or downloaded once.

    Args:
        bbox: Area of interest, WGS84 degrees. Pad it by a few cells beyond the fire grid so
            the reprojection has data on every edge.
        resolution_m: Source resolution, m (10, 30 or 60 are served directly by 3DEP).
        cache_root: Directory holding ``dem-<key>/`` directories.
        source: Downloader; None uses :func:`py3dep_source`.

    Raises:
        ValueError: A cached file does not match its manifest hash.
    """
    directory = Path(cache_root) / dem_cache_key(bbox, resolution_m)
    path = directory / "dem.tif"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]["dem.tif"]:
            raise ValueError(f"{directory}: dem.tif does not match its manifest hash; delete the directory")
        return DemFile(path, manifest)
    directory.mkdir(parents=True, exist_ok=True)
    (source or py3dep_source)(bbox, resolution_m, path)
    manifest = {
        "dataset": "USGS 3D Elevation Program (3DEP) DEM",
        "access": "py3dep.get_dem" if source is None else getattr(source, "__name__", "custom source"),
        "resolution_m": resolution_m,
        "crs": DEM_CRS,
        "bbox_wgs84": {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north},
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sha256": {"dem.tif": hashlib.sha256(path.read_bytes()).hexdigest()},
        "attribution": "U.S. Geological Survey, 3D Elevation Program (public domain)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return DemFile(path, manifest)


# ----------------------------------------------------------------------------- onto the grid


@dataclass(frozen=True)
class TargetGrid:
    """The raster the fire grid lives on (the LANDFIRE grid).

    Attributes:
        crs: Anything rasterio accepts as a CRS (``"EPSG:5070"``, WKT, a ``CRS``).
        transform: North-up affine transform (``rasterio.Affine``) of the top-left corner.
        shape: ``(H, W)``.
    """

    crs: Any
    transform: Any
    shape: tuple[int, int]

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> TargetGrid:
        """From a rasterio profile, e.g. the second value of ``landfire.read_layer``."""
        return cls(profile["crs"], profile["transform"], (int(profile["height"]), int(profile["width"])))

    @property
    def cell_size(self) -> tuple[float, float]:
        """``(dx, dy)``, the cell width and height in CRS units (m), both > 0.

        Raises:
            ValueError: The transform is rotated or not north-up.
        """
        t = self.transform
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError("the grid must be north-up and unrotated")
        return float(t.a), float(-t.e)


def reproject_to(dem: DemFile | str | Path, target: TargetGrid) -> FloatArray:
    """The DEM resampled bilinearly onto ``target``, elevation in m, float64 ``(H, W)``.

    Raises:
        ValueError: Some target cell has no DEM data (the DEM does not cover the grid).
    """
    rasterio = optional_import("rasterio")
    warp = optional_import("rasterio.warp")
    Resampling, reproject = warp.Resampling, warp.reproject

    path = dem.path if isinstance(dem, DemFile) else Path(dem)
    out = np.full(target.shape, np.nan, dtype=np.float64)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_nodata=src.nodata,
            dst_transform=target.transform,
            dst_crs=target.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    if not np.all(np.isfinite(out)):
        missing = int(np.count_nonzero(~np.isfinite(out)))
        raise ValueError(f"the DEM does not cover {missing} grid cells; fetch a larger bounding box")
    return out


def grid_metadata(target: TargetGrid) -> dict[str, str]:
    """``Grid.metadata`` entries for a real grid: ``crs``, ``transform``, ``centroid_lon``, ``centroid_lat``."""
    CRS = optional_import("rasterio.crs").CRS
    transform_points = optional_import("rasterio.warp").transform

    crs = CRS.from_user_input(target.crs)
    t = target.transform
    h, w = target.shape
    cx, cy = t.c + t.a * w / 2 + t.b * h / 2, t.f + t.d * w / 2 + t.e * h / 2
    lon, lat = transform_points(crs, "EPSG:4326", [cx], [cy])
    return {
        "crs": crs.to_wkt(),
        "transform": " ".join(repr(float(v)) for v in (t.a, t.b, t.c, t.d, t.e, t.f)),
        "centroid_lon": f"{lon[0]:.6f}",
        "centroid_lat": f"{lat[0]:.6f}",
    }


@dataclass(frozen=True)
class Terrain:
    """Terrain layers on the fire grid.

    Attributes:
        elevation: ``(H, W)`` m.
        slope: ``(H, W)`` rise over run.
        aspect: ``(H, W)`` downhill bearing, degrees in [0, 360).
        metadata: ``Grid.metadata`` entries (:func:`grid_metadata`).
    """

    elevation: FloatArray
    slope: FloatArray
    aspect: FloatArray
    metadata: dict[str, str]


def terrain_for_grid(dem: DemFile | str | Path, target: TargetGrid) -> Terrain:
    """Elevation resampled onto ``target``, its slope and aspect, and the grid metadata."""
    elevation = reproject_to(dem, target)
    dx, dy = target.cell_size
    slope, aspect = slope_aspect(elevation, dx, dy)
    return Terrain(elevation, slope, aspect, grid_metadata(target))


def slope_angle_deg(slope: FloatArray) -> FloatArray:
    """Slope angle, degrees, from rise over run (for reports; the model uses rise over run)."""
    return np.degrees(np.arctan(np.asarray(slope, dtype=np.float64)))


__all__ = [
    "DEM_CRS",
    "DEM_RESOLUTION_M",
    "DemFile",
    "TargetGrid",
    "Terrain",
    "dem_cache_key",
    "fetch_dem",
    "grid_metadata",
    "py3dep_source",
    "reproject_to",
    "slope_angle_deg",
    "slope_aspect",
    "terrain_for_grid",
]
