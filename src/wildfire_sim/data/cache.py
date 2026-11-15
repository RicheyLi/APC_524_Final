"""Cached study regions: one command from a fire's name to a grid and its weather (WP 2.6).

A region (``configs/regions.yaml``) is one validation fire. :func:`build_region` assembles it:

1. **Perimeter.** The fire's final perimeter from MTBS (Monitoring Trends in Burn Severity), read
   from the USFS map service and cached (:func:`fetch_mtbs_perimeter`).
2. **Grid extent.** Every input is downloaded for the perimeter's bounding box (planned in
   EPSG:5070) plus ``buffer_m`` (at least 3 km) and a 1 km margin. The grid is the LANDFIRE
   raster's own 30 m lattice cut to the perimeter's bounding box plus ``buffer_m`` and one cell
   on every side, so the grid size follows the fire (docs/plan.md section 2).
3. **Fuels** (WP 2.2). LANDFIRE FBFM40 and canopy layers of the region's version, which must
   predate the fire. The Product Service delivers them in an Albers projection centred on the
   request (central meridian at the box centre), so grid north is true north to a fraction of
   a degree across the region. That matters: wind bearings and aspect are measured against grid
   north, and the simulator takes it to be true north. In EPSG:5070 (central meridian 96 W) grid
   north is rotated about 14.5 deg at 120 W, so :func:`grid_rotation_deg` is checked and a grid
   rotated more than :data:`MAX_GRID_ROTATION_DEG` is refused. ``grid_codes`` gives fuel model
   and land use; canopy cover goes into
   ``Grid.canopy_cover`` as a fraction. Canopy height, base height and bulk density have no
   ``Grid`` layer yet (section 13) and are written to ``canopy.npz`` in metres, metres and
   kg/m^3. LANDFIRE stores canopy cover in percent, height and base height in metres x 10 and
   bulk density in kg/m^3 x 100, with -9999 for no data (read 0: no canopy).
4. **Terrain** (WP 2.3). The 3DEP DEM resampled onto the grid; slope, aspect, CRS, transform and
   centroid.
5. **Roads, water, buildings** (WP 2.5). OpenStreetMap features with the barrier rule; break
   widths; asset value from buildings.
6. **Weather** (WP 2.4). Hourly gridMET at the grid centroid from ``weather_start_utc`` for
   ``weather_days``.

The region directory ``data/cache/<region>/`` then holds ``grid.npz``, ``canopy.npz``,
``weather.yaml``, ``perimeter.geojson`` and ``manifest.json`` (sources, versions, dates, the
manifests of every download, the SHA-256 of each file). Every download is cached by its own
module, so building again reads the caches and reproduces the same files byte for byte
(:func:`verify_region`). A 5 km x 5 km crop centred on the MTBS point of origin is written to
``tests/fixtures/`` for offline tests.

Everything here except the region table needs the ``geo`` extra (``uv sync --extra geo``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from wildfire_sim.config import WeatherConfig, configs_dir, dump_yaml, load_costs, load_yaml
from wildfire_sim.data.landfire import (
    DEFAULT_CACHE,
    BBox,
    LandfireLayers,
    UrllibHttp,
    check_version,
    fetch_landfire,
    grid_codes,
    optional_import,
)
from wildfire_sim.data.osm import OsmFeatures, apply_to_grid, fetch_osm, rasterize_features
from wildfire_sim.data.terrain import DemFile, TargetGrid, fetch_dem, grid_metadata, terrain_for_grid
from wildfire_sim.data.weather import gridmet_cache_key, hourly_weather, solar_dates_needed
from wildfire_sim.grid import LAYER_SPECS, Grid
from wildfire_sim.types import FloatArray, IntArray
from wildfire_sim.urban import load_urban_model

MTBS_SERVICE = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer"
"""USFS Enterprise Data Warehouse map service of MTBS fire occurrences and burned area boundaries."""

GRID_CRS = "EPSG:5070"
"""NAD83 / Conus Albers, LANDFIRE's native projection: used to plan the download box (the grid
itself takes the CRS of the delivered LANDFIRE raster)."""

MAX_GRID_ROTATION_DEG = 1.0
"""Largest angle allowed between grid north and true north anywhere on the grid, degrees."""

DOWNLOAD_MARGIN_M = 1000.0
"""Extra margin around the grid for downloads, so resampling never runs out of DEM or fuel."""

CROP_SIZE_M = 5000.0
"""Side of the committed test crop (plan WP 2.6)."""

FIXTURES_DIR = Path("tests") / "fixtures"

_FIRE_ID = re.compile(r"^[A-Z]{2}\d{19}$")


# ----------------------------------------------------------------------------- regions


class Region(BaseModel):
    """One study region (``configs/regions.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["A", "B"]
    description: Annotated[str, Field(min_length=1)]
    mtbs_fire_id: str
    mtbs_year: Annotated[int, Field(ge=1984)]
    fire_date: dt.date
    landfire_version: str
    buffer_m: Annotated[float, Field(ge=3000, description="buffer around the perimeter, m; the plan requires 3 km")]
    weather_start_utc: AwareDatetime
    weather_days: Annotated[int, Field(ge=1)]
    source: Annotated[str, Field(min_length=1)]
    note: str = ""

    @field_validator("mtbs_fire_id")
    @classmethod
    def _fire_id(cls, value: str) -> str:
        if not _FIRE_ID.match(value):
            raise ValueError("mtbs_fire_id must look like CA3478712005220240705 (state, 19 digits)")
        return value

    @field_validator("weather_start_utc")
    @classmethod
    def _utc(cls, value: dt.datetime) -> dt.datetime:
        return value.astimezone(dt.UTC)

    @model_validator(mode="after")
    def _check(self) -> Region:
        check_version(self.landfire_version, self.fire_date)  # the fuels must predate the fire
        if self.mtbs_year != self.fire_date.year:
            raise ValueError("mtbs_year must be the year of fire_date")
        return self


class Regions(BaseModel):
    """Schema of ``configs/regions.yaml``."""

    model_config = ConfigDict(extra="forbid")

    regions: dict[str, Region]


def load_regions(path: str | Path | None = None) -> Regions:
    """Load the region table; default ``configs_dir() / "regions.yaml"``."""
    return load_yaml(configs_dir() / "regions.yaml" if path is None else path, Regions)


# ----------------------------------------------------------------------------- MTBS perimeter


@dataclass(frozen=True)
class Perimeter:
    """A fire's MTBS perimeter.

    Attributes:
        geometry: GeoJSON geometry, WGS84 degrees.
        properties: MTBS attributes (``fire_id``, ``fire_name``, ``acres``, ``ig_date``,
            ``latitude`` and ``longitude`` of the point of origin).
        path: The cached GeoJSON file.
        manifest: The cache manifest.
    """

    geometry: dict[str, Any]
    properties: dict[str, Any]
    path: Path
    manifest: dict[str, Any]

    @property
    def origin(self) -> tuple[float, float]:
        """``(lat, lon)`` of the MTBS point of origin."""
        return float(self.properties["latitude"]), float(self.properties["longitude"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_mtbs_perimeter(
    fire_id: str,
    year: int,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
    http: Any = None,
    service: str = MTBS_SERVICE,
) -> Perimeter:
    """The MTBS perimeter of one fire, from the cache or the USFS map service.

    Finds the layer named ``"<year> Burned Area Boundaries"`` and queries it by ``fire_id``
    (GeoJSON, WGS84). Cached as ``<cache_root>/mtbs-<fire_id>/perimeter.geojson`` with a manifest.

    Raises:
        ValueError: A malformed fire id, no such layer, or not exactly one matching feature.
    """
    if not _FIRE_ID.match(fire_id):
        raise ValueError(f"not an MTBS fire id: {fire_id!r}")
    directory = Path(cache_root) / f"mtbs-{fire_id}"
    path, manifest_path = directory / "perimeter.geojson", directory / "manifest.json"
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _sha256(path) != manifest["sha256"]["perimeter.geojson"]:
            raise ValueError(f"{directory}: perimeter.geojson does not match its manifest hash; delete the directory")
        feature = json.loads(path.read_text(encoding="utf-8"))["features"][0]
        return Perimeter(feature["geometry"], feature["properties"], path, manifest)
    client = UrllibHttp() if http is None else http
    layers = client.get_json(f"{service}?f=json").get("layers", [])
    name = f"{year} Burned Area Boundaries"
    ids = [lyr["id"] for lyr in layers if lyr.get("name") == name]
    if len(ids) != 1:
        raise ValueError(f"MTBS service has no layer {name!r}")
    query = urllib.parse.urlencode(
        {
            "where": f"fire_id = '{fire_id}'",
            "outFields": "fire_id,fire_name,acres,ig_date,latitude,longitude,asmnt_type",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
    )
    url = f"{service}/{ids[0]}/query?{query}"
    collection = client.get_json(url)
    features = collection.get("features", [])
    if len(features) != 1:
        raise ValueError(f"MTBS layer {name!r} has {len(features)} features with fire_id {fire_id}")
    feature = {"type": "Feature", "properties": features[0]["properties"], "geometry": features[0]["geometry"]}
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]}, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset": "MTBS Burned Area Boundaries",
        "fire_id": fire_id,
        "layer": name,
        "url": url,
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sha256": {"perimeter.geojson": _sha256(path)},
        "attribution": "Monitoring Trends in Burn Severity (MTBS), USDA Forest Service and USGS (public domain)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return Perimeter(feature["geometry"], feature["properties"], path, manifest)


# ----------------------------------------------------------------------------- geometry


def _coords(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    """Every vertex of a GeoJSON Polygon or MultiPolygon."""
    rings = (
        geometry["coordinates"] if geometry["type"] == "Polygon" else [r for p in geometry["coordinates"] for r in p]
    )
    return [(float(x), float(y)) for ring in rings for x, y, *_ in ring]


def projected_bounds(geometry: dict[str, Any], crs: str = GRID_CRS) -> tuple[float, float, float, float]:
    """``(minx, miny, maxx, maxy)`` of a WGS84 GeoJSON geometry in ``crs``."""
    transform = optional_import("rasterio.warp").transform
    lon, lat = zip(*_coords(geometry), strict=True)
    xs, ys = transform("EPSG:4326", crs, list(lon), list(lat))
    return min(xs), min(ys), max(xs), max(ys)


def download_bbox(geometry: dict[str, Any], buffer_m: float, margin_m: float = DOWNLOAD_MARGIN_M) -> BBox:
    """WGS84 box covering the perimeter's projected bounding box plus ``buffer_m + margin_m``."""
    transform_bounds = optional_import("rasterio.warp").transform_bounds
    x0, y0, x1, y1 = projected_bounds(geometry)
    pad = buffer_m + margin_m
    w, s, e, n = transform_bounds(GRID_CRS, "EPSG:4326", x0 - pad, y0 - pad, x1 + pad, y1 + pad, densify_pts=21)
    return BBox(w, s, e, n)


@dataclass(frozen=True)
class GridWindow:
    """The region's grid as a window of a LANDFIRE raster.

    Attributes:
        target: CRS, transform and shape of the grid.
        row_off, col_off: Offset of the grid's first cell in the raster.
    """

    target: TargetGrid
    row_off: int
    col_off: int


def grid_window(
    geometry: dict[str, Any], buffer_m: float, raster_transform: Any, raster_shape: tuple[int, int], crs: Any
) -> GridWindow:
    """The perimeter's projected bounding box plus ``buffer_m``, snapped outward to the raster's cells.

    One more cell is added on every side, so the cells the perimeter touches, not only its
    vertices, are at least ``buffer_m`` from the edge.

    Raises:
        ValueError: The raster is not north-up or does not contain the whole window.
    """
    a, b, c, d, e, f = (raster_transform[i] for i in range(6))
    if b != 0 or d != 0 or a <= 0 or e >= 0:
        raise ValueError("the LANDFIRE raster must be north-up")
    buffer_m = buffer_m + max(a, -e)
    x0, y0, x1, y1 = projected_bounds(geometry, crs)
    col0 = math.floor((x0 - buffer_m - c) / a + 1e-9)
    col1 = math.ceil((x1 + buffer_m - c) / a - 1e-9)
    row0 = math.floor((f - (y1 + buffer_m)) / -e + 1e-9)
    row1 = math.ceil((f - (y0 - buffer_m)) / -e - 1e-9)
    if row0 < 0 or col0 < 0 or row1 > raster_shape[0] or col1 > raster_shape[1]:
        raise ValueError("the downloaded LANDFIRE raster does not contain the region's grid; enlarge the margin")
    affine = optional_import("affine").Affine
    transform = affine(a, 0.0, c + col0 * a, 0.0, e, f + row0 * e)
    return GridWindow(TargetGrid(crs, transform, (row1 - row0, col1 - col0)), row0, col0)


def grid_rotation_deg(target: TargetGrid) -> float:
    """Largest angle between grid north and true north at the grid's corners and centre, degrees.

    At each point the grid's up direction (a 100 m step up a column) is converted to WGS84 and
    its bearing taken on a local equirectangular approximation, exact enough at 100 m.
    """
    transform = optional_import("rasterio.warp").transform
    h, w = target.shape
    t = target.transform
    worst = 0.0
    for row, col in ((0, 0), (0, w), (h, 0), (h, w), (h / 2, w / 2)):
        x, y = t.c + t.a * col + t.b * row, t.f + t.d * col + t.e * row
        lon, lat = transform(target.crs, "EPSG:4326", [x, x], [y, y + 100.0])
        dlon = (lon[1] - lon[0]) * math.cos(math.radians(lat[0]))
        bearing = math.degrees(math.atan2(dlon, lat[1] - lat[0]))
        worst = max(worst, abs(bearing))
    return worst


def perimeter_mask(geometry: dict[str, Any], target: TargetGrid) -> np.ndarray:
    """Cells the perimeter polygon touches (``all_touched``), on the target grid."""
    rasterize = optional_import("rasterio.features").rasterize
    transform_geom = optional_import("rasterio.warp").transform_geom
    shape = transform_geom("EPSG:4326", target.crs, geometry)
    out = rasterize([(shape, 1)], out_shape=target.shape, transform=target.transform, fill=0, all_touched=True)
    return out.astype(bool)


def edge_distance_cells(mask: np.ndarray) -> int:
    """Smallest number of cells between a True cell and the grid edge (0 for a cell on the edge).

    Raises:
        ValueError: The mask is empty.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("empty mask")
    h, w = mask.shape
    return int(min(rows.min(), cols.min(), h - 1 - rows.max(), w - 1 - cols.max()))


# ----------------------------------------------------------------------------- LANDFIRE layers


def landfire_canopy(
    cc: np.ndarray, ch: np.ndarray, cbh: np.ndarray, cbd: np.ndarray, nodata: float | None = -9999
) -> dict[str, FloatArray]:
    """LANDFIRE canopy layers in ``Grid`` units.

    Canopy cover percent -> fraction in [0, 1]; height and base height metres x 10 -> m; bulk
    density kg/m^3 x 100 -> kg/m^3. ``nodata`` and negative values become 0 (no canopy).
    """

    def clean(x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64)
        bad = v < 0 if nodata is None else (v == nodata) | (v < 0)
        return np.where(bad, 0.0, v)

    return {
        "canopy_cover": np.clip(clean(cc) / 100.0, 0.0, 1.0).astype(np.float32),
        "canopy_height_m": (clean(ch) / 10.0).astype(np.float32),
        "canopy_base_height_m": (clean(cbh) / 10.0).astype(np.float32),
        "canopy_bulk_density_kg_m3": (clean(cbd) / 100.0).astype(np.float32),
    }


def _read_window(layers: LandfireLayers, product: str, window: GridWindow, reference: Any) -> tuple[np.ndarray, Any]:
    rasterio = optional_import("rasterio")
    windows = optional_import("rasterio.windows")
    path, band = layers.files[product]
    with rasterio.open(path) as src:
        if src.transform != reference:
            raise ValueError(f"LANDFIRE {product} is not on the FBFM40 grid")
        h, w = window.target.shape
        win = windows.Window.from_slices((window.row_off, window.row_off + h), (window.col_off, window.col_off + w))
        return src.read(band, window=win), src.nodata


# ----------------------------------------------------------------------------- sources


def _landfire_source(bbox: BBox, region: Region, root: Path) -> LandfireLayers:
    return fetch_landfire(bbox, region.landfire_version, region.fire_date, cache_root=root)


def _dem_source(bbox: BBox, root: Path) -> DemFile:
    return fetch_dem(bbox, cache_root=root)


def _osm_source(bbox: BBox, root: Path) -> OsmFeatures:
    return fetch_osm(bbox, cache_root=root)


def _perimeter_source(region: Region, root: Path) -> Perimeter:
    return fetch_mtbs_perimeter(region.mtbs_fire_id, region.mtbs_year, cache_root=root)


def _weather_source(lat: float, lon: float, start: dt.datetime, duration_s: float, root: Path) -> WeatherConfig:
    return hourly_weather(lat, lon, start, duration_s, cache_root=root)


@dataclass
class RegionSources:
    """Where each input comes from; the defaults download (and cache) the real data. Tests pass fakes."""

    perimeter: Callable[[Region, Path], Perimeter] = _perimeter_source
    landfire: Callable[[BBox, Region, Path], LandfireLayers] = _landfire_source
    dem: Callable[[BBox, Path], DemFile] = _dem_source
    osm: Callable[[BBox, Path], OsmFeatures] = _osm_source
    weather: Callable[[float, float, dt.datetime, float, Path], WeatherConfig] = _weather_source


# ----------------------------------------------------------------------------- build


OUTPUT_FILES = ("grid.npz", "canopy.npz", "weather.yaml", "perimeter.geojson")


@dataclass(frozen=True)
class RegionBuild:
    """A built region.

    Attributes:
        directory: The region directory.
        grid: The grid (also in ``grid.npz``).
        manifest: The contents of ``manifest.json``.
        crop: Path of the 5 km test crop, if written.
    """

    directory: Path
    grid: Grid
    manifest: dict[str, Any]
    crop: Path | None = None


def build_region(
    name: str,
    region: Region,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
    out_dir: str | Path | None = None,
    sources: RegionSources | None = None,
    fixtures_dir: str | Path | None = None,
) -> RegionBuild:
    """Download (or read from the caches) every input of a region and write the region directory.

    Args:
        name: Region name (a key of ``configs/regions.yaml``).
        region: The region.
        cache_root: Root of every download cache.
        out_dir: Region directory; default ``<cache_root>/<name>``.
        sources: Input sources; None downloads the real data.
        fixtures_dir: If given, also write the 5 km crop and its weather there.

    Raises:
        ValueError: Inputs do not line up, or the perimeter comes within the buffer of the edge.
    """
    src = RegionSources() if sources is None else sources
    root = Path(cache_root)
    directory = root / name if out_dir is None else Path(out_dir)

    perimeter = src.perimeter(region, root)
    bbox = download_bbox(perimeter.geometry, region.buffer_m)

    landfire = src.landfire(bbox, region, root)
    rasterio = optional_import("rasterio")
    fb_path, _ = landfire.files["FBFM40"]
    with rasterio.open(fb_path) as fb_src:
        reference, raster_shape, crs = fb_src.transform, (fb_src.height, fb_src.width), fb_src.crs
    window = grid_window(perimeter.geometry, region.buffer_m, reference, raster_shape, crs)
    target = window.target
    dx, dy = target.cell_size
    if not math.isclose(dx, dy):
        raise ValueError(f"LANDFIRE cells must be square, got {dx} x {dy} m")
    rotation = grid_rotation_deg(target)
    if rotation > MAX_GRID_ROTATION_DEG:
        raise ValueError(
            f"grid north is {rotation:.1f} deg from true north (limit {MAX_GRID_ROTATION_DEG} deg): wind and aspect "
            "would be applied rotated; request LANDFIRE in a projection centred on the region"
        )
    layers = {p: _read_window(landfire, p, window, reference) for p in ("FBFM40", "CC", "CH", "CBH", "CBD")}
    fbfm40, fb_nodata = layers["FBFM40"]
    fuel, land = grid_codes(fbfm40, load_urban_model().surrogate_fuel_model, fb_nodata)
    canopy = landfire_canopy(*(layers[p][0] for p in ("CC", "CH", "CBH", "CBD")), nodata=layers["CC"][1])

    dem = src.dem(bbox, root)
    terrain = terrain_for_grid(dem, target)

    osm = src.osm(bbox, root)
    osm_layers = rasterize_features(osm.read(), target)

    grid = Grid.empty(target.shape, dx)
    grid.fuel_model[:] = fuel
    grid.land_use[:] = land
    grid.canopy_cover[:] = canopy["canopy_cover"]
    grid.elevation[:] = terrain.elevation
    grid.slope[:] = terrain.slope
    grid.aspect[:] = terrain.aspect
    usd_per_building = load_costs(configs_dir() / "costs.yaml").usd_per_building.value
    apply_to_grid(grid, osm_layers, usd_per_building)  # also refreshes cell states
    grid.metadata = {
        **terrain.metadata,
        "region": name,
        "mtbs_fire_id": region.mtbs_fire_id,
        "landfire_version": region.landfire_version,
        "weather_start_utc": region.weather_start_utc.isoformat(),
    }
    grid.validate()

    mask = perimeter_mask(perimeter.geometry, target)
    margin = edge_distance_cells(mask)
    if margin * dx < region.buffer_m - 1e-6:
        raise ValueError(f"the perimeter comes within {margin} cells of the grid edge (buffer {region.buffer_m} m)")

    lat, lon = float(terrain.metadata["centroid_lat"]), float(terrain.metadata["centroid_lon"])
    duration_s = region.weather_days * 86400.0
    weather = src.weather(lat, lon, region.weather_start_utc, duration_s, root)

    directory.mkdir(parents=True, exist_ok=True)
    grid.save(directory / "grid.npz")
    np.savez_compressed(
        directory / "canopy.npz",
        canopy_height_m=canopy["canopy_height_m"],
        canopy_base_height_m=canopy["canopy_base_height_m"],
        canopy_bulk_density_kg_m3=canopy["canopy_bulk_density_kg_m3"],
    )
    dump_yaml(weather, directory / "weather.yaml")
    shutil.copyfile(perimeter.path, directory / "perimeter.geojson")

    n_hours = math.ceil(duration_s / 3600)
    first_day, last_day = solar_dates_needed(lon, region.weather_start_utc, n_hours)
    gridmet_dir = root / gridmet_cache_key(lat, lon, first_day, last_day)
    gridmet_manifest = (
        json.loads((gridmet_dir / "manifest.json").read_text(encoding="utf-8"))
        if (gridmet_dir / "manifest.json").exists()
        else None
    )
    manifest: dict[str, Any] = {
        "region": name,
        "role": region.role,
        "description": region.description,
        "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "grid": {
            "crs": str(crs),
            "transform": terrain.metadata["transform"],
            "shape": list(target.shape),
            "dx_m": dx,
            "buffer_m": region.buffer_m,
            "perimeter_edge_cells": margin,
            "max_rotation_from_true_north_deg": round(rotation, 4),
        },
        "fire": {**perimeter.properties, "fire_date": region.fire_date.isoformat()},
        "download_bbox_wgs84": {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north},
        "weather": {
            "point": {"lat": lat, "lon": lon},
            "start_utc": region.weather_start_utc.isoformat(),
            "days": region.weather_days,
        },
        "sources": {
            "mtbs": perimeter.manifest,
            "landfire": landfire.manifest,
            "dem": dem.manifest,
            "osm": osm.manifest,
            "gridmet": gridmet_manifest,
        },
        "sha256": {f: _sha256(directory / f) for f in OUTPUT_FILES},
    }
    crop_path = None
    if fixtures_dir is not None:
        crop_path = write_crop(name, grid, target, perimeter.origin, weather, Path(fixtures_dir))
        manifest["crop"] = {"file": crop_path.name, "sha256": _sha256(crop_path)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return RegionBuild(directory, grid, manifest, crop_path)


def verify_region(name: str, region: Region, *, cache_root: str | Path = DEFAULT_CACHE, **kwargs: Any) -> list[str]:
    """Rebuild a region from the caches into a scratch directory and compare file hashes.

    Returns:
        The files whose hash differs from the region's ``manifest.json`` (empty: reproduced).
    """
    root = Path(cache_root)
    recorded = json.loads((root / name / "manifest.json").read_text(encoding="utf-8"))["sha256"]
    scratch = root / f".verify-{name}"
    try:
        rebuilt = build_region(name, region, cache_root=root, out_dir=scratch, **kwargs).manifest["sha256"]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return sorted(f for f in recorded if recorded[f] != rebuilt.get(f))


# ----------------------------------------------------------------------------- test crop


def crop_window(center_rc: tuple[float, float], shape: tuple[int, int], size: int) -> tuple[int, int]:
    """Top-left ``(row, col)`` of a ``size`` x ``size`` window centred on a cell, kept inside the grid.

    Raises:
        ValueError: The grid is smaller than the window.
    """
    h, w = shape
    if size > h or size > w:
        raise ValueError(f"grid {shape} is smaller than the {size}-cell crop")
    r0 = int(round(center_rc[0] - size / 2))
    c0 = int(round(center_rc[1] - size / 2))
    return min(max(r0, 0), h - size), min(max(c0, 0), w - size)


def crop_grid(grid: Grid, target: TargetGrid, row0: int, col0: int, size: int) -> tuple[Grid, TargetGrid]:
    """A ``size`` x ``size`` piece of a grid, with its own transform and metadata."""
    affine = optional_import("affine").Affine
    t = target.transform
    sub = TargetGrid(target.crs, t @ affine.translation(col0, row0), (size, size))
    sl = (slice(row0, row0 + size), slice(col0, col0 + size))
    layers: dict[str, Any] = {name: getattr(grid, name)[sl].copy() for name in LAYER_SPECS}
    meta = {
        **grid.metadata,
        **grid_metadata(sub),
        "crop_of": grid.metadata.get("region", ""),
        "crop_offset": f"{row0} {col0}",
    }
    return Grid(dx=grid.dx, metadata=meta, **layers), sub


def write_crop(
    name: str,
    grid: Grid,
    target: TargetGrid,
    origin_latlon: tuple[float, float],
    weather: WeatherConfig,
    fixtures_dir: Path,
) -> Path:
    """Write the 5 km x 5 km crop centred on the point of origin, and the region's weather.

    Files: ``<fixtures_dir>/<name>_5km.npz`` and ``<fixtures_dir>/<name>_weather.yaml``.
    """
    transform = optional_import("rasterio.warp").transform
    xs, ys = transform("EPSG:4326", target.crs, [origin_latlon[1]], [origin_latlon[0]])
    inv = ~target.transform
    col = inv.a * xs[0] + inv.b * ys[0] + inv.c
    row = inv.d * xs[0] + inv.e * ys[0] + inv.f
    size = int(round(CROP_SIZE_M / grid.dx))
    r0, c0 = crop_window((row, col), grid.shape, size)
    crop, _ = crop_grid(grid, target, r0, c0, size)
    crop.validate()
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{name}_5km.npz"
    crop.save(path)
    dump_yaml(weather, fixtures_dir / f"{name}_weather.yaml")
    return path


def fuel_summary(fuel_model: IntArray) -> dict[int, int]:
    """Cell count per fuel model, largest first (for the build report)."""
    values, counts = np.unique(fuel_model, return_counts=True)
    order = np.argsort(-counts)
    return {int(values[i]): int(counts[i]) for i in order}


def email_configured() -> bool:
    """Whether the LANDFIRE Product Service email is set (needed for a fresh LANDFIRE download)."""
    from wildfire_sim.data.landfire import EMAIL_ENV

    return bool(os.environ.get(EMAIL_ENV))
