"""Roads, water and buildings from OpenStreetMap (docs/plan.md section 3, WP 2.5).

OpenStreetMap features in a bounding box become three things on the simulation grid:

* **Unburnable cells.** Water mapped as polygons (``natural=water``, ``waterway=riverbank``,
  ``landuse=reservoir``) becomes WATER; motorways and trunk roads, and any road or waterway at
  least ``unburnable_width_m`` wide, become ROAD (or WATER for a waterway). This is the barrier
  rule of section 3: a 30 m cell is wider than most roads and streams, so marking every road
  cell unburnable would create barriers that do not exist.
* **Breaks.** A narrower road or stream leaves the cell's fuel in place and stores the
  feature's width in ``Grid.break_width_m`` (m), which the breach model of WP 2.9 reads: a break
  holds or is crossed depending on the fire's intensity and the width (Wilson 1988). Where
  several cross one cell, the widest counts.
* **Asset value.** ``asset_value = buildings in the cell * usd_per_building`` (``costs.yaml``);
  a building is counted in the cell holding its representative point.

**Widths.** A line feature's width is its ``width`` tag (metres, feet converted), else for a road
its ``lanes`` tag times the lane width, else the default of its class in ``configs/osm.yaml``.
Classes without a default (footpaths, cycleways, steps, proposed roads, ...) are ignored.

**What is left out.** Features in a tunnel or culvert, or on a bridge, are not surface breaks
and are ignored. Intermittent streams and ponds are ignored unless ``include_intermittent`` is
set (they are dry in the fire season). Water polygons fill the cells whose centres they
contain; lines fill every cell they touch and are then made 4-connected
(:func:`close_diagonal_gaps`), so no stencil move, knight moves included, crosses a line
between two of its cells that meet only at a corner.

**Getting the data.** :func:`fetch_osm` queries the Overpass API through ``osmnx`` and caches the
features as GeoJSON with a SHA-256 manifest under ``data/cache/osm-<key>/``. Rasterizing needs
``rasterio`` and ``geopandas``; all three are in the ``geo`` extra (``uv sync --extra geo``).
The classification itself (:func:`classify`) is plain Python.

Data: (c) OpenStreetMap contributors, available under the Open Database Licence (ODbL).

References:
    Wilson, A. A. G. (1988). Width of firebreak that is necessary to stop grass fires: some
    field experiments. Can. J. For. Res. 18, 682-687.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildfire_sim.config import CitedValue, configs_dir, load_yaml
from wildfire_sim.data.landfire import DEFAULT_CACHE, BBox, optional_import
from wildfire_sim.data.terrain import TargetGrid
from wildfire_sim.grid import Grid
from wildfire_sim.types import BoolArray, FloatArray, IntArray, LandUse

OSM_TAGS: dict[str, bool | list[str]] = {
    "highway": True,
    "waterway": True,
    "natural": ["water"],
    "landuse": ["reservoir"],
    "building": True,
}
"""The Overpass query: every feature with one of these tags (``True``: any value)."""

KEEP_COLUMNS: tuple[str, ...] = (
    "highway",
    "waterway",
    "natural",
    "water",
    "landuse",
    "building",
    "width",
    "lanes",
    "tunnel",
    "bridge",
    "intermittent",
    "area",
)
"""The tags kept in the cache (everything :func:`classify` reads)."""

ATTRIBUTION = "(c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)"

FeatureKind = Literal["water", "road_barrier", "water_barrier", "break", "building"]


# ----------------------------------------------------------------------------- parameters


class OsmParams(BaseModel):
    """Schema of ``configs/osm.yaml`` (see that file for the meaning and source of each value)."""

    model_config = ConfigDict(extra="forbid")

    barrier_highways: list[str]
    barrier_highways_source: Annotated[str, Field(min_length=1)]
    unburnable_width_m: CitedValue
    lane_width_m: CitedValue
    highway_width_m: dict[str, CitedValue]
    waterway_width_m: dict[str, CitedValue]
    include_intermittent: bool
    include_intermittent_source: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _check(self) -> OsmParams:
        widths = [self.unburnable_width_m, self.lane_width_m, *self.highway_width_m.values()]
        if any(w.value <= 0 for w in [*widths, *self.waterway_width_m.values()]):
            raise ValueError("widths must be > 0")
        overlap = set(self.barrier_highways) & set(self.highway_width_m)
        if overlap:
            raise ValueError(f"highway classes both barrier and break: {sorted(overlap)}")
        return self


def load_osm_params(path: str | Path | None = None) -> OsmParams:
    """Load the parameters; default ``configs_dir() / "osm.yaml"``."""
    return load_yaml(configs_dir() / "osm.yaml" if path is None else path, OsmParams)


# ----------------------------------------------------------------------------- classification


def _tag(tags: Mapping[str, Any], key: str) -> str | None:
    """A tag's value as a stripped string, None when absent (pandas gives NaN for missing)."""
    v = tags.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def parse_width_m(value: str | None) -> float | None:
    """Metres from an OSM ``width`` value: ``"6"``, ``"6.5 m"``, ``"20 ft"``, ``"20'"``.

    Returns None for anything else (ranges, lists, feet and inches), so the next rule applies.
    """
    if value is None:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(m|meters?|metres?|ft|feet|')?\s*", value.replace(",", "."))
    if m is None:
        return None
    number = float(m.group(1))
    unit = m.group(2)
    metres = number * 0.3048 if unit in ("ft", "feet", "'") else number
    return metres if metres > 0 else None


def _lanes(value: str | None) -> int | None:
    if value is None or not value.isdigit():
        return None
    n = int(value)
    return n if n > 0 else None


@dataclass(frozen=True)
class Feature:
    """What an OSM feature becomes on the grid.

    Attributes:
        kind: ``water`` (polygon), ``road_barrier``, ``water_barrier`` (wide line), ``break``
            (narrow line) or ``building``.
        width_m: Width of a line feature, m; 0 for polygons and buildings.
    """

    kind: FeatureKind
    width_m: float = 0.0


def classify(tags: Mapping[str, Any], geom_type: str, params: OsmParams) -> Feature | None:
    """Classify one OSM feature by its tags and geometry type, or None to ignore it.

    Args:
        tags: OSM tags (missing tags may be absent, None or NaN).
        geom_type: Shapely geometry type (``"Polygon"``, ``"LineString"``, ``"Point"``, ...).
        params: Parameters (``configs/osm.yaml``).
    """
    polygon = geom_type in ("Polygon", "MultiPolygon")
    line = geom_type in ("LineString", "MultiLineString")
    intermittent = _tag(tags, "intermittent") in ("yes", "seasonal")
    if intermittent and not params.include_intermittent:
        return None
    if _tag(tags, "building") not in (None, "no"):
        return Feature("building") if polygon or geom_type in ("Point", "MultiPoint") else None
    if polygon:
        is_water = (
            _tag(tags, "natural") == "water"
            or _tag(tags, "waterway") == "riverbank"
            or _tag(tags, "landuse") == "reservoir"
        )
        return Feature("water") if is_water else None
    if not line:
        return None
    if _tag(tags, "tunnel") not in (None, "no") or _tag(tags, "bridge") not in (None, "no"):
        return None  # not on the ground surface where the fire is
    width = parse_width_m(_tag(tags, "width"))
    highway, waterway = _tag(tags, "highway"), _tag(tags, "waterway")
    if highway is not None:
        if highway in params.barrier_highways:
            return Feature("road_barrier", width or 0.0)
        lanes = _lanes(_tag(tags, "lanes"))
        if width is None and lanes is not None:
            width = lanes * params.lane_width_m.value
        if width is None and highway in params.highway_width_m:
            width = params.highway_width_m[highway].value
        if width is None or highway not in params.highway_width_m:
            return None  # a class the model does not represent (paths, steps, proposed, ...)
        kind: FeatureKind = "road_barrier" if width >= params.unburnable_width_m.value else "break"
        return Feature(kind, width)
    if waterway is not None and waterway in params.waterway_width_m:
        if width is None:
            width = params.waterway_width_m[waterway].value
        return Feature("water_barrier" if width >= params.unburnable_width_m.value else "break", width)
    return None


# ----------------------------------------------------------------------------- download and cache


OsmSource = Callable[[BBox, Mapping[str, bool | list[str]]], Any]
"""Returns the features of a bounding box as a GeoDataFrame in EPSG:4326 (tests pass a fake)."""


OVERPASS_URL = "https://overpass-api.de/api"
"""The main public Overpass API instance (osmnx's default)."""


def osmnx_source(
    bbox: BBox,
    tags: Mapping[str, bool | list[str]],
    *,
    overpass_url: str = OVERPASS_URL,
    attempts: int = 5,
    first_wait_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Query the Overpass API through ``osmnx`` (its own HTTP cache is switched off; we cache).

    The public instance answers "at capacity" (HTTP 502/504) when busy and asks clients to
    retry with exponential backoff, which this does: ``first_wait_s``, then twice that, and so
    on, ``attempts`` tries in all.
    """
    ox = optional_import("osmnx")
    gpd = optional_import("geopandas")
    errors = optional_import("osmnx._errors")
    ox.settings.use_cache = False
    ox.settings.overpass_url = overpass_url
    wait = first_wait_s
    for attempt in range(attempts):
        try:
            return ox.features.features_from_bbox((bbox.west, bbox.south, bbox.east, bbox.north), dict(tags))
        except errors.InsufficientResponseError:  # no feature in the box
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        except errors.ResponseStatusCodeError:
            if attempt == attempts - 1:
                raise
            sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class OsmFeatures:
    """Cached OSM features of one bounding box.

    Attributes:
        path: The GeoJSON file (EPSG:4326; the :data:`KEEP_COLUMNS` tags and the geometry).
        manifest: The contents of ``manifest.json``.
    """

    path: Path
    manifest: dict[str, Any]

    def read(self) -> Any:
        """The features as a GeoDataFrame. Needs ``geopandas``."""
        gpd = optional_import("geopandas")
        return gpd.read_file(self.path)


def osm_cache_key(bbox: BBox) -> str:
    """``osm-`` + 16 hex digits identifying a (bounding box, query) request."""
    canon = json.dumps({"aoi": bbox.aoi(), "tags": OSM_TAGS, "columns": KEEP_COLUMNS}, sort_keys=True)
    return "osm-" + hashlib.sha256(canon.encode()).hexdigest()[:16]


def fetch_osm(bbox: BBox, *, cache_root: str | Path = DEFAULT_CACHE, source: OsmSource | None = None) -> OsmFeatures:
    """The OSM features of ``bbox``, from the cache or Overpass.

    Writes ``features.geojson`` and ``manifest.json`` (query, access time, SHA-256, attribution)
    under ``<cache_root>/osm-<key>/``. OSM changes every day, so a cached region keeps the
    snapshot it was built from; delete the directory to refresh it.

    Raises:
        ValueError: The cached file does not match its manifest hash.
        ImportError: ``geopandas`` (or ``osmnx``, for a download) is not installed.
    """
    directory = Path(cache_root) / osm_cache_key(bbox)
    path, manifest_path = directory / "features.geojson", directory / "manifest.json"
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]["features.geojson"]:
            raise ValueError(f"{directory}: features.geojson does not match its manifest hash; delete the directory")
        return OsmFeatures(path, manifest)
    gpd = optional_import("geopandas")
    gdf = (osmnx_source if source is None else source)(bbox, OSM_TAGS)
    gdf = (gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf.set_crs("EPSG:4326")).reset_index(drop=True)
    n = len(gdf)
    data = {c: gdf[c].astype("string") if c in gdf.columns else [None] * n for c in KEEP_COLUMNS}
    out = gpd.GeoDataFrame(data, geometry=list(gdf.geometry), crs="EPSG:4326")
    directory.mkdir(parents=True, exist_ok=True)
    out.to_file(path, driver="GeoJSON")
    manifest = {
        "dataset": "OpenStreetMap",
        "bbox_wgs84": {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north},
        "query_tags": OSM_TAGS,
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "n_features": len(out),
        "sha256": {"features.geojson": hashlib.sha256(path.read_bytes()).hexdigest()},
        "attribution": ATTRIBUTION,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return OsmFeatures(path, manifest)


# ----------------------------------------------------------------------------- rasterize


@dataclass(frozen=True)
class OsmLayers:
    """OSM features on the target grid.

    Attributes:
        water: Cells covered by water polygons or wide waterways.
        road: Cells of barrier roads (motorways, trunk roads, roads at least the threshold wide).
        break_width_m: Width of the widest break crossing each cell, m; 0 where none. Zero on
            water and road cells.
        buildings: Number of buildings whose representative point lies in the cell.
    """

    water: BoolArray
    road: BoolArray
    break_width_m: FloatArray
    buildings: IntArray


def rasterize_features(gdf: Any, target: TargetGrid, params: OsmParams | None = None) -> OsmLayers:
    """Classify the features of a GeoDataFrame and burn them into the target grid.

    Args:
        gdf: Features with OSM tag columns and a CRS (any; reprojected to the target's).
        target: The LANDFIRE grid (``wildfire_sim.data.terrain.TargetGrid``).
        params: Parameters; None loads ``configs/osm.yaml``.
    """
    rasterize = optional_import("rasterio.features").rasterize
    p = load_osm_params() if params is None else params
    shape = target.shape
    feats = gdf.to_crs(target.crs) if len(gdf) else gdf
    polys: list[Any] = []
    wide_road: list[Any] = []
    wide_water: list[Any] = []
    breaks: list[tuple[Any, float]] = []
    points: list[tuple[float, float]] = []
    columns = [c for c in KEEP_COLUMNS if c in feats.columns]
    for row, geom in zip(feats[columns].to_dict("records"), feats.geometry, strict=True):
        if geom is None or geom.is_empty:
            continue
        f = classify(row, geom.geom_type, p)
        if f is None:
            continue
        if f.kind == "water":
            polys.append(geom)
        elif f.kind == "road_barrier":
            wide_road.append(geom)
        elif f.kind == "water_barrier":
            wide_water.append(geom)
        elif f.kind == "break":
            breaks.append((geom, f.width_m))
        else:
            pt = geom.representative_point() if geom.geom_type != "Point" else geom
            points.append((pt.x, pt.y))

    def burn(shapes: list[Any], lines: bool) -> BoolArray:
        if not shapes:
            return np.zeros(shape, dtype=bool)
        out = rasterize(
            [(g, 1) for g in shapes], out_shape=shape, transform=target.transform, fill=0, all_touched=lines
        )
        return close_diagonal_gaps(out.astype(np.float32)) > 0 if lines else out.astype(bool)

    water = burn(polys, lines=False) | burn(wide_water, lines=True)
    road = burn(wide_road, lines=True) & ~water
    width = np.zeros(shape, dtype=np.float32)
    if breaks:
        # later shapes overwrite earlier ones, so sorting by width leaves the widest in each cell
        ordered = sorted(breaks, key=lambda b: b[1])
        width = rasterize(
            ordered, out_shape=shape, transform=target.transform, fill=0.0, all_touched=True, dtype="float32"
        )
        width = close_diagonal_gaps(width)
    width[water | road] = 0.0
    return OsmLayers(water, road, width.astype(np.float32), count_points(points, target))


def close_diagonal_gaps(values: FloatArray) -> FloatArray:
    """Make a rasterized line 4-connected: where two line cells (value > 0) touch only at a corner,
    fill one of the two cells between them with the larger of their values.

    A line through cell corners rasterizes as a diagonal of single cells, and the 16-neighbour
    stencil's knight moves can step across such a line past one of its cells. Filling the corner
    makes the line a staircase that every stencil move must cross.
    """
    v = np.array(values, dtype=np.float64, copy=True)
    on = v > 0
    # down-right pairs (r, c) - (r + 1, c + 1); fill (r + 1, c)
    gap = on[:-1, :-1] & on[1:, 1:] & ~on[1:, :-1] & ~on[:-1, 1:]
    rr, cc = np.nonzero(gap)
    v[rr + 1, cc] = np.maximum(v[rr, cc], v[rr + 1, cc + 1])
    on = v > 0
    # down-left pairs (r, c + 1) - (r + 1, c); fill (r + 1, c + 1)
    gap = on[:-1, 1:] & on[1:, :-1] & ~on[:-1, :-1] & ~on[1:, 1:]
    rr, cc = np.nonzero(gap)
    v[rr + 1, cc + 1] = np.maximum(v[rr, cc + 1], v[rr + 1, cc])
    return v.astype(np.asarray(values).dtype)


def count_points(points: list[tuple[float, float]], target: TargetGrid) -> IntArray:
    """Number of points (x, y in the target CRS) in each cell; points outside the grid are dropped."""
    counts = np.zeros(target.shape, dtype=np.int32)
    if not points:
        return counts
    xy = np.asarray(points, dtype=np.float64)
    inv = ~target.transform  # world (x, y) -> fractional (col, row)
    cols = inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c
    rows = inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f
    r, c = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    inside = (r >= 0) & (r < target.shape[0]) & (c >= 0) & (c < target.shape[1])
    np.add.at(counts, (r[inside], c[inside]), 1)
    return counts


def apply_to_grid(grid: Grid, layers: OsmLayers, usd_per_building: float) -> None:
    """Write OSM layers into a grid in place.

    Water cells become WATER and road cells ROAD, both without fuel; ``break_width_m`` is set
    on the remaining cells; ``asset_value = buildings * usd_per_building``. Cell states are
    refreshed with :meth:`Grid.apply_unburnable`.

    Raises:
        ValueError: Shapes differ or ``usd_per_building`` is negative.
    """
    if layers.water.shape != grid.shape:
        raise ValueError(f"OSM layers have shape {layers.water.shape}, grid {grid.shape}")
    if usd_per_building < 0:
        raise ValueError("usd_per_building must be >= 0")
    grid.land_use[layers.water] = int(LandUse.WATER)
    grid.land_use[layers.road] = int(LandUse.ROAD)
    gone = layers.water | layers.road
    grid.fuel_model[gone] = 0
    grid.break_width_m[:] = np.where(gone, 0.0, layers.break_width_m).astype(np.float32)
    grid.asset_value[:] = (layers.buildings * usd_per_building).astype(np.float32)
    grid.apply_unburnable()
