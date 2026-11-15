"""Tests for wildfire_sim.data.osm and the optional ``Grid.break_width_m`` layer (WP 2.5).

Acceptance (docs/plan.md WP 2.5): a known polygon rasterizes to the expected cells; a 6 m road
creates no unburnable cell and a line of cells with ``break_width_m = 6``. The classification
tests run everywhere; the rasterizing tests need the ``geo`` extra (skipped without it); one live
Overpass download is marked ``network``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wildfire_sim.data.landfire import BBox
from wildfire_sim.data.osm import (
    Feature,
    OsmLayers,
    OsmParams,
    apply_to_grid,
    classify,
    load_osm_params,
    parse_width_m,
)
from wildfire_sim.data.terrain import TargetGrid
from wildfire_sim.grid import Grid
from wildfire_sim.types import CellState, LandUse

CRS = "EPSG:5070"


@pytest.fixture(scope="module")
def params() -> OsmParams:
    return load_osm_params()


# ----------------------------------------------------------------------------- Grid.break_width_m


def test_break_width_layer_is_optional_and_round_trips(tmp_path: Path) -> None:
    grid = Grid.empty((4, 5), 30.0)
    assert grid.break_width_m.shape == (4, 5) and grid.break_width_m.dtype == np.float32
    assert not grid.break_width_m.any()
    grid.break_width_m[2, :] = 6.0
    grid.save(tmp_path / "g.npz")
    assert np.array_equal(Grid.load(tmp_path / "g.npz").break_width_m, grid.break_width_m)
    assert grid.copy() == grid


def test_grid_saved_before_the_layer_still_loads(tmp_path: Path) -> None:
    grid = Grid.empty((3, 3), 30.0)
    grid.save(tmp_path / "g.npz")
    with np.load(tmp_path / "g.npz") as data:
        old = {k: data[k] for k in data.files if k != "break_width_m"}
    np.savez_compressed(tmp_path / "old.npz", **old)
    loaded = Grid.load(tmp_path / "old.npz")
    assert loaded.break_width_m.shape == (3, 3) and not loaded.break_width_m.any()


def test_negative_break_width_is_invalid() -> None:
    grid = Grid.empty((2, 2), 30.0)
    grid.break_width_m[0, 0] = -1.0
    with pytest.raises(ValueError, match="break_width_m"):
        grid.validate()


# ----------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("value", "metres"),
    [("6", 6.0), ("6.5 m", 6.5), ("7,5", 7.5), ("20 ft", 6.096), ("20'", 6.096), ("12 metres", 12.0)],
)
def test_parse_width(value: str, metres: float) -> None:
    assert parse_width_m(value) == pytest.approx(metres)


@pytest.mark.parametrize("value", [None, "", "4-6", "6;8", "narrow", "0", "12'6\""])
def test_parse_width_gives_up(value: str | None) -> None:
    assert parse_width_m(value) is None


def test_roads(params: OsmParams) -> None:
    line = "LineString"
    assert classify({"highway": "motorway"}, line, params) == Feature("road_barrier")
    assert classify({"highway": "trunk", "width": "11"}, line, params) == Feature("road_barrier", 11.0)
    assert classify({"highway": "unclassified", "width": "6"}, line, params) == Feature("break", 6.0)
    assert classify({"highway": "residential"}, line, params) == Feature("break", 8.0)
    four = classify({"highway": "secondary", "lanes": "4"}, line, params)  # 4 x 3.6 m
    assert four is not None and four.kind == "break" and four.width_m == pytest.approx(14.4)
    five = classify({"highway": "primary", "lanes": "5"}, line, params)  # 18 m: over the 15 m threshold
    assert five is not None and five.kind == "road_barrier" and five.width_m == pytest.approx(18.0)
    assert classify({"highway": "primary", "width": "16"}, line, params) == Feature("road_barrier", 16.0)
    assert classify({"highway": "track", "width": "8", "lanes": "1"}, line, params) == Feature("break", 8.0)
    assert classify({"highway": "path"}, line, params) is None
    assert classify({"highway": "footway", "width": "3"}, line, params) is None
    assert classify({"highway": "secondary", "bridge": "yes"}, line, params) is None
    assert classify({"highway": "service", "tunnel": "yes"}, line, params) is None
    assert classify({"highway": "service", "bridge": "no"}, line, params) == Feature("break", 4.0)
    assert classify({"highway": "pedestrian", "area": "yes"}, "Polygon", params) is None


def test_water(params: OsmParams) -> None:
    line, poly = "LineString", "Polygon"
    assert classify({"waterway": "stream"}, line, params) == Feature("break", 2.0)
    assert classify({"waterway": "river"}, line, params) == Feature("break", 10.0)
    assert classify({"waterway": "river", "width": "40"}, line, params) == Feature("water_barrier", 40.0)
    assert classify({"waterway": "stream", "tunnel": "culvert"}, line, params) is None
    assert classify({"waterway": "stream", "intermittent": "yes"}, line, params) is None
    assert classify({"waterway": "dam"}, line, params) is None
    assert classify({"natural": "water"}, poly, params) == Feature("water")
    assert classify({"natural": "water", "water": "lake"}, "MultiPolygon", params) == Feature("water")
    assert classify({"waterway": "riverbank"}, poly, params) == Feature("water")
    assert classify({"landuse": "reservoir"}, poly, params) == Feature("water")
    assert classify({"natural": "water", "intermittent": "yes"}, poly, params) is None
    assert classify({"natural": "wetland"}, poly, params) is None
    wet = params.model_copy(update={"include_intermittent": True})
    assert classify({"waterway": "stream", "intermittent": "yes"}, line, wet) == Feature("break", 2.0)


def test_buildings_and_missing_tags(params: OsmParams) -> None:
    assert classify({"building": "house"}, "Polygon", params) == Feature("building")
    assert classify({"building": "yes"}, "Point", params) == Feature("building")
    assert classify({"building": "no", "natural": "water"}, "Polygon", params) == Feature("water")
    assert classify({"building": "yes"}, "LineString", params) is None
    nan = {k: math.nan for k in ("highway", "waterway", "building", "width", "tunnel", "bridge")}
    assert classify({**nan, "highway": "track"}, "LineString", params) == Feature("break", 3.0)
    assert classify(nan, "LineString", params) is None


def test_params_validation(params: OsmParams) -> None:
    raw = params.model_dump()
    raw["barrier_highways"] = ["motorway", "primary"]
    with pytest.raises(ValueError, match="both barrier and break"):
        OsmParams.model_validate(raw)
    raw = params.model_dump()
    raw["lane_width_m"]["value"] = 0.0
    with pytest.raises(ValueError, match="> 0"):
        OsmParams.model_validate(raw)


# ----------------------------------------------------------------------------- apply to a grid


def test_apply_to_grid() -> None:
    grid = Grid.empty((3, 4), 30.0)
    grid.fuel_model[:] = 102
    grid.apply_unburnable()
    water = np.zeros((3, 4), dtype=bool)
    water[0, 0] = True
    road = np.zeros((3, 4), dtype=bool)
    road[2, :] = True
    width = np.zeros((3, 4), dtype=np.float32)
    width[:, 1] = 6.0  # a stream crossing the water and the road too
    buildings = np.zeros((3, 4), dtype=np.int32)
    buildings[1, 3] = 2
    apply_to_grid(grid, OsmLayers(water, road, width, buildings), usd_per_building=400_000.0)
    assert grid.land_use[0, 0] == LandUse.WATER and grid.fuel_model[0, 0] == 0
    assert (grid.land_use[2] == LandUse.ROAD).all() and not grid.fuel_model[2].any()
    assert (grid.state[2] == CellState.UNBURNABLE).all() and grid.state[0, 0] == CellState.UNBURNABLE
    assert grid.break_width_m[1, 1] == 6.0 and grid.break_width_m[0, 1] == 6.0
    assert grid.break_width_m[2, 1] == 0.0  # not on a road cell
    assert grid.state[1, 1] == CellState.FUEL and grid.fuel_model[1, 1] == 102  # a break keeps its fuel
    assert grid.asset_value[1, 3] == 800_000.0 and grid.asset_value.sum() == 800_000.0
    grid.validate()
    with pytest.raises(ValueError):
        apply_to_grid(Grid.empty((2, 2), 30.0), OsmLayers(water, road, width, buildings), 1.0)
    with pytest.raises(ValueError):
        apply_to_grid(grid, OsmLayers(water, road, width, buildings), -1.0)


# ----------------------------------------------------------------------------- rasterizing (geo extra)


def _target(n: int = 10) -> TargetGrid:
    """n x n cells of 30 m, top-left corner at (0, 300) in EPSG:5070."""
    from_origin = pytest.importorskip("rasterio.transform").from_origin
    return TargetGrid(CRS, from_origin(0.0, 30.0 * n, 30.0, 30.0), (n, n))


def _gdf(rows: list[tuple[dict[str, Any], Any]], crs: str = CRS) -> Any:
    gpd = pytest.importorskip("geopandas")
    tags = [t for t, _ in rows]
    return gpd.GeoDataFrame(tags, geometry=[g for _, g in rows], crs=crs)


def _row_line(row: int, n: int = 10) -> Any:
    shapely = pytest.importorskip("shapely")
    y = 30.0 * n - 30.0 * row - 15.0  # through the centres of the row's cells
    return shapely.LineString([(0.0, y), (30.0 * n, y)])


def test_known_polygon_rasterizes_to_expected_cells(params: OsmParams) -> None:
    """Acceptance: a lake over x 90-210 m, y 150-240 m covers the centres of rows 2-4, columns 3-6."""
    pytest.importorskip("rasterio")
    shapely = pytest.importorskip("shapely")
    from wildfire_sim.data.osm import rasterize_features

    lake = shapely.box(90.0, 150.0, 210.0, 240.0)
    layers = rasterize_features(_gdf([({"natural": "water"}, lake)]), _target(), params)
    expected = np.zeros((10, 10), dtype=bool)
    expected[2:5, 3:7] = True
    assert np.array_equal(layers.water, expected)
    assert not layers.road.any() and not layers.break_width_m.any() and not layers.buildings.any()


def test_six_metre_road_is_a_break_not_a_barrier(params: OsmParams) -> None:
    """Acceptance: no unburnable cell, and a line of cells with break_width_m = 6."""
    pytest.importorskip("rasterio")
    from wildfire_sim.data.osm import rasterize_features

    layers = rasterize_features(_gdf([({"highway": "unclassified", "width": "6"}, _row_line(5))]), _target(), params)
    assert not layers.water.any() and not layers.road.any()
    expected = np.zeros((10, 10), dtype=np.float32)
    expected[5, :] = 6.0
    assert np.array_equal(layers.break_width_m, expected)
    grid = Grid.empty((10, 10), 30.0)
    grid.fuel_model[:] = 102
    apply_to_grid(grid, layers, 400_000.0)
    assert grid.burnable_mask().all()
    assert (grid.break_width_m[5] == 6.0).all() and grid.break_width_m.sum() == 60.0


def test_motorway_is_a_barrier(params: OsmParams) -> None:
    pytest.importorskip("rasterio")
    from wildfire_sim.data.osm import rasterize_features

    layers = rasterize_features(_gdf([({"highway": "motorway"}, _row_line(5))]), _target(), params)
    assert layers.road[5].all() and layers.road.sum() == 10
    assert not layers.break_width_m.any()


def test_widest_break_wins_and_breaks_stop_at_water(params: OsmParams) -> None:
    pytest.importorskip("rasterio")
    shapely = pytest.importorskip("shapely")
    from wildfire_sim.data.osm import rasterize_features

    column = shapely.LineString([(165.0, 0.0), (165.0, 300.0)])  # column 5
    rows = [
        ({"highway": "residential"}, _row_line(5)),  # 8 m
        ({"waterway": "stream"}, column),  # 2 m
        ({"natural": "water"}, shapely.box(150.0, 0.0, 180.0, 60.0)),  # rows 8-9 of column 5
    ]
    layers = rasterize_features(_gdf(rows), _target(), params)
    w = layers.break_width_m
    assert w[5, 5] == 8.0 and w[5, 0] == 8.0 and w[0, 5] == 2.0
    assert layers.water[8:, 5].all() and w[8, 5] == 0.0 and w[9, 5] == 0.0


def _reachable(barrier: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Cells fire can reach from ``start`` with the engine's 16-neighbour moves and blocking rule."""
    from wildfire_sim.sim import _blocked_links
    from wildfire_sim.types import STENCIL_16

    blocked = _blocked_links(barrier, STENCIL_16)
    h, w = barrier.shape
    seen = np.zeros_like(barrier)
    seen[start] = True
    todo = [start]
    while todo:
        r, c = todo.pop()
        for k, (dr, dc) in enumerate(STENCIL_16.offsets.tolist()):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not barrier[rr, cc] and not seen[rr, cc] and not blocked[k, r, c]:
                seen[rr, cc] = True
                todo.append((rr, cc))
    return seen


@pytest.mark.parametrize("end", [(300.0, 0.0), (300.0, 110.0), (130.0, 0.0)])
def test_barrier_line_is_not_crossed_by_any_stencil_move(params: OsmParams, end: tuple[float, float]) -> None:
    """A line through cell corners (or at any angle) stops every 8- and 16-neighbour move."""
    pytest.importorskip("rasterio")
    shapely = pytest.importorskip("shapely")
    from wildfire_sim.data.osm import close_diagonal_gaps, rasterize_features

    line = shapely.LineString([(0.0, 300.0), end])
    layers = rasterize_features(_gdf([({"highway": "motorway"}, line)]), _target(), params)
    fire = _reachable(layers.road, (0, 9))
    assert not fire[9, 0]
    assert fire[~layers.road].sum() < (~layers.road).sum()
    # without closing the corners, the corner-to-corner diagonal leaks through a knight move
    diagonal = np.eye(10, dtype=bool)  # cells (k, k): corner to corner
    assert _reachable(diagonal, (0, 9))[9, 0]
    assert not _reachable(close_diagonal_gaps(diagonal.astype(np.float32)) > 0, (0, 9))[9, 0]


def test_buildings_and_reprojection(params: OsmParams) -> None:
    """Features in WGS84 land in the same cells as the same features given in the grid's CRS."""
    pytest.importorskip("rasterio")
    shapely = pytest.importorskip("shapely")
    from wildfire_sim.data.osm import rasterize_features

    rows = [
        ({"building": "house"}, shapely.box(40.0, 250.0, 50.0, 260.0)),  # row 1, column 1
        ({"building": "yes"}, shapely.Point(45.0, 255.0)),
        ({"building": "shed"}, shapely.box(275.0, 5.0, 285.0, 15.0)),  # row 9, column 9
        ({"building": "yes"}, shapely.Point(-500.0, 255.0)),  # outside the grid
        ({"natural": "water"}, shapely.box(90.0, 150.0, 210.0, 240.0)),
    ]
    direct = rasterize_features(_gdf(rows), _target(), params)
    assert direct.buildings[1, 1] == 2 and direct.buildings[9, 9] == 1 and direct.buildings.sum() == 3
    wgs84 = _gdf(rows).to_crs("EPSG:4326")
    again = rasterize_features(wgs84, _target(), params)
    assert np.array_equal(again.buildings, direct.buildings)
    assert np.array_equal(again.water, direct.water)


def test_empty_features(params: OsmParams) -> None:
    pytest.importorskip("rasterio")
    from wildfire_sim.data.osm import rasterize_features

    layers = rasterize_features(_gdf([]), _target(), params)
    assert not (layers.water.any() or layers.road.any() or layers.break_width_m.any() or layers.buildings.any())


# ----------------------------------------------------------------------------- cache


def test_fetch_osm_caches_with_a_manifest(tmp_path: Path) -> None:
    gpd = pytest.importorskip("geopandas")
    pd = pytest.importorskip("pandas")
    shapely = pytest.importorskip("shapely")
    from wildfire_sim.data.osm import ATTRIBUTION, OSM_TAGS, fetch_osm

    calls: list[Any] = []

    def source(bbox: BBox, tags: Any) -> Any:
        calls.append((bbox, tags))
        index = pd.MultiIndex.from_tuples([("way", 1), ("way", 2), ("node", 3)], names=["element", "id"])
        return gpd.GeoDataFrame(
            {
                "highway": ["track", None, None],
                "natural": [None, "water", None],
                "building": [None, None, "yes"],
                "nodes": [[1, 2], [3, 4, 5, 3], None],  # a list column, as osmnx returns
            },
            geometry=[
                shapely.LineString([(-120.9, 35.7), (-120.8, 35.7)]),
                shapely.box(-120.9, 35.6, -120.8, 35.65),
                shapely.Point(-120.85, 35.68),
            ],
            index=index,
            crs="EPSG:4326",
        )

    bbox = BBox(-121.0, 35.5, -120.7, 35.8)
    got = fetch_osm(bbox, cache_root=tmp_path, source=source)
    assert calls == [(bbox, OSM_TAGS)]
    assert got.manifest["attribution"] == ATTRIBUTION and got.manifest["n_features"] == 3
    gdf = got.read()
    assert list(gdf["highway"])[0] == "track" and len(gdf) == 3
    assert "nodes" not in gdf.columns
    again = fetch_osm(bbox, cache_root=tmp_path, source=source)
    assert len(calls) == 1 and again.manifest == got.manifest
    manifest = json.loads((got.path.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bbox_wgs84"]["west"] == -121.0
    got.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        fetch_osm(bbox, cache_root=tmp_path, source=source)


def test_overpass_retries_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    ox = pytest.importorskip("osmnx")
    errors = pytest.importorskip("osmnx._errors")
    from wildfire_sim.data.osm import osmnx_source

    tries: list[int] = []

    def flaky(bbox: Any, tags: Any) -> str:
        tries.append(1)
        if len(tries) < 3:
            raise errors.ResponseStatusCodeError("502 at capacity")
        return "features"

    waits: list[float] = []
    monkeypatch.setattr(ox.features, "features_from_bbox", flaky)
    bbox = BBox(-121.0, 35.5, -120.7, 35.8)
    assert osmnx_source(bbox, {}, sleep=waits.append) == "features"
    assert waits == [15.0, 30.0]
    tries.clear()
    with pytest.raises(errors.ResponseStatusCodeError):
        osmnx_source(bbox, {}, attempts=2, sleep=waits.append)


# ----------------------------------------------------------------------------- live service


@pytest.mark.network
def test_live_overpass_lake_nacimiento(tmp_path: Path, params: OsmParams) -> None:
    """A 2 km box at Lake Nacimiento (next to the Chimney Fire): the lake, roads, buildings."""
    pytest.importorskip("osmnx")
    pytest.importorskip("rasterio")
    transformer = pytest.importorskip("pyproj").Transformer.from_crs(4326, 5070, always_xy=True)
    from_origin = pytest.importorskip("rasterio.transform").from_origin
    from wildfire_sim.data.osm import fetch_osm, rasterize_features

    bbox = BBox(-120.90, 35.75, -120.87, 35.77)
    features = fetch_osm(bbox, cache_root=tmp_path)
    x0, y1 = transformer.transform(bbox.west, bbox.north)
    x1, y0 = transformer.transform(bbox.east, bbox.south)
    shape = (int((y1 - y0) // 30) + 1, int((x1 - x0) // 30) + 1)
    layers = rasterize_features(features.read(), TargetGrid(CRS, from_origin(x0, y1, 30.0, 30.0), shape), params)
    assert layers.water.sum() > 100  # the lake
    assert (layers.break_width_m > 0).sum() > 50  # roads
    assert layers.buildings.sum() > 0
