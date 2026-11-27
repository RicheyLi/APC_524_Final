"""Tests for wildfire_sim.data.cache and scripts/download_data.py (WP 2.6).

Acceptance (docs/plan.md WP 2.6): the script runs end to end; the cached grid validates; no cell
of the observed perimeter lies within the buffer of the grid edge; fixtures under 2 MB each;
rebuilding from cache reproduces the manifest hashes. The end-to-end tests here use fake sources
(synthetic LANDFIRE rasters, a DEM plane, one road and one building around a 1 km square "fire")
and need the ``geo`` extra; the committed fixtures are checked without it; the real build of
every region is a ``network`` test that also needs the LANDFIRE email.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wildfire_sim.config import WeatherConfig, load_yaml
from wildfire_sim.data.cache import (
    FIXTURES_DIR,
    Perimeter,
    Region,
    crop_window,
    edge_distance_cells,
    fetch_mtbs_perimeter,
    landfire_canopy,
    load_regions,
)
from wildfire_sim.grid import Grid
from wildfire_sim.types import LandUse

REPO = Path(__file__).resolve().parents[1]
UTC = dt.UTC


def _region(**overrides: Any) -> Region:
    raw: dict[str, Any] = {
        "role": "A",
        "description": "test region",
        "mtbs_fire_id": "CA3478712005220240705",
        "mtbs_year": 2024,
        "fire_date": "2024-07-05",
        "landfire_version": "LF2023",
        "buffer_m": 3000,
        "weather_start_utc": "2024-07-05T07:00:00Z",
        "weather_days": 1,
        "source": "test",
    }
    return Region.model_validate({**raw, **overrides})


# ----------------------------------------------------------------------------- regions


def test_region_table() -> None:
    regions = load_regions().regions
    lake = regions["lake_2024"]
    assert lake.role == "A" and lake.landfire_version == "LF2023"
    assert lake.buffer_m >= 3000
    assert lake.weather_start_utc == dt.datetime(2024, 7, 5, 7, tzinfo=UTC)
    assert sum(r.role == "A" for r in regions.values()) == 1


def test_region_validation() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 3000"):
        _region(buffer_m=2000)
    with pytest.raises(ValueError, match="through 2024"):
        _region(landfire_version="LF2024")  # its fuels may show the fire
    with pytest.raises(ValueError, match="mtbs_fire_id"):
        _region(mtbs_fire_id="CA' OR 1=1 --")
    with pytest.raises(ValueError, match="mtbs_year"):
        _region(mtbs_year=2023)
    assert _region(weather_start_utc="2024-07-05T00:00:00-07:00").weather_start_utc.tzinfo == UTC


# ----------------------------------------------------------------------------- pure helpers


def test_landfire_canopy_units() -> None:
    cc = np.array([[0, 45, 95, -9999]])
    ch = np.array([[0, 228, 510, -9999]])
    cbh = np.array([[0, 25, 100, -9999]])
    cbd = np.array([[0, 12, 45, -9999]])
    out = landfire_canopy(cc, ch, cbh, cbd)
    assert np.allclose(out["canopy_cover"], [[0, 0.45, 0.95, 0]])
    assert np.allclose(out["canopy_height_m"], [[0, 22.8, 51.0, 0]])
    assert np.allclose(out["canopy_base_height_m"], [[0, 2.5, 10.0, 0]])
    assert np.allclose(out["canopy_bulk_density_kg_m3"], [[0, 0.12, 0.45, 0]])
    assert all(v.dtype == np.float32 for v in out.values())


def test_edge_distance_and_crop_window() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:6, 4:9] = True
    assert edge_distance_cells(mask) == 3  # rows 3..5 of 10: 3 above, 4 below; cols 4..8 of 12: 4 and 3
    with pytest.raises(ValueError):
        edge_distance_cells(np.zeros((3, 3), dtype=bool))
    assert crop_window((50.0, 50.0), (200, 300), 20) == (40, 40)
    assert crop_window((2.0, 299.0), (200, 300), 20) == (0, 280)  # kept inside
    with pytest.raises(ValueError):
        crop_window((5.0, 5.0), (10, 10), 20)


# ----------------------------------------------------------------------------- MTBS perimeter


class FakeMtbs:
    def __init__(self, n_features: int = 1) -> None:
        self.n = n_features
        self.urls: list[str] = []

    def get_json(self, url: str) -> dict[str, Any]:
        self.urls.append(url)
        if url.endswith("?f=json"):
            return {
                "layers": [
                    {"id": 3, "name": "2023 Burned Area Boundaries"},
                    {"id": 1, "name": "2024 Burned Area Boundaries"},
                ]
            }
        ring = [[-120.0, 34.75], [-119.99, 34.75], [-119.99, 34.76], [-120.0, 34.76], [-120.0, 34.75]]
        feature = {
            "type": "Feature",
            "properties": {
                "fire_id": "CA3478712005220240705",
                "acres": 41216.0,
                "latitude": 34.755,
                "longitude": -119.995,
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }
        return {"type": "FeatureCollection", "features": [feature] * self.n}


def test_fetch_mtbs_perimeter_caches(tmp_path: Path) -> None:
    http = FakeMtbs()
    p = fetch_mtbs_perimeter("CA3478712005220240705", 2024, cache_root=tmp_path, http=http)
    assert "/1/query?" in http.urls[1] and "fire_id+%3D+%27CA3478712005220240705%27" in http.urls[1]
    assert p.origin == (34.755, -119.995) and p.geometry["type"] == "Polygon"
    again = fetch_mtbs_perimeter("CA3478712005220240705", 2024, cache_root=tmp_path, http=http)
    assert len(http.urls) == 2 and again.properties == p.properties
    p.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        fetch_mtbs_perimeter("CA3478712005220240705", 2024, cache_root=tmp_path, http=http)
    with pytest.raises(ValueError, match="no layer"):
        fetch_mtbs_perimeter("CA3478712005220240705", 2019, cache_root=tmp_path / "x", http=FakeMtbs())
    with pytest.raises(ValueError, match="2 features"):
        fetch_mtbs_perimeter("CA3478712005220240705", 2024, cache_root=tmp_path / "y", http=FakeMtbs(2))
    with pytest.raises(ValueError, match="not an MTBS fire id"):
        fetch_mtbs_perimeter("x'; drop", 2024, cache_root=tmp_path, http=http)


# ----------------------------------------------------------------------------- end to end (geo extra)


LOCAL_ALBERS = "+proj=aea +lat_0=34.75 +lon_0=-120.0 +lat_1=34.6 +lat_2=34.9 +datum=NAD83 +units=m +no_defs"
"""The kind of projection the LANDFIRE Product Service delivers: Albers centred on the request."""


def _fake_sources(tmp: Path, landfire_crs: str = LOCAL_ALBERS) -> tuple[Any, dict[str, int]]:
    """Sources around a 1 km square fire at 34.75 N, 120.0 W; counts calls."""
    rasterio = pytest.importorskip("rasterio")
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    from_origin = pytest.importorskip("rasterio.transform").from_origin
    transform_bounds = pytest.importorskip("rasterio.warp").transform_bounds
    from wildfire_sim.data.cache import RegionSources
    from wildfire_sim.data.landfire import LandfireLayers
    from wildfire_sim.data.osm import OsmFeatures
    from wildfire_sim.data.terrain import DemFile

    calls = {"perimeter": 0, "landfire": 0, "dem": 0, "osm": 0, "weather": 0}
    ring = [[-120.0, 34.75], [-119.989, 34.75], [-119.989, 34.759], [-120.0, 34.759], [-120.0, 34.75]]
    geometry = {"type": "Polygon", "coordinates": [ring]}
    props = {"fire_id": "CA3478712005220240705", "latitude": 34.7545, "longitude": -119.9945, "acres": 247.0}

    def perimeter(region: Region, root: Path) -> Perimeter:
        calls["perimeter"] += 1
        path = tmp / "perimeter.geojson"
        feature = {"type": "Feature", "properties": props, "geometry": geometry}
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}, sort_keys=True), "utf-8")
        return Perimeter(geometry, props, path, {"dataset": "fake MTBS"})

    def bounds_in(crs: str, bbox: Any, cell: float) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = transform_bounds("EPSG:4326", crs, bbox.west, bbox.south, bbox.east, bbox.north)
        return (
            np.floor(x0 / cell) * cell,
            np.floor(y0 / cell) * cell,
            np.ceil(x1 / cell) * cell,
            np.ceil(y1 / cell) * cell,
        )

    def landfire(bbox: Any, region: Region, root: Path) -> Any:
        calls["landfire"] += 1
        x0, y0, x1, y1 = bounds_in(landfire_crs, bbox, 30.0)
        w, h = int((x1 - x0) / 30), int((y1 - y0) / 30)
        values = {"FBFM40": 102, "CC": 45, "CH": 228, "CBH": 25, "CBD": 12}
        files = {}
        d = tmp / "landfire"
        d.mkdir(exist_ok=True)
        for product, v in values.items():
            data = np.full((h, w), v, dtype=np.int16)
            if product == "FBFM40":
                data[:, w // 4] = 98  # a lake column on the west side
            if product == "CC":
                data[0, 0] = -9999
            path = d / f"LF2023_{product}.tif"
            profile = dict(
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="int16",
                crs=landfire_crs,
                transform=from_origin(x0, y1, 30.0, 30.0),
                nodata=-9999,
            )
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            files[product] = (path, 1)
        return LandfireLayers(d, files, {"dataset": "fake LANDFIRE", "version": region.landfire_version})

    def dem(bbox: Any, root: Path) -> Any:
        calls["dem"] += 1
        x0, y0, x1, y1 = bounds_in(landfire_crs, bbox, 10.0)  # same frame, so the plane keeps its aspect
        w, h = int((x1 - x0) / 10), int((y1 - y0) / 10)
        x = x0 + 10.0 * (np.arange(w) + 0.5)
        z = np.tile(100.0 + 0.1 * (x - x0), (h, 1)).astype(np.float32)  # rises to the east: slope 0.1, aspect 270
        path = tmp / "dem.tif"
        profile = dict(
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            crs=landfire_crs,
            transform=from_origin(x0, y1, 10.0, 10.0),
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(z, 1)
        return DemFile(path, {"dataset": "fake 3DEP"})

    def osm(bbox: Any, root: Path) -> Any:
        calls["osm"] += 1
        road = shapely.LineString([(bbox.west, 34.7545), (bbox.east, 34.7545)])
        house = shapely.box(-119.9946, 34.7546, -119.9944, 34.7548)
        gdf = gpd.GeoDataFrame(
            {"highway": ["unclassified", None], "width": ["6", None], "building": [None, "house"]},
            geometry=[road, house],
            crs="EPSG:4326",
        )
        path = tmp / "osm.geojson"
        gdf.to_file(path, driver="GeoJSON")
        return OsmFeatures(path, {"dataset": "fake OSM"})

    def weather(
        lat: float, lon: float, start: dt.datetime, duration_s: float, root: Path, source: str = "gridmet"
    ) -> WeatherConfig:
        calls["weather"] += 1
        assert 34.7 < lat < 34.8 and -120.1 < lon < -119.9 and duration_s == 86400.0
        return WeatherConfig()

    return RegionSources(perimeter, landfire, dem, osm, weather), calls


def test_build_region_end_to_end(tmp_path: Path) -> None:
    """Acceptance: the build runs, the grid validates, the perimeter keeps the buffer, crop < 2 MB."""
    sources, calls = _fake_sources(tmp_path)
    from wildfire_sim.data.cache import build_region, perimeter_mask, verify_region

    region = _region()
    built = build_region("t", region, cache_root=tmp_path / "cache", sources=sources, fixtures_dir=tmp_path / "fx")
    d = built.directory
    assert d == tmp_path / "cache" / "t"
    grid = Grid.load(d / "grid.npz")  # validates
    m = built.manifest
    # about 1 km of fire plus 3 km and one cell on every side (a little more: the box is rotated in Albers)
    assert 230 <= grid.shape[0] <= 250 and 230 <= grid.shape[1] <= 250
    assert m["grid"]["perimeter_edge_cells"] * grid.dx >= region.buffer_m
    from wildfire_sim.data.terrain import TargetGrid

    affine = pytest.importorskip("affine").Affine
    t = [float(v) for v in grid.metadata["transform"].split()]
    target = TargetGrid(grid.metadata["crs"], affine(*t), grid.shape)
    assert m["grid"]["max_rotation_from_true_north_deg"] < 0.1
    live = m["weather"]["live_moisture_samples"]  # which measured live moisture samples the build used
    assert live["file"] == "configs/live_fuel_moisture.yaml" and len(live["sha256"]) == 64
    mask = perimeter_mask(built_perimeter(tmp_path), target)
    assert mask.sum() > 1000 and edge_distance_cells(mask) * grid.dx >= 3000
    # fuels, canopy, terrain, roads, assets
    assert (grid.land_use == LandUse.WATER).any() and (grid.fuel_model == 102).sum() > 0.9 * grid.n_cells
    assert np.allclose(grid.canopy_cover, 0.45)  # percent -> fraction (no-data handling: test above)
    canopy = np.load(d / "canopy.npz")
    assert np.isclose(canopy["canopy_height_m"][5, 5], 22.8) and np.isclose(
        canopy["canopy_bulk_density_kg_m3"][5, 5], 0.12
    )
    assert np.allclose(grid.slope[5:-5, 5:-5], 0.1, atol=1e-3) and np.allclose(grid.aspect[5:-5, 5:-5], 270.0, atol=0.5)
    assert (grid.break_width_m == 6.0).sum() >= grid.shape[1] - 2
    assert grid.asset_value.sum() == 400_000.0
    assert grid.metadata["region"] == "t" and grid.metadata["landfire_version"] == "LF2023"
    assert isinstance(load_yaml(d / "weather.yaml", WeatherConfig), WeatherConfig)
    assert set(m["sha256"]) == {"grid.npz", "canopy.npz", "weather.yaml", "perimeter.geojson"}
    assert m["sources"]["landfire"]["version"] == "LF2023"
    # the crop: 5 km, centred on the point of origin, under 2 MB
    assert built.crop is not None
    crop = Grid.load(built.crop)
    assert crop.shape == (167, 167) and built.crop.stat().st_size < 2_000_000
    assert (tmp_path / "fx" / "t_weather.yaml").exists()
    assert crop.metadata["crop_of"] == "t" and crop.metadata["transform"] != grid.metadata["transform"]
    assert crop.asset_value.sum() == 400_000.0  # the house is at the origin
    # rebuilding reproduces every file byte for byte
    assert verify_region("t", region, cache_root=tmp_path / "cache", sources=sources) == []
    assert calls["landfire"] == 2


def built_perimeter(tmp: Path) -> dict[str, Any]:
    return json.loads((tmp / "perimeter.geojson").read_text(encoding="utf-8"))["features"][0]["geometry"]


def test_grid_north_must_be_true_north(tmp_path: Path) -> None:
    """EPSG:5070 is rotated about 14.5 deg from true north at 120 W; such a grid is refused."""
    from_origin = pytest.importorskip("rasterio.transform").from_origin
    transform = pytest.importorskip("rasterio.warp").transform
    from wildfire_sim.data.cache import build_region, grid_rotation_deg
    from wildfire_sim.data.terrain import TargetGrid

    xs, ys = transform("EPSG:4326", "EPSG:5070", [-120.0], [34.75])
    conus = TargetGrid("EPSG:5070", from_origin(xs[0], ys[0], 30.0, 30.0), (100, 100))
    assert 14.0 < grid_rotation_deg(conus) < 15.0  # n (lon - lon0) with n about 0.6 and 24 deg
    local = TargetGrid(LOCAL_ALBERS, from_origin(-1500.0, 1500.0, 30.0, 30.0), (100, 100))
    assert grid_rotation_deg(local) < 0.05
    sources, _ = _fake_sources(tmp_path, landfire_crs="EPSG:5070")
    with pytest.raises(ValueError, match="from true north"):
        build_region("t", _region(), cache_root=tmp_path / "cache", sources=sources)


def test_verify_reports_a_changed_file(tmp_path: Path) -> None:
    sources, _ = _fake_sources(tmp_path)
    from wildfire_sim.data.cache import build_region, verify_region

    region = _region()
    build_region("t", region, cache_root=tmp_path / "cache", sources=sources)
    manifest_path = tmp_path / "cache" / "t" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"]["grid.npz"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_region("t", region, cache_root=tmp_path / "cache", sources=sources) == ["grid.npz"]


def test_script_rejects_unknown_region(capsys: pytest.CaptureFixture[str]) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("download_data", REPO / "scripts" / "download_data.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.main(["no_such_fire"])
    assert "unknown" in capsys.readouterr().err


# ----------------------------------------------------------------------------- committed fixtures


@pytest.mark.parametrize("name", sorted(load_regions().regions))
def test_committed_fixture(name: str) -> None:
    """Acceptance: each region's 5 km crop is committed, under 2 MB, and a valid grid with fuel."""
    path = REPO / FIXTURES_DIR / f"{name}_5km.npz"
    assert path.exists(), f"run: uv run python scripts/download_data.py {name}"
    assert path.stat().st_size < 2_000_000
    grid = Grid.load(path)
    assert grid.shape == (167, 167) and grid.dx == 30.0
    assert grid.burnable_mask().sum() > 0.5 * grid.n_cells
    assert grid.metadata["crop_of"] == name
    weather = load_yaml(REPO / FIXTURES_DIR / f"{name}_weather.yaml", WeatherConfig)
    region = load_regions().regions[name]
    assert len(weather.series) == 24 * region.weather_days


# ----------------------------------------------------------------------------- live


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(load_regions().regions))
def test_live_region_build(name: str, tmp_path: Path) -> None:
    """The real build of every region, from nothing, then rebuilt from its caches."""
    pytest.importorskip("osmnx")
    pytest.importorskip("py3dep")
    from wildfire_sim.data.cache import build_region, verify_region
    from wildfire_sim.data.landfire import EMAIL_ENV

    if not os.environ.get(EMAIL_ENV):
        pytest.skip(f"set {EMAIL_ENV} to run the live region build")
    region = load_regions().regions[name]
    built = build_region(name, region, cache_root=tmp_path, fixtures_dir=tmp_path / "fixtures")
    assert built.manifest["grid"]["perimeter_edge_cells"] * built.grid.dx >= region.buffer_m
    assert built.crop is not None and built.crop.stat().st_size < 2_000_000
    assert verify_region(name, region, cache_root=tmp_path) == []
