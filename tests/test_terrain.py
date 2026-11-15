"""Tests for terrain (WP 2.3, ``wildfire_sim.data.terrain``).

Slope and aspect run everywhere (NumPy only). The DEM cache uses a fake downloader; resampling
onto a grid and the grid metadata need ``rasterio`` (skipped without it). The live 3DEP test is
marked ``network``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from wildfire_sim.data.landfire import BBox
from wildfire_sim.data.synthetic import slope_ramp
from wildfire_sim.data.terrain import (
    TargetGrid,
    dem_cache_key,
    fetch_dem,
    slope_angle_deg,
    slope_aspect,
)

BBOX = BBox(-121.10, 35.55, -121.05, 35.60)


def angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest absolute difference between bearings, degrees."""
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


# ----------------------------------------------------------------------------- slope and aspect


@pytest.mark.parametrize("aspect", [0.0, 30.0, 90.0, 117.3, 180.0, 245.0, 270.0, 359.6])
@pytest.mark.parametrize("slope", [0.05, 0.37, 1.2])
def test_recovers_slope_ramp(slope: float, aspect: float) -> None:
    """WP 2.3 acceptance: slope and aspect of ``synthetic.slope_ramp``'s elevation recovered to
    1e-3 and 0.5 deg on every cell, edges included (measured: 2e-6 and 2e-4 deg; the residue is the
    float32 elevation layer)."""
    grid = slope_ramp((31, 47), 30.0, slope=slope, aspect_deg=aspect)
    s, a = slope_aspect(grid.elevation, grid.dx)
    np.testing.assert_allclose(s, slope, atol=1e-3)
    assert angle_diff(a, aspect).max() <= 0.5
    # and agrees with the layers the generator wrote
    np.testing.assert_allclose(s, grid.slope, atol=1e-3)
    assert angle_diff(a - grid.aspect, 0.0).max() <= 0.5


def test_flat_ground_has_no_slope_and_aspect_zero() -> None:
    s, a = slope_aspect(np.full((5, 6), 312.5), 30.0)
    assert np.all(s == 0.0) and np.all(a == 0.0)


def test_aspect_points_downhill_on_a_hill() -> None:
    """A cone peaked in the centre: every cell's aspect points away from the peak."""
    n, dx = 41, 30.0
    rows, cols = np.mgrid[0:n, 0:n]
    east, north = (cols - 20) * dx, (20 - rows) * dx
    z = 1000.0 - 0.3 * np.hypot(east, north)
    s, a = slope_aspect(z, dx)
    away = np.degrees(np.arctan2(east, north)) % 360.0
    ring = (np.hypot(east, north) > 3 * dx) & (rows > 0) & (rows < n - 1) & (cols > 0) & (cols < n - 1)
    assert angle_diff(a[ring] - away[ring], 0.0).max() < 5.0
    np.testing.assert_allclose(s[ring], 0.3, atol=0.02)


def test_rectangular_cells() -> None:
    """dz/dx and dz/dy use their own cell sizes: a plane rising 0.2 north over 10 m x 40 m cells."""
    rows = np.mgrid[0:6, 0:7][0]
    z = 0.2 * (5 - rows) * 40.0  # rises northward; rows are 40 m tall
    s, a = slope_aspect(z, dx=10.0, dy=40.0)
    np.testing.assert_allclose(s, 0.2)
    assert angle_diff(a, 180.0).max() < 1e-9  # downhill is south


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        slope_aspect(np.array([[1.0, np.nan], [1.0, 1.0]]), 30.0)
    with pytest.raises(ValueError, match="2 x 2"):
        slope_aspect(np.ones((1, 5)), 30.0)
    with pytest.raises(ValueError, match="> 0"):
        slope_aspect(np.ones((3, 3)), 0.0)


def test_aspect_just_below_360_stays_below_360_in_float32() -> None:
    """Found building the Lake Fire grid (WP 2.6): a bearing of 360 - 6e-6 deg cast to float32 is 360.0."""
    from wildfire_sim.data.terrain import aspect_float32

    # downhill almost due north, a hair west of it
    z = np.array([[0.0, 0.0], [1.0, 1.0 + 2e-7]]) * 30.0
    _, aspect = slope_aspect(z, 30.0)
    assert np.all(aspect < 360.0) and np.float32(aspect.max()) == np.float32(360.0)  # the trap
    a32 = aspect_float32(aspect)
    assert a32.dtype == np.float32 and np.all((a32 >= 0) & (a32 < 360))
    assert aspect_float32(np.array([0.0, 90.0, 359.9])).tolist() == pytest.approx([0.0, 90.0, 359.9])


def test_slope_angle() -> None:
    np.testing.assert_allclose(slope_angle_deg(np.array([0.0, 1.0])), [0.0, 45.0])


# ----------------------------------------------------------------------------- the DEM cache


def test_fetch_dem_caches_with_a_manifest(tmp_path: Path) -> None:
    calls: list[tuple[BBox, int]] = []

    def fake_source(bbox: BBox, resolution_m: int, path: Path) -> None:
        calls.append((bbox, resolution_m))
        path.write_bytes(b"pretend GeoTIFF")

    dem = fetch_dem(BBOX, cache_root=tmp_path, source=fake_source)
    assert calls == [(BBOX, 10)]
    assert dem.path == tmp_path / dem_cache_key(BBOX, 10) / "dem.tif"
    manifest = json.loads((dem.path.parent / "manifest.json").read_text())
    assert manifest["resolution_m"] == 10 and manifest["crs"] == "EPSG:5070" and "sha256" in manifest
    again = fetch_dem(BBOX, cache_root=tmp_path, source=fake_source)
    assert len(calls) == 1 and again.path == dem.path  # read from the cache
    dem.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest hash"):
        fetch_dem(BBOX, cache_root=tmp_path, source=fake_source)
    assert dem_cache_key(BBOX, 30) != dem_cache_key(BBOX, 10)


def test_target_grid_cell_size() -> None:
    affine = pytest.importorskip("affine")
    assert TargetGrid("EPSG:5070", affine.Affine(30, 0, 0, 0, -30, 0), (4, 5)).cell_size == (30.0, 30.0)
    with pytest.raises(ValueError, match="north-up"):
        _ = TargetGrid("EPSG:5070", affine.Affine(30, 0, 0, 0, 30, 0), (4, 5)).cell_size


# ----------------------------------------------------------------------------- onto the grid (rasterio)


def plane_dem(path: Path, west: float, north: float, cell: float, shape: tuple[int, int], gx: float, gy: float) -> None:
    """A GeoTIFF in EPSG:5070 of the plane ``z = 500 + gx * x + gy * y`` (x east, y north, m)."""
    rasterio = pytest.importorskip("rasterio")
    from_origin = pytest.importorskip("rasterio.transform").from_origin

    h, w = shape
    rows, cols = np.mgrid[0:h, 0:w]
    x = west + (cols + 0.5) * cell
    y = north - (rows + 0.5) * cell
    z = (500.0 + gx * (x - west) + gy * (y - north)).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:5070",
        "transform": from_origin(west, north, cell, cell),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z, 1)


def test_terrain_on_a_coarser_grid(tmp_path: Path) -> None:
    """A 10 m plane resampled onto a 30 m grid inside it keeps its slope and aspect, and the grid
    gets its CRS, transform and centroid."""
    pytest.importorskip("rasterio")
    from_origin = pytest.importorskip("rasterio.transform").from_origin

    from wildfire_sim.data.terrain import terrain_for_grid

    west, north = -2_150_000.0, 1_700_000.0  # EPSG:5070, central California
    gx, gy = 0.12, -0.25  # rises east, falls north: slope 0.277, downhill toward bearing ~334 deg
    dem_path = tmp_path / "dem.tif"
    plane_dem(dem_path, west, north, 10.0, (300, 300), gx, gy)
    target = TargetGrid("EPSG:5070", from_origin(west + 300.0, north - 300.0, 30.0, 30.0), (60, 70))
    terrain = terrain_for_grid(dem_path, target)
    assert terrain.elevation.shape == (60, 70)
    np.testing.assert_allclose(terrain.slope, math.hypot(gx, gy), atol=1e-4)
    downhill = math.degrees(math.atan2(-gx, -gy)) % 360.0
    assert angle_diff(terrain.aspect, downhill).max() < 0.05
    md = terrain.metadata
    assert set(md) == {"crs", "transform", "centroid_lon", "centroid_lat"}
    assert "Albers" in md["crs"] or "5070" in md["crs"]
    assert [float(v) for v in md["transform"].split()] == [30.0, 0.0, west + 300.0, 0.0, -30.0, north - 300.0]
    lon, lat = float(md["centroid_lon"]), float(md["centroid_lat"])
    assert -125 < lon < -114 and 32 < lat < 42  # California


def test_reproject_refuses_a_dem_that_does_not_cover_the_grid(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    from_origin = pytest.importorskip("rasterio.transform").from_origin

    from wildfire_sim.data.terrain import reproject_to

    dem_path = tmp_path / "dem.tif"
    plane_dem(dem_path, 0.0, 1000.0, 10.0, (100, 100), 0.1, 0.0)
    target = TargetGrid("EPSG:5070", from_origin(500.0, 1000.0, 30.0, 30.0), (10, 30))  # runs 400 m past the DEM
    with pytest.raises(ValueError, match="does not cover"):
        reproject_to(dem_path, target)


# ----------------------------------------------------------------------------- live


@pytest.mark.network
def test_live_3dep_dem(tmp_path: Path) -> None:
    """Fetches the 10 m 3DEP DEM of a ~1 km box near the Chimney Fire and checks plausible values."""
    pytest.importorskip("py3dep")
    rasterio = pytest.importorskip("rasterio")

    box = BBox(-121.080, 35.570, -121.069, 35.579)
    dem = fetch_dem(box, cache_root=tmp_path)
    with rasterio.open(dem.path) as src:
        z = src.read(1, masked=True)
        assert src.crs.to_epsg() == 5070
    assert 0 < float(z.min()) < float(z.max()) < 1500  # Santa Lucia Range foothills, m
