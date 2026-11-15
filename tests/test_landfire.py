"""Tests for LANDFIRE layers (WP 2.2, ``wildfire_sim.data.landfire``).

Offline: the LANDFIRE Product Service is replaced by :class:`FakeHttp`; the archive loader runs
on small GeoTIFFs written here (skipped without ``rasterio``). The live test is marked
``network`` and also needs an email address in ``WILDFIRE_SIM_LANDFIRE_EMAIL``.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wildfire_sim.data.landfire import (
    EMAIL_ENV,
    LFPS_API,
    PRODUCTS,
    BBox,
    JobStatus,
    assign_bands,
    build_job_request,
    cache_key,
    check_version,
    fetch_landfire,
    grid_codes,
    load_versions,
    wait_for_job,
)
from wildfire_sim.types import LandUse
from wildfire_sim.urban import load_urban_model

CHIMNEY = dt.date(2016, 8, 13)  # fire A, the Chimney Fire (plan section 7)
BBOX = BBox(-121.10, 35.55, -121.05, 35.60)
EMAIL = "someone@example.org"


# ----------------------------------------------------------------------------- versions


def test_version_table_is_cited_and_complete() -> None:
    table = load_versions()
    assert "landfire.gov/data" in table.source
    on_service = sorted(n for n, v in table.versions.items() if v.on_service)
    assert on_service == ["LF2016", "LF2022", "LF2023", "LF2024", "LF2025"]  # https://lfps.usgs.gov/api/products
    assert table.versions["LF2016"].disturbance_years == (2015, 2016)
    assert table.versions["LF2014"].disturbance_years == (2013, 2014)


def test_refuses_a_version_newer_than_the_fire() -> None:
    """WP 2.2 acceptance: LF2016 includes 2016 disturbances, so it is refused for the Chimney Fire."""
    with pytest.raises(ValueError, match="newest that predates it is LF2014"):
        check_version("LF2016", CHIMNEY)
    assert check_version("LF2014", CHIMNEY).layer_prefix == "LF2014"
    assert check_version("LF2016", dt.date(2017, 1, 10)).on_service  # the year after is fine
    with pytest.raises(ValueError, match="through 2024"):
        check_version("LF2024", dt.date(2024, 9, 1))
    with pytest.raises(ValueError, match="unknown LANDFIRE version"):
        check_version("LF1999", CHIMNEY)
    with pytest.raises(ValueError, match="before the fire$"):
        check_version("LF2012", dt.date(2012, 6, 1))  # nothing older is listed


def test_layer_names() -> None:
    v = load_versions().versions["LF2024"]
    assert [v.layer(p) for p in PRODUCTS] == ["LF2024_FBFM40", "LF2024_CC", "LF2024_CH", "LF2024_CBH", "LF2024_CBD"]
    with pytest.raises(ValueError, match="unknown LANDFIRE product"):
        v.layer("EVT")


# ----------------------------------------------------------------------------- area of interest


def test_bbox_validation_and_aoi() -> None:
    assert BBOX.aoi() == "-121.100000 35.550000 -121.050000 35.600000"  # W S E N
    for bad in (
        (-121.0, 35.5, -121.1, 35.6),
        (-121.1, 35.6, -121.0, 35.5),
        (-190.0, 0.0, 0.0, 1.0),
        (0, 0, float("nan"), 1),
    ):
        with pytest.raises(ValueError):
            BBox(*bad)


def test_cache_key_identifies_the_request() -> None:
    k = cache_key(BBOX, "LF2022")
    assert len(k) == 16 and k == cache_key(BBox(-121.10, 35.55, -121.05, 35.60), "LF2022")
    assert k != cache_key(BBOX, "LF2023") and k != cache_key(BBOX, "LF2022", ("FBFM40",))


# ----------------------------------------------------------------------------- the request builder


def test_job_request_body() -> None:
    """WP 2.2 acceptance: the request the LFPS web page itself sends (``POST /api/job/submit``)."""
    v = load_versions().versions["LF2022"]
    body = build_job_request(BBOX, v, EMAIL)
    assert body == {
        "Layer_List": "LF2022_FBFM40;LF2022_CC;LF2022_CH;LF2022_CBH;LF2022_CBD",
        "Area_of_Interest": BBOX.aoi(),
        "Email": EMAIL,
    }
    body = build_job_request(BBOX, v, EMAIL, ("FBFM40",), output_projection=5070, resolution_m=60)
    assert body["Layer_List"] == "LF2022_FBFM40"
    assert body["Output_Projection"] == "5070" and body["Resample_Resolution"] == 60


def test_job_request_refuses_bad_input() -> None:
    table = load_versions()
    with pytest.raises(ValueError, match="archived"):
        build_job_request(BBOX, table.versions["LF2014"], EMAIL)
    v = table.versions["LF2022"]
    with pytest.raises(ValueError, match="email"):
        build_job_request(BBOX, v, "nobody")
    with pytest.raises(ValueError, match="multiple of 30"):
        build_job_request(BBOX, v, EMAIL, resolution_m=45)
    with pytest.raises(ValueError, match="at least one"):
        build_job_request(BBOX, v, EMAIL, products=())


# ----------------------------------------------------------------------------- the job API, mocked


def zip_bytes(files: Mapping[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class FakeHttp:
    """Stands in for LFPS: a job id, ``n_running`` status replies without a file, then the zip."""

    def __init__(self, archive: bytes, n_running: int = 2, final: dict[str, Any] | None = None) -> None:
        self.archive, self.n_running = archive, n_running
        self.final = final or {"status": "Succeeded", "outputFile": "https://lfps.usgs.gov/out/j1.zip"}
        self.calls: list[tuple[str, str, Any]] = []

    def post_json(self, url: str, body: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append(("POST", url, dict(body)))
        return {"jobId": "j1"}

    def get_json(self, url: str) -> dict[str, Any]:
        self.calls.append(("GET", url, None))
        polls = sum(1 for c in self.calls if "status" in c[1])
        if polls <= self.n_running:
            return {"status": "Executing", "messages": ["clipping"], "queuePosition": 0, "outputFile": None}
        return self.final

    def get_bytes(self, url: str) -> bytes:
        self.calls.append(("GET", url, None))
        return self.archive


def per_layer_zip(version: str = "LF2022") -> bytes:
    files = {f"j1/{version}_{p}.tif": f"tif {p}".encode() for p in PRODUCTS}
    files["j1/metadata.xml"] = b"<xml/>"
    return zip_bytes(files)


def test_fetch_submits_polls_downloads_and_caches(tmp_path: Path) -> None:
    http = FakeHttp(per_layer_zip())
    layers = fetch_landfire(
        BBOX, "LF2022", CHIMNEY.replace(year=2023), email=EMAIL, cache_root=tmp_path, http=http, sleep=lambda s: None
    )
    post = [c for c in http.calls if c[0] == "POST"]
    assert post == [
        ("POST", f"{LFPS_API}/job/submit", build_job_request(BBOX, load_versions().versions["LF2022"], EMAIL))
    ]
    assert sum("status/?JobId=j1" in c[1] for c in http.calls) == 3
    assert http.calls[-1][1] == "https://lfps.usgs.gov/out/j1.zip"
    assert layers.directory == tmp_path / cache_key(BBOX, "LF2022")
    for p in PRODUCTS:
        path, band = layers.files[p]
        assert band == 1 and path.read_bytes() == f"tif {p}".encode()
    manifest = json.loads((layers.directory / "manifest.json").read_text())
    assert manifest["version"] == "LF2022" and manifest["source"]["job_id"] == "j1"
    assert set(manifest["sha256"]) == {f"LF2022_{p}.tif" for p in PRODUCTS}
    assert EMAIL not in (layers.directory / "manifest.json").read_text()  # the address is not cached
    # a second request reads the cache and makes no call
    again = FakeHttp(b"")
    cached = fetch_landfire(BBOX, "LF2022", dt.date(2023, 8, 1), cache_root=tmp_path, http=again)
    assert again.calls == [] and cached.files == layers.files


def test_fetch_reads_the_email_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EMAIL_ENV, raising=False)
    with pytest.raises(ValueError, match=EMAIL_ENV):
        fetch_landfire(BBOX, "LF2022", dt.date(2023, 1, 1), cache_root=tmp_path, http=FakeHttp(per_layer_zip()))
    monkeypatch.setenv(EMAIL_ENV, EMAIL)
    http = FakeHttp(per_layer_zip(), n_running=0)
    fetch_landfire(BBOX, "LF2022", dt.date(2023, 1, 1), cache_root=tmp_path, http=http)
    assert http.calls[0][2]["Email"] == EMAIL


def test_fetch_refuses_too_new_or_archived_versions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LF2014"):
        fetch_landfire(BBOX, "LF2016", CHIMNEY, email=EMAIL, cache_root=tmp_path, http=FakeHttp(b""))
    with pytest.raises(ValueError, match="ingest_archived"):
        fetch_landfire(BBOX, "LF2014", CHIMNEY, email=EMAIL, cache_root=tmp_path, http=FakeHttp(b""))


def test_cache_detects_a_changed_file(tmp_path: Path) -> None:
    layers = fetch_landfire(
        BBOX,
        "LF2022",
        dt.date(2023, 1, 1),
        email=EMAIL,
        cache_root=tmp_path,
        http=FakeHttp(per_layer_zip(), n_running=0),
    )
    layers.files["CC"][0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest hash"):
        fetch_landfire(BBOX, "LF2022", dt.date(2023, 1, 1), cache_root=tmp_path, http=FakeHttp(b""))


def test_single_multiband_result_uses_bands_in_request_order(tmp_path: Path) -> None:
    http = FakeHttp(zip_bytes({"j1.tif": b"one file"}), n_running=0)
    layers = fetch_landfire(BBOX, "LF2023", dt.date(2024, 7, 1), email=EMAIL, cache_root=tmp_path, http=http)
    assert [layers.files[p][1] for p in PRODUCTS] == [1, 2, 3, 4, 5]
    assert {layers.files[p][0].name for p in PRODUCTS} == {"j1.tif"}


def test_assign_bands_refuses_ambiguous_results() -> None:
    with pytest.raises(ValueError, match="cannot match"):
        assign_bands(["a.tif", "b.tif"], ["LF2022_FBFM40", "LF2022_CC"])
    assert assign_bands(["x/LF2022_CC.TIF", "LF2022_FBFM40.tif"], ["LF2022_FBFM40", "LF2022_CC"]) == {
        "LF2022_FBFM40": ("LF2022_FBFM40.tif", 1),
        "LF2022_CC": ("x/LF2022_CC.TIF", 1),
    }


def test_wait_for_job_failure_and_timeout() -> None:
    failed = FakeHttp(b"", n_running=0, final={"status": "Failed", "messages": ["AOI too large"]})
    with pytest.raises(RuntimeError, match="AOI too large"):
        wait_for_job("j1", failed, sleep=lambda s: None)
    stuck = FakeHttp(b"", n_running=10**6)
    with pytest.raises(TimeoutError):
        wait_for_job("j1", stuck, poll_s=5, timeout_s=20, sleep=lambda s: None)
    assert sum("status" in c[1] for c in stuck.calls) == 5


def test_job_status_parsing() -> None:
    s = JobStatus.from_json({"status": "Queued", "messages": "waiting", "queuePosition": 3})
    assert (s.status, s.messages, s.queue_position, s.output_file, s.failed) == ("Queued", ("waiting",), 3, None, False)
    assert JobStatus.from_json({"status": "Cancelled"}).failed


# ----------------------------------------------------------------------------- codes


def test_grid_codes_map_non_burnable_fuel_models() -> None:
    """WP 2.2: 91 urban (with the surrogate fuel of configs/urban.yaml), 98 water, 92/93/99 no fuel."""
    surrogate = int(load_urban_model().fuel_model.value)
    fbfm = np.array([[101, 165, 91, 92], [93, 98, 99, -9999]], dtype=np.int16)
    fuel, land = grid_codes(fbfm, surrogate, nodata=-9999)
    assert fuel.dtype == np.int16 and land.dtype == np.int8
    assert fuel.tolist() == [[101, 165, surrogate, 0], [0, 0, 0, 0]]
    w, u, wa = int(LandUse.WILDLAND), int(LandUse.URBAN), int(LandUse.WATER)
    assert land.tolist() == [[w, w, u, w], [w, wa, w, w]]
    with pytest.raises(ValueError):
        grid_codes(fbfm, 0)


# ----------------------------------------------------------------------------- archived versions


def write_geotiff(path: Path, data: np.ndarray, west: float, north: float, cell_deg: float) -> None:
    rasterio = pytest.importorskip("rasterio")
    from_origin = pytest.importorskip("rasterio.transform").from_origin

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": str(data.dtype),
        "crs": "EPSG:4326",
        "transform": from_origin(west, north, cell_deg, cell_deg),
        "nodata": -9999,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_ingest_archived_clips_and_caches(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    from wildfire_sim.data.landfire import ingest_archived, read_layer

    data = np.arange(100, dtype=np.int16).reshape(10, 10) + 101
    src = tmp_path / "fbfm40_conus.tif"
    write_geotiff(src, data, west=-121.2, north=35.7, cell_deg=0.02)  # covers -121.2..-121.0, 35.5..35.7
    layers = ingest_archived({"FBFM40": src}, BBOX, "LF2014", CHIMNEY, cache_root=tmp_path / "cache")
    clipped, profile = read_layer(layers, "FBFM40")
    # bbox -121.10..-121.05, 35.55..35.60 -> columns 5..7 (whole cells), rows 5..7
    assert clipped.shape == (3, 3)
    np.testing.assert_array_equal(clipped, data[5:8, 5:8])
    assert profile["nodata"] == -9999
    manifest = json.loads((layers.directory / "manifest.json").read_text())
    assert manifest["source"] == {"archived": {"FBFM40": "fbfm40_conus.tif"}} and manifest["version"] == "LF2014"
    with pytest.raises(ValueError, match="LF2014"):
        ingest_archived({"FBFM40": src}, BBOX, "LF2016", CHIMNEY, cache_root=tmp_path / "cache")
    with pytest.raises(ValueError, match="overlap"):
        ingest_archived({"FBFM40": src}, BBox(-100.0, 40.0, -99.9, 40.1), "LF2014", CHIMNEY, cache_root=tmp_path / "c2")
    with pytest.raises(ValueError, match="products"):
        ingest_archived({"EVT": src}, BBOX, "LF2014", CHIMNEY, cache_root=tmp_path / "c3")


# ----------------------------------------------------------------------------- live


@pytest.mark.network
def test_live_lfps_job(tmp_path: Path) -> None:
    """Submits a real job for a 1 km x 1 km box (fuel models only) and checks the GeoTIFF."""
    if not os.environ.get(EMAIL_ENV):
        pytest.skip(f"set {EMAIL_ENV} to run the live LANDFIRE test")
    pytest.importorskip("rasterio")
    from wildfire_sim.data.landfire import read_layer

    box = BBox(-121.080, 35.570, -121.069, 35.579)
    layers = fetch_landfire(
        box, "LF2022", dt.date(2023, 6, 1), products=("FBFM40",), cache_root=tmp_path, timeout_s=1800
    )
    fbfm, _ = read_layer(layers, "FBFM40")
    assert fbfm.size > 0 and np.isin(fbfm, [*range(91, 100), *range(101, 205), -9999, 32767]).all()
