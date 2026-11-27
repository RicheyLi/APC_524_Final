"""Tests for wildfire_sim.data.validation (WP 2.7).

Acceptance (docs/plan.md WP 2.7): mocked inputs rasterize correctly; a place detected on the
third overpass after a cloudy second one gets ``t_clear`` from the first; live test marked
``network``. The detection rule and the time conversion run everywhere; putting swaths and
perimeters on a grid needs the ``geo`` extra.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wildfire_sim.data.validation import (
    BOWTIE,
    CLEAR_LAND,
    CLOUD,
    FIRE_HIGH,
    FIRE_LOW,
    FIRE_NOMINAL,
    GLINT,
    NOT_VIEWED,
    TOKEN_ENV,
    WATER,
    Detection,
    EarthdataHttp,
    PerimeterVersion,
    RangeFile,
    SwathWindow,
    earthdata_token,
    extract_window,
    fetch_daily_perimeters,
    fetch_ignition,
    find_overpasses,
    first_detections,
    locate_row,
    tai93_to_utc,
)

UTC = dt.UTC
IRWIN = "{748283A9-D659-4FC4-9938-55DDE2484A19}"
LOCAL_ALBERS = "+proj=aea +lat_0=34.75 +lon_0=-120.0 +lat_1=34.6 +lat_2=34.9 +datum=NAD83 +units=m +no_defs"


# ----------------------------------------------------------------------------- time


def test_tai93_scan_time_to_utc() -> None:
    """VIIRS scan times are TAI seconds since 1993-01-01: 10 leap seconds since then by 2017."""
    epoch = dt.datetime(1993, 1, 1, tzinfo=UTC)
    t = dt.datetime(2024, 7, 6, 9, 37, 24, tzinfo=UTC)
    assert tai93_to_utc((t - epoch).total_seconds() + 10) == t
    early = dt.datetime(1993, 3, 1, tzinfo=UTC)
    assert tai93_to_utc((early - epoch).total_seconds()) == early
    y2k = dt.datetime(2000, 1, 1, tzinfo=UTC)  # after the 1993, 1994, 1996, 1997 and 1999 leap seconds
    assert tai93_to_utc((y2k - epoch).total_seconds() + 5) == y2k


# ----------------------------------------------------------------------------- the detection rule


def _stack(*grids: list[list[int]]) -> list[np.ndarray]:
    return [np.array(g, dtype=np.uint8) for g in grids]


def test_detected_after_a_cloudy_pass_keeps_the_clear_view_before_it() -> None:
    """Acceptance: clear on pass 1, cloud on pass 2, fire on pass 3 -> t_clear is pass 1."""
    classes = _stack([[CLEAR_LAND]], [[CLOUD]], [[FIRE_NOMINAL]])
    out = first_detections(classes, [1000.0, 2000.0, 3000.0], [[], [], [(0, 0)]], ignition_s=0.0)
    assert out == [Detection(1000.0, 3000.0, 0, 0)]


def test_detection_rule_details() -> None:
    times = [100.0, 200.0, 300.0, 400.0]
    # a cell clear twice: the later clear view counts; glint is not a clear view
    classes = _stack(
        [[CLEAR_LAND, CLEAR_LAND, WATER]],
        [[CLEAR_LAND, GLINT, CLEAR_LAND]],
        [[FIRE_HIGH, FIRE_NOMINAL, CLEAR_LAND]],
        [[FIRE_HIGH, FIRE_HIGH, FIRE_NOMINAL]],
    )
    cells = [[], [], [(0, 0), (0, 1)], [(0, 0), (0, 1), (0, 2)]]
    out = first_detections(classes, times, cells, ignition_s=50.0)
    assert out == [Detection(200.0, 300.0, 0, 0), Detection(100.0, 300.0, 0, 1), Detection(300.0, 400.0, 0, 2)]
    # an interval never starts before the ignition
    out = first_detections(classes, times, cells, ignition_s=250.0)
    assert out[0] == Detection(250.0, 300.0, 0, 0) and out[2] == Detection(300.0, 400.0, 0, 2)
    # fire seen before the ignition is ignored (another heat source or a false alarm)
    early = first_detections(_stack([[FIRE_HIGH]], [[FIRE_HIGH]]), [10.0, 20.0], [[(0, 0)], [(0, 0)]], 15.0)
    assert early == [Detection(15.0, 20.0, 0, 0)]
    # order of the input does not matter
    shuffled = first_detections(classes[::-1], times[::-1], cells[::-1], ignition_s=50.0)
    assert shuffled == first_detections(classes, times, cells, ignition_s=50.0)


def test_confidence_threshold() -> None:
    classes = _stack([[CLEAR_LAND]], [[FIRE_LOW]], [[FIRE_NOMINAL]])
    cells = [[], [(0, 0)], [(0, 0)]]
    assert first_detections(classes, [1.0, 2.0, 3.0], cells, 0.0) == [Detection(1.0, 2.0, 0, 0)]
    # at nominal confidence the low-confidence pass counts as neither fire nor clear
    assert first_detections(classes, [1.0, 2.0, 3.0], [[], [], [(0, 0)]], 0.0, FIRE_NOMINAL) == [
        Detection(1.0, 3.0, 0, 0)
    ]
    with pytest.raises(ValueError):
        first_detections(classes, [1.0, 2.0, 3.0], cells, 0.0, min_confidence=5)
    with pytest.raises(ValueError):
        first_detections(classes, [1.0, 2.0], cells, 0.0)
    assert first_detections([], [], [], 0.0) == []


def test_cells_outside_the_grid_are_ignored() -> None:
    classes = _stack([[FIRE_HIGH]])
    assert first_detections(classes, [5.0], [[(0, 0), (3, 3), (-1, 0)]], 0.0) == [Detection(0.0, 5.0, 0, 0)]


# ----------------------------------------------------------------------------- swath geometry


def _swath(lines: int = 640, pixels: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic north-south swath: latitude grows down the track, longitude across it."""
    rows, cols = np.mgrid[0:lines, 0:pixels]
    return (30.0 + rows * 0.01).astype(np.float32), (-125.0 + cols * 0.02 + rows * 0.001).astype(np.float32)


def test_locate_row_and_extract_window() -> None:
    lat, lon = _swath()
    row, dist = locate_row(lat, lon, 34.0, -122.0)
    assert abs(row - 400) <= 16 and dist < 1.5
    scan_times = np.arange(20, dtype=float) * 1.78 + 1000.0
    found = extract_window(lat, lon, scan_times, (-122.1, 33.9, -121.9, 34.1))
    assert found is not None
    rows, cols, t = found
    la, lo = lat[rows, cols], lon[rows, cols]
    assert la.min() <= 33.92 and la.max() >= 34.08 and lo.min() <= -121.9 and lo.max() >= -122.1
    assert t == scan_times[((rows.start + rows.stop) // 2) // 32]
    assert extract_window(lat, lon, scan_times, (-80.0, 10.0, -79.0, 11.0)) is None  # not in this swath


def test_range_file_reads_hdf5_by_byte_ranges() -> None:
    h5py = pytest.importorskip("h5py")
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset(
            "geolocation_data/latitude",
            data=np.arange(64 * 50, dtype=np.float32).reshape(64, 50),
            chunks=(32, 50),
            compression="gzip",
        )
    blob = buf.getvalue()
    calls: list[str] = []

    def get(url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        calls.append(headers["Range"])
        a, b = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
        return 206, {"Content-Range": f"bytes {a}-{b}/{len(blob)}"}, blob[a : b + 1]

    rf = RangeFile("https://example/x.nc", get, block=4096)
    with h5py.File(rf, "r") as f:
        assert f["geolocation_data/latitude"][40, 7] == 40 * 50 + 7
    assert rf.size == len(blob) and len(calls) >= 2
    with pytest.raises(ValueError, match="byte ranges"):
        RangeFile("https://example/x.nc", lambda u, h: (200, {}, blob))


def test_swath_window_round_trip(tmp_path: Path) -> None:
    w = SwathWindow(
        "NOAA-20",
        "VJ114IMG.A2024188.0936.002.x.nc",
        dt.datetime(2024, 7, 6, 9, 37, 24, tzinfo=UTC),
        np.zeros((2, 3), np.float32),
        np.ones((2, 3), np.float32),
        np.full((2, 3), CLEAR_LAND, np.uint8),
        np.array([34.8], np.float32),
        np.array([-120.1], np.float32),
        np.array([FIRE_HIGH], np.uint8),
    )
    w.save(tmp_path / "w.npz")
    back = SwathWindow.load(tmp_path / "w.npz")
    assert back.time_utc == w.time_utc and back.satellite == "NOAA-20" and np.array_equal(back.mask, w.mask)


# ----------------------------------------------------------------------------- on a grid (geo extra)


def _grid(n: int = 60) -> Any:
    """A grid of n x n 30 m cells centred on (34.75 N, 120.0 W) in a request-centred Albers."""
    pytest.importorskip("rasterio")
    from wildfire_sim.grid import Grid

    grid = Grid.empty((n, n), 30.0)
    half = n * 15.0
    grid.metadata = {
        "crs": LOCAL_ALBERS,
        "transform": f"30.0 0.0 {-half} 0.0 -30.0 {half}",
        "weather_start_utc": "2024-07-05T07:00:00+00:00",
    }
    return grid


def _cell_latlon(grid: Any, rows: np.ndarray, cols: np.ndarray, centre: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 of cell centres (or, with centre=False, of the cells' top-left corners)."""
    transform = pytest.importorskip("rasterio.warp").transform
    half = grid.shape[0] * 15.0
    off = 0.5 if centre else 0.0
    x = -half + 30.0 * (cols + off)
    y = half - 30.0 * (rows + off)
    lon, lat = transform(LOCAL_ALBERS, "EPSG:4326", x.ravel().tolist(), y.ravel().tolist())
    return np.array(lat).reshape(x.shape), np.array(lon).reshape(x.shape)


def _window(grid: Any, mask_fn: Any, time: dt.datetime, fires: list[tuple[int, int]], step: int = 12) -> SwathWindow:
    """Pixels every `step` cells (360 m) over the grid, class from mask_fn(row, col)."""
    rows, cols = np.mgrid[0 : grid.shape[0] : step, 0 : grid.shape[1] : step]
    lat, lon = _cell_latlon(grid, rows, cols)
    mask = np.vectorize(mask_fn)(rows, cols).astype(np.uint8)
    fr = np.array([r for r, _ in fires], dtype=float)
    fc = np.array([c for _, c in fires], dtype=float)
    flat, flon = _cell_latlon(grid, fr, fc) if fires else (np.array([]), np.array([]))
    return SwathWindow(
        "NOAA-20",
        "test.nc",
        time,
        lat.astype(np.float32),
        lon.astype(np.float32),
        mask,
        flat.astype(np.float32),
        flon.astype(np.float32),
        np.full(len(fires), FIRE_HIGH, np.uint8),
    )


def test_classes_on_grid() -> None:
    """Mocked swath: cells take the nearest pixel's class; cells beyond the swath are not viewed."""
    pytest.importorskip("scipy")
    from wildfire_sim.data.validation import classes_on_grid, target_of

    grid = _grid()

    # pixels only over the west half; the north-west pixel is cloud
    def cls(r: int, c: int) -> int:
        return CLOUD if (r, c) == (0, 0) else CLEAR_LAND

    w = _window(grid, cls, dt.datetime(2024, 7, 6, tzinfo=UTC), [])
    keep = w.lon <= np.median(w.lon)
    w = SwathWindow(
        w.satellite,
        w.granule,
        w.time_utc,
        np.where(keep, w.lat, np.nan),
        w.lon,
        w.mask,
        w.fire_lat,
        w.fire_lon,
        w.fire_confidence,
    )
    out = classes_on_grid(w, target_of(grid))
    assert out[2, 2] == CLOUD and out[30, 5] == CLEAR_LAND
    assert out[30, 59] == NOT_VIEWED  # more than 600 m east of the last pixel column
    # a bowtie-deleted pixel never decides when a real pixel is close
    b = SwathWindow(
        w.satellite,
        w.granule,
        w.time_utc,
        w.lat,
        w.lon,
        np.where(w.mask == CLOUD, BOWTIE, w.mask).astype(np.uint8),
        w.fire_lat,
        w.fire_lon,
        w.fire_confidence,
    )
    assert classes_on_grid(b, target_of(grid))[2, 2] == CLEAR_LAND


def test_observations_on_a_grid() -> None:
    """Mocked fire: perimeters rasterize to the expected cells; detections get their intervals."""
    pytest.importorskip("scipy")
    from wildfire_sim.data.cache import Perimeter
    from wildfire_sim.data.validation import FireObservations, Ignition

    grid = _grid()
    rows, cols = np.mgrid[10:30, 10:30]
    _lat, _lon = _cell_latlon(grid, rows, cols)

    def box(r0: int, r1: int, c0: int, c1: int) -> dict[str, Any]:
        # corners on cell edges: covers the centres of cells r0..r1-1, c0..c1-1 and no others
        la, lo = _cell_latlon(grid, np.array([r0, r0, r1, r1]), np.array([c0, c1, c1, c0]), centre=False)
        ring = [[float(x), float(y)] for x, y in zip(lo, la, strict=True)]
        return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}

    t0 = dt.datetime(2024, 7, 5, 22, 48, tzinfo=UTC)
    daily = [
        PerimeterVersion(t0 + dt.timedelta(hours=10), t0 + dt.timedelta(hours=20), box(20, 30, 20, 30), 20.0),
        PerimeterVersion(
            t0 + dt.timedelta(hours=20), t0 + dt.timedelta(hours=22), box(25, 26, 25, 26), 0.1
        ),  # placeholder
        PerimeterVersion(t0 + dt.timedelta(hours=22), None, box(10, 40, 10, 40), 80.0),
    ]
    origin_lat, origin_lon = _cell_latlon(grid, np.array([25]), np.array([25]))
    ign = Ignition(t0, float(origin_lat[0]), float(origin_lon[0]), None, None, "2024-TEST-1", IRWIN, "test")
    final = Perimeter(box(5, 50, 5, 50), {}, Path("x"), {})
    passes = [
        _window(grid, lambda r, c: CLEAR_LAND, t0 + dt.timedelta(hours=1), []),
        _window(grid, lambda r, c: CLOUD, t0 + dt.timedelta(hours=11), []),
        _window(
            grid,
            lambda r, c: FIRE_HIGH if 18 <= r <= 30 and 18 <= c <= 30 else CLEAR_LAND,
            t0 + dt.timedelta(hours=12),
            [(24, 24)],
        ),
    ]
    obs = FireObservations("t", ign, final, daily, passes)
    assert obs.ignition_cell(grid) == (25, 25)
    assert obs.seconds(grid, t0) == (t0 - dt.datetime(2024, 7, 5, 7, tzinfo=UTC)).total_seconds()
    # final perimeter: cells 5..49; daily: monotone, the placeholder un-burns nothing
    final_mask = obs.observed_burn_mask(grid)
    assert final_mask.sum() == 45 * 45 and final_mask[5, 5] and final_mask[49, 49] and not final_mask[50, 50]
    assert not obs.observed_burn_mask(grid, t0 + dt.timedelta(hours=5)).any()
    m15 = obs.observed_burn_mask(grid, t0 + dt.timedelta(hours=15))
    assert m15[20:30, 20:30].all() and m15.sum() == 100
    m21 = obs.observed_burn_mask(grid, t0 + dt.timedelta(hours=21))
    assert np.array_equal(m21, m15)  # the one-cell placeholder lies inside: nothing changes
    assert obs.observed_burn_mask(grid, t0 + dt.timedelta(hours=23)).sum() == 30 * 30
    with pytest.raises(ValueError):
        obs.observed_burn_mask(grid, dt.datetime(2024, 7, 6))
    # detection on the third pass after a cloudy second: t_clear is the first pass
    dets = obs.detections(grid)
    assert dets == [
        Detection(obs.seconds(grid, t0 + dt.timedelta(hours=1)), obs.seconds(grid, t0 + dt.timedelta(hours=12)), 24, 24)
    ]


# ----------------------------------------------------------------------------- services (mocked)


class FakeJson:
    def __init__(self, replies: dict[str, Any]) -> None:
        self.replies, self.urls = replies, []

    def get_json(self, url: str) -> dict[str, Any]:
        self.urls.append(url)
        for key, reply in self.replies.items():
            if key in url:
                return reply(url) if callable(reply) else reply
        raise AssertionError(f"unexpected URL {url}")


def test_fetch_ignition(tmp_path: Path) -> None:
    record = {
        "features": [
            {
                "attributes": {
                    "UniqueFireIdentifier": "2024-CALPF-001542",
                    "FireDiscoveryDateTime": 1720219680000,
                    "InitialLatitude": 34.785,
                    "InitialLongitude": -120.044833,
                },
                "geometry": {"x": -120.05201254554999, "y": 34.78683782813522},
            }
        ]
    }
    http = FakeJson({"WFIGS_Incident_Locations": record})
    ign = fetch_ignition(IRWIN, cache_root=tmp_path, http=http)
    assert ign.time_utc == dt.datetime(2024, 7, 5, 22, 48, tzinfo=UTC)
    assert (ign.lat, ign.lon) == (34.78683782813522, -120.05201254554999) and ign.initial_lat == 34.785
    assert "IRWIN" in ign.source
    assert fetch_ignition(IRWIN, cache_root=tmp_path, http=http) == ign and len(http.urls) == 1
    with pytest.raises(ValueError, match="IRWIN id"):
        fetch_ignition("x' OR 1=1", cache_root=tmp_path, http=http)
    with pytest.raises(ValueError, match="0 records"):
        fetch_ignition(
            "{00000000-0000-0000-0000-000000000000}", cache_root=tmp_path, http=FakeJson({"WFIGS": {"features": []}})
        )


def test_fetch_daily_perimeters_pages(tmp_path: Path) -> None:
    def feature(ms: int, acres: float) -> dict[str, Any]:
        ring = [[-120.0, 34.7], [-119.9, 34.7], [-119.9, 34.8], [-120.0, 34.7]]
        return {
            "type": "Feature",
            "properties": {"GDB_FROM_DATE": ms, "GDB_TO_DATE": None, "GISAcres": acres},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }

    def query(url: str) -> dict[str, Any]:
        assert "Wildfire+Daily+Fire+Perimeter" in url and "Approved" in url
        if "resultOffset=0" in url:
            return {"features": [feature(1720300000000, 4674.0)], "exceededTransferLimit": True}
        return {"features": [feature(1720400000000, 12227.0)]}

    http = FakeJson(
        {
            "?f=json": {"layers": [{"id": 2, "name": "Event Point"}, {"id": 6, "name": "Event Polygon"}]},
            "/6/query": query,
        }
    )
    versions = fetch_daily_perimeters(IRWIN, 2024, cache_root=tmp_path, http=http)
    assert [v.acres for v in versions] == [4674.0, 12227.0]
    assert versions[0].valid_from == dt.datetime.fromtimestamp(1720300000, UTC) and versions[0].valid_to is None
    assert versions[0].current_at(dt.datetime(2024, 7, 10, tzinfo=UTC))
    assert len(fetch_daily_perimeters(IRWIN, 2024, cache_root=tmp_path, http=http)) == 2 and len(http.urls) == 3


def test_no_archive_for_the_year_means_no_daily_perimeters(tmp_path: Path) -> None:
    """The Operational Data Archive starts in 2022: for a 2020 fire (fire B) none are published."""
    http = FakeJson({"Operational_Data_Archive_2020": {"error": {"code": 400, "message": "Invalid URL"}}})
    assert fetch_daily_perimeters(IRWIN, 2020, cache_root=tmp_path, http=http) == []
    assert fetch_daily_perimeters(IRWIN, 2020, cache_root=tmp_path, http=http) == [] and len(http.urls) == 1


def test_find_overpasses_pairs_fire_and_geolocation() -> None:
    def entry(name: str, start: str) -> dict[str, Any]:
        host = "lpdaac" if "14IMG" in name else "laadsdaac"
        return {
            "title": name,
            "time_start": start,
            "links": [
                {"href": f"https://data.{host}.earthdatacloud.nasa.gov/x/{name}.nc"},
                {"href": "https://doi.org/x"},
            ],
        }

    def cmr(url: str) -> dict[str, Any]:
        if "short_name=VJ114IMG" in url:
            return {
                "feed": {
                    "entry": [
                        entry("VJ114IMG.A2024188.0936.002.2024188143639", "2024-07-06T09:36:00.000Z"),
                        entry("VJ114IMG.A2024188.2054.002.2024189000000", "2024-07-06T20:54:00.000Z"),
                    ]
                }
            }
        if "short_name=VJ103IMG" in url:
            return {"feed": {"entry": [entry("VJ103IMG.A2024188.0936.021.2024188135525", "2024-07-06T09:36:00.000Z")]}}
        return {"feed": {"entry": []}}

    pairs = find_overpasses(
        -120.2,
        34.6,
        -119.8,
        34.9,
        dt.datetime(2024, 7, 6, tzinfo=UTC),
        dt.datetime(2024, 7, 7, tzinfo=UTC),
        http=FakeJson({"cmr": cmr}),
    )
    assert len(pairs) == 1  # the 20:54 pass has no geolocation granule listed
    p = pairs[0]
    assert (p.satellite, p.stamp) == ("NOAA-20", "A2024188.0936") and "VJ103IMG" in p.geo_url


# ----------------------------------------------------------------------------- Earthdata access


class FakeResponse:
    def __init__(self, status: int, headers: dict[str, str] | None = None, content: bytes = b"") -> None:
        self.status_code, self.headers, self.content = status, headers or {}, content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeRequests:
    ConnectionError = ConnectionError

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses, self.calls = responses, []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs.get("headers", {})))
        return self.responses.pop(0)


def test_earthdata_http_retries_and_keeps_the_token_to_nasa() -> None:
    pytest.importorskip("requests")
    waits: list[float] = []
    http = EarthdataHttp("TOKEN", sleep=waits.append)
    fake = FakeRequests(
        [
            FakeResponse(502),
            FakeResponse(303, {"location": "https://cdn.example/signed"}),
            FakeResponse(200, content=b"data"),
        ]
    )
    http._requests = fake
    assert http.get_bytes("https://data.lpdaac.earthdatacloud.nasa.gov/f.nc") == b"data"
    assert waits == [5.0]
    assert fake.calls[0][1]["Authorization"] == "Bearer TOKEN"
    assert "Authorization" not in fake.calls[2][1]  # the signed URL gets no token
    http._requests = FakeRequests([FakeResponse(302, {"location": "https://urs.earthdata.nasa.gov/oauth/authorize"})])
    with pytest.raises(PermissionError, match="refused the token"):
        http.signed_url("https://data.lpdaac.earthdatacloud.nasa.gov/f.nc")
    # a redirect that only mentions the login host elsewhere in its URL is not a refusal
    lookalike = "https://cdn.example/signed?next=urs.earthdata.nasa.gov"
    http._requests = FakeRequests([FakeResponse(302, {"location": lookalike})])
    assert http.signed_url("https://data.lpdaac.earthdatacloud.nasa.gov/f.nc") == lookalike
    # and the token is never sent to a host that is not an Earthdata data host
    unsent = FakeRequests([FakeResponse(200)])
    http._requests = unsent
    with pytest.raises(PermissionError, match="token is not sent"):
        http.signed_url("https://data.example.org/f.nc?x=.earthdatacloud.nasa.gov")
    assert unsent.calls == []


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://data.lpdaac.earthdatacloud.nasa.gov/a/b.nc", True),
        ("https://DATA.laadsdaac.earthdatacloud.nasa.gov/b.nc", True),
        ("http://data.lpdaac.earthdatacloud.nasa.gov/b.nc", False),  # not https
        ("https://data.example.org/b.nc", False),
        ("https://data.example.org/.earthdatacloud.nasa.gov/b.nc", False),  # the domain only in the path
        ("https://data.lpdaac.earthdatacloud.nasa.gov.example.org/b.nc", False),  # the domain as a prefix
        ("https://data.lpdaac.earthdatacloud.nasa.gov@example.org/b.nc", False),  # the domain as user info
        ("https://lpdaac.earthdatacloud.nasa.gov/b.nc", False),  # not a data host
        ("not a url", False),
    ],
)
def test_earthdata_data_urls_are_recognised_by_host(url: str, ok: bool) -> None:
    from wildfire_sim.data.validation import _https_nc, is_earthdata_data_url

    assert is_earthdata_data_url(url) is ok
    assert _https_nc({"links": [{"href": url}]}) == (url if ok else None)


def test_earthdata_token_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(ValueError, match="Earthdata token"):
        earthdata_token()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / ".earthdata_token").write_text("from-file\n", encoding="utf-8")
    assert earthdata_token() == "from-file"
    monkeypatch.setenv(TOKEN_ENV, "from-env")
    assert earthdata_token() == "from-env"


# ----------------------------------------------------------------------------- live


def test_fetch_viirs_reads_a_cached_period_without_the_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second call for the same box and period lists nothing and downloads nothing: the
    catalogue's answer and every overpass come from the cache (a compute node without internet)."""
    from wildfire_sim.data import validation
    from wildfire_sim.data.validation import GranulePair, SwathWindow, fetch_viirs

    start = dt.datetime(2024, 7, 6, 8, tzinfo=UTC)
    end = start + dt.timedelta(hours=2)
    box = (-120.2, 34.6, -119.9, 34.9)
    pairs = [
        GranulePair("VNP14IMG", "Suomi NPP", "A2024188.0930", "https://f/a.nc", "https://g/a.nc", start),
        GranulePair("VJ114IMG", "NOAA-20", "A2024188.1018", "https://f/b.nc", "https://g/b.nc", end),
    ]
    calls = {"list": 0, "read": 0}

    def find(*args: object, **kwargs: object) -> list[GranulePair]:
        calls["list"] += 1
        return pairs

    def read(pair: GranulePair, box: object, http: object) -> SwathWindow | None:
        calls["read"] += 1
        if pair.satellite == "NOAA-20":
            return None  # this overpass does not cover the box
        z = np.zeros((2, 2))
        return SwathWindow(pair.satellite, "a.nc", pair.time_start, z, z, z.astype(np.uint8), z[0], z[0], z[0])

    monkeypatch.setattr(validation, "find_overpasses", find)
    monkeypatch.setattr(validation, "read_overpass", read)
    monkeypatch.setattr(validation, "EarthdataHttp", lambda token: None)  # the real client needs requests
    first = fetch_viirs("x", box, start, end, cache_root=tmp_path, token="t")
    assert calls == {"list": 1, "read": 2} and [w.satellite for w in first] == ["Suomi NPP"]

    def offline(*args: object, **kwargs: object) -> None:
        raise AssertionError("the cached period must not touch the network")

    monkeypatch.setattr(validation, "find_overpasses", offline)
    monkeypatch.setattr(validation, "read_overpass", offline)
    again = fetch_viirs("x", box, start, end, cache_root=tmp_path)  # no token either
    assert [(w.satellite, w.time_utc) for w in again] == [(w.satellite, w.time_utc) for w in first]

    monkeypatch.setattr(validation, "find_overpasses", find)  # another period asks the catalogue again
    fetch_viirs("x", box, start, end + dt.timedelta(hours=1), cache_root=tmp_path)
    assert calls["list"] == 2 and calls["read"] == 2


@pytest.mark.network
def test_live_lake_fire_observations(tmp_path: Path) -> None:
    """Ignition, daily perimeters and the first VIIRS night passes of the Lake Fire."""
    pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    from wildfire_sim.data.cache import fetch_mtbs_perimeter, load_regions
    from wildfire_sim.data.validation import fetch_viirs, region_box

    try:
        earthdata_token()
    except ValueError:
        pytest.skip(f"set {TOKEN_ENV} (or data/.earthdata_token) to run the live VIIRS test")
    region = load_regions().regions["lake_2024"]
    assert region.irwin_id is not None
    ign = fetch_ignition(region.irwin_id, cache_root=tmp_path)
    assert ign.time_utc == dt.datetime(2024, 7, 5, 22, 48, tzinfo=UTC) and ign.unique_fire_id == "2024-CALPF-001542"
    daily = fetch_daily_perimeters(region.irwin_id, 2024, cache_root=tmp_path)
    assert max(v.acres for v in daily) > 38000
    final = fetch_mtbs_perimeter(region.mtbs_fire_id, region.mtbs_year, cache_root=tmp_path)
    start = dt.datetime(2024, 7, 6, 8, 30, tzinfo=UTC)
    windows = fetch_viirs("lake", region_box(final), start, start + dt.timedelta(hours=1), cache_root=tmp_path)
    assert windows and all(w.fire_lat.size > 50 for w in windows)  # the night passes saw the fire
    assert json.loads((tmp_path / "viirs-lake" / "manifest.json").read_text(encoding="utf-8"))["granules"]
