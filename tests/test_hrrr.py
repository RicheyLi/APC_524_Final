"""Tests for wildfire_sim.data.hrrr and the HRRR source of data.weather (WP 2.4 follow-up).

Offline: synthetic GRIB2 headers built byte by byte, a fake downloader and decoder. One live
download of real analyses is marked ``network``.
"""

from __future__ import annotations

import datetime as dt
import email.message
import math
import urllib.error
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from wildfire_sim.data.hrrr import (
    FIELDS,
    Field,
    HrrrHour,
    LambertGrid,
    analysis_url,
    byte_ranges,
    fetch_hrrr_point,
    lambert_from_grib,
    to_earth_relative,
    wind_from,
)
from wildfire_sim.data.weather import DailyWeather, hourly_from_hrrr, hourly_weather, state_of_weather_from_cloud

UTC = dt.UTC
LAKE = (34.7586, -120.0350)


def grib2_header(
    flags: int = 0b1000,
    lov_udeg: int = 262_500_000,
    latin1_udeg: int = 38_500_000,
    latin2_udeg: int | None = None,
    template: int = 30,
) -> bytes:
    """A GRIB2 message with sections 0, 1 and 3 (template 3.30) and the octets the reader needs."""
    sec3 = bytearray(81)
    sec3[0:4] = (81).to_bytes(4, "big")
    sec3[4] = 3
    sec3[12:14] = template.to_bytes(2, "big")

    def put(first_octet: int, value: int, n: int = 4) -> None:
        sec3[first_octet - 1 : first_octet - 1 + n] = value.to_bytes(n, "big")

    put(47, flags, 1)
    put(52, lov_udeg)
    put(66, latin1_udeg)
    put(70, latin1_udeg if latin2_udeg is None else latin2_udeg)
    sec1 = (21).to_bytes(4, "big") + bytes([1]) + bytes(16)
    total = 16 + len(sec1) + len(sec3) + 4
    sec0 = b"GRIB" + bytes(2) + bytes([0, 2]) + total.to_bytes(8, "big")
    return sec0 + sec1 + bytes(sec3) + b"7777"


# ----------------------------------------------------------------------------- the index and the header


IDX = """1:0:d=2024070600:REFC:entire atmosphere:anl:
2:100:d=2024070600:TMP:2 m above ground:anl:
3:250:d=2024070600:RH:2 m above ground:anl:
4:400:d=2024070600:UGRD:10 m above ground:anl:
5:600:d=2024070600:VGRD:10 m above ground:anl:
6:800:d=2024070600:GUST:surface:anl:
7:900:d=2024070600:TCDC:entire atmosphere:anl:
"""


def test_analysis_url_and_byte_ranges() -> None:
    t = dt.datetime(2024, 7, 6, 0, tzinfo=UTC)
    assert analysis_url(t).endswith("/hrrr.20240706/conus/hrrr.t00z.wrfsfcf00.grib2")
    r = byte_ranges(IDX)
    assert r["temperature_c"] == (100, 249) and r["v_grid"] == (600, 799)
    assert r["cloud_pct"] == (900, None)  # the last message runs to the end of the file
    with pytest.raises(ValueError, match="0 messages"):
        byte_ranges(IDX.replace("TCDC", "XXXX"))
    with pytest.raises(ValueError, match="2 messages"):
        byte_ranges(IDX + "8:999:d=2024070600:TMP:2 m above ground:anl:\n")


def test_lambert_header() -> None:
    g = lambert_from_grib(grib2_header())
    assert g == LambertGrid(lov_deg=-97.5, latin1_deg=38.5, latin2_deg=38.5, grid_relative_winds=True)
    assert g.cone == pytest.approx(math.sin(math.radians(38.5)))
    assert not lambert_from_grib(grib2_header(flags=0b1010_0000)).grid_relative_winds
    negative = lambert_from_grib(grib2_header(latin1_udeg=(1 << 31) | 30_000_000, latin2_udeg=(1 << 31) | 30_000_000))
    assert negative.latin1_deg == -30.0  # GRIB2 sign-and-magnitude
    with pytest.raises(ValueError, match="3.20"):
        lambert_from_grib(grib2_header(template=20))
    with pytest.raises(ValueError, match="GRIB edition 2"):
        lambert_from_grib(b"not grib at all" * 4)


def test_cone_constant_of_two_parallels() -> None:
    """Snyder (1987, p. 296): standard parallels 33 and 45 N give n = 0.6304777."""
    assert LambertGrid(-96.0, 33.0, 45.0, True).cone == pytest.approx(0.6304777, abs=1e-7)


# ----------------------------------------------------------------------------- winds


def test_rotation_to_true_north() -> None:
    hrrr = LambertGrid(-97.5, 38.5, 38.5, True)
    # on the central meridian grid north is true north
    assert to_earth_relative(1.0, 2.0, -97.5, hrrr) == pytest.approx((1.0, 2.0))
    # at the Lake Fire, 22.5 deg west of it, grid north points 14 deg west of true north
    u, v = to_earth_relative(0.0, 1.0, LAKE[1], hrrr)
    bearing = math.degrees(math.atan2(u, v)) % 360
    assert bearing == pytest.approx((math.sin(math.radians(38.5)) * (LAKE[1] + 97.5)) % 360, abs=1e-6)
    assert 345.0 < bearing < 347.0  # about 14 deg west of true north
    assert math.hypot(u, v) == pytest.approx(1.0)
    earth = LambertGrid(-97.5, 38.5, 38.5, False)
    assert to_earth_relative(1.0, 2.0, LAKE[1], earth) == (1.0, 2.0)


def test_wind_from() -> None:
    assert wind_from(0.0, -5.0) == pytest.approx((5.0, 0.0))  # blowing south: a north wind
    assert wind_from(5.0, 0.0) == pytest.approx((5.0, 270.0))  # blowing east: a west wind
    assert wind_from(0.0, 0.0) == (0.0, 0.0)


def test_bilinear_sampling() -> None:
    pytest.importorskip("rasterio")
    from_origin = pytest.importorskip("rasterio.transform").from_origin
    transform = pytest.importorskip("rasterio.warp").transform
    from wildfire_sim.data.hrrr import sample_bilinear

    x0, y0 = transform("EPSG:4326", "EPSG:5070", [LAKE[1]], [LAKE[0]])
    t = from_origin(x0[0] - 15000, y0[0] + 15000, 3000, 3000)  # 10 x 10 cells of 3 km around the point
    cols, rows = np.meshgrid(np.arange(10), np.arange(10))
    values = 2.0 * cols + 5.0 * rows  # linear: bilinear interpolation is exact
    v = sample_bilinear(Field(values, t, "EPSG:5070"), *LAKE)
    # the point sits at the corner shared by cells (4, 4), (4, 5), (5, 4), (5, 5): fractional centre 4.5
    assert v == pytest.approx(2.0 * 4.5 + 5.0 * 4.5, abs=1e-6)
    with pytest.raises(ValueError, match="outside"):
        sample_bilinear(Field(values, t, "EPSG:5070"), 40.0, -100.0)


# ----------------------------------------------------------------------------- fetching (fake archive)


class FakeArchive:
    """Serves the index and one tagged message per field; a decoder turns the tag into a field."""

    VALUES = {"temperature_c": 40.0, "relative_humidity_pct": 9.0, "u_grid": 0.0, "v_grid": -5.0, "cloud_pct": 30.0}

    def __init__(self, missing: dt.datetime | None = None) -> None:
        self.calls = 0
        self.missing = missing
        self.header = grib2_header()

    def get(self, url: str, byte_range: str | None) -> bytes:
        self.calls += 1
        if self.missing is not None and analysis_url(self.missing) in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), None)
        if url.endswith(".idx"):
            return IDX.encode()
        first = int(byte_range.split("=")[1].split("-")[0]) if byte_range else 0
        name = {100: "temperature_c", 250: "relative_humidity_pct", 400: "u_grid", 600: "v_grid", 900: "cloud_pct"}[
            first
        ]
        return self.header + name.encode()

    def decode(self, message: bytes) -> Field:
        name = message[len(self.header) :].decode()
        values = np.full((4, 4), self.VALUES[name])
        from_origin = pytest.importorskip("rasterio.transform").from_origin
        transform = pytest.importorskip("rasterio.warp").transform
        x0, y0 = transform("EPSG:4326", "EPSG:5070", [LAKE[1]], [LAKE[0]])
        return Field(values, from_origin(x0[0] - 6000, y0[0] + 6000, 3000, 3000), "EPSG:5070")


def test_fetch_hrrr_point_samples_rotates_and_caches(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    archive = FakeArchive()
    start = dt.datetime(2024, 7, 6, 0, tzinfo=UTC)
    series = fetch_hrrr_point(*LAKE, start, 3, cache_root=tmp_path, get=archive.get, decode=archive.decode)
    h = series.hours[0]
    assert [x.time_utc for x in series.hours] == [start + dt.timedelta(hours=i) for i in range(3)]
    assert (h.temperature_c, h.relative_humidity, h.cloud_fraction) == (40.0, pytest.approx(0.09), 0.3)
    # 5 m/s blowing toward grid south: at the Lake Fire that is from 14 deg east of grid north's bearing
    assert h.wind_speed == pytest.approx(5.0)
    assert h.wind_from_deg == pytest.approx((0.0 + math.sin(math.radians(38.5)) * (LAKE[1] + 97.5)) % 360)
    assert archive.calls == 3 * (1 + len(FIELDS))
    again = fetch_hrrr_point(*LAKE, start, 3, cache_root=tmp_path, get=archive.get, decode=archive.decode)
    assert again.hours == series.hours and archive.calls == 3 * (1 + len(FIELDS))
    assert list(series.manifest["files"][start.isoformat()]["byte_ranges"]["cloud_pct"]) == [900, None]
    (series.directory / "hourly.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        fetch_hrrr_point(*LAKE, start, 3, cache_root=tmp_path, get=archive.get, decode=archive.decode)


def test_fetch_hrrr_point_refuses_bad_requests(tmp_path: Path) -> None:
    pytest.importorskip("rasterio")
    start = dt.datetime(2024, 7, 6, 0, tzinfo=UTC)
    archive = FakeArchive(missing=start + dt.timedelta(hours=1))
    with pytest.raises(ValueError, match="no HRRR analysis for 2024-07-06 01:00"):
        fetch_hrrr_point(*LAKE, start, 3, cache_root=tmp_path, get=archive.get, decode=archive.decode)
    with pytest.raises(ValueError, match="on the hour"):
        fetch_hrrr_point(*LAKE, start + dt.timedelta(minutes=30), 2, cache_root=tmp_path, get=archive.get)
    with pytest.raises(ValueError, match="timezone"):
        fetch_hrrr_point(*LAKE, start.replace(tzinfo=None), 2, cache_root=tmp_path, get=archive.get)


# ----------------------------------------------------------------------------- into the weather series


def test_cloud_cover_to_state_of_weather() -> None:
    assert [state_of_weather_from_cloud(c) for c in (0.0, 0.04, 0.1, 0.5, 0.6, 0.9, 0.96, 1.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_hourly_from_hrrr_takes_100h_moisture_from_gridmet() -> None:
    day = DailyWeather(dt.date(2024, 7, 5), 40.0, 21.0, 0.77, 0.10, 2.4, 216.0, 0.105, 321.0)
    hours = [
        HrrrHour(dt.datetime(2024, 7, 5, 22, tzinfo=UTC), 40.2, 0.09, 4.6, 357.2, 0.3),  # 14:00 local solar time
        HrrrHour(dt.datetime(2024, 7, 6, 6, tzinfo=UTC), 22.0, 0.70, 1.0, 360.0, 0.3),  # 22:00 local solar time
    ]
    out = hourly_from_hrrr(hours, [day], *LAKE)
    assert out[0].moisture_100h == 0.105 and out[0].state_of_weather == 1  # 3/10 cloud, daylight
    assert out[1].state_of_weather is None and out[1].wind_direction == 0.0  # night; 360 wraps to 0
    with pytest.raises(ValueError, match="gridMET day"):
        hourly_from_hrrr(hours, [], *LAKE)
    unknown = cast(Any, "era5")  # deliberately not one of the accepted sources
    with pytest.raises(ValueError, match="unknown weather source"):
        hourly_weather(*LAKE, dt.datetime(2024, 7, 5, 22, tzinfo=UTC), 3600, source=unknown)


# ----------------------------------------------------------------------------- live


@pytest.mark.network
def test_live_hrrr_lake_fire(tmp_path: Path) -> None:
    """Two real analyses at the Lake Fire: the afternoon of the ignition was hot and dry."""
    pytest.importorskip("rasterio")
    start = dt.datetime(2024, 7, 5, 22, tzinfo=UTC)
    series = fetch_hrrr_point(*LAKE, start, 2, cache_root=tmp_path)
    for h in series.hours:
        assert 35 < h.temperature_c < 45 and h.relative_humidity < 0.2
        assert 0 <= h.wind_from_deg < 360 and 0 < h.wind_speed < 15
