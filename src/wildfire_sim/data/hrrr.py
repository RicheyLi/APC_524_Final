"""Hourly weather at a point from the HRRR analysis (WP 2.4 follow-up, docs/plan.md section 13).

The High-Resolution Rapid Refresh (HRRR; Dowell et al. 2022) is NOAA's 3 km hourly model of the
contiguous US; its hourly *analysis* (forecast hour 0, ``wrfsfcf00``) is the best estimate of
each hour's weather. The plan uses it for both validation fires because its archive has every
hour of their weather windows, while gridMET's daily means cannot represent winds such as the
Santa Ana that drove fire B.

**Reading only what is needed.** Each hourly file on NOAA's AWS archive is about 130 MB, but a
published index (``.idx``) gives each field's byte offset, so only five fields are downloaded
by HTTP byte range, 1-2 MB each: 2 m temperature (``TMP``) and relative humidity (``RH``), the
10 m wind components (``UGRD``, ``VGRD``) and total cloud cover (``TCDC``). They are decoded with
the GRIB2 driver of the GDAL that ``rasterio`` ships (no further dependency) and sampled at the
grid centroid by bilinear interpolation on the 3 km grid.

**Winds are grid-relative.** The HRRR's Lambert conformal grid is centred on 97.5 W, and its
GRIB2 files flag the wind components as relative to the grid's axes (flag table 3.3, bit 5), so
at 120 W they are about 14 deg off true north. They are rotated by the standard formula for a
Lambert grid: the angle ``n (lon - LoV)``, with ``n`` the cone constant from the standard
parallels; the projection parameters and the flag are read from each file's own header
(:func:`lambert_from_grib`), not assumed.

Each region's hourly values are cached as ``hrrr-<key>/hourly.csv`` with a manifest (URLs, byte
ranges, SHA-256). The archive is public; no account is needed.

References:
    Dowell, D. C. et al. (2022). The High-Resolution Rapid Refresh (HRRR): an hourly updating
    convection-allowing forecast model. Part I. Weather and Forecasting 37, 1371-1395.
    WMO (2023). Manual on Codes, GRIB edition 2: template 3.30 (Lambert conformal) and flag
    table 3.3.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_sim.data.landfire import DEFAULT_CACHE, optional_import

HRRR_BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
"""NOAA Open Data Dissemination archive of the HRRR on AWS (public)."""

FIELDS: dict[str, str] = {
    "temperature_c": ":TMP:2 m above ground:",
    "relative_humidity_pct": ":RH:2 m above ground:",
    "u_grid": ":UGRD:10 m above ground:",
    "v_grid": ":VGRD:10 m above ground:",
    "cloud_pct": ":TCDC:entire atmosphere:",
}
"""Field -> the record text that identifies it in the ``.idx`` file."""

HOUR = dt.timedelta(hours=1)

Get = Callable[[str, str | None], bytes]
"""``get(url, byte_range)`` -> the response body; ``byte_range`` like ``"bytes=0-99"`` or None."""


def analysis_url(t: dt.datetime) -> str:
    """URL of the CONUS surface analysis valid at the UTC hour ``t``."""
    t = t.astimezone(dt.UTC)
    return f"{HRRR_BUCKET}/hrrr.{t:%Y%m%d}/conus/hrrr.t{t:%H}z.wrfsfcf00.grib2"


def byte_ranges(idx_text: str, fields: dict[str, str] = FIELDS) -> dict[str, tuple[int, int | None]]:
    """Each field's ``(first, last)`` byte in the GRIB2 file (last None: to the end), from its index.

    Index lines read ``n:offset:d=YYYYMMDDHH:VAR:level:anl:``; a message ends where the next begins.

    Raises:
        ValueError: A field is missing or matches more than one message.
    """
    lines = [ln for ln in idx_text.strip().splitlines() if ln.strip()]
    offsets = [int(ln.split(":")[1]) for ln in lines]
    out: dict[str, tuple[int, int | None]] = {}
    for name, pattern in fields.items():
        hits = [i for i, ln in enumerate(lines) if pattern in ln]
        if len(hits) != 1:
            raise ValueError(f"HRRR index: {len(hits)} messages match {pattern!r}")
        i = hits[0]
        out[name] = (offsets[i], offsets[i + 1] - 1 if i + 1 < len(offsets) else None)
    return out


# ----------------------------------------------------------------------------- grid and winds


@dataclass(frozen=True)
class LambertGrid:
    """What a GRIB2 Lambert conformal grid says about its winds.

    Attributes:
        lov_deg: Orientation longitude (where grid north is true north), degrees east in [-180, 180).
        latin1_deg, latin2_deg: Standard parallels, degrees.
        grid_relative_winds: Whether u and v are relative to the grid's axes (flag table 3.3, bit 5).
    """

    lov_deg: float
    latin1_deg: float
    latin2_deg: float
    grid_relative_winds: bool

    @property
    def cone(self) -> float:
        """Cone constant ``n`` (sin of the standard parallel when there is one)."""
        p1, p2 = math.radians(self.latin1_deg), math.radians(self.latin2_deg)
        if math.isclose(p1, p2):
            return math.sin(p1)
        return math.log(math.cos(p1) / math.cos(p2)) / math.log(
            math.tan(math.pi / 4 + p2 / 2) / math.tan(math.pi / 4 + p1 / 2)
        )


def _signed(octets: bytes) -> int:
    """A GRIB2 signed integer (sign and magnitude, most significant bit the sign)."""
    v = int.from_bytes(octets, "big")
    top = 1 << (8 * len(octets) - 1)
    return -(v & (top - 1)) if v & top else v


def lambert_from_grib(message: bytes) -> LambertGrid:
    """Read grid template 3.30 (Lambert conformal) from a GRIB2 message's section 3.

    Raises:
        ValueError: Not GRIB2, or section 3 is not template 3.30.
    """
    if message[:4] != b"GRIB" or message[7] != 2:
        raise ValueError("not a GRIB edition 2 message")
    pos = 16  # section 0 is 16 octets
    while pos + 5 <= len(message):
        length = int.from_bytes(message[pos : pos + 4], "big")
        if length <= 0:
            break
        if message[pos + 4] == 3:
            template = int.from_bytes(message[pos + 12 : pos + 14], "big")
            if template != 30:
                raise ValueError(f"grid template 3.{template}, expected 3.30 (Lambert conformal)")

            sec = message[pos : pos + length]  # octet k of section 3 is sec[k - 1]
            lov = _signed(sec[51:55]) / 1e6  # octets 52-55
            return LambertGrid(
                lov_deg=(lov + 180.0) % 360.0 - 180.0,
                latin1_deg=_signed(sec[65:69]) / 1e6,  # octets 66-69
                latin2_deg=_signed(sec[69:73]) / 1e6,  # octets 70-73
                grid_relative_winds=bool(sec[46] & 0b1000),  # octet 47, flag table 3.3 bit 5
            )
        pos += length
    raise ValueError("no section 3 in the GRIB2 message")


def to_earth_relative(u: float, v: float, lon_deg: float, grid: LambertGrid) -> tuple[float, float]:
    """Rotate grid-relative wind components to east/north components at longitude ``lon_deg``.

    ``alpha = n (lon - LoV)``; ``u_e = cos(alpha) u + sin(alpha) v``, ``v_e = -sin(alpha) u +
    cos(alpha) v`` (the rotation NCEP applies to its Lambert grids). Unchanged if the grid says
    its winds are already earth-relative.
    """
    if not grid.grid_relative_winds:
        return u, v
    dlon = (lon_deg - grid.lov_deg + 180.0) % 360.0 - 180.0
    alpha = math.radians(grid.cone * dlon)
    c, s = math.cos(alpha), math.sin(alpha)
    return c * u + s * v, -s * u + c * v


def wind_from(u: float, v: float) -> tuple[float, float]:
    """Speed (m/s) and the bearing the wind blows from (degrees in [0, 360)) of east/north components."""
    speed = math.hypot(u, v)
    return speed, (math.degrees(math.atan2(-u, -v)) % 360.0 if speed > 0 else 0.0)


# ----------------------------------------------------------------------------- decoding and sampling


@dataclass(frozen=True)
class Field:
    """A decoded 2-D field: values, the affine transform of its cells and its CRS."""

    values: np.ndarray
    transform: Any
    crs: Any


def decode_message(message: bytes) -> Field:
    """Decode one GRIB2 message with GDAL's GRIB driver (through ``rasterio``)."""
    memfile = optional_import("rasterio.io").MemoryFile
    with memfile(message) as mem, mem.open() as src:
        return Field(src.read(1, masked=True).filled(np.nan).astype(np.float64), src.transform, src.crs)


def sample_bilinear(field: Field, lat: float, lon: float) -> float:
    """Bilinear interpolation of a field at a WGS84 point (between the four nearest cell centres).

    Raises:
        ValueError: The point is outside the field.
    """
    transform = optional_import("rasterio.warp").transform
    xs, ys = transform("EPSG:4326", field.crs, [lon], [lat])
    inv = ~field.transform
    col = inv.a * xs[0] + inv.b * ys[0] + inv.c - 0.5  # in units of cells, centres at integers
    row = inv.d * xs[0] + inv.e * ys[0] + inv.f - 0.5
    r0, c0 = math.floor(row), math.floor(col)
    h, w = field.values.shape
    if not (0 <= r0 < h - 1 and 0 <= c0 < w - 1):
        raise ValueError(f"({lat}, {lon}) lies outside the HRRR grid")
    fr, fc = row - r0, col - c0
    v = field.values[r0 : r0 + 2, c0 : c0 + 2]
    return float(v[0, 0] * (1 - fr) * (1 - fc) + v[0, 1] * (1 - fr) * fc + v[1, 0] * fr * (1 - fc) + v[1, 1] * fr * fc)


# ----------------------------------------------------------------------------- hourly values


@dataclass(frozen=True)
class HrrrHour:
    """HRRR analysis at a point for one hour.

    Attributes:
        time_utc: Valid time (the hour).
        temperature_c: 2 m temperature, deg C.
        relative_humidity: 2 m relative humidity, fraction in [0, 1].
        wind_speed: 10 m wind speed, m/s.
        wind_from_deg: Bearing the 10 m wind blows from, degrees (true north).
        cloud_fraction: Total cloud cover, fraction in [0, 1].
    """

    time_utc: dt.datetime
    temperature_c: float
    relative_humidity: float
    wind_speed: float
    wind_from_deg: float
    cloud_fraction: float


def hour_values(
    messages: dict[str, bytes],
    t: dt.datetime,
    lat: float,
    lon: float,
    decode: Callable[[bytes], Field] = decode_message,
) -> HrrrHour:
    """One hour's values at a point from its five GRIB2 messages (:data:`FIELDS`)."""
    sampled = {name: sample_bilinear(decode(messages[name]), lat, lon) for name in FIELDS}
    grid = lambert_from_grib(messages["u_grid"])
    u, v = to_earth_relative(sampled["u_grid"], sampled["v_grid"], lon, grid)
    speed, direction = wind_from(u, v)
    return HrrrHour(
        time_utc=t,
        temperature_c=sampled["temperature_c"],
        relative_humidity=min(1.0, max(0.0, sampled["relative_humidity_pct"] / 100.0)),
        wind_speed=speed,
        wind_from_deg=direction,
        cloud_fraction=min(1.0, max(0.0, sampled["cloud_pct"] / 100.0)),
    )


def urllib_get(url: str, byte_range: str | None, attempts: int = 4, first_wait_s: float = 2.0) -> bytes:
    """GET with an optional ``Range`` header, retried with backoff on server errors."""
    headers = {"User-Agent": "wildfire-sim (research model)"}
    if byte_range:
        headers["Range"] = byte_range
    wait = first_wait_s
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(wait)
        wait *= 2
    raise AssertionError("unreachable")


def fetch_hour(
    t: dt.datetime, lat: float, lon: float, get: Get, decode: Callable[[bytes], Field]
) -> tuple[HrrrHour, dict[str, Any]]:
    """Download and sample one analysis hour; also returns its provenance (URL, byte ranges)."""
    url = analysis_url(t)
    ranges = byte_ranges(get(url + ".idx", None).decode("utf-8"))
    messages = {}
    for name, (first, last) in ranges.items():
        messages[name] = get(url, f"bytes={first}-{'' if last is None else last}")
    return hour_values(messages, t, lat, lon, decode), {"url": url, "byte_ranges": ranges}


_COLUMNS = ("time_utc", "temperature_c", "relative_humidity", "wind_speed", "wind_from_deg", "cloud_fraction")


@dataclass(frozen=True)
class HrrrSeries:
    """Cached hourly HRRR values at a point.

    Attributes:
        hours: One record per hour, in time order.
        directory: ``<cache root>/hrrr-<key>/``.
        manifest: The contents of ``manifest.json``.
    """

    hours: list[HrrrHour]
    directory: Path
    manifest: dict[str, Any]


def hrrr_cache_key(lat: float, lon: float, start: dt.datetime, n_hours: int) -> str:
    """``hrrr-`` + 16 hex digits identifying a (point, hours) request."""
    canon = json.dumps(
        {"lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "start": start.astimezone(dt.UTC).isoformat(), "n": n_hours},
        sort_keys=True,
    )
    return "hrrr-" + hashlib.sha256(canon.encode()).hexdigest()[:16]


def _write(path: Path, hours: Sequence[HrrrHour]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(_COLUMNS)
        for h in hours:
            w.writerow([h.time_utc.isoformat(), *(repr(getattr(h, c)) for c in _COLUMNS[1:])])


def _read(path: Path) -> list[HrrrHour]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            HrrrHour(
                time_utc=dt.datetime.fromisoformat(r["time_utc"]),
                temperature_c=float(r["temperature_c"]),
                relative_humidity=float(r["relative_humidity"]),
                wind_speed=float(r["wind_speed"]),
                wind_from_deg=float(r["wind_from_deg"]),
                cloud_fraction=float(r["cloud_fraction"]),
            )
            for r in csv.DictReader(fh)
        ]


def fetch_hrrr_point(
    lat: float,
    lon: float,
    start_utc: dt.datetime,
    n_hours: int,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
    get: Get | None = None,
    decode: Callable[[bytes], Field] = decode_message,
    workers: int = 8,
) -> HrrrSeries:
    """HRRR analysis values at a point for ``n_hours`` hours from ``start_utc`` (on the hour).

    Raises:
        ValueError: ``start_utc`` is not on the hour or not timezone-aware, an hour's file is
            missing from the archive, or a cached file does not match its hash.
    """
    if start_utc.tzinfo is None:
        raise ValueError("start_utc must be timezone-aware")
    start = start_utc.astimezone(dt.UTC)
    if (start.minute, start.second, start.microsecond) != (0, 0, 0):
        raise ValueError("start_utc must be on the hour (HRRR analyses are hourly)")
    if n_hours < 1:
        raise ValueError("n_hours must be >= 1")
    directory = Path(cache_root) / hrrr_cache_key(lat, lon, start, n_hours)
    data, manifest_path = directory / "hourly.csv", directory / "manifest.json"
    if data.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(data.read_bytes()).hexdigest() != manifest["sha256"]["hourly.csv"]:
            raise ValueError(f"{directory}: hourly.csv does not match its manifest hash; delete the directory")
        return HrrrSeries(_read(data), directory, manifest)

    fetch = urllib_get if get is None else get
    times = [start + i * HOUR for i in range(n_hours)]

    def one(t: dt.datetime) -> tuple[HrrrHour, dict[str, Any]]:
        try:
            return fetch_hour(t, lat, lon, fetch, decode)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                raise ValueError(f"no HRRR analysis for {t:%Y-%m-%d %H}:00 UTC in the archive") from None
            raise

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, times))
    hours = [h for h, _ in results]
    directory.mkdir(parents=True, exist_ok=True)
    _write(data, hours)
    manifest = {
        "dataset": "HRRR analysis (wrfsfcf00), NOAA, AWS Open Data",
        "point_wgs84": {"lat": lat, "lon": lon},
        "start_utc": start.isoformat(),
        "n_hours": n_hours,
        "fields": FIELDS,
        "sampling": "bilinear at the point; 10 m wind rotated from grid-relative to true north",
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "files": {t.isoformat(): p for t, (_, p) in zip(times, results, strict=True)},
        "sha256": {"hourly.csv": hashlib.sha256(data.read_bytes()).hexdigest()},
        "attribution": "NOAA High-Resolution Rapid Refresh (HRRR), NOAA Open Data Dissemination program",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return HrrrSeries(hours, directory, manifest)
