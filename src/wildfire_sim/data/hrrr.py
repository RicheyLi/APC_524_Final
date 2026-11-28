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

**Wind on the native grid (WP 3.4).** The same UGRD and VGRD messages also give the wind on every
HRRR node of a small window around a fire grid (:func:`fetch_hrrr_wind_field`): the nodes
covering the grid's cell centres plus one node of margin on every side, each node's components
rotated to earth-relative at that node's own longitude (the rotation angle ``n (lon - LoV)``
changes by about 0.02 deg per km east-west), and each cell centre's fractional position in the
window, so that the run can interpolate with numpy alone (:mod:`wildfire_sim.wind_field`). It is
cached beside the point values as ``hrrr-<key>/wind-<grid key>.npz`` with its own manifest
``wind-<grid key>.json``. When neither is cached yet, both are made from one download.

References:
    Dowell, D. C. et al. (2022). The High-Resolution Rapid Refresh (HRRR): an hourly updating
    convection-allowing forecast model. Part I. Weather and Forecasting 37, 1371-1395.
    WMO (2023). Manual on Codes, GRIB edition 2: template 3.30 (Lambert conformal) and flag
    table 3.3.
"""

from __future__ import annotations

import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_sim.data.landfire import DEFAULT_CACHE, optional_import
from wildfire_sim.data.terrain import TargetGrid

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


def to_earth_relative_array(
    u: np.ndarray, v: np.ndarray, lon_deg: np.ndarray, grid: LambertGrid
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`to_earth_relative` element-wise: each node rotated at its own longitude (degrees east).

    Returns float64 east and north components, m/s, of the shape of the inputs.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if not grid.grid_relative_winds:
        return u.copy(), v.copy()
    dlon = (np.asarray(lon_deg, dtype=np.float64) - grid.lov_deg + 180.0) % 360.0 - 180.0
    alpha = np.radians(grid.cone * dlon)
    c, s = np.cos(alpha), np.sin(alpha)
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


def download_messages(
    t: dt.datetime, get: Get, fields: dict[str, str] = FIELDS
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Download the GRIB2 messages of ``fields`` for the analysis hour ``t``, and their provenance."""
    url = analysis_url(t)
    ranges = byte_ranges(get(url + ".idx", None).decode("utf-8"), fields)
    messages = {}
    for name, (first, last) in ranges.items():
        messages[name] = get(url, f"bytes={first}-{'' if last is None else last}")
    return messages, {"url": url, "byte_ranges": ranges}


def _decode_once(decode: Callable[[bytes], Field]) -> Callable[[bytes], Field]:
    """``decode`` that remembers its results, so the point and the window share one decoding."""
    done: dict[int, tuple[bytes, Field]] = {}

    def wrapped(message: bytes) -> Field:
        hit = done.get(id(message))
        if hit is None or hit[0] is not message:
            hit = (message, decode(message))
            done[id(message)] = hit
        return hit[1]

    return wrapped


def fetch_hour(
    t: dt.datetime, lat: float, lon: float, get: Get, decode: Callable[[bytes], Field]
) -> tuple[HrrrHour, dict[str, Any]]:
    """Download and sample one analysis hour; also returns its provenance (URL, byte ranges)."""
    messages, provenance = download_messages(t, get)
    return hour_values(messages, t, lat, lon, decode), provenance


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
    wind_target: TargetGrid | None = None,
) -> HrrrSeries:
    """HRRR analysis values at a point for ``n_hours`` hours from ``start_utc`` (on the hour).

    With ``wind_target`` (a fire grid) and nothing cached yet, the same download also gives that
    grid's wind field on the native HRRR nodes, cached for :func:`fetch_hrrr_wind_field`.

    Raises:
        ValueError: ``start_utc`` is not on the hour or not timezone-aware, an hour's file is
            missing from the archive, or a cached file does not match its hash.
    """
    start = _check_start(start_utc, n_hours)
    directory = Path(cache_root) / hrrr_cache_key(lat, lon, start, n_hours)
    data, manifest_path = directory / "hourly.csv", directory / "manifest.json"
    if data.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(data.read_bytes()).hexdigest() != manifest["sha256"]["hourly.csv"]:
            raise ValueError(f"{directory}: hourly.csv does not match its manifest hash; delete the directory")
        return HrrrSeries(_read(data), directory, manifest)

    fetch = urllib_get if get is None else get
    times = [start + i * HOUR for i in range(n_hours)]
    geometry: list[WindGeometry] = []  # set from the first hour when a wind field is wanted

    def one(t: dt.datetime) -> tuple[HrrrHour, dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
        with _archive_errors(t):
            messages, provenance = download_messages(t, fetch)
        dec = _decode_once(decode)
        hour = hour_values(messages, t, lat, lon, dec)
        if wind_target is None:
            return hour, provenance, None
        if not geometry:
            geometry.append(wind_geometry(dec(messages["u_grid"]), wind_target))
        return hour, provenance, window_wind(messages, geometry[0], dec)

    first = [one(times[0])]  # alone, so the wind window is known before the others run
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = first + list(pool.map(one, times[1:]))
    hours = [h for h, _, _ in results]
    directory.mkdir(parents=True, exist_ok=True)
    if wind_target is not None:
        _write_wind(
            directory,
            wind_target,
            times,
            geometry[0],
            [w for _, _, w in results if w is not None],
            {
                t.isoformat(): {**p, "byte_ranges": {k: p["byte_ranges"][k] for k in WIND_FIELDS}}
                for t, (_, p, _) in zip(times, results, strict=True)
            },
        )
    _write(data, hours)
    manifest = {
        "dataset": "HRRR analysis (wrfsfcf00), NOAA, AWS Open Data",
        "point_wgs84": {"lat": lat, "lon": lon},
        "start_utc": start.isoformat(),
        "n_hours": n_hours,
        "fields": FIELDS,
        "sampling": "bilinear at the point; 10 m wind rotated from grid-relative to true north",
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "files": {t.isoformat(): p for t, (_, p, _) in zip(times, results, strict=True)},
        "sha256": {"hourly.csv": hashlib.sha256(data.read_bytes()).hexdigest()},
        "attribution": "NOAA High-Resolution Rapid Refresh (HRRR), NOAA Open Data Dissemination program",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return HrrrSeries(hours, directory, manifest)


def _check_start(start_utc: dt.datetime, n_hours: int) -> dt.datetime:
    """``start_utc`` in UTC, after checking it is aware and on the hour and ``n_hours >= 1``."""
    if start_utc.tzinfo is None:
        raise ValueError("start_utc must be timezone-aware")
    start = start_utc.astimezone(dt.UTC)
    if (start.minute, start.second, start.microsecond) != (0, 0, 0):
        raise ValueError("start_utc must be on the hour (HRRR analyses are hourly)")
    if n_hours < 1:
        raise ValueError("n_hours must be >= 1")
    return start


@contextlib.contextmanager
def _archive_errors(t: dt.datetime) -> Generator[None]:
    """Turn the archive's 403/404 for a missing hour into a ``ValueError`` naming the hour."""
    try:
        yield
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            raise ValueError(f"no HRRR analysis for {t:%Y-%m-%d %H}:00 UTC in the archive") from None
        raise


# ----------------------------------------------------------------------------- wind on the native grid (WP 3.4)


WIND_FIELDS: dict[str, str] = {"u_grid": FIELDS["u_grid"], "v_grid": FIELDS["v_grid"]}
"""The two fields of the wind window."""

WINDOW_MARGIN_NODES = 1
"""HRRR nodes added on every side of the nodes that surround the grid's cell centres."""


@dataclass(frozen=True)
class WindGeometry:
    """Where a fire grid's cells lie on the HRRR grid.

    Attributes:
        row0, col0: First node row and column of the window in the HRRR raster (as GDAL reads it).
        cell_i, cell_j: ``(H, W)`` float64, fractional window row and column of each cell centre
            (node centres at integers).
        node_lon, node_lat: ``(ny, nx)`` float64, WGS84 longitude and latitude of the window's
            nodes, degrees.
        hrrr_shape: ``(rows, cols)`` of the HRRR raster.
        hrrr_transform: Its affine transform ``(a, b, c, d, e, f)``, m.
        hrrr_crs: Its CRS, WKT.
    """

    row0: int
    col0: int
    cell_i: np.ndarray
    cell_j: np.ndarray
    node_lon: np.ndarray
    node_lat: np.ndarray
    hrrr_shape: tuple[int, int]
    hrrr_transform: tuple[float, ...]
    hrrr_crs: str

    @property
    def window_shape(self) -> tuple[int, int]:
        """``(ny, nx)`` of the node window."""
        return (int(self.node_lon.shape[0]), int(self.node_lon.shape[1]))

    @property
    def slices(self) -> tuple[slice, slice]:
        """Row and column slices of the window in the HRRR raster."""
        ny, nx = self.window_shape
        return slice(self.row0, self.row0 + ny), slice(self.col0, self.col0 + nx)


def _transform_tuple(transform: Any) -> tuple[float, ...]:
    return tuple(float(transform[i]) for i in range(6))


def cell_positions(field: Field, target: TargetGrid) -> tuple[np.ndarray, np.ndarray]:
    """Fractional ``(row, col)`` of every target cell centre on the field's raster.

    Cell centres are projected from the target's CRS to the field's (PROJ, through rasterio);
    node centres of the field are at integers, as in :func:`sample_bilinear`.

    Returns:
        Two ``(H, W)`` float64 arrays.
    """
    transform = optional_import("rasterio.warp").transform
    h, w = target.shape
    a, b, c, d, e, f = _transform_tuple(target.transform)
    jj, ii = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
    xs = c + a * jj + b * ii
    ys = f + d * jj + e * ii
    hx, hy = transform(target.crs, field.crs, xs.ravel(), ys.ravel())
    hx, hy = np.asarray(hx, dtype=np.float64), np.asarray(hy, dtype=np.float64)
    inv = ~field.transform
    col = inv.a * hx + inv.b * hy + inv.c - 0.5
    row = inv.d * hx + inv.e * hy + inv.f - 0.5
    return row.reshape(h, w), col.reshape(h, w)


def wind_geometry(field: Field, target: TargetGrid, margin: int = WINDOW_MARGIN_NODES) -> WindGeometry:
    """The node window around a fire grid and the cells' positions in it.

    The window spans the nodes around every cell centre (``floor`` of the smallest to ``ceil``
    of the largest fractional row and column) plus ``margin`` nodes on every side.

    Raises:
        ValueError: The window does not fit inside the HRRR raster.
    """
    transform = optional_import("rasterio.warp").transform
    rows, cols = cell_positions(field, target)
    r0 = math.floor(float(rows.min())) - margin
    c0 = math.floor(float(cols.min())) - margin
    r1 = math.ceil(float(rows.max())) + margin
    c1 = math.ceil(float(cols.max())) + margin
    hh, ww = field.values.shape
    if r0 < 0 or c0 < 0 or r1 > hh - 1 or c1 > ww - 1:
        raise ValueError("the fire grid (plus the margin) is not inside the HRRR grid")
    nr, nc = np.meshgrid(np.arange(r0, r1 + 1, dtype=np.float64) + 0.5, np.arange(c0, c1 + 1, dtype=np.float64) + 0.5)
    nr, nc = nr.T, nc.T  # (ny, nx)
    t = field.transform
    xs = t.c + t.a * nc + t.b * nr
    ys = t.f + t.d * nc + t.e * nr
    lon, lat = transform(field.crs, "EPSG:4326", xs.ravel(), ys.ravel())
    crs = optional_import("rasterio.crs").CRS.from_user_input(field.crs)
    return WindGeometry(
        row0=r0,
        col0=c0,
        cell_i=rows - r0,
        cell_j=cols - c0,
        node_lon=np.asarray(lon, dtype=np.float64).reshape(nr.shape),
        node_lat=np.asarray(lat, dtype=np.float64).reshape(nr.shape),
        hrrr_shape=(int(hh), int(ww)),
        hrrr_transform=_transform_tuple(t),
        hrrr_crs=crs.to_wkt(),
    )


def window_wind(
    messages: dict[str, bytes], geometry: WindGeometry, decode: Callable[[bytes], Field] = decode_message
) -> tuple[np.ndarray, np.ndarray]:
    """Earth-relative 10 m wind on the window's nodes from one hour's UGRD and VGRD messages.

    Each node's grid-relative components are rotated at its own longitude
    (:func:`to_earth_relative_array`), with the projection read from the message's header.

    Returns:
        ``(u, v)``, east and north components, ``(ny, nx)`` float64, m/s.

    Raises:
        ValueError: The hour's raster is not the one the window was cut from, or has missing values.
    """
    fu, fv = decode(messages["u_grid"]), decode(messages["v_grid"])
    for f in (fu, fv):
        if f.values.shape != geometry.hrrr_shape or _transform_tuple(f.transform) != geometry.hrrr_transform:
            raise ValueError("the HRRR raster changed between hours; the wind window no longer applies")
    rows, cols = geometry.slices
    u_grid, v_grid = fu.values[rows, cols], fv.values[rows, cols]
    if not (np.all(np.isfinite(u_grid)) and np.all(np.isfinite(v_grid))):
        raise ValueError("the HRRR wind has missing values in the window")
    return to_earth_relative_array(u_grid, v_grid, geometry.node_lon, lambert_from_grib(messages["u_grid"]))


@dataclass(frozen=True)
class HrrrWindField:
    """Cached hourly HRRR wind on the node window of a fire grid.

    Attributes:
        times_utc: Valid time of each hour.
        u, v: ``(T, ny, nx)`` float32, earth-relative east and north 10 m wind, m/s.
        geometry: The window and the cells' positions in it.
        path: The cached ``wind-<grid key>.npz``.
        manifest: The contents of its manifest.
    """

    times_utc: list[dt.datetime]
    u: np.ndarray
    v: np.ndarray
    geometry: WindGeometry
    path: Path
    manifest: dict[str, Any]


def wind_grid_key(target: TargetGrid) -> str:
    """16 hex digits identifying a fire grid (CRS, transform, shape) for the wind cache."""
    crs = optional_import("rasterio.crs").CRS.from_user_input(target.crs).to_wkt()
    canon = json.dumps(
        {"crs": crs, "transform": [repr(x) for x in _transform_tuple(target.transform)], "shape": list(target.shape)},
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _write_wind(
    directory: Path,
    target: TargetGrid,
    times: Sequence[dt.datetime],
    geometry: WindGeometry,
    winds: Sequence[tuple[np.ndarray, np.ndarray]],
    files: dict[str, Any],
) -> None:
    key = wind_grid_key(target)
    data = directory / f"wind-{key}.npz"
    np.savez_compressed(
        data,
        times_utc_s=np.array([int(t.timestamp()) for t in times], dtype=np.int64),
        u=np.stack([u for u, _ in winds]).astype(np.float32),
        v=np.stack([v for _, v in winds]).astype(np.float32),
        cell_i=geometry.cell_i.astype(np.float32),
        cell_j=geometry.cell_j.astype(np.float32),
        node_lon=geometry.node_lon,
        node_lat=geometry.node_lat,
    )
    ny, nx = geometry.window_shape
    manifest = {
        "dataset": "HRRR analysis (wrfsfcf00), NOAA, AWS Open Data: 10 m wind on the native 3 km grid",
        "grid_key": key,
        "fire_grid": {
            "crs": optional_import("rasterio.crs").CRS.from_user_input(target.crs).to_wkt(),
            "transform": list(_transform_tuple(target.transform)),
            "shape": list(target.shape),
        },
        "window": {
            "row0": geometry.row0,
            "col0": geometry.col0,
            "ny": ny,
            "nx": nx,
            "margin_nodes": WINDOW_MARGIN_NODES,
            "lon_range": [float(geometry.node_lon.min()), float(geometry.node_lon.max())],
            "lat_range": [float(geometry.node_lat.min()), float(geometry.node_lat.max())],
        },
        "hrrr_grid": {
            "crs": geometry.hrrr_crs,
            "transform": list(geometry.hrrr_transform),
            "shape": list(geometry.hrrr_shape),
        },
        "start_utc": times[0].isoformat(),
        "n_hours": len(times),
        "fields": WIND_FIELDS,
        "processing": "grid-relative u, v rotated to earth-relative at each node's longitude; float32",
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "files": files,
        "sha256": {data.name: hashlib.sha256(data.read_bytes()).hexdigest()},
        "attribution": "NOAA High-Resolution Rapid Refresh (HRRR), NOAA Open Data Dissemination program",
    }
    (directory / f"wind-{key}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _read_wind(data: Path, manifest_path: Path) -> HrrrWindField:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest["sha256"][data.name]:
        raise ValueError(f"{data}: does not match its manifest hash; delete it and {manifest_path.name}")
    window, hrrr_grid = manifest["window"], manifest["hrrr_grid"]
    with np.load(data, allow_pickle=False) as z:
        times = [dt.datetime.fromtimestamp(int(s), dt.UTC) for s in z["times_utc_s"]]
        geometry = WindGeometry(
            row0=int(window["row0"]),
            col0=int(window["col0"]),
            cell_i=np.asarray(z["cell_i"], dtype=np.float64),
            cell_j=np.asarray(z["cell_j"], dtype=np.float64),
            node_lon=np.asarray(z["node_lon"]),
            node_lat=np.asarray(z["node_lat"]),
            hrrr_shape=(int(hrrr_grid["shape"][0]), int(hrrr_grid["shape"][1])),
            hrrr_transform=tuple(float(x) for x in hrrr_grid["transform"]),
            hrrr_crs=str(hrrr_grid["crs"]),
        )
        return HrrrWindField(times, np.asarray(z["u"]), np.asarray(z["v"]), geometry, data, manifest)


def fetch_hrrr_wind_field(
    lat: float,
    lon: float,
    start_utc: dt.datetime,
    n_hours: int,
    target: TargetGrid,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
    get: Get | None = None,
    decode: Callable[[bytes], Field] = decode_message,
    workers: int = 8,
) -> HrrrWindField:
    """Hourly HRRR 10 m wind on the native nodes around a fire grid (no downscaling).

    Cached in the point request's directory (``lat``, ``lon``, ``start_utc``, ``n_hours`` as in
    :func:`fetch_hrrr_point`, so a region's point values and its field sit together). When the
    point values are not cached either, one download makes both; otherwise only the UGRD and
    VGRD messages are downloaded.

    Args:
        lat, lon: The region's weather point (grid centroid), degrees.
        start_utc: First hour, timezone-aware, on the hour.
        n_hours: Number of hours.
        target: The fire grid: CRS, affine transform of its top-left corner and ``(H, W)``.
        cache_root: Directory holding ``hrrr-<key>/``.
        get: Download function; None uses urllib.
        decode: GRIB2 decoder.
        workers: Parallel downloads.

    Raises:
        ValueError: As :func:`fetch_hrrr_point`; the grid is not inside the HRRR domain; a
            cached file does not match its hash.
    """
    start = _check_start(start_utc, n_hours)
    directory = Path(cache_root) / hrrr_cache_key(lat, lon, start, n_hours)
    key = wind_grid_key(target)
    data, manifest_path = directory / f"wind-{key}.npz", directory / f"wind-{key}.json"
    if data.exists() and manifest_path.exists():
        return _read_wind(data, manifest_path)
    if not (directory / "hourly.csv").exists():
        fetch_hrrr_point(
            lat, lon, start, n_hours, cache_root=cache_root, get=get, decode=decode, workers=workers, wind_target=target
        )
        return _read_wind(data, manifest_path)

    fetch = urllib_get if get is None else get
    times = [start + i * HOUR for i in range(n_hours)]
    geometry: list[WindGeometry] = []

    def one(t: dt.datetime) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        with _archive_errors(t):
            messages, provenance = download_messages(t, fetch, WIND_FIELDS)
        dec = _decode_once(decode)
        if not geometry:
            geometry.append(wind_geometry(dec(messages["u_grid"]), target))
        return window_wind(messages, geometry[0], dec), provenance

    first = [one(times[0])]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = first + list(pool.map(one, times[1:]))
    _write_wind(
        directory,
        target,
        times,
        geometry[0],
        [w for w, _ in results],
        {t.isoformat(): p for t, (_, p) in zip(times, results, strict=True)},
    )
    return _read_wind(data, manifest_path)
