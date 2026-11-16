"""Observations of a validation fire: ignition, perimeters and satellite detections (WP 2.7).

Calibration and validation (docs/plan.md section 7) compare simulated fires with what was
observed. This module gets the observations of a region (``configs/regions.yaml``) and puts
them on its grid:

* **Ignition** (:func:`fetch_ignition`): time and point of origin from the incident record in
  IRWIN, read through NIFC's WFIGS Incident Locations service: ``FireDiscoveryDateTime`` and
  the record's point geometry (the point of origin); the initially reported position is kept
  too.
* **Final perimeter**: MTBS, as in WP 2.6 (:func:`wildfire_sim.data.cache.fetch_mtbs_perimeter`).
* **Daily perimeters** (:func:`fetch_daily_perimeters`): the incident's mapped perimeters (mostly
  from infrared flights) in NIFC's Operational Data Archive, which keeps every version of the
  incident's "Wildfire Daily Fire Perimeter" with the time it became current
  (``GDB_FROM_DATE``) and was replaced (``GDB_TO_DATE``). :meth:`FireObservations.observed_burn_mask`
  with ``until`` takes every cell inside some approved perimeter posted by then: those had burned
  (cells outside may have burned and not yet been mapped). It does not use only the version
  current at ``until``, because the archive sometimes shows a shrunken version for a few hours
  (the Lake Fire's 4,674-acre perimeter was replaced by a 1-acre placeholder on 6 July), and
  burned ground stays burned.
* **VIIRS detections** (:func:`fetch_viirs`): the 375 m active-fire product (VNP14IMG, VJ114IMG,
  VJ214IMG; Schroeder et al. 2014) of Suomi NPP, NOAA-20 and NOAA-21. Its *fire mask* gives
  every pixel of every overpass a class (``legend`` attribute of the files: 3 water, 4 cloud,
  5 clear land, 7-9 fire of low, nominal and high confidence, 0-2 and 6 not usable), and the
  matching geolocation product (VNP03IMG, VJ103IMG, VJ203IMG) the pixel positions and scan
  times. Overpasses are found through NASA's CMR catalogue; the fire mask (about 2 MB) is
  downloaded whole, the geolocation (about 145 MB) only in the rows that cover the region,
  read by HTTP byte ranges. Both need a NASA Earthdata token (:data:`TOKEN_ENV`, or the file
  :data:`TOKEN_FILE`), which is sent only to NASA's data servers and never stored.

**Detections and their time interval** (:func:`first_detections`). A detection does not say when
the fire arrived, only that it had arrived by the detecting overpass. Each overpass's classes
are put on the grid (every cell takes the class of the nearest pixel within
:data:`VIEW_RADIUS_M`, else "not viewed"). A grid cell holding the centre of a fire pixel that
no earlier overpass saw burning is a first detection; its ``t_clear`` is the latest earlier
overpass that saw the cell clear (water or clear land: no cloud, no fire), or the ignition
time if none did or that pass came before the ignition. Cloud, glint, bowtie and unclassified
views never count as clear, so a place detected after a cloudy pass keeps the clear view
before it. Times are in seconds from the region's ``weather_start_utc`` (the scenario's
``start_time_utc``, WP 1.14).

References:
    Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I. A. (2014). The New VIIRS 375 m active fire
    detection data product: algorithm description and initial assessment. Remote Sens. Environ.
    143, 85-96.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from wildfire_sim.data.cache import Perimeter, Region, fetch_mtbs_perimeter, perimeter_mask
from wildfire_sim.data.landfire import DEFAULT_CACHE, UrllibHttp, optional_import
from wildfire_sim.data.terrain import TargetGrid
from wildfire_sim.grid import Grid
from wildfire_sim.types import BoolArray

NIFC_SERVICES = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
"""NIFC's ArcGIS Online organisation (WFIGS and the Operational Data Archive)."""

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
"""NASA Common Metadata Repository granule search (public)."""

VIIRS_PRODUCTS: dict[str, tuple[str, str]] = {
    "VNP14IMG": ("Suomi NPP", "VNP03IMG"),
    "VJ114IMG": ("NOAA-20", "VJ103IMG"),
    "VJ214IMG": ("NOAA-21", "VJ203IMG"),
}
"""375 m active-fire product -> (satellite, geolocation product)."""

TOKEN_ENV = "EARTHDATA_TOKEN"
"""Environment variable holding a NASA Earthdata bearer token."""

TOKEN_FILE = Path("data") / ".earthdata_token"
"""File read when :data:`TOKEN_ENV` is not set (git-ignored)."""

# fire mask classes (the products' own `legend` attribute)
NOT_PROCESSED, BOWTIE, GLINT, WATER, CLOUD, CLEAR_LAND, UNCLASSIFIED = 0, 1, 2, 3, 4, 5, 6
FIRE_LOW, FIRE_NOMINAL, FIRE_HIGH = 7, 8, 9
NOT_VIEWED = 255
"""Our class for a cell no pixel of an overpass covers."""

VIEW_RADIUS_M = 600.0
"""A cell takes the class of the nearest pixel within this distance, else it was not viewed.
VIIRS I-band pixels grow from 375 m at nadir to about 800 m at the swath edge, so inside the
swath the nearest pixel centre is at most about half a pixel diagonal (under 570 m) away."""

LINES_PER_SCAN = 32
"""VIIRS I-band detector rows per scan; the geolocation file is chunked by scan."""

_GUID = re.compile(r"^\{[0-9A-Fa-f-]{36}\}$")

# Leap seconds after 1993-01-01 (effective dates); VIIRS scan times count TAI seconds since then.
_LEAP_SECONDS = (
    dt.datetime(1993, 7, 1, tzinfo=dt.UTC),
    dt.datetime(1994, 7, 1, tzinfo=dt.UTC),
    dt.datetime(1996, 1, 1, tzinfo=dt.UTC),
    dt.datetime(1997, 7, 1, tzinfo=dt.UTC),
    dt.datetime(1999, 1, 1, tzinfo=dt.UTC),
    dt.datetime(2006, 1, 1, tzinfo=dt.UTC),
    dt.datetime(2009, 1, 1, tzinfo=dt.UTC),
    dt.datetime(2012, 7, 1, tzinfo=dt.UTC),
    dt.datetime(2015, 7, 1, tzinfo=dt.UTC),
    dt.datetime(2017, 1, 1, tzinfo=dt.UTC),
)
_TAI93 = dt.datetime(1993, 1, 1, tzinfo=dt.UTC)


def tai93_to_utc(seconds: float) -> dt.datetime:
    """UTC time of a VIIRS scan time (TAI seconds since 1993-01-01, so leap seconds included)."""
    naive = _TAI93 + dt.timedelta(seconds=float(seconds))
    utc = naive
    for _ in range(2):  # the count depends on the UTC time itself; two passes settle it
        utc = naive - dt.timedelta(seconds=sum(1 for d in _LEAP_SECONDS if utc >= d))
    return utc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ms(value: Any) -> dt.datetime | None:
    return None if value is None else dt.datetime.fromtimestamp(float(value) / 1000.0, dt.UTC)


# ----------------------------------------------------------------------------- grid georeference


def target_of(grid: Grid) -> TargetGrid:
    """The grid's CRS, transform and shape from its metadata (written by WPs 2.3 and 2.6).

    Raises:
        ValueError: The grid has no ``crs`` or ``transform`` metadata.
    """
    if "crs" not in grid.metadata or "transform" not in grid.metadata:
        raise ValueError("the grid has no crs/transform metadata (build it with scripts/download_data.py)")
    affine = optional_import("affine").Affine
    t = [float(v) for v in grid.metadata["transform"].split()]
    return TargetGrid(grid.metadata["crs"], affine(*t), grid.shape)


def latlon_to_cell(
    target: TargetGrid, lat: Sequence[float] | np.ndarray, lon: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column (int, may lie outside the grid) of WGS84 points."""
    transform = optional_import("rasterio.warp").transform
    xs, ys = transform("EPSG:4326", target.crs, list(map(float, lon)), list(map(float, lat)))
    inv = ~target.transform
    x, y = np.asarray(xs), np.asarray(ys)
    cols = inv.a * x + inv.b * y + inv.c
    rows = inv.d * x + inv.e * y + inv.f
    return np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)


# ----------------------------------------------------------------------------- ignition


@dataclass(frozen=True)
class Ignition:
    """Ignition of a fire from its incident record.

    Attributes:
        time_utc: Discovery time (``FireDiscoveryDateTime``).
        lat, lon: Point of origin (the record's point geometry), WGS84.
        initial_lat, initial_lon: Initially reported position, WGS84.
        unique_fire_id: IRWIN unique fire identifier, e.g. ``2024-CALPF-001542``.
        irwin_id: IRWIN record id.
        source: Where it came from.
    """

    time_utc: dt.datetime
    lat: float
    lon: float
    initial_lat: float | None
    initial_lon: float | None
    unique_fire_id: str
    irwin_id: str
    source: str

    def to_json(self) -> dict[str, Any]:
        return {**self.__dict__, "time_utc": self.time_utc.isoformat()}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Ignition:
        return cls(
            time_utc=dt.datetime.fromisoformat(d["time_utc"]),
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            initial_lat=d.get("initial_lat"),
            initial_lon=d.get("initial_lon"),
            unique_fire_id=str(d["unique_fire_id"]),
            irwin_id=str(d["irwin_id"]),
            source=str(d["source"]),
        )


def fetch_ignition(irwin_id: str, *, cache_root: str | Path = DEFAULT_CACHE, http: Any = None) -> Ignition:
    """The ignition of an incident from IRWIN (NIFC WFIGS Incident Locations), cached as JSON.

    Raises:
        ValueError: A malformed IRWIN id, or not exactly one record.
    """
    if not _GUID.match(irwin_id):
        raise ValueError(f"not an IRWIN id: {irwin_id!r}")
    path = Path(cache_root) / f"irwin-{irwin_id.strip('{}').lower()}" / "ignition.json"
    if path.exists():
        return Ignition.from_json(json.loads(path.read_text(encoding="utf-8")))
    client = UrllibHttp() if http is None else http
    url = f"{NIFC_SERVICES}/WFIGS_Incident_Locations/FeatureServer/0/query?" + urllib.parse.urlencode(
        {
            "where": f"IrwinID = '{irwin_id}'",
            "outFields": (
                "IrwinID,UniqueFireIdentifier,IncidentName,FireDiscoveryDateTime,InitialLatitude,InitialLongitude"
            ),
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "json",
        }
    )
    features = client.get_json(url).get("features", [])
    if len(features) != 1:
        raise ValueError(f"WFIGS has {len(features)} records with IrwinID {irwin_id}")
    a, g = features[0]["attributes"], features[0]["geometry"]
    time = _ms(a["FireDiscoveryDateTime"])
    if time is None:
        raise ValueError(f"the record {irwin_id} has no discovery time")
    ignition = Ignition(
        time_utc=time,
        lat=float(g["y"]),
        lon=float(g["x"]),
        initial_lat=a.get("InitialLatitude"),
        initial_lon=a.get("InitialLongitude"),
        unique_fire_id=str(a["UniqueFireIdentifier"]),
        irwin_id=irwin_id,
        source=f"IRWIN via NIFC WFIGS Incident Locations, {url.split('?')[0]}, accessed "
        f"{dt.datetime.now(dt.UTC).date().isoformat()}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ignition.to_json(), indent=2) + "\n", encoding="utf-8")
    return ignition


# ----------------------------------------------------------------------------- daily perimeters


@dataclass(frozen=True)
class PerimeterVersion:
    """One version of an incident's daily perimeter.

    Attributes:
        valid_from: When it became the current perimeter (``GDB_FROM_DATE``).
        valid_to: When it was replaced (``GDB_TO_DATE``), None if never.
        geometry: GeoJSON polygon or multipolygon, WGS84.
        acres: Mapped area (``GISAcres``).
    """

    valid_from: dt.datetime
    valid_to: dt.datetime | None
    geometry: dict[str, Any]
    acres: float

    def current_at(self, t: dt.datetime) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


def fetch_daily_perimeters(
    irwin_id: str, year: int, *, cache_root: str | Path = DEFAULT_CACHE, http: Any = None
) -> list[PerimeterVersion]:
    """Every approved version of an incident's daily perimeter (NIFC Operational Data Archive).

    Cached as GeoJSON under ``<cache_root>/irwin-<id>/daily_perimeters_<year>.geojson``. The
    archive starts in 2022; for an earlier year there is none, and the result is empty (the plan
    asks for daily perimeters "where published").

    Raises:
        ValueError: A malformed IRWIN id, or the archive has no ``Event Polygon`` layer.
    """
    if not _GUID.match(irwin_id):
        raise ValueError(f"not an IRWIN id: {irwin_id!r}")
    path = Path(cache_root) / f"irwin-{irwin_id.strip('{}').lower()}" / f"daily_perimeters_{year}.geojson"
    if not path.exists():
        client = UrllibHttp() if http is None else http
        service = f"{NIFC_SERVICES}/Operational_Data_Archive_{year}/FeatureServer"
        info = client.get_json(f"{service}?f=json")
        if "error" in info:  # no archive for that year (it starts in 2022): none published
            path.parent.mkdir(parents=True, exist_ok=True)
            empty = {"type": "FeatureCollection", "features": [], "note": f"no {service}"}
            path.write_text(json.dumps(empty) + "\n", encoding="utf-8")
            return []
        layers = {lyr["name"]: lyr["id"] for lyr in info.get("layers", [])}
        if "Event Polygon" not in layers:
            raise ValueError(f"no 'Event Polygon' layer in {service}")
        where = (
            f"IRWINID = '{irwin_id}' AND FeatureCategory = 'Wildfire Daily Fire Perimeter' "
            "AND FeatureStatus = 'Approved'"
        )
        features: list[dict[str, Any]] = []
        offset = 0
        while True:
            query = urllib.parse.urlencode(
                {
                    "where": where,
                    "outFields": "GDB_FROM_DATE,GDB_TO_DATE,GISAcres",
                    "returnGeometry": "true",
                    "outSR": "4326",
                    "orderByFields": "GDB_FROM_DATE",
                    "resultOffset": offset,
                    "f": "geojson",
                }
            )
            page = client.get_json(f"{service}/{layers['Event Polygon']}/query?{query}")
            batch = page.get("features", [])
            features += batch
            if not batch or not (
                page.get("exceededTransferLimit") or page.get("properties", {}).get("exceededTransferLimit")
            ):
                break
            offset += len(batch)
        path.parent.mkdir(parents=True, exist_ok=True)
        collection = {"type": "FeatureCollection", "features": [f for f in features if f.get("geometry")]}
        path.write_text(json.dumps(collection, sort_keys=True) + "\n", encoding="utf-8")
    collection = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for f in collection["features"]:
        p = f["properties"]
        start = _ms(p.get("GDB_FROM_DATE"))
        if start is None:
            continue
        out.append(PerimeterVersion(start, _ms(p.get("GDB_TO_DATE")), f["geometry"], float(p.get("GISAcres") or 0.0)))
    return sorted(out, key=lambda v: v.valid_from)


# ----------------------------------------------------------------------------- VIIRS


def earthdata_token() -> str:
    """The NASA Earthdata token from :data:`TOKEN_ENV` or :data:`TOKEN_FILE`.

    Raises:
        ValueError: Neither is set.
    """
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(
            f"a NASA Earthdata token is needed: set {TOKEN_ENV} or save it in {TOKEN_FILE} "
            "(https://urs.earthdata.nasa.gov, profile, Generate Token)"
        )
    return token


@dataclass(frozen=True)
class GranulePair:
    """One overpass: the fire-mask granule and its geolocation granule.

    Attributes:
        product: Fire product short name (``VJ114IMG``...).
        satellite: Satellite name.
        stamp: Acquisition stamp shared by both files, e.g. ``A2024188.0936``.
        fire_url, geo_url: HTTPS download URLs (NASA Earthdata cloud).
        time_start: Granule start time.
    """

    product: str
    satellite: str
    stamp: str
    fire_url: str
    geo_url: str
    time_start: dt.datetime


def _cmr(params: dict[str, Any], http: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    page = 1
    while True:
        q = urllib.parse.urlencode({**params, "page_size": 2000, "page_num": page})
        batch = http.get_json(f"{CMR_GRANULES}?{q}")["feed"]["entry"]
        entries += batch
        if len(batch) < 2000:
            return entries
        page += 1


def _https_nc(entry: dict[str, Any]) -> str | None:
    for link in entry.get("links", []):
        href = link.get("href", "")
        if href.startswith("https://data.") and href.endswith(".nc"):
            return href
    return None


def find_overpasses(
    west: float, south: float, east: float, north: float, start: dt.datetime, end: dt.datetime, http: Any = None
) -> list[GranulePair]:
    """Every 375 m active-fire overpass of the box in ``[start, end]``, paired with its geolocation."""
    client = UrllibHttp() if http is None else http
    bbox = f"{west},{south},{east},{north}"
    temporal = f"{start.astimezone(dt.UTC):%Y-%m-%dT%H:%M:%SZ},{end.astimezone(dt.UTC):%Y-%m-%dT%H:%M:%SZ}"
    pairs = []
    for product, (satellite, geo_product) in VIIRS_PRODUCTS.items():
        fires = _cmr({"short_name": product, "bounding_box": bbox, "temporal": temporal}, client)
        geos = _cmr({"short_name": geo_product, "bounding_box": bbox, "temporal": temporal}, client)
        geo_by_stamp = {}
        for g in geos:
            url = _https_nc(g)
            if url is not None:  # file names: <product>.A<yyyyddd>.<hhmm>.<collection>.<production>.nc
                geo_by_stamp[".".join(Path(url).name.split(".")[1:3])] = url
        for f in fires:
            url = _https_nc(f)
            if url is None:
                continue
            stamp = ".".join(Path(url).name.split(".")[1:3])
            if stamp in geo_by_stamp:
                time = dt.datetime.fromisoformat(f["time_start"].replace("Z", "+00:00"))
                pairs.append(GranulePair(product, satellite, stamp, url, geo_by_stamp[stamp], time))
    return sorted(pairs, key=lambda p: p.time_start)


class RangeFile(io.RawIOBase):
    """A read-only file over HTTP byte ranges, for ``h5py`` (blocks are cached).

    Args:
        url: A URL that answers ``Range`` requests (a signed URL needs no further auth).
        get: ``get(url, headers) -> (status, headers, bytes)``.
        block: Block size, bytes.
    """

    def __init__(
        self, url: str, get: Callable[[str, dict[str, str]], tuple[int, dict[str, str], bytes]], block: int = 1 << 20
    ) -> None:
        super().__init__()
        self.url, self._get, self.block = url, get, block
        status, headers, _ = get(url, {"Range": "bytes=0-0"})
        if status != 206:
            raise ValueError(f"the server does not serve byte ranges (status {status})")
        self.size = int({k.lower(): v for k, v in headers.items()}["content-range"].split("/")[1])
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            a = i * self.block
            b = min(self.size, a + self.block) - 1
            _, _, data = self._get(self.url, {"Range": f"bytes={a}-{b}"})
            self._cache[i] = data
            self.bytes_read += len(data)
        return self._cache[i]

    def readinto(self, buffer: Any) -> int:
        n = max(0, min(len(buffer), self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            i, o = divmod(self.pos + len(out), self.block)
            out += self._block(i)[o : o + n - len(out)]
        buffer[:n] = out
        self.pos += n
        return n


class EarthdataHttp:
    """HTTPS access to NASA Earthdata cloud files with a bearer token (``requests``).

    The token goes only to the first (``data.*.earthdatacloud.nasa.gov``) request; it answers
    with a redirect to a signed URL, which is followed without it.
    """

    def __init__(
        self,
        token: str,
        timeout_s: float = 300.0,
        attempts: int = 5,
        first_wait_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._requests = optional_import("requests")
        self._token = token
        self.timeout_s = timeout_s
        self.attempts, self.first_wait_s, self._sleep = attempts, first_wait_s, sleep
        self._ua = {"User-Agent": "wildfire-sim (research model)"}

    def _get(self, url: str, **kwargs: Any) -> Any:
        """GET, retried with exponential backoff on 5xx answers and connection errors."""
        wait = self.first_wait_s
        for attempt in range(self.attempts):
            try:
                r = self._requests.get(url, **kwargs)
                if r.status_code < 500 or attempt == self.attempts - 1:
                    return r
            except self._requests.ConnectionError:
                if attempt == self.attempts - 1:
                    raise
            self._sleep(wait)
            wait *= 2
        raise AssertionError("unreachable")

    def signed_url(self, url: str) -> str:
        r = self._get(
            url, headers={**self._ua, "Authorization": f"Bearer {self._token}"}, allow_redirects=False, timeout=60
        )
        if r.status_code in (301, 302, 303, 307, 308) and "location" in r.headers:
            if "urs.earthdata.nasa.gov" in r.headers["location"]:
                raise PermissionError("NASA Earthdata refused the token (expired? tokens last 60 days)")
            return r.headers["location"]
        if r.status_code == 200:
            return url
        raise PermissionError(f"NASA Earthdata answered {r.status_code} for {url}")

    def get_bytes(self, url: str) -> bytes:
        r = self._get(self.signed_url(url), headers=self._ua, timeout=self.timeout_s)
        r.raise_for_status()
        return r.content

    def get(self, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        r = self._get(url, headers={**self._ua, **headers}, timeout=self.timeout_s)
        return r.status_code, dict(r.headers), r.content


@dataclass(frozen=True)
class SwathWindow:
    """The part of one overpass that covers a region.

    Attributes:
        satellite: Satellite name.
        granule: Fire granule file name.
        time_utc: Scan time at the window's centre line.
        lat, lon: ``(h, w)`` pixel centres, WGS84.
        mask: ``(h, w)`` fire-mask classes.
        fire_lat, fire_lon, fire_confidence: Fire pixels in the window (centres and mask class 7-9).
    """

    satellite: str
    granule: str
    time_utc: dt.datetime
    lat: np.ndarray
    lon: np.ndarray
    mask: np.ndarray
    fire_lat: np.ndarray
    fire_lon: np.ndarray
    fire_confidence: np.ndarray

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            satellite=np.array(self.satellite),
            granule=np.array(self.granule),
            time_utc=np.array(self.time_utc.isoformat()),
            lat=self.lat,
            lon=self.lon,
            mask=self.mask,
            fire_lat=self.fire_lat,
            fire_lon=self.fire_lon,
            fire_confidence=self.fire_confidence,
        )

    @classmethod
    def load(cls, path: Path) -> SwathWindow:
        with np.load(path, allow_pickle=False) as d:
            return cls(
                str(d["satellite"]),
                str(d["granule"]),
                dt.datetime.fromisoformat(str(d["time_utc"])),
                d["lat"],
                d["lon"],
                d["mask"],
                d["fire_lat"],
                d["fire_lon"],
                d["fire_confidence"],
            )


def _min_distance_row(lat: Any, lon: Any, row: int, lat0: float, lon0: float) -> float:
    """Smallest distance (km, equirectangular) from (lat0, lon0) to the pixels of one row."""
    la = np.asarray(lat[row], dtype=np.float64)
    lo = np.asarray(lon[row], dtype=np.float64)
    ok = (la >= -90) & (la <= 90)
    if not ok.any():
        return math.inf
    dy = (la[ok] - lat0) * 111.2
    dx = (lo[ok] - lon0) * 111.2 * math.cos(math.radians(lat0))
    return float(np.sqrt(dx * dx + dy * dy).min())


def locate_row(lat: Any, lon: Any, lat0: float, lon0: float) -> tuple[int, float]:
    """The swath line nearest a point, and its distance (km).

    Searches one line per scan (each read costs one chunk of the geolocation file) with a
    ternary search, then refines within the best scan; the distance to the point is unimodal
    along the track of one granule.
    """
    n = int(lat.shape[0])
    rows = list(range(LINES_PER_SCAN // 2, n, LINES_PER_SCAN)) or [n // 2]
    lo_i, hi_i = 0, len(rows) - 1
    cache: dict[int, float] = {}

    def f(i: int) -> float:
        if i not in cache:
            cache[i] = _min_distance_row(lat, lon, rows[i], lat0, lon0)
        return cache[i]

    while hi_i - lo_i > 2:
        m1 = lo_i + (hi_i - lo_i) // 3
        m2 = hi_i - (hi_i - lo_i) // 3
        if f(m1) <= f(m2):
            hi_i = m2
        else:
            lo_i = m1
    best = min(range(lo_i, hi_i + 1), key=f)
    return rows[best], f(best)


def extract_window(
    lat: Any,
    lon: Any,
    scan_times: Any,
    box: tuple[float, float, float, float],
    half_rows: int = 160,
) -> tuple[slice, slice, float] | None:
    """Rows and columns of the swath that cover a WGS84 box, and the scan time at its centre.

    Args:
        lat: ``(lines, pixels)`` latitudes, an array or HDF5 dataset.
        lon: ``(lines, pixels)`` longitudes, an array or HDF5 dataset.
        scan_times: Per-scan times (TAI93 seconds), one per :data:`LINES_PER_SCAN` lines.
        box: ``(west, south, east, north)``.
        half_rows: Lines read on each side of the line nearest the box centre.

    Returns:
        ``(rows, cols, scan_time)`` or None if the box is not in the swath.
    """
    w, s, e, n = box
    lat0, lon0 = (s + n) / 2, (w + e) / 2
    row, dist = locate_row(lat, lon, lat0, lon0)
    if dist > 50.0:
        return None
    r0, r1 = max(0, row - half_rows), min(int(lat.shape[0]), row + half_rows)
    la = np.asarray(lat[r0:r1], dtype=np.float64)
    lo = np.asarray(lon[r0:r1], dtype=np.float64)
    inside = (la >= s) & (la <= n) & (lo >= w) & (lo <= e)
    if not inside.any():
        return None
    rr, cc = np.nonzero(inside)
    rows = slice(r0 + int(rr.min()), r0 + int(rr.max()) + 1)
    cols = slice(int(cc.min()), int(cc.max()) + 1)
    centre_scan = ((rows.start + rows.stop) // 2) // LINES_PER_SCAN
    return rows, cols, float(np.asarray(scan_times[centre_scan]))


def read_overpass(pair: GranulePair, box: tuple[float, float, float, float], http: EarthdataHttp) -> SwathWindow | None:
    """Download one overpass's fire mask and the geolocation rows covering ``box`` (needs ``h5py``)."""
    h5py = optional_import("h5py")
    fire = h5py.File(io.BytesIO(http.get_bytes(pair.fire_url)), "r")
    geo_file = RangeFile(http.signed_url(pair.geo_url), http.get, block=256 * 1024)
    geo = h5py.File(geo_file, "r")
    lat, lon = geo["geolocation_data/latitude"], geo["geolocation_data/longitude"]
    scan_times = geo["scan_line_attributes/ev_mid_time"][:]
    found = extract_window(lat, lon, scan_times, box)
    if found is None:
        return None
    rows, cols, scan_time = found
    la = np.asarray(lat[rows, cols], dtype=np.float32)
    lo = np.asarray(lon[rows, cols], dtype=np.float32)
    mask = np.asarray(fire["fire mask"][rows, cols], dtype=np.uint8)
    fp = {k: np.asarray(fire[k][:]) for k in ("FP_line", "FP_sample", "FP_latitude", "FP_longitude")}
    keep = (
        (fp["FP_line"] >= rows.start)
        & (fp["FP_line"] < rows.stop)
        & (fp["FP_sample"] >= cols.start)
        & (fp["FP_sample"] < cols.stop)
    )
    conf = mask[fp["FP_line"][keep] - rows.start, fp["FP_sample"][keep] - cols.start]
    return SwathWindow(
        pair.satellite,
        Path(pair.fire_url).name,
        tai93_to_utc(scan_time),
        la,
        lo,
        mask,
        fp["FP_latitude"][keep].astype(np.float32),
        fp["FP_longitude"][keep].astype(np.float32),
        conf.astype(np.uint8),
    )


def fetch_viirs(
    name: str,
    box: tuple[float, float, float, float],
    start: dt.datetime,
    end: dt.datetime,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
    token: str | None = None,
    http: Any = None,
    earthdata: EarthdataHttp | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[SwathWindow]:
    """Every VIIRS 375 m overpass of ``box`` in ``[start, end]``, cached per granule.

    Cached under ``<cache_root>/viirs-<name>/`` (one ``.npz`` per overpass that covers the box,
    and ``manifest.json`` listing every overpass found, with or without coverage). The catalogue's
    answer for this box and period is kept in the manifest too (``listings``), so a second call
    with the same arguments reads everything from disk and needs no network; delete the
    manifest's ``listings`` entry to ask the catalogue again.
    """
    directory = Path(cache_root) / f"viirs-{name}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"granules": {}}
    listings = manifest.setdefault("listings", {})
    key = "|".join([",".join(repr(float(v)) for v in box), *(t.astimezone(dt.UTC).isoformat() for t in (start, end))])
    if key in listings:
        pairs = [
            GranulePair(
                p["product"],
                p["satellite"],
                p["stamp"],
                p["fire_url"],
                p["geo_url"],
                dt.datetime.fromisoformat(p["time_start"]),
            )
            for p in listings[key]
        ]
    else:
        pairs = find_overpasses(*box, start, end, http=http)
        listings[key] = [{**dataclasses.asdict(p), "time_start": p.time_start.isoformat()} for p in pairs]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    client = earthdata
    windows = []
    for i, pair in enumerate(pairs):
        path = directory / f"{pair.stamp}.{pair.product}.npz"
        entry = manifest["granules"].get(path.name)
        if entry is None:
            if client is None:
                client = EarthdataHttp(earthdata_token() if token is None else token)
            if progress:
                progress(f"  overpass {i + 1}/{len(pairs)}: {pair.satellite} {pair.time_start:%Y-%m-%d %H:%M}")
            window = read_overpass(pair, box, client)
            if window is not None:
                window.save(path)
            entry = {
                "fire_url": pair.fire_url,
                "geo_url": pair.geo_url,
                "satellite": pair.satellite,
                "covers_region": window is not None,
                "sha256": _sha256(path) if window is not None else None,
            }
            manifest["granules"][path.name] = entry
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if entry["covers_region"]:
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"{path} does not match its manifest hash; delete it")
            windows.append(SwathWindow.load(path))
    return sorted(windows, key=lambda w: w.time_utc)


# ----------------------------------------------------------------------------- on the grid


def classes_on_grid(window: SwathWindow, target: TargetGrid, radius_m: float = VIEW_RADIUS_M) -> np.ndarray:
    """Each cell's class in one overpass: the nearest pixel's within ``radius_m``, else :data:`NOT_VIEWED`."""
    transform = optional_import("rasterio.warp").transform
    cKDTree = optional_import("scipy.spatial").cKDTree
    # bowtie-deleted pixels repeat ground the neighbouring scan sees; the nearest real pixel decides
    ok = np.isfinite(window.lat) & (np.abs(window.lat) <= 90) & (window.mask != BOWTIE)
    xs, ys = transform(
        "EPSG:4326", target.crs, window.lon[ok].astype(float).tolist(), window.lat[ok].astype(float).tolist()
    )
    tree = cKDTree(np.column_stack([xs, ys]))
    h, w = target.shape
    t = target.transform
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cx = t.a * cols + t.b * rows + t.c
    cy = t.d * cols + t.e * rows + t.f
    dist, idx = tree.query(np.column_stack([cx.ravel(), cy.ravel()]), distance_upper_bound=radius_m)
    classes = window.mask[ok]
    out = np.full(h * w, NOT_VIEWED, dtype=np.uint8)
    hit = np.isfinite(dist)
    out[hit] = classes[idx[hit]]
    return out.reshape(h, w)


class Detection(NamedTuple):
    """A first detection: the fire reached ``(row, col)`` between ``t_clear`` and ``t_detect`` (s)."""

    t_clear: float
    t_detect: float
    row: int
    col: int


def first_detections(
    classes: Sequence[np.ndarray],
    times_s: Sequence[float],
    fire_cells: Sequence[Sequence[tuple[int, int]]],
    ignition_s: float,
    min_confidence: int = FIRE_LOW,
) -> list[Detection]:
    """First detections with their arrival intervals (rule in the module docstring).

    Args:
        classes: Per overpass, the ``(H, W)`` classes on the grid (:func:`classes_on_grid`).
        times_s: Per overpass, its time, s.
        fire_cells: Per overpass, the cells holding the centres of its fire pixels.
        ignition_s: Ignition time, s.
        min_confidence: Lowest fire-mask class counted as fire (7 low, 8 nominal, 9 high);
            weaker fire classes count as neither fire nor clear.

    Returns:
        Detections ordered by detection time, then row and column.
    """
    if not (len(classes) == len(times_s) == len(fire_cells)):
        raise ValueError("classes, times_s and fire_cells need one entry per overpass")
    if not FIRE_LOW <= min_confidence <= FIRE_HIGH:
        raise ValueError("min_confidence must be 7, 8 or 9")
    if not classes:
        return []
    shape = classes[0].shape
    seen = np.zeros(shape, dtype=bool)
    last_clear = np.full(shape, -np.inf)
    out: list[Detection] = []
    for k in sorted(range(len(classes)), key=lambda i: times_s[i]):
        c, t = classes[k], float(times_s[k])
        is_fire = (c >= min_confidence) & (c <= FIRE_HIGH)
        if t >= ignition_s:
            for r, col in sorted(set(fire_cells[k])):
                if 0 <= r < shape[0] and 0 <= col < shape[1] and not seen[r, col]:
                    out.append(Detection(max(float(last_clear[r, col]), ignition_s), t, int(r), int(col)))
                    seen[r, col] = True
            seen |= is_fire
        clear = (c == WATER) | (c == CLEAR_LAND)
        last_clear[clear & ~seen] = t
    return out


# ----------------------------------------------------------------------------- one fire


@dataclass
class FireObservations:
    """Everything observed of one fire, ready to be put on its grid.

    Attributes:
        region: Region name.
        ignition: The incident record's ignition.
        final: The MTBS perimeter.
        daily: Versions of the daily perimeter.
        overpasses: VIIRS overpasses that cover the region.
    """

    region: str
    ignition: Ignition
    final: Perimeter
    daily: list[PerimeterVersion]
    overpasses: list[SwathWindow]
    _first_mapped: tuple[Any, np.ndarray] | None = field(default=None, repr=False, compare=False)

    def first_mapped(self, grid: Grid) -> np.ndarray:
        """POSIX time at which each cell was first inside an approved daily perimeter (inf: never).

        Burned ground stays burned, so a cell counts from the first perimeter that enclosed it,
        even if a later version (an edit, or a placeholder the archive briefly shows) leaves it out.
        """
        key = (grid.shape, grid.metadata.get("transform"))
        if self._first_mapped is None or self._first_mapped[0] != key:
            rasterize = optional_import("rasterio.features").rasterize
            transform_geom = optional_import("rasterio.warp").transform_geom
            target = target_of(grid)
            out = np.full(grid.shape, np.inf)
            # latest first: each earlier perimeter overwrites, so a cell keeps its earliest time
            shapes = [
                (transform_geom("EPSG:4326", target.crs, v.geometry), v.valid_from.timestamp())
                for v in sorted(self.daily, key=lambda v: v.valid_from, reverse=True)
            ]
            if shapes:
                out = rasterize(shapes, out_shape=grid.shape, transform=target.transform, fill=np.inf, dtype="float64")
            self._first_mapped = (key, out)
        return self._first_mapped[1]

    def observed_burn_mask(self, grid: Grid, until: dt.datetime | None = None) -> BoolArray:
        """Cells observed burned.

        Without ``until``: inside the final (MTBS) perimeter. With ``until``: inside some approved
        daily perimeter posted at or before ``until``, so burned by then (cells burned but not yet
        mapped are not included).
        """
        if until is None:
            return perimeter_mask(self.final.geometry, target_of(grid))
        if until.tzinfo is None:
            raise ValueError("until must be timezone-aware")
        return self.first_mapped(grid) <= until.timestamp()

    def ignition_cell(self, grid: Grid) -> tuple[int, int]:
        """Grid cell of the point of origin.

        Raises:
            ValueError: It lies outside the grid.
        """
        rows, cols = latlon_to_cell(target_of(grid), [self.ignition.lat], [self.ignition.lon])
        r, c = int(rows[0]), int(cols[0])
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            raise ValueError("the point of origin lies outside the grid")
        return r, c

    def seconds(self, grid: Grid, t: dt.datetime) -> float:
        """Simulated seconds of a UTC time: from the grid's ``weather_start_utc``."""
        start = dt.datetime.fromisoformat(grid.metadata["weather_start_utc"])
        return (t - start).total_seconds()

    def detections(self, grid: Grid, min_confidence: int = FIRE_LOW) -> list[Detection]:
        """First VIIRS detections on the grid with their arrival intervals, in simulated seconds."""
        target = target_of(grid)
        classes, times, cells = [], [], []
        for w in self.overpasses:
            classes.append(classes_on_grid(w, target))
            times.append(self.seconds(grid, w.time_utc))
            strong = w.fire_confidence >= min_confidence
            rows, cols = latlon_to_cell(target, w.fire_lat[strong], w.fire_lon[strong])
            cells.append(list(zip(rows.tolist(), cols.tolist(), strict=True)))
        return first_detections(classes, times, cells, self.seconds(grid, self.ignition.time_utc), min_confidence)


def region_box(final: Perimeter, margin_deg: float = 0.02) -> tuple[float, float, float, float]:
    """WGS84 box ``(west, south, east, north)`` of a perimeter plus a margin (for the overpass search)."""
    g = final.geometry
    polygons = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    lon = [float(p[0]) for poly in polygons for ring in poly for p in ring]
    lat = [float(p[1]) for poly in polygons for ring in poly for p in ring]
    return min(lon) - margin_deg, min(lat) - margin_deg, max(lon) + margin_deg, max(lat) + margin_deg


def fetch_observations(
    name: str,
    region: Region,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
    token: str | None = None,
    http: Any = None,
    viirs: bool = True,
    progress: Callable[[str], None] | None = None,
) -> FireObservations:
    """Ignition, final and daily perimeters and VIIRS overpasses of a region, from caches or services.

    VIIRS covers the region's box from 12 h before the ignition to the end of its weather window.

    Raises:
        ValueError: The region has no ``irwin_id``.
    """
    if region.irwin_id is None:
        raise ValueError(f"region {name} has no irwin_id in configs/regions.yaml")
    ignition = fetch_ignition(region.irwin_id, cache_root=cache_root, http=http)
    final = fetch_mtbs_perimeter(region.mtbs_fire_id, region.mtbs_year, cache_root=cache_root, http=http)
    daily = fetch_daily_perimeters(region.irwin_id, region.fire_date.year, cache_root=cache_root, http=http)
    windows: list[SwathWindow] = []
    if viirs:
        start = ignition.time_utc - dt.timedelta(hours=12)
        end = region.weather_start_utc + dt.timedelta(days=region.weather_days)
        windows = fetch_viirs(
            name, region_box(final), start, end, cache_root=cache_root, token=token, http=http, progress=progress
        )
    return FireObservations(name, ignition, final, daily, windows)
