"""LANDFIRE fuel and canopy layers for a bounding box (docs/plan.md section 3, WP 2.2).

LANDFIRE maps the Scott & Burgan (2005) fuel models and canopy structure of the US at 30 m.
This module gets five of its layers for a lat/lon bounding box and a **named version**:

=========  ==========================================  ==============
Product    Layer                                       LFPS name
=========  ==========================================  ==============
FBFM40     40 Scott and Burgan fire behaviour fuel models  ``LF2024_FBFM40``
CC         Forest canopy cover                         ``LF2024_CC``
CH         Forest canopy height                        ``LF2024_CH``
CBH        Forest canopy base height                   ``LF2024_CBH``
CBD        Forest canopy bulk density                  ``LF2024_CBD``
=========  ==========================================  ==============

**The version must predate the fire.** A version's fuels reflect the disturbances through the
last year of its annual disturbance window (``configs/landfire_versions.yaml``, from the
LANDFIRE version comparison table), so :func:`check_version` refuses a version whose window
reaches the fire's year: its fuels may already show the fire's scar.

**Two ways to get the data.**

* :func:`fetch_landfire` uses the LANDFIRE Product Service (LFPS, https://lfps.usgs.gov), which
  serves only the current versions (LF2016 and later). A job is submitted with ``POST
  /api/job/submit`` (a JSON body with ``Layer_List``, ``Area_of_Interest`` as ``"W S E N"`` in
  WGS84 degrees, optional ``Output_Projection`` and ``Resample_Resolution``, and a required
  ``Email``), polled with ``GET /api/job/status/?JobId=...`` until it reports an
  ``outputFile`` URL, and that file (a zip) is downloaded. These are the calls the LFPS web
  page itself makes (checked 2026-11-14); the service publishes no API reference, so the client
  keeps them in one place. The email address is read from the environment variable
  :data:`EMAIL_ENV` and is never written to the cache.
* :func:`ingest_archived` takes GeoTIFFs the user downloaded from an archived version (for
  example LF2014 from USGS ScienceBase, the newest version before fire A), clips them to the
  bounding box and writes the same cache layout. It needs ``rasterio`` (``uv sync --extra geo``).

Either way the layers land in ``data/cache/<key>/`` with a ``manifest.json`` recording the
source, version, bounding box, access time, the file and band of each product, and a SHA-256
per file. ``<key>`` hashes the bounding box, version and products, so a second request for the
same data reads the cache.

**Codes.** :func:`grid_codes` turns FBFM40 values into ``Grid.fuel_model`` and
``Grid.land_use``: the burnable models (101-204) are kept; urban (91, NB1) becomes an urban
cell with the surrogate fuel model of ``configs/urban.yaml`` (WP 1.13), without which it could
not burn; open water (98, NB8) becomes water; snow and ice (92, NB2), agricultural land kept
non-burnable (93, NB3) and bare ground (99, NB9) get no fuel (Scott & Burgan 2005). Converting
the canopy layers to ``Grid`` units and assembling the grid is WP 2.6.

References:
    Scott, J. H. & Burgan, R. E. (2005). Standard fire behavior fuel models. USDA For. Serv. Gen.
    Tech. Rep. RMRS-GTR-153.
    LANDFIRE Product Version Comparison Table, https://landfire.gov/data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import io
import json
import math
import os
import time
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildfire_sim.config import configs_dir, load_yaml
from wildfire_sim.types import IntArray, LandUse

PRODUCTS: tuple[str, ...] = ("FBFM40", "CC", "CH", "CBH", "CBD")
"""The LANDFIRE products the simulator uses, in the order they are requested."""

LFPS_API = "https://lfps.usgs.gov/api"
"""Base URL of the LANDFIRE Product Service job API."""

EMAIL_ENV = "WILDFIRE_SIM_LANDFIRE_EMAIL"
"""Environment variable holding the email address LFPS requires with every job."""

DEFAULT_CACHE = Path("data") / "cache"
"""Cache root, relative to the working directory (the repository root; ``/data/`` is git-ignored)."""

URBAN_CODE = 91  # NB1, urban/developed
WATER_CODE = 98  # NB8, open water
NO_FUEL_CODES = (92, 93, 99)  # NB2 snow/ice, NB3 agricultural (non-burnable), NB9 bare ground
"""FBFM40 non-burnable codes (Scott & Burgan 2005) other than urban and water: no fuel."""


def optional_import(module: str) -> Any:
    """Import a module of the optional ``geo`` extra.

    Imported by name, as ``sim_fast`` imports numba, so the type checker and the default
    install (without the extra) do not need it.

    Raises:
        ImportError: The module is missing, with the install command.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"{module} is needed here: install the geo extra (uv sync --extra geo)") from exc


# ----------------------------------------------------------------------------- versions


class LandfireVersion(BaseModel):
    """One LANDFIRE version (``configs/landfire_versions.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    disturbance_years: tuple[int, int]
    on_service: bool
    layer_prefix: Annotated[str, Field(min_length=1)]
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> LandfireVersion:
        if self.disturbance_years[1] < self.disturbance_years[0]:
            raise ValueError("disturbance_years must be [first, last]")
        return self

    @property
    def last_disturbance_year(self) -> int:
        """Latest year whose disturbances the version's fuels reflect."""
        return self.disturbance_years[1]

    def layer(self, product: str) -> str:
        """LFPS layer name of a product, e.g. ``LF2024_FBFM40``."""
        if product not in PRODUCTS:
            raise ValueError(f"unknown LANDFIRE product {product!r}; known: {PRODUCTS}")
        return f"{self.layer_prefix}_{product}"


class LandfireVersions(BaseModel):
    """Schema of ``configs/landfire_versions.yaml``."""

    model_config = ConfigDict(extra="forbid")

    source: Annotated[str, Field(min_length=1)]
    versions: dict[str, LandfireVersion]


def load_versions(path: str | Path | None = None) -> LandfireVersions:
    """Load the version table; default ``configs_dir() / "landfire_versions.yaml"``."""
    return load_yaml(configs_dir() / "landfire_versions.yaml" if path is None else path, LandfireVersions)


def check_version(version: str, fire_date: dt.date, table: LandfireVersions | None = None) -> LandfireVersion:
    """The named version, if its fuels predate a fire on ``fire_date``.

    Raises:
        ValueError: The version is unknown, or its last disturbance year is the fire's year or
            later (the message names the newest version that would do).
    """
    table = load_versions() if table is None else table
    if version not in table.versions:
        raise ValueError(f"unknown LANDFIRE version {version!r}; known: {sorted(table.versions)}")
    v = table.versions[version]
    if v.last_disturbance_year >= fire_date.year:
        allowed = [n for n, x in table.versions.items() if x.last_disturbance_year < fire_date.year]
        newest = max(allowed, key=lambda n: table.versions[n].last_disturbance_year) if allowed else None
        hint = f"; the newest that predates it is {newest}" if newest else ""
        raise ValueError(
            f"LANDFIRE {version} includes disturbances through {v.last_disturbance_year}, which may already "
            f"show a fire of {fire_date.isoformat()}: use a version from before the fire{hint}"
        )
    return v


# ----------------------------------------------------------------------------- area of interest


@dataclass(frozen=True)
class BBox:
    """A bounding box in WGS84 degrees (longitude ``west``/``east``, latitude ``south``/``north``)."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        values = (self.west, self.south, self.east, self.north)
        if not all(np.isfinite(values)):
            raise ValueError("bounding box values must be finite")
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("need -180 <= west < east <= 180 degrees")
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("need -90 <= south < north <= 90 degrees")

    def aoi(self) -> str:
        """``"W S E N"``, the LFPS ``Area_of_Interest`` format."""
        return " ".join(f"{v:.6f}" for v in (self.west, self.south, self.east, self.north))


def cache_key(bbox: BBox, version: str, products: Sequence[str] = PRODUCTS) -> str:
    """16 hex digits identifying a (bounding box, version, products) request: the cache directory name."""
    canon = json.dumps({"aoi": bbox.aoi(), "version": version, "products": list(products)}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- the LFPS job API


def build_job_request(
    bbox: BBox,
    version: LandfireVersion,
    email: str,
    products: Sequence[str] = PRODUCTS,
    output_projection: str | int | None = None,
    resolution_m: int = 30,
) -> dict[str, Any]:
    """The JSON body of an LFPS job submission.

    Args:
        bbox: Area of interest.
        version: A version the service offers (``on_service``).
        email: The address LFPS requires (it may email the result).
        products: Products to request, a subset of :data:`PRODUCTS`.
        output_projection: EPSG code (WKID) or WKT of the output; None keeps LANDFIRE's own.
        resolution_m: Output cell size, a multiple of 30 m.

    Raises:
        ValueError: The version is archived, the email or resolution is invalid, or no product
            is requested.
    """
    if not version.on_service:
        raise ValueError(
            f"{version.layer_prefix} is archived and not on the LANDFIRE Product Service; use ingest_archived"
        )
    if "@" not in email or email.strip() != email or len(email) < 3:
        raise ValueError("a valid email address is required by the LANDFIRE Product Service")
    if resolution_m < 30 or resolution_m % 30 != 0:
        raise ValueError("resolution_m must be a positive multiple of 30")
    if not products:
        raise ValueError("request at least one product")
    body: dict[str, Any] = {
        "Layer_List": ";".join(version.layer(p) for p in products),
        "Area_of_Interest": bbox.aoi(),
        "Email": email,
    }
    if output_projection is not None:
        body["Output_Projection"] = str(output_projection)
    if resolution_m != 30:
        body["Resample_Resolution"] = resolution_m
    return body


class Http(Protocol):
    """The three HTTP calls the client makes; tests pass a fake."""

    def post_json(self, url: str, body: Mapping[str, Any]) -> dict[str, Any]: ...

    def get_json(self, url: str) -> dict[str, Any]: ...

    def get_bytes(self, url: str) -> bytes: ...


class UrllibHttp:
    """:class:`Http` on the standard library (no extra dependency).

    Args:
        timeout_s: Timeout of each request, s.
    """

    def __init__(self, timeout_s: float = 120.0) -> None:
        self.timeout_s = timeout_s
        self._headers = {"User-Agent": "wildfire-sim (research model)", "Accept": "application/json"}

    def _open(self, request: urllib.request.Request) -> bytes:
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return response.read()

    def post_json(self, url: str, body: Mapping[str, Any]) -> dict[str, Any]:
        data = json.dumps(dict(body)).encode()
        headers = {**self._headers, "Content-Type": "application/json"}
        return json.loads(self._open(urllib.request.Request(url, data=data, headers=headers, method="POST")))

    def get_json(self, url: str) -> dict[str, Any]:
        return json.loads(self._open(urllib.request.Request(url, headers=self._headers)))

    def get_bytes(self, url: str) -> bytes:
        return self._open(urllib.request.Request(url, headers={"User-Agent": self._headers["User-Agent"]}))


@dataclass(frozen=True)
class JobStatus:
    """What ``/api/job/status`` reports.

    Attributes:
        status: The service's status string (not documented by LFPS; shown as it comes).
        messages: Progress or error messages.
        queue_position: Position in the queue, or None.
        output_file: URL of the result when the job has finished, else None.
    """

    status: str
    messages: tuple[str, ...]
    queue_position: int | None
    output_file: str | None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> JobStatus:
        raw = data.get("messages") or ()
        messages = tuple(str(m) for m in raw) if isinstance(raw, list | tuple) else (str(raw),)
        queue = data.get("queuePosition")
        return cls(
            status=str(data.get("status", "")),
            messages=messages,
            queue_position=int(queue) if isinstance(queue, int | float) else None,
            output_file=str(data["outputFile"]) if data.get("outputFile") else None,
        )

    @property
    def failed(self) -> bool:
        """Whether the status reads as a failure or cancellation."""
        return any(word in self.status.lower() for word in ("fail", "error", "cancel"))


def submit_job(body: Mapping[str, Any], http: Http, api: str = LFPS_API) -> str:
    """Submit a job; returns its id.

    Raises:
        RuntimeError: The response carries no ``jobId``.
    """
    reply = http.post_json(f"{api}/job/submit", body)
    job_id = reply.get("jobId")
    if not job_id:
        raise RuntimeError(f"LFPS did not return a job id: {reply}")
    return str(job_id)


def job_status(job_id: str, http: Http, api: str = LFPS_API) -> JobStatus:
    """Current status of a job."""
    return JobStatus.from_json(http.get_json(f"{api}/job/status/?JobId={job_id}"))


def wait_for_job(
    job_id: str,
    http: Http,
    *,
    api: str = LFPS_API,
    poll_s: float = 5.0,
    timeout_s: float = 3600.0,
    sleep: Callable[[float], None] = time.sleep,
) -> JobStatus:
    """Poll a job until it has an output file.

    Raises:
        RuntimeError: The job failed or was cancelled.
        TimeoutError: No output file after ``timeout_s`` seconds of polling.
    """
    waited = 0.0
    while True:
        status = job_status(job_id, http, api)
        if status.output_file:
            return status
        if status.failed:
            raise RuntimeError(f"LFPS job {job_id} {status.status}: {' '.join(status.messages)}")
        if waited >= timeout_s:
            raise TimeoutError(f"LFPS job {job_id} still {status.status!r} after {timeout_s:g} s")
        sleep(poll_s)
        waited += poll_s


# ----------------------------------------------------------------------------- the cache


@dataclass(frozen=True)
class LandfireLayers:
    """The cached layers of one request.

    Attributes:
        directory: ``<cache root>/<key>/``.
        files: Product -> (GeoTIFF path, 1-based band).
        manifest: The contents of ``manifest.json``.
    """

    directory: Path
    files: dict[str, tuple[Path, int]]
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    bbox: BBox, version: str, products: Sequence[str], files: dict[str, tuple[Path, int]], source: dict[str, Any]
) -> dict[str, Any]:
    names = sorted({p.name for p, _ in files.values()})
    directory = next(iter(files.values()))[0].parent
    return {
        "dataset": "LANDFIRE",
        "version": version,
        "products": list(products),
        "bbox_wgs84": {"west": bbox.west, "south": bbox.south, "east": bbox.east, "north": bbox.north},
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": source,
        "files": {p: {"file": f.name, "band": b} for p, (f, b) in files.items()},
        "sha256": {n: _sha256(directory / n) for n in names},
        "attribution": "LANDFIRE, U.S. Department of the Interior and U.S. Department of Agriculture (public domain)",
    }


def _read_cache(directory: Path, products: Sequence[str]) -> LandfireLayers | None:
    path = directory / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = {p: (directory / v["file"], int(v["band"])) for p, v in manifest["files"].items()}
    if list(manifest["products"]) != list(products) or not all(f.exists() for f, _ in files.values()):
        return None
    if any(_sha256(directory / n) != h for n, h in manifest["sha256"].items()):
        raise ValueError(f"{directory}: a cached file does not match its manifest hash; delete the directory")
    return LandfireLayers(directory, files, manifest)


def _write_cache(directory: Path, files: dict[str, tuple[Path, int]], manifest: dict[str, Any]) -> LandfireLayers:
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return LandfireLayers(directory, files, manifest)


def assign_bands(names: Sequence[str], layers: Sequence[str]) -> dict[str, tuple[str, int]]:
    """Which file and band of an LFPS result holds each layer.

    Each layer is the file whose name contains its layer name (band 1). If not every layer
    matches and the result is a single GeoTIFF, its bands are taken in request order.

    Args:
        names: GeoTIFF file names in the result.
        layers: Requested layer names in request order.

    Raises:
        ValueError: The files cannot be matched to the layers.
    """
    tifs = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
    matched: dict[str, tuple[str, int]] = {}
    for layer in layers:
        hits = [n for n in tifs if layer.lower() in Path(n).name.lower()]
        if len(hits) == 1:
            matched[layer] = (hits[0], 1)
    if len(matched) == len(layers):
        return matched
    if len(tifs) == 1:
        return {layer: (tifs[0], i + 1) for i, layer in enumerate(layers)}
    raise ValueError(f"cannot match the LFPS result files {tifs} to the layers {list(layers)}")


def fetch_landfire(
    bbox: BBox,
    version: str,
    fire_date: dt.date,
    *,
    email: str | None = None,
    products: Sequence[str] = PRODUCTS,
    cache_root: str | Path = DEFAULT_CACHE,
    http: Http | None = None,
    output_projection: str | int | None = None,
    poll_s: float = 5.0,
    timeout_s: float = 3600.0,
    sleep: Callable[[float], None] = time.sleep,
) -> LandfireLayers:
    """Get the layers of ``version`` for ``bbox`` from LFPS, or from the cache if already there.

    Args:
        bbox: Area of interest, WGS84 degrees.
        version: Version name in ``configs/landfire_versions.yaml``, e.g. ``"LF2022"``.
        fire_date: Date of the fire to be simulated; the version must predate it.
        email: Address for LFPS; None reads :data:`EMAIL_ENV`.
        products: Subset of :data:`PRODUCTS`.
        cache_root: Directory holding ``<key>/`` directories.
        http: HTTP client; None uses :class:`UrllibHttp`.
        output_projection: Optional output CRS (EPSG code or WKT).
        poll_s: Seconds between status checks.
        timeout_s: Give up after this much polling, s.
        sleep: Sleep function (tests pass a no-op).

    Raises:
        ValueError: The version is unknown, too new for the fire, or archived; or no email.
    """
    table = load_versions()
    v = check_version(version, fire_date, table)
    directory = Path(cache_root) / cache_key(bbox, version, products)
    cached = _read_cache(directory, products)
    if cached is not None:
        return cached
    address = email if email is not None else os.environ.get(EMAIL_ENV, "")
    if not address:
        raise ValueError(f"the LANDFIRE Product Service needs an email address: pass email= or set {EMAIL_ENV}")
    body = build_job_request(bbox, v, address, products, output_projection)
    client = UrllibHttp() if http is None else http
    job_id = submit_job(body, client)
    status = wait_for_job(job_id, client, poll_s=poll_s, timeout_s=timeout_s, sleep=sleep)
    assert status.output_file is not None
    archive = client.get_bytes(status.output_file)
    layers = [v.layer(p) for p in products]
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        bands = assign_bands(members, layers)
        for member in members:
            if member.lower().endswith((".tif", ".tiff", ".xml")):
                (directory / Path(member).name).write_bytes(zf.read(member))
    files = {
        p: (directory / Path(bands[layer][0]).name, bands[layer][1]) for p, layer in zip(products, layers, strict=True)
    }
    source = {
        "service": "LANDFIRE Product Service",
        "api": LFPS_API,
        "job_id": job_id,
        "layer_list": body["Layer_List"],
        "area_of_interest": body["Area_of_Interest"],
        "output_projection": body.get("Output_Projection"),
    }
    return _write_cache(directory, files, _manifest(bbox, version, products, files, source))


def ingest_archived(
    sources: Mapping[str, str | Path],
    bbox: BBox,
    version: str,
    fire_date: dt.date,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
) -> LandfireLayers:
    """Clip user-downloaded GeoTIFFs of an archived version to ``bbox`` and cache them.

    For versions the service no longer offers, such as LF2014 (fire A). Each source raster is
    clipped to the bounding box (transformed into the raster's CRS, expanded to whole cells) and
    written as ``<version>_<product>.tif`` with the source's CRS, cell size and nodata value.

    Args:
        sources: Product -> path of the downloaded GeoTIFF (band 1 is used).
        bbox: Area of interest, WGS84 degrees.
        version: Version name in ``configs/landfire_versions.yaml``.
        fire_date: Date of the fire; the version must predate it.
        cache_root: Directory holding ``<key>/`` directories.

    Raises:
        ValueError: The version is too new for the fire, a product is unknown, or the bounding
            box does not overlap a source raster.
        ImportError: ``rasterio`` is not installed.
    """
    rasterio = optional_import("rasterio")
    WindowError = optional_import("rasterio.errors").WindowError
    transform_bounds = optional_import("rasterio.warp").transform_bounds
    windows = optional_import("rasterio.windows")
    Window, from_bounds = windows.Window, windows.from_bounds

    check_version(version, fire_date)
    products = tuple(p for p in PRODUCTS if p in sources)
    unknown = set(sources) - set(PRODUCTS)
    if unknown or not products:
        raise ValueError(f"sources must name products from {PRODUCTS}; got {sorted(sources)}")
    directory = Path(cache_root) / cache_key(bbox, version, products)
    cached = _read_cache(directory, products)
    if cached is not None:
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[Path, int]] = {}
    for product in products:
        src_path = Path(sources[product])
        with rasterio.open(src_path) as src:
            bounds = transform_bounds(
                "EPSG:4326", src.crs, bbox.west, bbox.south, bbox.east, bbox.north, densify_pts=21
            )
            exact = from_bounds(*bounds, transform=src.transform)
            # every cell the box touches: floor the start, ceil the end (1e-9 absorbs rounding)
            c0, r0 = math.floor(exact.col_off + 1e-9), math.floor(exact.row_off + 1e-9)
            c1 = math.ceil(exact.col_off + exact.width - 1e-9)
            r1 = math.ceil(exact.row_off + exact.height - 1e-9)
            try:
                box = Window.from_slices((r0, r1), (c0, c1))
                window = box.intersection(Window.from_slices((0, src.height), (0, src.width)))
            except WindowError:
                raise ValueError(f"{src_path}: the bounding box does not overlap the raster") from None
            data = src.read(1, window=window)
            profile = {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "count": 1,
                "dtype": src.dtypes[0],
                "crs": src.crs,
                "transform": src.window_transform(window),
                "nodata": src.nodata,
                "compress": "deflate",
            }
        out = directory / f"{version}_{product}.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data, 1)
        files[product] = (out, 1)
    source = {"archived": {p: Path(s).name for p, s in sources.items()}}
    return _write_cache(directory, files, _manifest(bbox, version, products, files, source))


def read_layer(layers: LandfireLayers, product: str) -> tuple[np.ndarray, dict[str, Any]]:
    """One product as an array with its raster profile (CRS, transform, nodata). Needs ``rasterio``."""
    rasterio = optional_import("rasterio")

    path, band = layers.files[product]
    with rasterio.open(path) as src:
        return src.read(band), dict(src.profile)


# ----------------------------------------------------------------------------- codes


def grid_codes(fbfm40: IntArray, urban_fuel_model: int, nodata: int | None = None) -> tuple[IntArray, IntArray]:
    """``Grid.fuel_model`` and ``Grid.land_use`` from FBFM40 values.

    Burnable models are kept as they are (Scott & Burgan numbers, 101-204). Urban (91) becomes
    an URBAN cell with ``urban_fuel_model`` (``configs/urban.yaml``); open water (98) becomes
    WATER with no fuel; snow and ice (92), non-burnable agriculture (93), bare ground (99), the
    raster's ``nodata`` value and any other code become WILDLAND with no fuel (fuel model 0),
    which no rule spreads into.

    Args:
        fbfm40: FBFM40 raster values, any integer shape.
        urban_fuel_model: Surrogate fuel model for urban cells (>= 1).
        nodata: The raster's nodata value, if any.

    Returns:
        ``(fuel_model, land_use)``: ``int16`` and ``int8`` arrays of the input's shape.
    """
    if urban_fuel_model < 1:
        raise ValueError("urban_fuel_model must be >= 1")
    codes = np.asarray(fbfm40).astype(np.int64)
    burnable = (codes >= 101) & (codes <= 204)
    if nodata is not None:
        burnable &= codes != nodata
    fuel = np.where(burnable, codes, 0)
    land = np.full(codes.shape, int(LandUse.WILDLAND), dtype=np.int8)
    land[codes == URBAN_CODE] = int(LandUse.URBAN)
    fuel = np.where(codes == URBAN_CODE, urban_fuel_model, fuel)
    land[codes == WATER_CODE] = int(LandUse.WATER)
    return fuel.astype(np.int16), land
