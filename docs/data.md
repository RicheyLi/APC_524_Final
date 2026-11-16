# Data

Every model input comes from a free, public US source. A study region is built with one
command, and every download is cached with a manifest so a build can be reproduced and
checked. Downloaded data is never committed; only a 5 km x 5 km crop of each region is, for
offline tests.

## Building a region

The regions are defined in `configs/regions.yaml`, one per validation fire:

| Region | Role | Fire | Pre-fire fuels | Weather window | Grid |
| --- | --- | --- | --- | --- | --- |
| `lake_2024` | A (calibration) | 2024 Lake Fire, Santa Barbara County, CA (MTBS `CA3478712005220240705`, 41,216 acres) | LANDFIRE LF2023 | 14 days from 5 July 2024, 00:00 PDT | 815 x 978 cells of 30 m (24.5 x 29.3 km), 97 % burnable; perimeter 101 cells from the edge; grid north within 0.09 deg of true north |
| `silverado_2020` | B (test, no retuning) | 2020 Silverado Fire, Orange County, CA (MTBS `CA3375311768720201026`, 12,504 acres) | LANDFIRE LF2016 | 7 days from 26 October 2020, 00:00 PDT | 573 x 543 cells of 30 m (17.2 x 16.3 km), 96 % burnable; perimeter 101 cells from the edge; 34 % urban (Lake Forest, Irvine) |

Fire B's grid takes in the edge of the city: 106,844 urban cells carry the urban surrogate
fuel (TL3, `configs/urban.yaml`), 77,439 buildings are worth 31.0 billion USD, and 68,575 of its
100,420 break cells are city streets. Inside the fire's perimeter it is 97 % burnable wildland
(grass-shrub and chaparral) and 5.5 % urban. It has 192 first VIIRS detections (79 % inside the
MTBS perimeter, 99 % within 1 km; intervals 0.8 to 11.5 h) and no daily perimeters, which NIFC
publishes only from 2022. Its HRRR winds show the Santa Ana: from the north-east at 13.0 m/s on
average over the 12 hours after ignition, up to 20.3 m/s, with humidity down to 1.4 %. gridMET's
daily means, used before the HRRR reader existed, gave at most 4.8 m/s.

As built (2026-11-15) the Lake Fire grid is mostly chaparral and grass: SH5 (29 % of cells),
GR2 (20 %), GS2 (15 %), SH4 (15 %), GS3 and TL3; 36,576 cells carry a road or stream break and
261 buildings are worth 104.4 million USD. A three-hour Rothermel run on the committed crop,
ignited at the reported time in the afternoon weather of 5 July (40 deg C, 10 % humidity, 1-h
moisture 1.6 %), burns about 650 ha before reaching the crop's edge.

Both regions take their hourly weather from HRRR analyses (`weather_source: hrrr`; see
[Hourly weather](#hourly-weather) below).

```sh
uv sync --extra geo
$env:WILDFIRE_SIM_LANDFIRE_EMAIL = "you@example.edu"   # PowerShell; export ... in bash
uv run python scripts/download_data.py lake_2024        # or --all
uv run python scripts/download_data.py lake_2024 --verify
```

The build:

1. reads the fire's final perimeter from MTBS;
2. downloads LANDFIRE, 3DEP, OpenStreetMap, HRRR and gridMET for the perimeter's bounding box plus the
   buffer (at least 3 km) and a 1 km margin, each cached by its own module under `data/cache/`;
3. cuts the grid from the LANDFIRE raster's own 30 m lattice: the perimeter's bounding box plus
   the buffer and one cell on every side, so the grid size follows the fire;
4. assembles `grid.npz` (fuel model, land use, canopy cover, elevation, slope, aspect, asset
   value, break widths), `canopy.npz` (canopy height and base height in m, bulk density in
   kg/m^3, which have no `Grid` layer yet), `weather.yaml` (one point per hour) and a copy of the
   perimeter;
5. checks that the grid validates and that no cell the perimeter touches lies within the buffer
   of the edge;
6. writes `data/cache/<region>/manifest.json` and the crop `tests/fixtures/<region>_5km.npz`
   (with `<region>_weather.yaml`), centred on the MTBS point of origin.

A scenario on a region uses `grid: {kind: cached, path: data/cache/<region>/grid.npz}`, the
series of `weather.yaml`, and `start_time_utc` equal to the region's `weather_start_utc`
(WP 1.14), with the ignition time counted from it.

**Reproducing a build.** The manifest records every source (with each download's own
manifest: URL or query, version, access time, SHA-256) and the SHA-256 of each output file.
`--verify` rebuilds the region from the caches into a scratch directory and compares the
hashes; it reports any file that differs.

## Sources

| Input | Source and access | Licence | Attribution |
| --- | --- | --- | --- |
| Fuel model (FBFM40), canopy cover, height, base height, bulk density | [LANDFIRE](https://landfire.gov) Product Service job API (`https://lfps.usgs.gov/api`, submit, poll, download a zip of GeoTIFFs); needs an email address, read from `WILDFIRE_SIM_LANDFIRE_EMAIL` and never stored (`data/landfire.py`) | US Government work, public domain | LANDFIRE, U.S. Department of the Interior and U.S. Department of Agriculture |
| Elevation (slope and aspect computed) | USGS [3D Elevation Program](https://www.usgs.gov/3d-elevation-program) 10 m DEM through `py3dep` (`data/terrain.py`) | Public domain | U.S. Geological Survey, 3D Elevation Program |
| Hourly wind, temperature, humidity, cloud cover | NOAA [HRRR](https://rapidrefresh.noaa.gov/hrrr/) hourly analyses (`wrfsfcf00`) from the NOAA Open Data archive on AWS (`noaa-hrrr-bdp-pds`): five fields of each hour's GRIB2 file read by HTTP byte range from its `.idx` offsets, decoded with GDAL's GRIB driver through `rasterio`; no account (`data/hrrr.py`) | NOAA Open Data, no restrictions | NOAA; Dowell, D. C. et al. (2022), Weather and Forecasting 37, 1371-1395 |
| Daily wind, temperature, humidity, radiation (alternative to HRRR), 100-h fuel moisture | [gridMET](https://www.climatologylab.org/gridmet.html) daily, read at a point through the THREDDS NetCDF Subset Service of `thredds.northwestknowledge.net` (`data/weather.py`) | Stated variously as free of known copyright restrictions (Google Earth Engine catalogue) and CC BY 4.0 (UCAR GDEX); we attribute as CC BY 4.0 requires | Abatzoglou, J. T. (2013), Int. J. Climatol. 33, 121-131; Climatology Lab, UC Merced |
| Roads, waterways, water bodies, buildings | [OpenStreetMap](https://www.openstreetmap.org) through the Overpass API with `osmnx` (`data/osm.py`); the public Overpass server asks automated clients to retry with backoff when busy, which the client does | [Open Database Licence (ODbL) 1.0](https://www.openstreetmap.org/copyright) | (c) OpenStreetMap contributors |
| Fire perimeter and point of origin | [MTBS](https://www.mtbs.gov) Burned Area Boundaries through the USFS map service `EDW_MTBS_01` (`data/cache.py`) | Public domain | Monitoring Trends in Burn Severity, USDA Forest Service and USGS |
| Ignition time and point of origin | IRWIN incident record through NIFC's WFIGS Incident Locations service (`FireDiscoveryDateTime`, the record's point geometry); public (`data/validation.py`) | Public domain | National Interagency Fire Center, IRWIN |
| Daily perimeters | NIFC Operational Data Archive (`Operational_Data_Archive_<year>`, layer "Event Polygon", approved "Wildfire Daily Fire Perimeter" versions with the time each became current); public; the archive starts in 2022 | Public domain | National Interagency Fire Center |
| Active fire detections, clear and cloudy views | VIIRS 375 m active fire products of Suomi NPP, NOAA-20 and NOAA-21 (VNP14IMG, VJ114IMG, VJ214IMG, collection 2; NASA LP DAAC) and their geolocation (VNP03IMG, VJ103IMG, VJ203IMG; NASA LAADS DAAC), found through NASA's CMR catalogue and read from the Earthdata cloud: the fire mask whole (about 2 MB), the geolocation only in the rows over the region, by HTTP byte ranges. Needs a free NASA Earthdata token (`EARTHDATA_TOKEN`, or the git-ignored file `data/.earthdata_token`; valid 60 days), sent only to NASA's servers (`data/validation.py`) | NASA open data, no restrictions | Schroeder, W. & Giglio, L., NASA VIIRS Land Science Investigator Processing System; cite the product DOIs, e.g. 10.5067/VIIRS/VNP14IMG.002 |
| Hourly analysis coverage check | NOAA HRRR archive on AWS (`noaa-hrrr-bdp-pds`), listing only (`data/weather.py`, `hrrr_missing_hours`) | NOAA Open Data, no restrictions | NOAA |
| Live fuel moisture | National Fuel Moisture Database, now in [FEMS](https://www.wildfire.gov/application/fems); no public API, so samples are passed in and a seasonal default is used until then (plan section 13) | Public domain | USDA Forest Service |

**Projection: grid north is true north.** The simulator measures wind bearings (from true
north, as gridMET gives them and as HRRR's are rotated to) and aspect against the grid's up direction, so grid north must be
true north. The LANDFIRE Product Service delivers its rasters in an Albers projection centred on
the request (central meridian at the box's centre, standard parallels at its south and north
edges), where that holds to 0.09 deg across the Lake Fire grid. LANDFIRE's national projection,
EPSG:5070, would not do: its central meridian is 96 W, so at 120 W its grid north is rotated
about 14.5 deg and every wind would be applied 14.5 deg off. The build measures the rotation
(`grid_rotation_deg`), records it in the manifest and refuses a grid rotated by more than 1 deg.
The price is one resampling of LANDFIRE's categorical layers on its server.

**Units.** LANDFIRE stores canopy cover in percent, canopy height and base height in metres x
10 and bulk density in kg/m^3 x 100, with -9999 for no data; the build converts them
(`landfire_canopy`). gridMET's point service returns the packed integers of its files; the
client applies each variable's `scale_factor` and `add_offset`.

## Hourly weather

A region's weather is one value per hour at the grid's centroid, from `weather_start_utc` for
`weather_days` days. With `weather_source: hrrr` (both regions) each hour is the HRRR analysis
of that hour (`wildfire_sim.data.hrrr`):

* **Only what is needed is downloaded.** An hourly file is about 130 MB; its `.idx` index gives
  each field's byte offset, so only 2 m temperature (`TMP`) and relative humidity (`RH`), the
  10 m wind components (`UGRD`, `VGRD`) and total cloud cover (`TCDC`) are fetched, 1-2 MB per
  hour, eight hours at a time. They are decoded with the GRIB2 driver of the GDAL that
  `rasterio` ships and interpolated bilinearly to the centroid on the 3 km grid. A region's
  168-336 hours download in one to two minutes; the values are cached as
  `data/cache/hrrr-<key>/hourly.csv` with a manifest (every URL and byte range, SHA-256).
* **Winds are rotated to true north.** HRRR's wind components are relative to its Lambert
  conformal grid, centred on 97.5 W, as each file's flag table 3.3 (bit 5) states, so at 120 W
  they point about 14 deg off. They are rotated by the angle `n (lon - LoV)`, with the cone
  constant `n` from the standard parallels (Snyder 1987, eq. 15-3), all read from the file's
  own header. On the Silverado Santa Ana the rotated winds are closer to the ASOS airport
  observations at all four nearby stations (mean direction error 27-47 deg, against 33-49 deg
  unrotated).
* **The rest of the weather.** The cloudiness code for the 1-h moisture comes from the hour's
  cloud cover in tenths (Deeming et al. 1977) in daylight and is absent at night; the 100-h
  moisture is gridMET's (`fm100`), since HRRR has no fuel moisture. The 1-h and 10-h moisture
  and live moisture are computed as for gridMET.

With `weather_source: gridmet` the hours are gridMET's daily values spread over the day by the
diurnal profiles documented in `data/weather.py`. Daily means cannot show a Santa Ana's gusts:
for Silverado gridMET gave at most 4.8 m/s, HRRR 20.3 m/s.

## Observations of the fire

`uv run python scripts/download_data.py lake_2024 --observations` also fetches what was
observed of the fire (WP 2.7, `wildfire_sim.data.validation`): the ignition record, the MTBS
final perimeter, the daily perimeters and every VIIRS overpass from 12 hours before the
ignition to the end of the weather window, cached under `data/cache/` (`irwin-<id>/`,
`mtbs-<id>/`, `viirs-<region>/` with one small `.npz` per overpass and a manifest). The first
build of the Lake Fire reads 100 overpasses in about an hour; later builds read the cache.

On the grid (`FireObservations`):

* `observed_burn_mask(grid)` is the MTBS perimeter; `observed_burn_mask(grid, until)` the cells
  inside some approved daily perimeter posted by `until`, so burned by then. Burned ground stays
  burned: the archive briefly replaced the Lake Fire's 4,674-acre perimeter of 6 July by a
  1-acre placeholder, which must not un-burn anything.
* `detections(grid)` gives each place's first VIIRS detection as `(t_clear, t_detect, row,
  col)`, in seconds from the region's `weather_start_utc`: the fire arrived after `t_clear`, the
  last earlier overpass that saw the cell clear (no cloud, no fire), or the ignition, and by
  `t_detect`. Each overpass's fire-mask classes are put on the grid by nearest pixel within
  600 m; bowtie-deleted pixels never decide; sun glint, cloud and unclassified views never count
  as clear. On the Lake Fire whole overpasses are glint-flagged in daytime (Suomi NPP on 5 July
  at 20:51 UTC, NOAA-20 on 6 July at 20:56 UTC), and the rule skips them.

As built (2026-11-15), the Lake Fire has 97 overpasses over the region and 654 first detections
between 6 and 13 July; 87 % lie inside the MTBS perimeter and 98 % within 1 km of it (a 375 m
pixel's centre can fall just outside the mapped edge); their intervals are 0.4 to 14 h wide,
median 9.7 h. The detections lead the mapped perimeters, as they should: 252 first detections
by 12 UTC on 7 July, when the first infrared perimeter enclosed 4,959 ha. The ignition record
gives 5 July 2024, 22:48 UTC (15:48 PDT), at 34.7868 N, 120.0520 W (IRWIN `2024-CALPF-001542`;
the initially reported position, 34.785 N, 120.0448 W, is 660 m away and is kept in the cache).

## Running a fire

Once a region and its observations are built, one command runs the fire on its real grid,
ignited at the recorded point and time, and compares it with what was observed:

```sh
uv run python scripts/demo_real_fire.py lake_2024 --hours 24
uv run python scripts/demo_real_fire.py silverado_2020 --hours 12
```

It prints the simulated and observed burned area, their overlap (Jaccard and Sorensen), the
share of VIIRS detections whose interval contains the simulated arrival and the median error,
and writes a map to `runs/demo_<region>.png`. About a minute per simulated day of fire A with the
Numba engine (`uv sync --extra fast`). These runs are **not calibrated** (Phase 3) and have no
suppression (Phase 4). With HRRR weather (2026-11-15) Silverado's 12-hour run burns 4,404 ha
south-west before the Santa Ana, against 5,183 ha inside the final perimeter (Jaccard 0.45; 27 %
of detections inside their interval). The Lake Fire's 24-hour run puts 60 % of detections inside
their interval (median error 0.0 h) but burns far too much: 27,089 ha, reaching the grid's edge
after 11.6 hours, against 1,891 ha in the daily perimeter posted 24 hours after ignition
(Jaccard 0.07). The list of overpasses is cached with them
(`listings` in the VIIRS manifest), so with everything cached the command needs no network.

## The LANDFIRE version must predate the fire

A LANDFIRE version's fuels reflect the disturbances through the last year of its disturbance
window, so a version from the fire's year or later may already show the fire's own scar
(`configs/landfire_versions.yaml`, `check_version`). This was checked directly for the first
fire A, the 2016 Chimney Fire: inside its MTBS perimeter 98 % of LF2016's cells carry a fire
disturbance (`LF2016_FDist`; under 1 % in a 3 km ring outside), and the fuels there are
post-fire (GS1 most common, against GR2, TL3 and SH2 outside). Its pre-fire version, LF2014, is
not obtainable (the map viewer offers LF2016 and later, the Product Service serves only LF2014's
disturbance layers, and ScienceBase has no LF2014 fuel model), so fire A became the 2024 Lake
Fire, whose pre-fire version LF2023 is on the service and shows no disturbance inside the
perimeter (plan section 13).

## Network tests

Tests that call these services are marked `network` and run with `--run-network`. A scheduled
CI job (`.github/workflows/network.yml`, Mondays) runs them so a changed service is noticed. The
LANDFIRE tests and the full region build need the repository secret
`WILDFIRE_SIM_LANDFIRE_EMAIL`, and the VIIRS test the secret `EARTHDATA_TOKEN` (renewed every 60
days); each skips without its secret.
