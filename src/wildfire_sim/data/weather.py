"""Hourly weather for a site and a time window (docs/plan.md section 3, WP 2.4).

The simulator takes one spatially uniform :class:`~wildfire_sim.grid.Weather` per hour
(``WeatherConfig``, one point per hour, aligned with ``ScenarioConfig.start_time_utc``, WP 1.14).
This module builds it for the centroid of the study area.

**Source.** The plan uses an hourly analysis (HRRR) if its archive covers both validation fires
and disaggregated gridMET for both otherwise (section 13). The HRRR archive misses 2-14 of the
24 analysis hours on every day of the 2016 Chimney Fire (checked with
:func:`hrrr_missing_hours`), fire A when this module was written, so it implements gridMET
(Abatzoglou 2013), a 1/24 degree (about 4 km) daily dataset. Fire A is now the 2024 Lake Fire
and fire B the 2020 Silverado Fire, which HRRR covers completely, so both now use HRRR
(``source="hrrr"``, :mod:`wildfire_sim.data.hrrr`, :func:`hourly_from_hrrr`); gridMET remains
the alternative and supplies the 100-h moisture in either case. :func:`fetch_gridmet` reads
gridMET at a point through the THREDDS NetCDF Subset Service of the University of Idaho and
caches it with a SHA-256 manifest. The service returns the packed integers stored in the
files, so the client applies each variable's ``scale_factor`` and ``add_offset`` from the
dataset's attributes.

**From days to hours** (:func:`disaggregate`). Hours are placed in local solar time, UTC +
longitude / 15 h (the equation of time, at most 16 minutes, is neglected), and an hour belongs to
the gridMET day of its local solar date. Sunrise and sunset come from the FAO-56 solar formulas
(Allen et al. 1998, eqs. 21-25 and 34).

* *Temperature*: Parton & Logan (1981). A sine from the daily minimum, reached at sunrise + c,
  to sunset, peaking a hours after solar noon (+ c); after sunset an exponential decay towards the
  next day's minimum, ``T = Tmin' + (T_sunset - Tmin') exp(-b n / Z)`` with ``n`` the hours since
  sunset and ``Z`` the length of the night. As in the original model the night curve does not
  quite reach the minimum by the next morning, so the temperature steps down by about
  ``exp(-b)`` = 11 % of the evening fall at that moment. The day's maximum and minimum are those
  of gridMET; the model fixes their timing, not a daily mean (gridMET has none).
* *Relative humidity*: the temperature curve inverted. By day ``RH = RHmax - (RHmax - RHmin) f``
  where ``T = Tmin + (Tmax - Tmin) f``; after sunset it decays with the same factor as the
  temperature towards the next day's maximum. So the humidity maximum falls at the temperature
  minimum and the minimum at the temperature maximum, and both are gridMET's own, the pairing
  FARSITE uses for its daily weather (Finney 1998). Holding the vapour pressure constant
  instead (FAO-56 eq. 17) cannot honour both extremes when they disagree: at the Chimney Fire
  on 13 August 2016 a humid marine night (77 % at 21 deg C) and a dry afternoon (10 % at 40 deg
  C) average to 18 % in the afternoon, and the afternoon minimum is what drives the 1-h moisture.
* *Wind speed*: a cosine around gridMET's daily mean, ``v = v_mean + A cos(2 pi (t - t_peak) /
  24 h)``, with the amplitude ``A`` and the early-afternoon peak of the land climatology of Dai &
  Deser (1999), ``A`` limited to the daily mean. The 24 hours of a day average to the daily mean
  exactly.
* *Wind direction*, *100-h fuel moisture*: gridMET's daily values, held for the day. gridMET's
  direction is the bearing the wind blows from, as in :class:`~wildfire_sim.grid.Weather`.
* *Cloudiness* (for the sunshine correction of the 1-h moisture, WP 2.1c): the relative sunshine
  duration from the Angstrom formula ``Rs = (a_s + b_s n/N) Ra`` (FAO-56 eq. 35) with gridMET's
  daily radiation ``Rs``, cloud cover ``1 - n/N`` in tenths, mapped to the NFDRS state-of-weather
  code (0 under 1/10, 1 for 1/10-5/10, 2 for 6/10-9/10, 3 above; Deeming et al. 1977). Daylight
  hours get the day's code; night hours get none (no sun correction).

The 1-h and 10-h dead fuel moisture are then computed every hour from the temperature,
humidity and code by :func:`wildfire_sim.spread.moisture.dead_fuel_moisture` (WP 2.1c; 10-h is
1.28 / 1.03 times 1-h). The 1000-h moisture is not used.

**Live fuel moisture** (:func:`live_fuel_moisture`): the mean of National Fuel Moisture
Database samples taken near the site around the date, per fuel class, when samples are given;
otherwise the Scott & Burgan (2005) live moisture scenario the month maps to in
``configs/weather.yaml`` (a flagged team assumption). The NFMD is now served through the Fire
Environment Mapping System (FEMS), whose API is not published, so samples are passed in as
:class:`NfmdSample` records (section 13).

All parameters, with sources, are in ``configs/weather.yaml``.

References:
    Abatzoglou, J. T. (2013). Development of gridded surface meteorological data for ecological
    applications and modelling. Int. J. Climatol. 33, 121-131.
    Allen, R. G., Pereira, L. S., Raes, D. & Smith, M. (1998). Crop evapotranspiration. FAO
    Irrigation and Drainage Paper 56.
    Dai, A. & Deser, C. (1999). Diurnal and semidiurnal variations in global surface wind and
    divergence fields. J. Geophys. Res. 104(D24), 31109-31125.
    Deeming, J. E., Burgan, R. E. & Cohen, J. D. (1977). The National Fire-Danger Rating
    System - 1978. USDA For. Serv. Gen. Tech. Rep. INT-39.
    Finney, M. A. (1998). FARSITE: Fire Area Simulator - model development and evaluation. USDA
    For. Serv. Res. Pap. RMRS-RP-4.
    Parton, W. J. & Logan, J. A. (1981). A model for diurnal variation in soil and air
    temperature. Agric. Meteorol. 23, 205-216.
    Scott, J. H. & Burgan, R. E. (2005). Standard fire behavior fuel models. USDA For. Serv. Gen.
    Tech. Rep. RMRS-GTR-153.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import math
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildfire_sim.config import CitedValue, WeatherConfig, WeatherPoint, configs_dir, load_yaml
from wildfire_sim.data.landfire import DEFAULT_CACHE, BBox, UrllibHttp
from wildfire_sim.spread.moisture import FuelMoistureParams, dead_fuel_moisture, load_fuel_moisture_params

GRIDMET_THREDDS = "http://thredds.northwestknowledge.net:8080/thredds"
"""THREDDS server of gridMET (University of Idaho / Climatology Lab)."""

GRIDMET_VARIABLES: dict[str, str] = {
    "tmmx": "daily_maximum_temperature",
    "tmmn": "daily_minimum_temperature",
    "rmax": "daily_maximum_relative_humidity",
    "rmin": "daily_minimum_relative_humidity",
    "vs": "daily_mean_wind_speed",
    "th": "daily_mean_wind_direction",
    "fm100": "dead_fuel_moisture_100hr",
    "srad": "daily_mean_shortwave_radiation_at_surface",
}
"""gridMET short name -> variable name inside its aggregated dataset ``agg_met_<short>_1979_CurrentYear_CONUS.nc``."""

HRRR_BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
"""NOAA Open Data Dissemination archive of the HRRR on AWS (from 2014-07-30)."""

SOLAR_CONSTANT = 0.0820
"""MJ m^-2 min^-1 (FAO-56, eq. 21)."""

HOUR = dt.timedelta(hours=1)


class GetBytes(Protocol):
    """The one HTTP call this module makes; tests pass a fake. :class:`UrllibHttp` satisfies it."""

    def get_bytes(self, url: str) -> bytes: ...


# ----------------------------------------------------------------------------- parameters


class PartonLogan(BaseModel):
    """Coefficients of the Parton & Logan (1981) diurnal temperature model."""

    model_config = ConfigDict(extra="forbid")

    a: CitedValue
    b: CitedValue
    c: CitedValue


class LiveScenario(BaseModel):
    """Live herbaceous and woody fuel moisture of one scenario, fractions."""

    model_config = ConfigDict(extra="forbid")

    herb: CitedValue
    woody: CitedValue


class SeasonalScenario(BaseModel):
    """Calendar month (1-12) -> live moisture scenario name."""

    model_config = ConfigDict(extra="forbid")

    source: Annotated[str, Field(min_length=1)]
    months: dict[int, str]


class WeatherParams(BaseModel):
    """Schema of ``configs/weather.yaml`` (see that file for the meaning and source of each value)."""

    model_config = ConfigDict(extra="forbid")

    parton_logan: PartonLogan
    wind_amplitude_m_s: CitedValue
    wind_peak_hour_lst: CitedValue
    angstrom_a: CitedValue
    angstrom_b: CitedValue
    live_scenarios: dict[str, LiveScenario]
    seasonal_scenario: SeasonalScenario
    nfmd_window_days: CitedValue
    nfmd_radius_km: CitedValue

    @model_validator(mode="after")
    def _check(self) -> WeatherParams:
        if self.parton_logan.a.value <= 0 or self.parton_logan.b.value <= 0:
            raise ValueError("parton_logan a and b must be > 0")
        if self.wind_amplitude_m_s.value < 0:
            raise ValueError("wind_amplitude_m_s must be >= 0")
        if not 0 <= self.wind_peak_hour_lst.value < 24:
            raise ValueError("wind_peak_hour_lst must lie in [0, 24)")
        if self.angstrom_a.value < 0 or self.angstrom_b.value <= 0:
            raise ValueError("angstrom_a must be >= 0 and angstrom_b > 0")
        if sorted(self.seasonal_scenario.months) != list(range(1, 13)):
            raise ValueError("seasonal_scenario.months needs every month 1-12")
        unknown = set(self.seasonal_scenario.months.values()) - set(self.live_scenarios)
        if unknown:
            raise ValueError(f"seasonal_scenario names unknown scenarios {sorted(unknown)}")
        if self.nfmd_window_days.value < 0 or self.nfmd_radius_km.value < 0:
            raise ValueError("nfmd_window_days and nfmd_radius_km must be >= 0")
        return self


def load_weather_params(path: str | Path | None = None) -> WeatherParams:
    """Load the parameters; default ``configs_dir() / "weather.yaml"``."""
    return load_yaml(configs_dir() / "weather.yaml" if path is None else path, WeatherParams)


# ----------------------------------------------------------------------------- records


@dataclass(frozen=True)
class DailyWeather:
    """One gridMET day at a point, in the simulator's units.

    Attributes:
        date: The gridMET day (a calendar day ending about midnight MST).
        tmax_c, tmin_c: Daily maximum and minimum 2 m temperature, deg C.
        rh_max, rh_min: Daily maximum and minimum 2 m relative humidity, fractions.
        wind_speed: Daily mean 10 m wind speed, m/s.
        wind_from_deg: Daily mean 10 m wind direction, bearing the wind blows from, degrees.
        moisture_100h: 100-h dead fuel moisture, fraction.
        srad_w_m2: Daily mean downward shortwave radiation at the surface, W/m^2.
    """

    date: dt.date
    tmax_c: float
    tmin_c: float
    rh_max: float
    rh_min: float
    wind_speed: float
    wind_from_deg: float
    moisture_100h: float
    srad_w_m2: float

    def __post_init__(self) -> None:
        values = (self.tmax_c, self.tmin_c, self.rh_max, self.rh_min, self.wind_speed, self.wind_from_deg)
        if not all(math.isfinite(v) for v in (*values, self.moisture_100h, self.srad_w_m2)):
            raise ValueError(f"{self.date}: gridMET values must be finite (missing data?)")
        if self.tmax_c < self.tmin_c:
            raise ValueError(f"{self.date}: tmax_c < tmin_c")
        if not 0 <= self.rh_min <= self.rh_max <= 1:
            raise ValueError(f"{self.date}: need 0 <= rh_min <= rh_max <= 1")
        if self.wind_speed < 0 or self.moisture_100h < 0 or self.srad_w_m2 < 0:
            raise ValueError(f"{self.date}: wind speed, 100-h moisture and radiation must be >= 0")


@dataclass(frozen=True)
class HourlyWeather:
    """Weather for the hour starting at ``time_utc``.

    Attributes:
        time_utc: Start of the hour, timezone-aware UTC.
        wind_speed: 10 m wind speed, m/s.
        wind_direction: Bearing the wind blows from, degrees in [0, 360).
        temperature_c: 2 m air temperature, deg C.
        relative_humidity: Fraction in [0, 1].
        moisture_100h: 100-h dead fuel moisture, fraction.
        state_of_weather: NFDRS cloudiness code 0-3 in daylight, None at night.
    """

    time_utc: dt.datetime
    wind_speed: float
    wind_direction: float
    temperature_c: float
    relative_humidity: float
    moisture_100h: float
    state_of_weather: int | None


@dataclass(frozen=True)
class NfmdSample:
    """One National Fuel Moisture Database sample.

    Attributes:
        date: Sampling date.
        lat, lon: Site location, WGS84 degrees.
        fuel: ``"herb"`` (live herbaceous) or ``"woody"`` (live woody: shrub foliage and twigs).
        moisture: Moisture content, fraction of dry weight (0.65 = 65 %).
    """

    date: dt.date
    lat: float
    lon: float
    fuel: Literal["herb", "woody"]
    moisture: float


@dataclass(frozen=True)
class LiveMoisture:
    """Live fuel moisture for a run and where each value came from."""

    herb: float
    woody: float
    herb_source: str
    woody_source: str


# ----------------------------------------------------------------------------- sun (FAO-56)


@dataclass(frozen=True)
class SunDay:
    """Sun on one day at one latitude.

    Attributes:
        sunrise_lst, sunset_lst: Local solar time, hours after midnight.
        day_length_h: Hours from sunrise to sunset (FAO-56 ``N``).
        ra_mj_m2: Extraterrestrial radiation, MJ m^-2 day^-1 (FAO-56 ``Ra``).
    """

    sunrise_lst: float
    sunset_lst: float
    day_length_h: float
    ra_mj_m2: float


def sun_day(lat_deg: float, date: dt.date) -> SunDay:
    """Day length and extraterrestrial radiation, FAO-56 eqs. 21-25 and 34 (Allen et al. 1998).

    Args:
        lat_deg: Latitude, degrees (north positive).
        date: Calendar date.
    """
    if not -90 <= lat_deg <= 90:
        raise ValueError("latitude must lie in [-90, 90]")
    j = date.timetuple().tm_yday
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * j / 365)
    delta = 0.409 * math.sin(2 * math.pi * j / 365 - 1.39)
    omega = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta))))
    ra = (
        24
        * 60
        / math.pi
        * SOLAR_CONSTANT
        * dr
        * (omega * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(omega))
    )
    n = 24 / math.pi * omega
    return SunDay(sunrise_lst=12 - n / 2, sunset_lst=12 + n / 2, day_length_h=n, ra_mj_m2=max(ra, 0.0))


def state_of_weather_code(srad_w_m2: float, sun: SunDay, params: WeatherParams) -> int:
    """NFDRS state-of-weather code (0-3) of a day from its mean shortwave radiation.

    Relative sunshine ``n/N = (Rs / Ra - a_s) / b_s`` (Angstrom; FAO-56 eq. 35), clipped to [0, 1];
    cloud cover ``1 - n/N`` rounded to tenths; 0 tenths -> 0 (clear), 1-5 -> 1 (scattered), 6-9 -> 2
    (broken), 10 -> 3 (overcast) (Deeming et al. 1977).
    """
    if sun.ra_mj_m2 <= 0:
        return 3
    rs = srad_w_m2 * 0.0864  # W/m^2 averaged over the day -> MJ m^-2 day^-1
    sunshine = (rs / sun.ra_mj_m2 - params.angstrom_a.value) / params.angstrom_b.value
    return state_of_weather_from_cloud(1 - min(1.0, max(0.0, sunshine)))


def state_of_weather_from_cloud(cloud_fraction: float) -> int:
    """NFDRS state-of-weather code (0-3) of a cloud cover fraction (Deeming et al. 1977).

    Cover rounded to tenths: 0 -> 0 (clear), 1-5 -> 1 (scattered), 6-9 -> 2 (broken), 10 -> 3
    (overcast).
    """
    tenths = round(10 * min(1.0, max(0.0, cloud_fraction)))
    if tenths == 0:
        return 0
    if tenths <= 5:
        return 1
    if tenths <= 9:
        return 2
    return 3


# ----------------------------------------------------------------------------- disaggregation


def _local_solar(t_utc: dt.datetime, lon_deg: float) -> tuple[dt.date, float]:
    """Local solar date and hour of day (0-24) of a UTC instant."""
    lst = t_utc.astimezone(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=lon_deg / 15)
    return lst.date(), lst.hour + lst.minute / 60 + lst.second / 3600 + lst.microsecond / 3.6e9


class _Days:
    """Daily records by date, with the sun of each day, for the disaggregation."""

    def __init__(self, days: Sequence[DailyWeather], lat_deg: float, params: WeatherParams) -> None:
        self.by_date = {d.date: d for d in days}
        if len(self.by_date) != len(days):
            raise ValueError("daily records must have distinct dates")
        self.lat = lat_deg
        self.params = params
        self._sun: dict[dt.date, SunDay] = {}

    def day(self, date: dt.date) -> DailyWeather:
        try:
            return self.by_date[date]
        except KeyError:
            raise ValueError(f"no gridMET day {date.isoformat()}: fetch one more day on each side") from None

    def sun(self, date: dt.date) -> SunDay:
        if date not in self._sun:
            self._sun[date] = sun_day(self.lat, date)
        return self._sun[date]

    def _daytime(self, date: dt.date, tau: float) -> tuple[float, float]:
        """Temperature and humidity on the daytime sine of ``date``."""
        d, s, pl = self.day(date), self.sun(date), self.params.parton_logan
        m = tau - (s.sunrise_lst + pl.c.value)
        f = math.sin(math.pi * m / (s.day_length_h + 2 * pl.a.value))
        return d.tmin_c + (d.tmax_c - d.tmin_c) * f, d.rh_max - (d.rh_max - d.rh_min) * f

    def air(self, date: dt.date, tau: float) -> tuple[float, float]:
        """Temperature (deg C) and relative humidity (fraction) at local solar hour ``tau`` of ``date``.

        Temperature: Parton & Logan (1981). Humidity follows the same curve inverted, from the
        day's maximum at the temperature minimum to its minimum at the temperature maximum, and
        decays after sunset with the same factor towards the next day's maximum.
        """
        s, pl = self.sun(date), self.params.parton_logan
        t_min = s.sunrise_lst + pl.c.value
        if t_min <= tau <= s.sunset_lst:
            return self._daytime(date, tau)
        if tau > s.sunset_lst:  # evening: decay towards tomorrow's minimum
            evening, n, morning = date, tau - s.sunset_lst, self.day(date + dt.timedelta(days=1))
        else:  # before dawn: still the decay that began at yesterday's sunset
            evening = date - dt.timedelta(days=1)
            n, morning = tau + 24 - self.sun(evening).sunset_lst, self.day(date)
        se = self.sun(evening)
        t_sunset, rh_sunset = self._daytime(evening, se.sunset_lst)
        g = math.exp(-pl.b.value * n / (24 - se.day_length_h))
        return morning.tmin_c + (t_sunset - morning.tmin_c) * g, morning.rh_max + (rh_sunset - morning.rh_max) * g


def disaggregate(
    days: Sequence[DailyWeather],
    lat_deg: float,
    lon_deg: float,
    start_utc: dt.datetime,
    n_hours: int,
    params: WeatherParams | None = None,
) -> list[HourlyWeather]:
    """Hourly weather from daily gridMET records (profiles in the module docstring).

    Args:
        days: Daily records; must include the local solar dates of every hour and one day on
            each side (the night before the first dawn and after the last dusk).
        lat_deg: Latitude of the site, WGS84 degrees.
        lon_deg: Longitude of the site, WGS84 degrees.
        start_utc: Start of the first hour (timezone-aware; normally on the hour).
        n_hours: Number of hours.
        params: Parameters; None loads ``configs/weather.yaml``.

    Raises:
        ValueError: ``start_utc`` has no timezone, ``n_hours < 1``, or a needed day is missing.
    """
    if start_utc.tzinfo is None:
        raise ValueError("start_utc must be timezone-aware")
    if n_hours < 1:
        raise ValueError("n_hours must be >= 1")
    p = load_weather_params() if params is None else params
    table = _Days(days, lat_deg, p)
    amplitude, peak = p.wind_amplitude_m_s.value, p.wind_peak_hour_lst.value
    hours: list[HourlyWeather] = []
    for k in range(n_hours):
        t = start_utc.astimezone(dt.UTC) + k * HOUR
        date, tau = _local_solar(t, lon_deg)
        day, sun = table.day(date), table.sun(date)
        temperature, rh = table.air(date, tau)
        a = min(amplitude, day.wind_speed)
        wind = day.wind_speed + a * math.cos(2 * math.pi * (tau - peak) / 24)
        daylight = sun.sunrise_lst <= tau <= sun.sunset_lst
        hours.append(
            HourlyWeather(
                time_utc=t,
                wind_speed=max(wind, 0.0),
                wind_direction=day.wind_from_deg % 360.0,
                temperature_c=temperature,
                relative_humidity=min(1.0, max(0.0, rh)),
                moisture_100h=day.moisture_100h,
                state_of_weather=state_of_weather_code(day.srad_w_m2, sun, p) if daylight else None,
            )
        )
    return hours


def solar_dates_needed(lon_deg: float, start_utc: dt.datetime, n_hours: int) -> tuple[dt.date, dt.date]:
    """First and last gridMET day :func:`disaggregate` needs for these hours (one extra each side)."""
    first, _ = _local_solar(start_utc, lon_deg)
    last, _ = _local_solar(start_utc + (n_hours - 1) * HOUR, lon_deg)
    return first - dt.timedelta(days=1), last + dt.timedelta(days=1)


# ----------------------------------------------------------------------------- live moisture


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (haversine, mean Earth radius 6371.0 km)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, h)))


def live_fuel_moisture(
    date: dt.date,
    lat_deg: float,
    lon_deg: float,
    samples: Sequence[NfmdSample] = (),
    params: WeatherParams | None = None,
) -> LiveMoisture:
    """Live herbaceous and woody fuel moisture for a site and date.

    For each fuel class, the mean of the samples taken within ``nfmd_window_days`` of ``date``
    at sites within ``nfmd_radius_km`` of the site; with none, the seasonal default (the Scott &
    Burgan scenario of the month in ``configs/weather.yaml``).
    """
    p = load_weather_params() if params is None else params
    window = dt.timedelta(days=p.nfmd_window_days.value)
    near = [
        s
        for s in samples
        if abs(s.date - date) <= window and _distance_km(lat_deg, lon_deg, s.lat, s.lon) <= p.nfmd_radius_km.value
    ]
    name = p.seasonal_scenario.months[date.month]
    scenario = p.live_scenarios[name]
    result: dict[str, tuple[float, str]] = {}
    for fuel, default in (("herb", scenario.herb.value), ("woody", scenario.woody.value)):
        values = [s.moisture for s in near if s.fuel == fuel]
        if values:
            result[fuel] = (sum(values) / len(values), f"NFMD: mean of {len(values)} samples")
        else:
            result[fuel] = (default, f"seasonal default: Scott & Burgan scenario {name} for month {date.month}")
    return LiveMoisture(result["herb"][0], result["woody"][0], result["herb"][1], result["woody"][1])


# ----------------------------------------------------------------------------- WeatherConfig


def to_weather_config(
    hours: Sequence[HourlyWeather],
    start_time_utc: dt.datetime,
    live: LiveMoisture,
    moisture_params: FuelMoistureParams | None = None,
) -> WeatherConfig:
    """One :class:`WeatherPoint` per hour, with ``time_s`` counted from ``start_time_utc``.

    The first hour must contain the start (it becomes the point at ``time_s = 0``); later hours
    start at ``time_s = hour - start``. The 1-h and 10-h moisture are computed per hour from its
    temperature, humidity and cloudiness code (WP 2.1c).

    Raises:
        ValueError: The hours are empty, not consecutive, or the first does not contain the start.
    """
    if not hours:
        raise ValueError("no hours")
    if start_time_utc.tzinfo is None:
        raise ValueError("start_time_utc must be timezone-aware")
    if any(b.time_utc - a.time_utc != HOUR for a, b in itertools.pairwise(hours)):
        raise ValueError("hours must be consecutive, one hour apart")
    first = hours[0].time_utc
    if not first <= start_time_utc < first + HOUR:
        raise ValueError("the first hour must contain start_time_utc")
    mp = load_fuel_moisture_params() if moisture_params is None else moisture_params
    points = []
    for i, h in enumerate(hours):
        m1, m10 = dead_fuel_moisture(h.temperature_c, h.relative_humidity, h.state_of_weather, mp)
        points.append(
            WeatherPoint(
                time_s=0.0 if i == 0 else (h.time_utc - start_time_utc).total_seconds(),
                wind_speed=h.wind_speed,
                wind_direction=h.wind_direction,
                moisture_1h=float(m1),
                moisture_10h=float(m10),
                moisture_100h=h.moisture_100h,
                moisture_live_herb=live.herb,
                moisture_live_woody=live.woody,
                temperature_c=h.temperature_c,
                relative_humidity=h.relative_humidity,
            )
        )
    return WeatherConfig(series=points)


# ----------------------------------------------------------------------------- gridMET client


@dataclass(frozen=True)
class Packing:
    """How a gridMET variable is stored: ``value = raw * scale + offset``; ``missing`` raw values are absent."""

    scale: float = 1.0
    offset: float = 0.0
    missing: tuple[float, ...] = ()
    unsigned: bool = False

    def unpack(self, raw: float) -> float:
        if raw in self.missing:
            return math.nan
        if self.unsigned and raw < 0:  # a 16-bit unsigned value read as signed
            raw += 65536
        return raw * self.scale + self.offset


def parse_packing(das: str, variable: str) -> Packing:
    """The packing attributes of ``variable`` from an OPeNDAP DAS response.

    Raises:
        ValueError: The variable has no attribute block.
    """
    block = re.search(rf"^\s*{re.escape(variable)}\s*\{{(.*?)^\s*\}}", das, flags=re.M | re.S)
    if block is None:
        raise ValueError(f"no attributes for {variable!r} in the DAS response")
    text = block.group(1)

    def number(name: str) -> float | None:
        m = re.search(rf"\b{name}\s+(-?[\d.eE+-]+)\s*;", text)
        return float(m.group(1)) if m else None

    scale, offset = number("scale_factor"), number("add_offset")
    missing = tuple(v for v in (number("_FillValue"), number("missing_value")) if v is not None)
    unsigned = re.search(r'_Unsigned\s+"true"', text) is not None
    return Packing(
        scale=1.0 if scale is None else scale,
        offset=0.0 if offset is None else offset,
        missing=missing,
        unsigned=unsigned,
    )


def parse_ncss_csv(text: str) -> dict[dt.date, float]:
    """Date -> raw value from a NetCDF Subset Service point CSV (time first, value last).

    Raises:
        ValueError: The response has no data rows.
    """
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) < 2 or not rows[0] or rows[0][0] != "time":
        raise ValueError(f"unexpected NCSS response: {text[:200]!r}")
    return {dt.date.fromisoformat(r[0][:10]): float(r[-1]) for r in rows[1:] if r}


def gridmet_urls(
    short: str, lat_deg: float, lon_deg: float, start: dt.date, end: dt.date, base: str = GRIDMET_THREDDS
) -> tuple[str, str]:
    """The NCSS point-query URL and the DAS URL of one gridMET variable."""
    dataset = f"agg_met_{short}_1979_CurrentYear_CONUS.nc"
    query = urllib.parse.urlencode(
        {
            "var": GRIDMET_VARIABLES[short],
            "latitude": f"{lat_deg:.6f}",
            "longitude": f"{lon_deg:.6f}",
            "time_start": f"{start.isoformat()}T00:00:00Z",
            "time_end": f"{end.isoformat()}T00:00:00Z",
            "accept": "csv",
        }
    )
    return f"{base}/ncss/{dataset}?{query}", f"{base}/dodsC/{dataset}.das"


def daily_from_gridmet(values: Mapping[str, Mapping[dt.date, float]]) -> list[DailyWeather]:
    """:class:`DailyWeather` records from unpacked gridMET values (gridMET units), by date.

    Temperatures K -> deg C; humidity and 100-h moisture percent -> fraction (humidity clipped
    to [0, 1]); direction taken modulo 360.

    Raises:
        ValueError: A variable is missing, or the variables cover different dates.
    """
    missing = set(GRIDMET_VARIABLES) - set(values)
    if missing:
        raise ValueError(f"missing gridMET variables {sorted(missing)}")
    dates = sorted(values["tmmx"])
    if any(sorted(values[k]) != dates for k in GRIDMET_VARIABLES):
        raise ValueError("gridMET variables cover different dates")

    def rh(x: float) -> float:
        return min(1.0, max(0.0, x / 100))

    return [
        DailyWeather(
            date=d,
            tmax_c=values["tmmx"][d] - 273.15,
            tmin_c=values["tmmn"][d] - 273.15,
            rh_max=rh(values["rmax"][d]),
            rh_min=rh(values["rmin"][d]),
            wind_speed=values["vs"][d],
            wind_from_deg=values["th"][d] % 360.0,
            moisture_100h=values["fm100"][d] / 100,
            srad_w_m2=values["srad"][d],
        )
        for d in dates
    ]


_CSV_FIELDS = (
    "date",
    "tmax_c",
    "tmin_c",
    "rh_max",
    "rh_min",
    "wind_speed",
    "wind_from_deg",
    "moisture_100h",
    "srad_w_m2",
)


def _write_daily(path: Path, days: Sequence[DailyWeather]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(_CSV_FIELDS)
        for d in days:
            w.writerow([d.date.isoformat(), *(repr(getattr(d, f)) for f in _CSV_FIELDS[1:])])


def _read_daily(path: Path) -> list[DailyWeather]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            DailyWeather(
                date=dt.date.fromisoformat(r["date"]),
                tmax_c=float(r["tmax_c"]),
                tmin_c=float(r["tmin_c"]),
                rh_max=float(r["rh_max"]),
                rh_min=float(r["rh_min"]),
                wind_speed=float(r["wind_speed"]),
                wind_from_deg=float(r["wind_from_deg"]),
                moisture_100h=float(r["moisture_100h"]),
                srad_w_m2=float(r["srad_w_m2"]),
            )
            for r in csv.DictReader(fh)
        ]


@dataclass(frozen=True)
class GridmetDaily:
    """Cached gridMET days at a point.

    Attributes:
        days: The records, by date.
        directory: ``<cache root>/gridmet-<key>/``.
        manifest: The contents of ``manifest.json``.
    """

    days: list[DailyWeather]
    directory: Path
    manifest: dict[str, Any]


def gridmet_cache_key(lat_deg: float, lon_deg: float, start: dt.date, end: dt.date) -> str:
    """``gridmet-`` + 16 hex digits identifying a (point, date range) request."""
    canon = json.dumps(
        {"lat": f"{lat_deg:.6f}", "lon": f"{lon_deg:.6f}", "start": start.isoformat(), "end": end.isoformat()},
        sort_keys=True,
    )
    return "gridmet-" + hashlib.sha256(canon.encode()).hexdigest()[:16]


def fetch_gridmet(
    lat_deg: float,
    lon_deg: float,
    start: dt.date,
    end: dt.date,
    *,
    cache_root: str | Path = DEFAULT_CACHE,
    http: GetBytes | None = None,
    base: str = GRIDMET_THREDDS,
) -> GridmetDaily:
    """gridMET days ``start`` to ``end`` (inclusive) at a point, from the cache or the service.

    Writes ``daily.csv`` (simulator units) and ``manifest.json`` (source URLs, packing, access
    time, SHA-256) under ``<cache_root>/gridmet-<key>/``.

    Raises:
        ValueError: ``end`` before ``start``, the service's answer is incomplete, or a cached
            file does not match its hash.
    """
    if end < start:
        raise ValueError("end must not be before start")
    directory = Path(cache_root) / gridmet_cache_key(lat_deg, lon_deg, start, end)
    manifest_path, data_path = directory / "manifest.json", directory / "daily.csv"
    if manifest_path.exists() and data_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["sha256"]["daily.csv"]:
            raise ValueError(f"{directory}: daily.csv does not match its manifest hash; delete the directory")
        return GridmetDaily(_read_daily(data_path), directory, manifest)

    client: GetBytes = UrllibHttp() if http is None else http
    values: dict[str, dict[dt.date, float]] = {}
    sources: dict[str, Any] = {}
    for short, name in GRIDMET_VARIABLES.items():
        url, das_url = gridmet_urls(short, lat_deg, lon_deg, start, end, base)
        packing = parse_packing(client.get_bytes(das_url).decode("utf-8"), name)
        raw = parse_ncss_csv(client.get_bytes(url).decode("utf-8"))
        values[short] = {d: packing.unpack(v) for d, v in raw.items()}
        sources[short] = {"url": url, "scale_factor": packing.scale, "add_offset": packing.offset}
    expected = {start + dt.timedelta(days=i) for i in range((end - start).days + 1)}
    for short, by_date in values.items():
        if set(by_date) != expected:
            raise ValueError(f"gridMET {short}: expected {len(expected)} days, got {len(by_date)}")
    days = daily_from_gridmet(values)
    directory.mkdir(parents=True, exist_ok=True)
    _write_daily(data_path, days)
    manifest = {
        "dataset": "gridMET",
        "point_wgs84": {"lat": lat_deg, "lon": lon_deg},
        "start": start.isoformat(),
        "end": end.isoformat(),
        "accessed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": sources,
        "sha256": {"daily.csv": hashlib.sha256(data_path.read_bytes()).hexdigest()},
        "attribution": "gridMET, Abatzoglou (2013), Int. J. Climatol. 33, 121-131; Climatology Lab, University of "
        "California Merced",
        "note": "days after the dataset's last permanent slice are provisional and may change",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return GridmetDaily(days, directory, manifest)


# ----------------------------------------------------------------------------- the whole chain


def bbox_centroid(bbox: BBox) -> tuple[float, float]:
    """``(lat, lon)`` of the middle of a bounding box, degrees (adequate for boxes of tens of km)."""
    return (bbox.south + bbox.north) / 2, (bbox.west + bbox.east) / 2


def hourly_from_hrrr(
    hrrr: Sequence[Any],
    days: Sequence[DailyWeather],
    lat_deg: float,
    lon_deg: float,
) -> list[HourlyWeather]:
    """Hourly weather from HRRR analyses (:mod:`wildfire_sim.data.hrrr`), with gridMET's 100-h moisture.

    Temperature, humidity and the 10 m wind are the hour's analysis at the site. The cloudiness
    code for the 1-h moisture comes from the hour's total cloud cover (an observation of that
    hour, better than :func:`state_of_weather_code`'s daily estimate) in daylight, none at night.
    The 100-h moisture is gridMET's for the hour's local solar day (HRRR has no fuel moisture).

    Args:
        hrrr: :class:`~wildfire_sim.data.hrrr.HrrrHour` records, one per hour.
        days: gridMET days covering the hours' local solar dates.
        lat_deg, lon_deg: The site, degrees.

    Raises:
        ValueError: A needed gridMET day is missing.
    """
    by_date = {d.date: d for d in days}
    out = []
    for h in hrrr:
        date, tau = _local_solar(h.time_utc, lon_deg)
        if date not in by_date:
            raise ValueError(f"no gridMET day {date.isoformat()} for the 100-h moisture")
        sun = sun_day(lat_deg, date)
        daylight = sun.sunrise_lst <= tau <= sun.sunset_lst
        out.append(
            HourlyWeather(
                time_utc=h.time_utc,
                wind_speed=h.wind_speed,
                wind_direction=h.wind_from_deg % 360.0,
                temperature_c=h.temperature_c,
                relative_humidity=h.relative_humidity,
                moisture_100h=by_date[date].moisture_100h,
                state_of_weather=state_of_weather_from_cloud(h.cloud_fraction) if daylight else None,
            )
        )
    return out


def hourly_weather(
    lat_deg: float,
    lon_deg: float,
    start_time_utc: dt.datetime,
    duration_s: float,
    *,
    source: Literal["gridmet", "hrrr"] = "gridmet",
    samples: Sequence[NfmdSample] = (),
    cache_root: str | Path = DEFAULT_CACHE,
    http: GetBytes | None = None,
    hrrr_get: Any = None,
    params: WeatherParams | None = None,
    moisture_params: FuelMoistureParams | None = None,
) -> WeatherConfig:
    """The ``WeatherConfig`` of a run at the site, one point per hour.

    With ``source="gridmet"`` the hours are gridMET's days disaggregated (:func:`disaggregate`);
    with ``source="hrrr"`` they are HRRR analyses (:func:`hourly_from_hrrr`), the 100-h moisture
    still from gridMET.

    Args:
        lat_deg: Latitude of the site, the grid centroid (``Grid.metadata["centroid_lat"]``,
            WP 2.3, or :func:`bbox_centroid`).
        lon_deg: Longitude of the site (``Grid.metadata["centroid_lon"]``).
        start_time_utc: ``ScenarioConfig.start_time_utc`` (timezone-aware).
        duration_s: ``ScenarioConfig.duration_s``.
        source: ``"gridmet"`` or ``"hrrr"``.
        samples: NFMD samples for the live fuel moisture; none uses the seasonal default.
        cache_root: Directory holding ``gridmet-<key>/`` (and ``hrrr-<key>/``).
        http: HTTP client for gridMET; None uses :class:`~wildfire_sim.data.landfire.UrllibHttp`.
        hrrr_get: Download function for HRRR (:data:`wildfire_sim.data.hrrr.Get`); None uses urllib.
        params: Parameters; None loads ``configs/weather.yaml``.
        moisture_params: Fuel moisture parameters; None loads ``configs/fuel_moisture.yaml``.
    """
    if source not in ("gridmet", "hrrr"):
        raise ValueError(f"unknown weather source {source!r}; expected 'gridmet' or 'hrrr'")
    if start_time_utc.tzinfo is None:
        raise ValueError("start_time_utc must be timezone-aware")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    p = load_weather_params() if params is None else params
    start = start_time_utc.astimezone(dt.UTC)
    first_hour = start.replace(minute=0, second=0, microsecond=0)
    n_hours = math.ceil((start - first_hour).total_seconds() / 3600 + duration_s / 3600)
    first_day, last_day = solar_dates_needed(lon_deg, first_hour, n_hours)
    daily = fetch_gridmet(lat_deg, lon_deg, first_day, last_day, cache_root=cache_root, http=http)
    if source == "hrrr":
        from wildfire_sim.data.hrrr import fetch_hrrr_point

        series = fetch_hrrr_point(lat_deg, lon_deg, first_hour, n_hours, cache_root=cache_root, get=hrrr_get)
        hours = hourly_from_hrrr(series.hours, daily.days, lat_deg, lon_deg)
    else:
        hours = disaggregate(daily.days, lat_deg, lon_deg, first_hour, n_hours, p)
    live = live_fuel_moisture(_local_solar(start, lon_deg)[0], lat_deg, lon_deg, samples, p)
    return to_weather_config(hours, start, live, moisture_params)


# ----------------------------------------------------------------------------- HRRR coverage


def hrrr_missing_hours(
    start: dt.datetime, end: dt.datetime, http: GetBytes | None = None, bucket: str = HRRR_BUCKET
) -> list[dt.datetime]:
    """UTC hours in ``[start, end)`` without an HRRR CONUS surface analysis (``wrfsfcf00``) in the archive.

    Lists the bucket one day at a time (following continuation tokens). Used to decide between
    HRRR and gridMET (section 13).
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    client: GetBytes = UrllibHttp() if http is None else http
    s3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    first = start.astimezone(dt.UTC).replace(minute=0, second=0, microsecond=0)
    if first < start:
        first += HOUR
    end_utc = end.astimezone(dt.UTC)
    wanted: list[dt.datetime] = []
    t = first
    while t < end_utc:
        wanted.append(t)
        t += HOUR
    present: set[dt.datetime] = set()
    for day in sorted({w.date() for w in wanted}):
        prefix = f"hrrr.{day:%Y%m%d}/conus/hrrr.t"
        token: str | None = None
        while True:
            query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                query["continuation-token"] = token
            root = ET.fromstring(client.get_bytes(f"{bucket}/?{urllib.parse.urlencode(query)}"))
            for key in root.iter(f"{s3}Key"):
                m = re.search(r"hrrr\.t(\d\d)z\.wrfsfcf00\.grib2$", key.text or "")
                if m:
                    present.add(dt.datetime(day.year, day.month, day.day, int(m.group(1)), tzinfo=dt.UTC))
            nxt = root.find(f"{s3}NextContinuationToken")
            token = nxt.text if nxt is not None else None
            if not token:
                break
    return [w for w in wanted if w not in present]
