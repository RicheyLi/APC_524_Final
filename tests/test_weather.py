"""Tests for wildfire_sim.data.weather (WP 2.4).

Acceptance (docs/plan.md WP 2.4): mocked datasets convert to a valid ``WeatherConfig``;
disaggregated daily data preserves the daily mean; live test marked ``network``. The mocked
gridMET responses use the values the service returned for the Chimney Fire origin (35.72 N,
121.02 W) on 12-16 August 2016, packed as the service packs them.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import urllib.parse
from pathlib import Path

import numpy as np
import pytest

from wildfire_sim.config import WeatherConfig
from wildfire_sim.data.landfire import BBox
from wildfire_sim.data.weather import (
    GRIDMET_VARIABLES,
    DailyWeather,
    NfmdSample,
    Packing,
    WeatherParams,
    _distance_km,
    bbox_centroid,
    daily_from_gridmet,
    disaggregate,
    fetch_gridmet,
    hourly_weather,
    hrrr_missing_hours,
    live_fuel_moisture,
    load_live_moisture_samples,
    load_weather_params,
    parse_ncss_csv,
    parse_packing,
    solar_dates_needed,
    state_of_weather_code,
    sun_day,
    to_weather_config,
)
from wildfire_sim.spread.moisture import load_fuel_moisture_params

LAT, LON = 35.72, -121.02
UTC = dt.UTC

# Raw (packed) values from the gridMET NCSS service, 2016-08-12 .. 2016-08-16.
RAW = {
    "tmmx": [900, 931, 929, 898, 901],
    "tmmn": [718, 746, 739, 727, 719],
    "rmax": [879, 766, 799, 827, 738],
    "rmin": [129, 103, 109, 127, 99],
    "vs": [24, 24, 25, 27, 24],
    "th": [238, 216, 242, 239, 220],
    "fm100": [117, 105, 100, 98, 92],
    "srad": [3227, 3211, 3245, 3244, 3265],
}
PACKING = {k: (0.1, 220.0) if k in ("tmmx", "tmmn") else (1.0, 0.0) if k == "th" else (0.1, 0.0) for k in RAW}
FIRST = dt.date(2016, 8, 12)


@pytest.fixture(scope="module")
def params() -> WeatherParams:
    return load_weather_params()


@pytest.fixture(scope="module")
def days() -> list[DailyWeather]:
    values = {
        k: {FIRST + dt.timedelta(days=i): r * PACKING[k][0] + PACKING[k][1] for i, r in enumerate(v)}
        for k, v in RAW.items()
    }
    return daily_from_gridmet(values)


def _das(name: str, scale: float, offset: float) -> str:
    return (
        'Attributes {\n    lat {\n        String units "degrees_north";\n    }\n'
        f'    {name} {{\n        Int16 _FillValue 32767;\n        String units "x";\n'
        f"        Int16 missing_value 32767;\n        Float64 scale_factor {scale};\n"
        f'        Float64 add_offset {offset};\n        String _Unsigned "true";\n    }}\n}}\n'
    )


def _csv(name: str, values: dict[dt.date, float]) -> str:
    rows = [f'time,latitude[unit="degrees_north"],longitude[unit="degrees_east"],{name}[unit="x"]']
    rows += [f"{d.isoformat()}T00:00:00Z,{LAT},{LON},{v}" for d, v in values.items()]
    return "\n".join(rows) + "\n"


class FakeGridmet:
    """Serves DAS and NCSS CSV responses for the eight variables; counts calls."""

    def __init__(self, first: dt.date = FIRST, n_days: int = 5) -> None:
        self.calls: list[str] = []
        self.first, self.n_days = first, n_days

    def get_bytes(self, url: str) -> bytes:
        self.calls.append(url)
        short = url.split("agg_met_")[1].split("_1979")[0]
        name = GRIDMET_VARIABLES[short]
        if url.endswith(".das"):
            return _das(name, *PACKING[short]).encode()
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start = dt.date.fromisoformat(q["time_start"][0][:10])
        end = dt.date.fromisoformat(q["time_end"][0][:10])
        n = (end - start).days + 1
        raw = RAW[short]
        return _csv(name, {start + dt.timedelta(days=i): float(raw[i % len(raw)]) for i in range(n)}).encode()


# ----------------------------------------------------------------------------- sun


def test_fao56_example_8_and_9() -> None:
    """FAO-56 examples 8 and 9: 20 deg S on 3 September, Ra = 32.2 MJ m^-2 day^-1, N = 11.7 h."""
    sun = sun_day(-20.0, dt.date(2015, 9, 3))
    assert sun.ra_mj_m2 == pytest.approx(32.2, abs=0.05)
    assert sun.day_length_h == pytest.approx(11.7, abs=0.05)
    assert sun.sunrise_lst + sun.sunset_lst == pytest.approx(24.0)


def test_day_length_is_longer_in_summer_and_polar_limits() -> None:
    assert sun_day(LAT, dt.date(2016, 6, 21)).day_length_h > 14 > sun_day(LAT, dt.date(2016, 12, 21)).day_length_h
    assert sun_day(80.0, dt.date(2016, 6, 21)).day_length_h == pytest.approx(24.0)
    assert sun_day(80.0, dt.date(2016, 12, 21)).day_length_h == pytest.approx(0.0)
    with pytest.raises(ValueError):
        sun_day(91.0, dt.date(2016, 1, 1))


def test_state_of_weather_from_radiation(params: WeatherParams) -> None:
    sun = sun_day(LAT, dt.date(2016, 8, 13))
    ra_w = sun.ra_mj_m2 / 0.0864  # extraterrestrial radiation as a daily mean, W/m^2
    a, b = params.angstrom_a.value, params.angstrom_b.value
    assert state_of_weather_code((a + b) * ra_w, sun, params) == 0  # full sunshine: clear
    assert state_of_weather_code((a + 0.7 * b) * ra_w, sun, params) == 1  # 3/10 cloud
    assert state_of_weather_code((a + 0.2 * b) * ra_w, sun, params) == 2  # 8/10
    assert state_of_weather_code(a * ra_w, sun, params) == 3  # no sunshine: overcast
    assert state_of_weather_code(0.0, sun, params) == 3


# ----------------------------------------------------------------------------- disaggregation


def _by_solar_day(hours: list, lon: float) -> dict[dt.date, list]:
    out: dict[dt.date, list] = {}
    for h in hours:
        lst = h.time_utc + dt.timedelta(hours=lon / 15)
        out.setdefault(lst.date(), []).append(h)
    return out


def _three_full_days(days: list[DailyWeather], params: WeatherParams) -> tuple[list, dict[dt.date, list]]:
    # local solar midnight starting 13 August at 121.02 W is 08:04 UTC, so 72 hours from 09 UTC
    # hold three whole solar days (13-15 August)
    start = dt.datetime(2016, 8, 13, 9, tzinfo=UTC)
    hours = disaggregate(days, LAT, LON, start, 72, params)
    groups = {d: hs for d, hs in _by_solar_day(hours, LON).items() if len(hs) == 24}
    return hours, groups


def test_disaggregation_preserves_the_daily_mean_wind(days: list[DailyWeather], params: WeatherParams) -> None:
    """Acceptance: each solar day's 24 hourly winds average to gridMET's daily mean."""
    _, groups = _three_full_days(days, params)
    assert len(groups) == 3
    by_date = {d.date: d for d in days}
    for date, hs in groups.items():
        assert np.mean([h.wind_speed for h in hs]) == pytest.approx(by_date[date].wind_speed, rel=1e-12)
        speeds = [h.wind_speed for h in hs]
        assert max(speeds) - min(speeds) == pytest.approx(2 * params.wind_amplitude_m_s.value, abs=0.05)


def test_wind_amplitude_is_limited_by_the_mean(days: list[DailyWeather], params: WeatherParams) -> None:
    calm = [dataclasses.replace(d, wind_speed=0.3) for d in days]
    _, groups = _three_full_days(calm, params)
    for hs in groups.values():
        speeds = [h.wind_speed for h in hs]
        assert min(speeds) >= 0
        assert np.mean(speeds) == pytest.approx(0.3, rel=1e-12)


def test_temperature_and_humidity_reach_gridmet_extremes(days: list[DailyWeather], params: WeatherParams) -> None:
    """Hourly maximum and minimum within the sampling of the curve; peak in the early afternoon."""
    _, groups = _three_full_days(days, params)
    by_date = {d.date: d for d in days}
    for date, hs in groups.items():
        d = by_date[date]
        temps = [h.temperature_c for h in hs]
        rhs = [h.relative_humidity for h in hs]
        assert d.tmax_c - 0.2 <= max(temps) <= d.tmax_c + 1e-9
        assert min(rhs) == pytest.approx(d.rh_min, abs=0.005)
        assert min(rhs) >= d.rh_min - 1e-9
        hottest = hs[int(np.argmax(temps))].time_utc + dt.timedelta(hours=LON / 15)
        assert 13 <= hottest.hour <= 15
        # dawn: within the night decay's undershoot of the minimum (11 % of the evening fall)
        assert min(temps) <= d.tmin_c + 0.15 * (d.tmax_c - d.tmin_c)
        assert np.corrcoef(temps, rhs)[0, 1] < -0.99


def test_daytime_curve_hits_the_extremes_exactly(days: list[DailyWeather], params: WeatherParams) -> None:
    """At the times the model puts them, the temperature and humidity equal gridMET's values."""
    d = days[1]
    sun = sun_day(LAT, d.date)
    pl = params.parton_logan
    t_min = sun.sunrise_lst + pl.c.value
    t_max = t_min + (sun.day_length_h + 2 * pl.a.value) / 2
    for tau, temp, rh in ((t_min, d.tmin_c, d.rh_max), (t_max, d.tmax_c, d.rh_min)):
        midnight = dt.datetime(d.date.year, d.date.month, d.date.day, tzinfo=UTC) - dt.timedelta(hours=LON / 15)
        (h,) = disaggregate(days, LAT, LON, midnight + dt.timedelta(hours=tau), 1, params)
        assert h.temperature_c == pytest.approx(temp, abs=1e-9)
        assert h.relative_humidity == pytest.approx(rh, abs=1e-9)


def test_continuous_at_sunset_and_midnight(days: list[DailyWeather], params: WeatherParams) -> None:
    d = days[1]
    sun = sun_day(LAT, d.date)
    midnight = dt.datetime(d.date.year, d.date.month, d.date.day, tzinfo=UTC) - dt.timedelta(hours=LON / 15)
    eps = dt.timedelta(seconds=1)
    for tau in (sun.sunset_lst, 24.0):
        t = midnight + dt.timedelta(hours=tau)
        before, after = (disaggregate(days, LAT, LON, x, 1, params)[0] for x in (t - eps, t + eps))
        # two seconds apart: at most a few thousandths of a degree from the slope, no step
        assert after.temperature_c == pytest.approx(before.temperature_c, abs=0.01)
        assert after.relative_humidity == pytest.approx(before.relative_humidity, abs=1e-3)


def test_daily_values_held_and_codes_only_by_day(days: list[DailyWeather], params: WeatherParams) -> None:
    hours, groups = _three_full_days(days, params)
    by_date = {d.date: d for d in days}
    for date, hs in groups.items():
        assert {h.wind_direction for h in hs} == {by_date[date].wind_from_deg}
        assert {h.moisture_100h for h in hs} == {by_date[date].moisture_100h}
        sun = sun_day(LAT, date)
        for h in hs:
            lst = h.time_utc + dt.timedelta(hours=LON / 15)
            tau = lst.hour + lst.minute / 60
            if sun.sunrise_lst + 0.5 < tau < sun.sunset_lst - 0.5:
                assert h.state_of_weather in (0, 1)  # clear August days
            elif tau < sun.sunrise_lst - 0.5 or tau > sun.sunset_lst + 0.5:
                assert h.state_of_weather is None
    assert all(h.time_utc.tzinfo is not None for h in hours)


def test_missing_day_and_bad_input(days: list[DailyWeather], params: WeatherParams) -> None:
    with pytest.raises(ValueError, match="no gridMET day"):
        disaggregate(days, LAT, LON, dt.datetime(2016, 8, 16, 12, tzinfo=UTC), 24, params)
    with pytest.raises(ValueError, match="timezone"):
        disaggregate(days, LAT, LON, dt.datetime(2016, 8, 13, 12), 3, params)
    with pytest.raises(ValueError):
        disaggregate(days, LAT, LON, dt.datetime(2016, 8, 13, 12, tzinfo=UTC), 0, params)
    with pytest.raises(ValueError, match="distinct"):
        disaggregate([days[1], days[1]], LAT, LON, dt.datetime(2016, 8, 13, 20, tzinfo=UTC), 1, params)
    with pytest.raises(ValueError, match="tmax_c < tmin_c"):
        DailyWeather(FIRST, 10.0, 20.0, 0.8, 0.2, 1.0, 0.0, 0.1, 300.0)
    with pytest.raises(ValueError, match="finite"):
        DailyWeather(FIRST, math.nan, 20.0, 0.8, 0.2, 1.0, 0.0, 0.1, 300.0)


def test_solar_dates_needed() -> None:
    first, last = solar_dates_needed(LON, dt.datetime(2016, 8, 13, 20, tzinfo=UTC), 24)
    assert (first, last) == (dt.date(2016, 8, 12), dt.date(2016, 8, 15))


# ----------------------------------------------------------------------------- WeatherConfig


def test_to_weather_config_is_valid_and_hourly(days: list[DailyWeather], params: WeatherParams) -> None:
    start = dt.datetime(2016, 8, 13, 20, 30, tzinfo=UTC)
    hours = disaggregate(days, LAT, LON, dt.datetime(2016, 8, 13, 20, tzinfo=UTC), 25, params)
    live = live_fuel_moisture(dt.date(2016, 8, 13), LAT, LON, params=params)
    cfg = to_weather_config(hours, start, live)
    assert isinstance(WeatherConfig.model_validate(cfg.model_dump()), WeatherConfig)
    times = [p.time_s for p in cfg.series]
    assert times[:3] == [0.0, 1800.0, 5400.0]
    assert len(cfg.series) == 25
    ratio = load_fuel_moisture_params().ten_hour_factor.value / load_fuel_moisture_params().one_hour_factor.value
    for p in cfg.series:
        assert p.moisture_10h == pytest.approx(ratio * p.moisture_1h)
        assert (p.moisture_live_herb, p.moisture_live_woody) == (0.30, 0.60)
    m1 = [p.moisture_1h for p in cfg.series]
    afternoon, dawn = m1[0], m1[13]  # 13:30 and 02:30 PDT
    assert afternoon < 0.05 < dawn
    series = cfg.to_series()
    assert series.at(0.0).wind_direction == hours[0].wind_direction


def test_to_weather_config_rejects_bad_hours(days: list[DailyWeather], params: WeatherParams) -> None:
    hours = disaggregate(days, LAT, LON, dt.datetime(2016, 8, 13, 20, tzinfo=UTC), 3, params)
    live = live_fuel_moisture(dt.date(2016, 8, 13), LAT, LON, params=params)
    with pytest.raises(ValueError, match="contain"):
        to_weather_config(hours, dt.datetime(2016, 8, 13, 21, tzinfo=UTC), live)
    with pytest.raises(ValueError, match="consecutive"):
        to_weather_config([hours[0], hours[2]], hours[0].time_utc, live)
    with pytest.raises(ValueError):
        to_weather_config([], hours[0].time_utc, live)


# ----------------------------------------------------------------------------- live moisture


def test_live_moisture_seasonal_default(params: WeatherParams) -> None:
    live = live_fuel_moisture(dt.date(2016, 8, 13), LAT, LON, params=params)
    assert (live.herb, live.woody) == (0.30, 0.60)
    assert "L1" in live.herb_source
    spring = live_fuel_moisture(dt.date(2016, 3, 1), LAT, LON, params=params)
    assert (spring.herb, spring.woody) == (1.20, 1.50)


def test_live_moisture_from_nearby_samples(params: WeatherParams) -> None:
    day = dt.date(2016, 8, 13)
    samples = [
        NfmdSample(day - dt.timedelta(days=5), LAT + 0.1, LON, "woody", 0.62),
        NfmdSample(day + dt.timedelta(days=10), LAT, LON - 0.1, "woody", 0.70),
        NfmdSample(day - dt.timedelta(days=90), LAT, LON, "woody", 1.40),  # too old
        NfmdSample(day, LAT + 2.0, LON, "woody", 1.40),  # about 220 km away
    ]
    live = live_fuel_moisture(day, LAT, LON, samples, params)
    assert live.woody == pytest.approx(0.66)
    assert live.woody_source.startswith("NFMD: mean of 2")
    assert live.herb == 0.30 and "seasonal" in live.herb_source


# Grid centroid of the built lake_2024 region (data/cache/lake_2024/manifest.json, "weather.point").
LAKE_CENTROID = (34.762242, -120.034626)
LAKE_WEATHER_DATE = dt.date(2024, 7, 5)  # local date of the region's weather start (configs/regions.yaml)


def test_live_moisture_samples_file_loads() -> None:
    samples = load_live_moisture_samples()
    assert len(samples) == 10
    assert {s.fuel for s in samples} == {"woody"}
    assert {s.date for s in samples} == {dt.date(2024, 6, 6), dt.date(2024, 7, 2)}
    assert all(0.5 < s.moisture < 1.5 for s in samples)


def test_lake_sample_sites_are_well_inside_the_radius(params: WeatherParams) -> None:
    # The site coordinates are estimates from the place names; they decide only whether a site
    # is within nfmd_radius_km, so each must be inside it by more than the estimate's error.
    margin_km = 5.0
    for s in load_live_moisture_samples():
        assert _distance_km(*LAKE_CENTROID, s.lat, s.lon) <= params.nfmd_radius_km.value - margin_km


def test_lake_2024_live_moisture_from_county_samples(params: WeatherParams) -> None:
    live = live_fuel_moisture(LAKE_WEATHER_DATE, *LAKE_CENTROID, load_live_moisture_samples(), params)
    # chamise, Santa Barbara County Fire Department: 6 June (100, 88, 99, 83, 94 %) and 2 July 2024
    # (83, 81, 95, 74, 86 %)
    assert live.woody == pytest.approx(0.883)
    assert live.woody_source.startswith("NFMD: mean of 10")
    assert live.herb == 0.30 and "seasonal" in live.herb_source  # no herbaceous samples


def test_silverado_keeps_the_seasonal_default(params: WeatherParams) -> None:
    live = live_fuel_moisture(dt.date(2020, 10, 26), 33.75, -117.69, load_live_moisture_samples(), params)
    assert (live.herb, live.woody) == (0.30, 0.60)


def test_live_moisture_samples_validation(tmp_path: Path) -> None:
    good = {
        "sources": {"s": "a report"},
        "sites": {"A": {"lat": 34.0, "lon": -120.0, "location_source": "estimate"}},
        "samples": [{"site": "A", "date": "2024-07-02", "fuel": "woody", "moisture": 0.8, "source": "s"}],
    }
    path = tmp_path / "lfm.yaml"
    for change, match in (
        ({"site": "B"}, "unknown site"),
        ({"source": "t"}, "unknown source"),
        ({"moisture": 0.0}, "greater than 0"),
        ({"fuel": "dead"}, "fuel"),
    ):
        raw = {**good, "samples": [{**good["samples"][0], **change}]}
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_live_moisture_samples(path)


def test_region_weather_uses_the_live_moisture_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from wildfire_sim.data import cache

    seen: dict[str, object] = {}

    def fake_hourly_weather(*args: object, **kwargs: object) -> str:
        seen.update(kwargs)
        return "weather"

    monkeypatch.setattr(cache, "hourly_weather", fake_hourly_weather)
    start = dt.datetime(2024, 7, 5, 7, tzinfo=UTC)
    assert cache._weather_source(*LAKE_CENTROID, start, 3600.0, tmp_path, "hrrr") == "weather"
    assert seen["samples"] == load_live_moisture_samples()
    assert seen["source"] == "hrrr"


def test_params_validation(params: WeatherParams) -> None:
    raw = params.model_dump()
    raw["seasonal_scenario"]["months"].pop(5)
    with pytest.raises(ValueError, match="every month"):
        WeatherParams.model_validate(raw)
    raw = params.model_dump()
    raw["seasonal_scenario"]["months"][5] = "L9"
    with pytest.raises(ValueError, match="unknown"):
        WeatherParams.model_validate(raw)


# ----------------------------------------------------------------------------- gridMET client


REAL_DAS = """Attributes {
    lat {
        String units "degrees_north";
        Int32 _ChunkSizes 585;
    }
    daily_maximum_temperature {
        Int16 _FillValue 32767;
        String units "K";
        Int16 missing_value 32767;
        Float64 scale_factor 0.1;
        Float64 add_offset 220.0;
        String _Unsigned "true";
        Int32 _ChunkSizes 61, 98, 231;
    }
    day {
        String units "days since 1900-01-01 00:00:00";
    }
}
"""


def test_parse_packing_and_unpack() -> None:
    p = parse_packing(REAL_DAS, "daily_maximum_temperature")
    assert p == Packing(scale=0.1, offset=220.0, missing=(32767.0, 32767.0), unsigned=True)
    assert p.unpack(931.0) == pytest.approx(313.1)
    assert math.isnan(p.unpack(32767.0))
    assert p.unpack(-1.0) == pytest.approx(65535 * 0.1 + 220)
    assert parse_packing(REAL_DAS, "day") == Packing()
    with pytest.raises(ValueError):
        parse_packing(REAL_DAS, "daily_minimum_temperature")


def test_parse_ncss_csv() -> None:
    text = _csv("x", {dt.date(2016, 8, 13): 931.0, dt.date(2016, 8, 14): 929.0})
    assert parse_ncss_csv(text) == {dt.date(2016, 8, 13): 931.0, dt.date(2016, 8, 14): 929.0}
    with pytest.raises(ValueError):
        parse_ncss_csv("<html>error</html>")


def test_fetch_gridmet_unpacks_and_caches(tmp_path: Path) -> None:
    http = FakeGridmet()
    got = fetch_gridmet(LAT, LON, FIRST, FIRST + dt.timedelta(days=4), cache_root=tmp_path, http=http)
    d = got.days[1]
    assert d.date == dt.date(2016, 8, 13)
    assert d.tmax_c == pytest.approx(39.95)  # 931 * 0.1 + 220 = 313.1 K
    assert d.tmin_c == pytest.approx(21.45)
    assert (d.rh_max, d.rh_min) == (pytest.approx(0.766), pytest.approx(0.103))
    assert (d.wind_speed, d.wind_from_deg) == (pytest.approx(2.4), 216.0)
    assert d.moisture_100h == pytest.approx(0.105)
    assert d.srad_w_m2 == pytest.approx(321.1)
    assert len(http.calls) == 16
    assert got.manifest["dataset"] == "gridMET"
    again = fetch_gridmet(LAT, LON, FIRST, FIRST + dt.timedelta(days=4), cache_root=tmp_path, http=http)
    assert len(http.calls) == 16 and again.days == got.days
    (got.directory / "daily.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        fetch_gridmet(LAT, LON, FIRST, FIRST + dt.timedelta(days=4), cache_root=tmp_path, http=http)
    manifest = json.loads((got.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["tmmx"]["add_offset"] == 220.0


def test_fetch_gridmet_rejects_an_incomplete_answer(tmp_path: Path) -> None:
    class Short(FakeGridmet):
        def get_bytes(self, url: str) -> bytes:
            text = super().get_bytes(url).decode()
            return ("\n".join(text.splitlines()[:-1]) + "\n").encode() if "/ncss/" in url else text.encode()

    with pytest.raises(ValueError, match="expected 5 days"):
        fetch_gridmet(LAT, LON, FIRST, FIRST + dt.timedelta(days=4), cache_root=tmp_path, http=Short())
    with pytest.raises(ValueError):
        fetch_gridmet(LAT, LON, FIRST, FIRST - dt.timedelta(days=1), cache_root=tmp_path, http=Short())


def test_hourly_weather_from_mocked_gridmet(tmp_path: Path) -> None:
    """Acceptance: mocked datasets convert to a valid WeatherConfig with one point per hour."""
    start = dt.datetime(2016, 8, 13, 23, 0, tzinfo=UTC)  # 16:00 PDT
    cfg = hourly_weather(LAT, LON, start, 12 * 3600, cache_root=tmp_path, http=FakeGridmet())
    assert isinstance(cfg, WeatherConfig)
    assert [p.time_s for p in cfg.series] == [3600.0 * i for i in range(12)]
    first = cfg.series[0]
    assert first.wind_direction == 216.0
    assert 0.1 <= first.relative_humidity <= 0.25
    assert first.moisture_100h == pytest.approx(0.105)
    with pytest.raises(ValueError):
        hourly_weather(LAT, LON, start.replace(tzinfo=None), 3600, cache_root=tmp_path, http=FakeGridmet())


def test_bbox_centroid() -> None:
    assert bbox_centroid(BBox(-121.1, 35.6, -120.9, 35.8)) == (pytest.approx(35.7), pytest.approx(-121.0))


# ----------------------------------------------------------------------------- HRRR coverage


class FakeS3:
    """Two pages for 2016-08-13: hours 00-11 then 12-23, with 06 and 13 missing."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def get_bytes(self, url: str) -> bytes:
        self.urls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        page2 = "continuation-token" in q
        hours = [h for h in (range(12, 24) if page2 else range(12)) if h not in (6, 13)]
        keys = "".join(
            f"<Contents><Key>hrrr.20160813/conus/hrrr.t{h:02d}z.wrfsfcf{f:02d}.grib2</Key></Contents>"
            for h in hours
            for f in (0, 1)
        )
        token = "" if page2 else "<NextContinuationToken>abc/=</NextContinuationToken>"
        xml = f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{keys}{token}</ListBucketResult>'
        return xml.encode()


def test_hrrr_missing_hours_follows_pages() -> None:
    http = FakeS3()
    start = dt.datetime(2016, 8, 13, tzinfo=UTC)
    missing = hrrr_missing_hours(start, start + dt.timedelta(days=1), http)
    assert missing == [start.replace(hour=6), start.replace(hour=13)]
    assert len(http.urls) == 2 and "continuation-token=abc%2F%3D" in http.urls[1]
    with pytest.raises(ValueError):
        hrrr_missing_hours(start.replace(tzinfo=None), start + dt.timedelta(days=1), http)


# ----------------------------------------------------------------------------- live services


@pytest.mark.network
def test_live_gridmet_chimney_fire(tmp_path: Path) -> None:
    got = fetch_gridmet(LAT, LON, dt.date(2016, 8, 12), dt.date(2016, 8, 16), cache_root=tmp_path)
    d = got.days[1]
    assert d.tmax_c == pytest.approx(39.95, abs=0.01)
    assert d.rh_min == pytest.approx(0.103, abs=0.0005)
    assert d.wind_from_deg == 216.0
    cfg = hourly_weather(LAT, LON, dt.datetime(2016, 8, 13, 23, tzinfo=UTC), 6 * 3600, cache_root=tmp_path)
    assert len(cfg.series) == 6


@pytest.mark.network
def test_live_hrrr_archive_has_gaps_in_august_2016() -> None:
    """The finding behind the choice of gridMET (section 13): analysis hours missing on 2016-08-13."""
    start = dt.datetime(2016, 8, 13, tzinfo=UTC)
    missing = hrrr_missing_hours(start, start + dt.timedelta(days=1))
    assert missing  # 06, 13 and 22 UTC were missing when checked on 2026-11-15
