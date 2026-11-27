"""Tests for dead fuel moisture from temperature and humidity (WP 2.1c, ``wildfire_sim.spread.moisture``)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from wildfire_sim.config import configs_dir
from wildfire_sim.grid import Weather, WeatherSeries
from wildfire_sim.spread.moisture import (
    apply_hourly,
    c_to_f,
    dead_fuel_moisture,
    equilibrium_moisture,
    fuel_level_air,
    load_fuel_moisture_params,
    with_dead_fuel_moisture,
)

# Wood Handbook, Table 4-2: moisture content of wood (%) in equilibrium with temperature and
# relative humidity. Glass, S. V. & Zelinka, S. L. (2010), Moisture relations and physical
# properties of wood, in Wood Handbook, USDA For. Serv. Gen. Tech. Rep. FPL-GTR-190, p. 4-4;
# the rows for 30-110 deg F, relative humidity 5 % to 95 % in steps of 5. Simard's (1968)
# equations are a regression on this table (Cohen & Deeming 1985, p. 2).
WOOD_HANDBOOK_RH = tuple(range(5, 100, 5))
WOOD_HANDBOOK = {
    30: (1.4, 2.6, 3.7, 4.6, 5.5, 6.3, 7.1, 7.9, 8.7, 9.5, 10.4, 11.3, 12.4, 13.5, 14.9, 16.5, 18.5, 21.0, 24.3),
    40: (1.4, 2.6, 3.7, 4.6, 5.5, 6.3, 7.1, 7.9, 8.7, 9.5, 10.4, 11.3, 12.3, 13.5, 14.9, 16.5, 18.5, 21.0, 24.3),
    50: (1.4, 2.6, 3.6, 4.6, 5.5, 6.3, 7.1, 7.9, 8.7, 9.5, 10.3, 11.2, 12.3, 13.4, 14.8, 16.4, 18.4, 20.9, 24.3),
    60: (1.3, 2.5, 3.6, 4.6, 5.4, 6.2, 7.0, 7.8, 8.6, 9.4, 10.2, 11.1, 12.1, 13.3, 14.6, 16.2, 18.2, 20.7, 24.1),
    70: (1.3, 2.5, 3.5, 4.5, 5.4, 6.2, 6.9, 7.7, 8.5, 9.2, 10.1, 11.0, 12.0, 13.1, 14.4, 16.0, 17.9, 20.5, 23.9),
    80: (1.3, 2.4, 3.5, 4.4, 5.3, 6.1, 6.8, 7.6, 8.3, 9.1, 9.9, 10.8, 11.7, 12.9, 14.2, 15.7, 17.7, 20.2, 23.6),
    90: (1.2, 2.3, 3.4, 4.3, 5.1, 5.9, 6.7, 7.4, 8.1, 8.9, 9.7, 10.5, 11.5, 12.6, 13.9, 15.4, 17.3, 19.8, 23.3),
    100: (1.2, 2.3, 3.3, 4.2, 5.0, 5.8, 6.5, 7.2, 7.9, 8.7, 9.5, 10.3, 11.2, 12.3, 13.6, 15.1, 17.0, 19.5, 22.9),
    110: (1.1, 2.2, 3.2, 4.0, 4.9, 5.6, 6.3, 7.0, 7.7, 8.4, 9.2, 10.0, 11.0, 12.0, 13.2, 14.7, 16.6, 19.1, 22.4),
}
TABLE_TOL = 1.0  # percentage points; measured worst 0.91 (90 deg F, 95 %), mean 0.23 over the 171 cells


def f_to_c(f: float) -> float:
    return (f - 32.0) / 1.8


# ----------------------------------------------------------------------------- equilibrium moisture


@pytest.mark.parametrize("temp_f", sorted(WOOD_HANDBOOK))
def test_reproduces_the_wood_handbook_table(temp_f: int) -> None:
    """WP 2.1c acceptance: Simard's EMC reproduces the source's tabulated values.

    Every cell from 30 to 110 deg F and 5 % to 95 % relative humidity within 1.0 percentage
    point (the regression's fit; measured worst 0.91, mean 0.23).
    """
    rh = np.array(WOOD_HANDBOOK_RH) / 100.0
    emc_percent = 100.0 * equilibrium_moisture(f_to_c(temp_f), rh)
    np.testing.assert_allclose(emc_percent, WOOD_HANDBOOK[temp_f], atol=TABLE_TOL, rtol=0)


def test_simard_equations_as_printed() -> None:
    """Cohen & Deeming (1985) eqs. 1a-1c, one point in each humidity range (T in deg F, H in %)."""
    cases = [  # (T F, H %, EMC %)
        (80.0, 5.0, 0.03229 + 0.281073 * 5 - 0.000578 * 80 * 5),
        (80.0, 30.0, 2.22749 + 0.160107 * 30 - 0.014784 * 80),
        (80.0, 70.0, 21.0606 + 0.005565 * 70**2 - 0.00035 * 70 * 80 - 0.483199 * 70),
    ]
    for t_f, h, expected in cases:
        assert 100.0 * float(equilibrium_moisture(f_to_c(t_f), h / 100.0)) == pytest.approx(expected, rel=1e-12)
    # the range boundaries: 10 % and 50 % belong to the upper equation
    assert 100.0 * float(equilibrium_moisture(f_to_c(80.0), 0.10)) == pytest.approx(
        2.22749 + 0.160107 * 10 - 0.014784 * 80
    )
    assert 100.0 * float(equilibrium_moisture(f_to_c(80.0), 0.50)) == pytest.approx(
        21.0606 + 0.005565 * 2500 - 0.00035 * 50 * 80 - 0.483199 * 50
    )


def test_emc_rises_with_humidity_and_falls_with_temperature() -> None:
    rh = np.linspace(0.0, 1.0, 101)
    for t_c in (0.0, 20.0, 40.0):
        emc = equilibrium_moisture(t_c, rh)
        assert np.all(emc >= 0) and emc[-1] > 0.2
        # monotone within each equation's range (Simard's three pieces meet with small steps)
        for lo, hi in ((0, 10), (10, 50), (50, 101)):
            assert np.all(np.diff(emc[lo:hi]) > 0)
    assert float(equilibrium_moisture(35.0, 0.3)) < float(equilibrium_moisture(5.0, 0.3))


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="relative_humidity"):
        equilibrium_moisture(20.0, 1.2)
    with pytest.raises(ValueError, match="relative_humidity"):
        equilibrium_moisture(20.0, -0.1)
    with pytest.raises(ValueError, match="finite"):
        equilibrium_moisture(np.nan, 0.3)
    with pytest.raises(ValueError, match="state_of_weather"):
        fuel_level_air(20.0, 0.3, 7)


def test_c_to_f() -> None:
    assert float(c_to_f(100.0)) == pytest.approx(212.0) and float(c_to_f(-40.0)) == pytest.approx(-40.0)


# ----------------------------------------------------------------------------- 1-h and 10-h


def test_one_and_ten_hour_are_nfdrs_multiples_of_emc() -> None:
    """``MC1 = 1.03 EMC``, ``MC10 = 1.28 EMC`` (Cohen & Deeming 1985, pp. 3-4): the 10-h offset."""
    p = load_fuel_moisture_params()
    assert (p.one_hour_factor.value, p.ten_hour_factor.value) == (1.03, 1.28)
    t, rh = np.array([5.0, 25.0, 38.0]), np.array([0.8, 0.3, 0.08])
    m1, m10 = dead_fuel_moisture(t, rh)
    emc = equilibrium_moisture(t, rh)
    np.testing.assert_allclose(m1, 1.03 * emc)
    np.testing.assert_allclose(m10, 1.28 * emc)
    np.testing.assert_allclose(m10 / m1, 1.28 / 1.03)


def test_sunshine_dries_the_fuel() -> None:
    """Clear sky adds 25 deg F and multiplies RH by 0.75; overcast adds 5 deg F (Cohen & Deeming 1985, p. 3)."""
    t, rh = fuel_level_air(20.0, 0.4, 0)
    assert float(t) == pytest.approx(20.0 + 25.0 / 1.8) and float(rh) == pytest.approx(0.30)
    t, rh = fuel_level_air(20.0, 0.4, 3)
    assert float(t) == pytest.approx(20.0 + 5.0 / 1.8) and float(rh) == pytest.approx(0.40)
    assert fuel_level_air(20.0, 0.4, None) == (20.0, 0.4)
    shade = dead_fuel_moisture(20.0, 0.4)[0]
    by_cloud = [float(dead_fuel_moisture(20.0, 0.4, code)[0]) for code in (0, 1, 2, 3)]
    assert by_cloud == sorted(by_cloud) and by_cloud[-1] < float(shade)


def test_with_dead_fuel_moisture_changes_only_the_dead_fine_fuels() -> None:
    w = Weather(temperature_c=30.0, relative_humidity=0.15, moisture_100h=0.09, wind_speed=4.0)
    out = with_dead_fuel_moisture(w)
    m1, m10 = dead_fuel_moisture(30.0, 0.15)
    assert (out.moisture_1h, out.moisture_10h) == (pytest.approx(float(m1)), pytest.approx(float(m10)))
    assert out.moisture_100h == 0.09 and out.wind_speed == 4.0 and out.moisture_live_herb == w.moisture_live_herb


# ----------------------------------------------------------------------------- diurnal cycle


def diurnal_series(days: int = 2) -> tuple[WeatherSeries, list[int | None]]:
    """Hourly weather: temperature 10-30 deg C peaking at 15:00, humidity 15-75 % lowest at 15:00.

    Clear sky (code 0) from 08:00 to 17:00, no sun correction otherwise. Time 0 is midnight.
    """
    times, values, codes = [], [], []
    for hour in range(24 * days):
        phase = math.cos(2 * math.pi * (hour % 24 - 15) / 24)  # 1 at 15:00, -1 at 03:00
        values.append(Weather(temperature_c=20.0 + 10.0 * phase, relative_humidity=0.45 - 0.30 * phase))
        times.append(3600.0 * hour)
        codes.append(0 if 8 <= hour % 24 <= 17 else None)
    return WeatherSeries(times_s=tuple(times), values=tuple(values)), codes


def test_moisture_rises_overnight_and_falls_in_the_afternoon() -> None:
    """WP 2.1c acceptance: on a synthetic diurnal series, 1-h moisture rises through the night to
    a pre-dawn high and falls to an afternoon low, every day."""
    series, codes = diurnal_series()
    out = apply_hourly(series, codes)
    m1 = np.array([w.moisture_1h for w in out.values])
    assert out.times_s == series.times_s
    for day in range(2):
        hours = m1[24 * day : 24 * day + 24]
        assert int(np.argmin(hours)) == 15  # afternoon low, at the warmest, driest hour
        assert int(np.argmax(hours)) == 3  # pre-dawn high, at the coolest, most humid hour
        night = hours[18:24].tolist() + m1[24 * day + 24 : 24 * day + 28].tolist()  # 18:00 to 03:00
        assert all(b >= a for a, b in itertools.pairwise(night)), "moisture must rise overnight"
        afternoon = hours[8:15]
        assert all(b <= a for a, b in itertools.pairwise(afternoon)), "moisture must fall to the afternoon"
    assert 0.02 < m1.min() < 0.06 < 0.12 < m1.max() < 0.20  # a dry afternoon and a moist night


def test_apply_hourly_without_codes_and_length_check() -> None:
    series, _ = diurnal_series(1)
    plain = apply_hourly(series)
    m1, _m10 = dead_fuel_moisture(series.values[15].temperature_c, series.values[15].relative_humidity)
    assert plain.values[15].moisture_1h == pytest.approx(float(m1))
    with pytest.raises(ValueError, match="one entry per breakpoint"):
        apply_hourly(series, [0, 1])


# ----------------------------------------------------------------------------- parameters


def test_parameter_file_is_validated(tmp_path: Path) -> None:
    text = (configs_dir() / "fuel_moisture.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "fuel_moisture.yaml"
    bad.write_text(text.replace("humidity_factor: {value: 0.75", "humidity_factor: {value: 1.5"), encoding="utf-8")
    with pytest.raises(ValueError, match="humidity_factor"):
        load_fuel_moisture_params(bad)
    bad.write_text(text.replace("  3:\n", "  4:\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="codes"):
        load_fuel_moisture_params(bad)
    bad.write_text(text.replace("value: 1.03", "value: 0.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="factors"):
        load_fuel_moisture_params(bad)
