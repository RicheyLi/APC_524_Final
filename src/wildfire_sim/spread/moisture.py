"""Dead fuel moisture from temperature and relative humidity (docs/plan.md section 2.2, WP 2.1c).

Fine dead fuels gain and lose water with the air around them, so the 1-hour fuel moisture that
drives Rothermel's moisture damping follows the daily cycle of temperature and humidity: low in
the afternoon, high before dawn. This module computes it hour by hour.

1. **Equilibrium moisture content** (EMC), the moisture dead wood reaches in steady air:
   Simard's (1968) regression on the Wood Handbook tables, as printed in Cohen & Deeming
   (1985, eqs. 1a-1c), with ``T`` in deg F and ``H`` the relative humidity in percent:

   .. math::

       EMC = \\begin{cases}
       0.03229 + 0.281073 H - 0.000578 H T & H < 10 \\\\
       2.22749 + 0.160107 H - 0.014784 T & 10 \\le H < 50 \\\\
       21.0606 + 0.005565 H^2 - 0.00035 H T - 0.483199 H & H \\ge 50
       \\end{cases}

   in percent. It reproduces the Wood Handbook's Table 4-2 (Glass & Zelinka 2010) within 1.0
   percentage point from 30 to 110 deg F (``tests/test_moisture.py``).
2. **Fuel-level air.** Sunshine heats the fuel surface, so the air touching the fuel is warmer
   and drier than at instrument height. The state-of-weather code (cloudiness) adds to the
   temperature and multiplies the humidity (Byram & Jemison 1943; Cohen & Deeming 1985, p. 3).
   Without a code (night, or no sky observation) the instrument values are used.
3. **1-h and 10-h moisture**: ``MC1 = 1.03 EMC`` and ``MC10 = 1.28 EMC`` with the EMC at the
   fuel, the rule of the 1978 National Fire-Danger Rating System without fuel moisture sticks
   (Cohen & Deeming 1985, pp. 3-4). The 10-h moisture is therefore ``1.28 / 1.03`` times the
   1-h moisture, the cited offset the plan asks for.

The 100-h moisture is not computed here; it comes from gridMET (WP 2.4).

**Limitation.** Both moistures are equilibrium values for the hour's weather, with no time lag.
For 1-h fuels that is the NFDRS assumption ("the response ... is so rapid that only the
potential moisture content ... is required"). For 10-h fuels it is the NFDRS rule for one
afternoon observation; applied every hour it swings the 10-h moisture through the full daily
cycle, which a 10-hour time lag would damp.

All parameters are in ``configs/fuel_moisture.yaml``.

References:
    Byram, G. M. & Jemison, G. M. (1943). Solar radiation and forest fuel moisture. J. Agric.
    Res. 67, 149-176.
    Cohen, J. D. & Deeming, J. E. (1985). The National Fire-Danger Rating System: basic
    equations. USDA For. Serv. Gen. Tech. Rep. PSW-82.
    Glass, S. V. & Zelinka, S. L. (2010). Moisture relations and physical properties of wood.
    In Wood Handbook, USDA For. Serv. Gen. Tech. Rep. FPL-GTR-190, chapter 4.
    Simard, A. J. (1968). The moisture content of forest fuels - I. Forest Fire Research
    Institute, Ottawa, Information Report FF-X-14.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from wildfire_sim.config import CitedValue, configs_dir, load_yaml
from wildfire_sim.grid import Weather, WeatherSeries
from wildfire_sim.types import FloatArray

STATE_OF_WEATHER_CODES = (0, 1, 2, 3)
"""0 clear, 1 scattered clouds, 2 broken clouds, 3 overcast (NFDRS state-of-weather codes)."""


class SkyCorrection(BaseModel):
    """Fuel-level correction for one state-of-weather code."""

    model_config = ConfigDict(extra="forbid")

    temperature_add_f: CitedValue
    humidity_factor: CitedValue


class FuelMoistureParams(BaseModel):
    """Schema of ``configs/fuel_moisture.yaml``.

    Attributes:
        one_hour_factor: ``MC1 / EMC`` (dimensionless).
        ten_hour_factor: ``MC10 / EMC`` (dimensionless).
        state_of_weather: Code (0-3) -> correction to temperature (deg F, added) and relative
            humidity (multiplied).
    """

    model_config = ConfigDict(extra="forbid")

    one_hour_factor: CitedValue
    ten_hour_factor: CitedValue
    state_of_weather: dict[int, SkyCorrection]

    @model_validator(mode="after")
    def _check(self) -> FuelMoistureParams:
        if not (self.one_hour_factor.value > 0 and self.ten_hour_factor.value > 0):
            raise ValueError("moisture factors must be > 0")
        if sorted(self.state_of_weather) != list(STATE_OF_WEATHER_CODES):
            raise ValueError(f"state_of_weather needs exactly the codes {STATE_OF_WEATHER_CODES}")
        for code, c in self.state_of_weather.items():
            if not 0 < c.humidity_factor.value <= 1:
                raise ValueError(f"state_of_weather {code}: humidity_factor must lie in (0, 1]")
        return self


def load_fuel_moisture_params(path: str | Path | None = None) -> FuelMoistureParams:
    """Load the parameters; default ``configs_dir() / "fuel_moisture.yaml"``."""
    return load_yaml(configs_dir() / "fuel_moisture.yaml" if path is None else path, FuelMoistureParams)


def c_to_f(temperature_c: float | FloatArray) -> FloatArray:
    """Degrees Celsius to degrees Fahrenheit."""
    return np.asarray(temperature_c, dtype=np.float64) * 1.8 + 32.0


def equilibrium_moisture(temperature_c: float | FloatArray, relative_humidity: float | FloatArray) -> FloatArray:
    """Equilibrium moisture content of dead wood (Simard 1968), fraction of oven-dry weight.

    Args:
        temperature_c: Dry-bulb temperature, deg C.
        relative_humidity: Relative humidity, fraction in [0, 1].

    Returns:
        EMC as a fraction (0.06 = 6 %), float64, broadcast shape of the inputs.

    Raises:
        ValueError: Relative humidity outside [0, 1] or a non-finite input.
    """
    t = c_to_f(temperature_c)
    rh = np.asarray(relative_humidity, dtype=np.float64)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(rh))):
        raise ValueError("temperature and relative humidity must be finite")
    if np.any(rh < 0) or np.any(rh > 1):
        raise ValueError("relative_humidity must lie in [0, 1]")
    h = rh * 100.0
    emc = np.where(
        h < 10.0,
        0.03229 + 0.281073 * h - 0.000578 * h * t,
        np.where(
            h < 50.0,
            2.22749 + 0.160107 * h - 0.014784 * t,
            21.0606 + 0.005565 * h**2 - 0.00035 * h * t - 0.483199 * h,
        ),
    )
    return np.maximum(emc, 0.0) / 100.0


def fuel_level_air(
    temperature_c: float | FloatArray,
    relative_humidity: float | FloatArray,
    state_of_weather: int | None,
    params: FuelMoistureParams | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Temperature (deg C) and relative humidity (fraction) of the air in contact with the fuel.

    With a state-of-weather code the instrument values are corrected for sunshine (Cohen &
    Deeming 1985, p. 3): clear sky adds 25 deg F and multiplies the humidity by 0.75, overcast
    adds 5 deg F. With ``None`` they are returned unchanged; use it at night, when the sun does
    not heat the fuel, and when cloudiness is unknown.

    Raises:
        ValueError: An unknown state-of-weather code.
    """
    t = np.asarray(temperature_c, dtype=np.float64)
    rh = np.asarray(relative_humidity, dtype=np.float64)
    if state_of_weather is None:
        return t, rh
    p = load_fuel_moisture_params() if params is None else params
    if state_of_weather not in p.state_of_weather:
        raise ValueError(f"state_of_weather must be one of {STATE_OF_WEATHER_CODES} or None")
    c = p.state_of_weather[state_of_weather]
    return t + c.temperature_add_f.value / 1.8, rh * c.humidity_factor.value


def dead_fuel_moisture(
    temperature_c: float | FloatArray,
    relative_humidity: float | FloatArray,
    state_of_weather: int | None = None,
    params: FuelMoistureParams | None = None,
) -> tuple[FloatArray, FloatArray]:
    """1-h and 10-h dead fuel moisture for one hour's weather, fractions of oven-dry weight.

    ``MC1 = 1.03 EMC`` and ``MC10 = 1.28 EMC``, the EMC taken at the fuel (:func:`fuel_level_air`)
    (Cohen & Deeming 1985, pp. 3-4).

    Args:
        temperature_c: Air temperature at instrument height, deg C.
        relative_humidity: Relative humidity at instrument height, fraction in [0, 1].
        state_of_weather: Cloudiness code 0-3 for daylight hours, or None (no sun correction).
        params: Parameters; None loads ``configs/fuel_moisture.yaml``.

    Returns:
        ``(one_hour, ten_hour)`` float64 arrays of the broadcast input shape.
    """
    p = load_fuel_moisture_params() if params is None else params
    t, rh = fuel_level_air(temperature_c, relative_humidity, state_of_weather, p)
    emc = equilibrium_moisture(t, rh)
    return p.one_hour_factor.value * emc, p.ten_hour_factor.value * emc


def with_dead_fuel_moisture(
    weather: Weather, state_of_weather: int | None = None, params: FuelMoistureParams | None = None
) -> Weather:
    """``weather`` with ``moisture_1h`` and ``moisture_10h`` computed from its temperature and humidity.

    Every other field, the 100-h and live moisture included, is unchanged.
    """
    m1, m10 = dead_fuel_moisture(weather.temperature_c, weather.relative_humidity, state_of_weather, params)
    return dataclasses.replace(weather, moisture_1h=float(m1), moisture_10h=float(m10))


def apply_hourly(
    series: WeatherSeries,
    state_of_weather: Sequence[int | None] | None = None,
    params: FuelMoistureParams | None = None,
) -> WeatherSeries:
    """Recompute the 1-h and 10-h moisture of every breakpoint of a weather series.

    The series is expected to hold one breakpoint per hour (WP 2.4 builds it so); each
    breakpoint's moisture is the equilibrium value for its temperature and humidity.

    Args:
        series: Piecewise-constant weather.
        state_of_weather: One code (0-3) or None per breakpoint; None for all means no sun
            correction anywhere.
        params: Parameters; None loads ``configs/fuel_moisture.yaml``.

    Raises:
        ValueError: ``state_of_weather`` has a different length from the series.
    """
    p = load_fuel_moisture_params() if params is None else params
    codes: Sequence[int | None] = [None] * len(series.values) if state_of_weather is None else state_of_weather
    if len(codes) != len(series.values):
        raise ValueError("state_of_weather needs one entry per breakpoint")
    values = tuple(with_dead_fuel_moisture(w, c, p) for w, c in zip(series.values, codes, strict=True))
    return WeatherSeries(times_s=series.times_s, values=values)
