"""Rothermel (1972) surface fire spread with the Albini (1976) revisions, vectorized over a grid.

This module computes the head-fire behaviour of a surface fuel bed: rate of spread, reaction
intensity, heat per unit area, residence time, Byram fireline intensity and flame length
(WP 2.1a), and registers :class:`RothermelRule` as the ``"rothermel"`` spread rule (WP 2.1b),
which turns the head rate into directional rates on the stencil with the helpers of
:mod:`wildfire_sim.spread.ellipse` (the 10 m to midflame wind reduction, the vector combination
of wind and slope, the fire ellipse).

The model, in Rothermel's own units (lb, ft, min, Btu), for a fuel bed with dead (1-h, 10-h,
100-h, cured herbaceous) and live (herbaceous, woody) particle classes:

.. math::

    R = \\frac{I_R\\, \\xi\\, (1 + \\phi_w + \\phi_s)}{\\rho_b\\, \\varepsilon\\, Q_{ig}}

* :math:`I_R = \\Gamma' \\sum_c w_{n,c}\\, h_c\\, \\eta_{M,c}\\, \\eta_{s,c}` is the reaction
  intensity, summed over the dead and live categories ``c``;
* :math:`\\xi` is the propagating flux ratio, :math:`\\phi_w` and :math:`\\phi_s` the wind and slope
  factors, and :math:`\\rho_b \\varepsilon Q_{ig}` the heat sink.

Every equation below follows Andrews (2018), who documents the model as BehavePlus implements
it, with the original papers cited in each function. The Albini (1976) revisions used are: the
optimum-reaction-velocity exponent ``A = 133 sigma^-0.7913``, net fuel load weighted by size
class rather than by surface area alone, and the live fuel moisture of extinction. For the
dynamic fuel models of Scott & Burgan (2005), part of the live herbaceous load is transferred
to a dead herbaceous class before anything else is computed (:func:`herbaceous_cured_fraction`).

The wind speed limit of Rothermel (1972), ``U <= 0.9 I_R`` (ft/min, with ``I_R`` in
Btu/ft^2/min), is optional (``wind_limit_on``) and off by default. Andrews et al. (2013) found it
too restrictive; the BehavePlus engine (the ``behave`` library) applies it by default, so the
comparisons with BehavePlus switch it on. Whether the spread rule applies it is an open decision
for WP 2.1b (docs/plan.md section 13).

Fuel parameters are read from ``configs/fuel_models.csv`` (:func:`load_fuel_models`).

References:
    Albini, F. A. (1976). Estimating wildfire behavior and effects. USDA For. Serv. Gen. Tech.
    Rep. INT-30.
    Anderson, H. E. (1969). Heat transfer and fire spread. USDA For. Serv. Res. Pap. INT-69.
    Anderson, H. E. (1982). Aids to determining fuel models for estimating fire behavior. USDA
    For. Serv. Gen. Tech. Rep. INT-122.
    Andrews, P. L. (2018). The Rothermel surface fire spread model and associated developments:
    a comprehensive explanation. USDA For. Serv. Gen. Tech. Rep. RMRS-GTR-371.
    Andrews, P. L., Cruz, M. G. & Rothermel, R. C. (2013). Examination of the wind speed limit
    function in the Rothermel surface fire spread model. Int. J. Wildland Fire 22, 959-969.
    Byram, G. M. (1959). Combustion of forest fuels. In Davis (ed.), Forest Fire: Control and Use.
    Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels.
    USDA For. Serv. Res. Pap. INT-115.
    Scott, J. H. & Burgan, R. E. (2005). Standard fire behavior fuel models. USDA For. Serv. Gen.
    Tech. Rep. RMRS-GTR-153.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from wildfire_sim.config import CitedValue, configs_dir, load_yaml
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import FlameLengthModel, flame_length_from_intensity, load_flame_length_model
from wildfire_sim.spread import register
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.spread.ellipse import (
    LatticeFitTable,
    combine_wind_and_slope,
    directional_rates,
    length_to_breadth,
    load_fit_table,
    twenty_foot_wind,
    wind_adjustment_factor,
)
from wildfire_sim.types import FloatArray, IntArray, Stencil

# ----------------------------------------------------------------------------- unit conversions

TONS_PER_ACRE_TO_LB_PER_FT2 = 2000.0 / 43560.0
"""1 short ton/acre = 2000 lb / 43,560 ft^2."""
FT = 0.3048
"""Metres per foot."""
BTU = 1.05505585262
"""Kilojoules per British thermal unit (International Table)."""
FT_PER_MIN_TO_M_PER_S = FT / 60.0
BTU_PER_FT2_TO_KJ_PER_M2 = BTU / FT**2
BTU_PER_FT2_MIN_TO_KW_PER_M2 = BTU_PER_FT2_TO_KJ_PER_M2 / 60.0

_DEAD, _LIVE = 0, 1
_N_DEAD = 4  # 1-h, 10-h, 100-h, cured (dead) herbaceous
_N_LIVE = 2  # live herbaceous, live woody


# ----------------------------------------------------------------------------- fuel models


@dataclass(frozen=True)
class FuelModelTable:
    """Parameters of the surface fuel models, one entry per model, in the source's US units.

    Every array has shape ``(n,)``, in the row order of ``numbers``.

    Attributes:
        numbers: Fuel model numbers as stored in ``Grid.fuel_model``.
        codes: Short codes (``"GR1"``, or ``"1"`` for Anderson's models).
        names: Descriptions.
        load_1h, load_10h, load_100h, load_live_herb, load_live_woody: Oven-dry loads, tons/acre.
        sav_1h, sav_10h, sav_100h, sav_live_herb, sav_live_woody: Surface-area-to-volume
            ratios, 1/ft.
        depth: Fuel bed depth, ft.
        mx_dead: Dead fuel moisture of extinction, fraction.
        heat_dead, heat_live: Low heat content, Btu/lb.
        dynamic: Whether live herbaceous load cures with moisture (Scott & Burgan 2005).
        particle_density: Oven-dry particle density, lb/ft^3.
        mineral_total: Total mineral content, fraction.
        mineral_effective: Effective (silica-free) mineral content, fraction.
    """

    numbers: IntArray
    codes: tuple[str, ...]
    names: tuple[str, ...]
    load_1h: FloatArray
    load_10h: FloatArray
    load_100h: FloatArray
    load_live_herb: FloatArray
    load_live_woody: FloatArray
    sav_1h: FloatArray
    sav_10h: FloatArray
    sav_100h: FloatArray
    sav_live_herb: FloatArray
    sav_live_woody: FloatArray
    depth: FloatArray
    mx_dead: FloatArray
    heat_dead: FloatArray
    heat_live: FloatArray
    dynamic: np.ndarray
    particle_density: FloatArray
    mineral_total: FloatArray
    mineral_effective: FloatArray

    def __len__(self) -> int:
        return int(self.numbers.size)

    def index(self, fuel_model: IntArray) -> IntArray:
        """Row of each fuel model number in this table, or -1 for numbers not listed.

        Args:
            fuel_model: Integer array of fuel model numbers, any shape.

        Returns:
            ``int64`` array of the same shape.
        """
        fm = np.asarray(fuel_model, dtype=np.int64)
        order = np.argsort(self.numbers)
        sorted_numbers = self.numbers[order]
        pos = np.clip(np.searchsorted(sorted_numbers, fm), 0, sorted_numbers.size - 1)
        found = sorted_numbers[pos] == fm
        return np.where(found, order[pos], -1).astype(np.int64)

    def row(self, number: int) -> int:
        """Row of one fuel model number.

        Raises:
            KeyError: The number is not in the table.
        """
        idx = int(self.index(np.array([number]))[0])
        if idx < 0:
            raise KeyError(f"fuel model {number} is not in the table")
        return idx


_FLOAT_COLUMNS = (
    "load_1h",
    "load_10h",
    "load_100h",
    "load_live_herb",
    "load_live_woody",
    "sav_1h",
    "sav_10h",
    "sav_100h",
    "sav_live_herb",
    "sav_live_woody",
    "depth",
    "mx_dead",
    "heat_dead",
    "heat_live",
    "particle_density",
    "mineral_total",
    "mineral_effective",
)


def load_fuel_models(path: str | Path | None = None) -> FuelModelTable:
    """Read and validate the table of surface fuel model parameters.

    The file is a CSV with one row per fuel model (Anderson 1982's 13 models and the 40 of
    Scott & Burgan 2005 in the shipped table). Lines whose first non-blank character is ``#``
    are comments and may appear anywhere. Values keep the source's US customary units (loads
    in tons/acre, SAV ratios in 1/ft, depth in ft, heat content in Btu/lb, particle density in
    lb/ft^3), except ``mx_dead``, which the file gives in percent and this function returns as a
    fraction. Unit conversion to SI happens later, in :func:`fuel_bed`.

    Args:
        path: CSV file to read. None (the default) reads ``configs_dir() / "fuel_models.csv"``.
            Required columns: ``number``, ``code``, ``name``, ``dynamic`` (``1`` for a dynamic
            model whose live herbaceous load cures, anything else for static), and the
            ``load_*``, ``sav_*``, ``depth``, ``mx_dead``, ``heat_dead``, ``heat_live``,
            ``particle_density``, ``mineral_total`` and ``mineral_effective`` columns of
            :class:`FuelModelTable`. Extra columns are ignored.

    Returns:
        A :class:`FuelModelTable` whose arrays have shape ``(n,)``, one entry per row of the
        file, in file order.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file has no data rows; a required column is missing; a fuel model
            number repeats; a ``number`` or numeric field cannot be parsed; ``mx_dead`` lies
            outside (0, 100] %; a load is negative; a class with load > 0 has SAV <= 0;
            ``depth``, ``heat_dead``, ``heat_live`` or ``particle_density`` is <= 0; or a model
            has no 1-h load.
        TypeError: A row has fewer fields than the header, so a value is missing.

    Examples:
        >>> table = load_fuel_models()
        >>> len(table)
        53
        >>> i = table.row(102)
        >>> table.codes[i], float(table.depth[i]), float(table.mx_dead[i])
        ('GR2', 1.0, 0.15)
    """
    p = configs_dir() / "fuel_models.csv" if path is None else Path(path)
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.lstrip().startswith("#")))
    if not rows:
        raise ValueError(f"{p}: no fuel models")
    required = {"number", "code", "name", "dynamic", *_FLOAT_COLUMNS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{p}: missing columns {sorted(missing)}")
    numbers = np.array([int(r["number"]) for r in rows], dtype=np.int64)
    if np.unique(numbers).size != numbers.size:
        raise ValueError(f"{p}: fuel model numbers repeat")
    cols = {c: np.array([float(r[c]) for r in rows], dtype=np.float64) for c in _FLOAT_COLUMNS}
    if not 0 < cols["mx_dead"].min() or cols["mx_dead"].max() > 100:
        raise ValueError(f"{p}: mx_dead must lie in (0, 100] percent")
    cols["mx_dead"] = cols["mx_dead"] / 100.0
    for c in ("load_1h", "load_10h", "load_100h", "load_live_herb", "load_live_woody"):
        if np.any(cols[c] < 0):
            raise ValueError(f"{p}: {c} must be >= 0")
        sav = cols[c.replace("load", "sav")]
        if np.any((cols[c] > 0) & (sav <= 0)):
            raise ValueError(f"{p}: {c.replace('load', 'sav')} must be > 0 where {c} > 0")
    for c in ("depth", "heat_dead", "heat_live", "particle_density"):
        if np.any(cols[c] <= 0):
            raise ValueError(f"{p}: {c} must be > 0")
    if np.any(cols["load_1h"] <= 0):
        raise ValueError(f"{p}: every fuel model needs a 1-h load")
    return FuelModelTable(
        numbers=numbers,
        codes=tuple(r["code"] for r in rows),
        names=tuple(r["name"] for r in rows),
        dynamic=np.array([r["dynamic"].strip() == "1" for r in rows], dtype=bool),
        **cols,
    )


# ----------------------------------------------------------------------------- moisture


@dataclass(frozen=True)
class FuelMoisture:
    """Fuel moisture contents, fractions of oven-dry weight (0.06 = 6 %).

    Each field is a float or an array that broadcasts against the fuel models being evaluated.
    """

    one_hour: float | FloatArray
    ten_hour: float | FloatArray
    hundred_hour: float | FloatArray
    live_herb: float | FloatArray
    live_woody: float | FloatArray

    @classmethod
    def from_weather(cls, weather: Weather) -> FuelMoisture:
        """Moisture contents carried by a :class:`~wildfire_sim.grid.Weather`."""
        return cls(
            weather.moisture_1h,
            weather.moisture_10h,
            weather.moisture_100h,
            weather.moisture_live_herb,
            weather.moisture_live_woody,
        )


def herbaceous_cured_fraction(moisture_live_herb: float | FloatArray) -> FloatArray:
    """Fraction of a dynamic model's live herbaceous load that is cured, i.e. treated as dead.

    Scott & Burgan (2005, p. 9): fully cured at a live herbaceous moisture of 30 % or less,
    fully green at 120 % or more, linear in between. The line is used in the rounded form of
    BehavePlus, ``T = 1.333 - 1.11 M`` clipped to [0, 1] (Andrews 2018), so results match the
    reference implementation; it leaves 0.1 % cured at 120 %, which matters only for a fuel
    near its moisture of extinction.

    Args:
        moisture_live_herb: Live herbaceous moisture, fraction.

    Returns:
        Cured fraction in [0, 1], float64.
    """
    m = np.asarray(moisture_live_herb, dtype=np.float64)
    return np.clip(1.333 - 1.11 * m, 0.0, 1.0)


# ----------------------------------------------------------------------------- fuel bed


@dataclass(frozen=True)
class FuelBed:
    """Moisture-dependent properties of each fuel model that do not depend on wind or slope.

    Arrays have shape ``(n,)``, one entry per row of the :class:`FuelModelTable` evaluated.
    Rates and intensities are SI; the wind-factor coefficients are in Rothermel's units and are
    used through :meth:`wind_factor`.

    Attributes:
        ros_no_wind: No-wind, no-slope rate of spread ``R_0``, m/s.
        reaction_intensity: ``I_R``, kW/m^2.
        heat_per_area: Heat released per unit area by the flaming front, ``I_R t_r``, kJ/m^2.
        residence_time: Flame residence time ``t_r = 384 / sigma`` min (Anderson 1969), in s.
        sigma: Characteristic surface-area-to-volume ratio of the fuel bed, 1/ft.
        packing_ratio: ``beta``, dimensionless.
        relative_packing_ratio: ``beta / beta_op``, dimensionless.
        mx_live: Live fuel moisture of extinction, fraction (0 where there is no live fuel).
        wind_c, wind_b, wind_e: Coefficients of :math:`\\phi_w = C U^B (\\beta/\\beta_{op})^{-E}`
            with ``U`` in ft/min.
    """

    ros_no_wind: FloatArray
    reaction_intensity: FloatArray
    heat_per_area: FloatArray
    residence_time: FloatArray
    sigma: FloatArray
    packing_ratio: FloatArray
    relative_packing_ratio: FloatArray
    mx_live: FloatArray
    wind_c: FloatArray
    wind_b: FloatArray
    wind_e: FloatArray

    def wind_factor(self, midflame_wind: float | FloatArray) -> FloatArray:
        """Rothermel's wind factor :math:`\\phi_w` (dimensionless) for a midflame wind, m/s.

        :math:`\\phi_w = C U^B (\\beta/\\beta_{op})^{-E}`, ``U`` in ft/min, with
        ``C = 7.47 exp(-0.133 sigma^0.55)``, ``B = 0.02526 sigma^0.54`` and
        ``E = 0.715 exp(-3.59e-4 sigma)`` (Rothermel 1972, eqs. 47-50). ``midflame_wind`` must
        broadcast against the fuel bed arrays.
        """
        u = np.maximum(np.asarray(midflame_wind, dtype=np.float64), 0.0) / FT_PER_MIN_TO_M_PER_S
        return self.wind_c * u**self.wind_b * self.relative_packing_ratio ** (-self.wind_e)

    def slope_factor(self, slope: float | FloatArray) -> FloatArray:
        """Rothermel's slope factor :math:`\\phi_s = 5.275 \\beta^{-0.3} \\tan^2\\theta` (eq. 51).

        Args:
            slope: Rise over run (``tan`` of the slope angle), >= 0; must broadcast against the
                fuel bed arrays.
        """
        tan = np.asarray(slope, dtype=np.float64)
        return 5.275 * self.packing_ratio ** (-0.3) * tan**2

    def effective_wind(self, phi: float | FloatArray) -> FloatArray:
        """Midflame wind, m/s, that alone would give the combined factor ``phi``.

        The inverse of :meth:`wind_factor` (Albini 1976): the *effective wind speed* used for
        the length-to-breadth ratio of the fire ellipse (WP 2.1b) and for the wind limit.
        """
        phi = np.maximum(np.asarray(phi, dtype=np.float64), 0.0)
        base = phi * self.relative_packing_ratio**self.wind_e / self.wind_c
        u = np.power(base, 1.0 / self.wind_b, out=np.zeros_like(base), where=base > 0)
        return u * FT_PER_MIN_TO_M_PER_S


def _eta_m(moisture: FloatArray, extinction: FloatArray) -> FloatArray:
    """Moisture damping coefficient (Rothermel 1972, eq. 29), 0 at or above extinction."""
    r = np.divide(moisture, extinction, out=np.ones_like(moisture), where=extinction > 0)
    eta = 1.0 - 2.59 * r + 5.11 * r**2 - 3.52 * r**3
    return np.where(r < 1.0, eta, 0.0)


def _size_class(sav: FloatArray) -> IntArray:
    """Albini's (1976) six SAV size classes for weighting net load: >= 1200, 192, 96, 48, 16, < 16 1/ft."""
    edges = np.array([16.0, 48.0, 96.0, 192.0, 1200.0])
    return 5 - np.searchsorted(edges, sav, side="right")


def _weights(area: FloatArray, sav: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Surface-area and size-class weights within one category.

    Args:
        area: ``(n, j)`` mean total surface area per unit fuel cell, ``sigma w_0 / rho_p``.
        sav: ``(n, j)`` SAV ratios, 1/ft.

    Returns:
        ``(f, g, total)``: ``f[n, j]`` the area fraction of class ``j`` in the category,
        ``g[n, j]`` the summed area fraction of every class in the same size class as ``j``
        (Albini 1976), and ``total[n]`` the category's surface area.
    """
    total = area.sum(axis=1)
    f = np.divide(area, total[:, None], out=np.zeros_like(area), where=total[:, None] > 0)
    cls = _size_class(sav)
    g = np.zeros_like(f)
    for j in range(f.shape[1]):
        same = cls == cls[:, j : j + 1]
        g[:, j] = np.where(same, f, 0.0).sum(axis=1)
    return f, g, total


def fuel_bed(table: FuelModelTable, moisture: FuelMoisture, rows: IntArray | None = None) -> FuelBed:
    """Evaluate the wind- and slope-independent part of Rothermel's model for each fuel model.

    For each selected fuel model this computes, in Rothermel's units and then converted to SI,
    the no-wind, no-slope rate of spread

    .. math::

        R_0 = \\frac{I_R\\, \\xi}{\\rho_b\\, \\varepsilon\\, Q_{ig}}

    together with the reaction intensity :math:`I_R`, the residence time and heat per unit
    area, and the coefficients of the wind and slope factors. It follows Rothermel (1972) with
    the Albini (1976) revisions as documented by Andrews (2018): surface-area weighting of the
    particle classes (eqs. 53-57), size-class weighted net load, the live fuel moisture of
    extinction (Andrews 2018, eqs. 88-91), optimum reaction velocity with
    ``A = 133 sigma^-0.7913``, the propagating flux ratio (eq. 42) and the heat sink (eqs. 12,
    14, 77). For dynamic models the cured part of the live herbaceous load
    (:func:`herbaceous_cured_fraction`) is moved to a dead class at the 1-h moisture first.

    The rate is then ``R = R_0 (1 + phi_w + phi_s)``: see :func:`head_fire` and
    :func:`surface_fire`.

    Args:
        table: Fuel model parameters, as returned by :func:`load_fuel_models`.
        moisture: Fuel moisture contents, fractions of oven-dry weight (0.06 = 6 %). Each field
            is a scalar (the same moisture for every model) or an ``(n,)`` array, one value per
            entry of ``rows``.
        rows: ``(n,)`` integer row indices into ``table`` (not fuel model numbers; convert
            those with :meth:`FuelModelTable.index`). None (the default) evaluates every row,
            so ``n = len(table)``.

    Returns:
        A :class:`FuelBed` whose arrays have shape ``(n,)``, in the order of ``rows``.
        ``ros_no_wind`` is in m/s, ``reaction_intensity`` in kW/m^2, ``heat_per_area`` in
        kJ/m^2, ``residence_time`` in s and ``sigma`` in 1/ft. A model whose moisture is at or
        above its moisture of extinction has ``ros_no_wind = 0``.

    Raises:
        ValueError: A row index is negative or ``>= len(table)``; a moisture value is
            negative, NaN or infinite; or a moisture array does not broadcast to ``(n,)``.

    Examples:
        Short grass (fuel model 1) and GR2 at Scott & Burgan's D2L2 moisture scenario:

        >>> import numpy as np
        >>> table = load_fuel_models()
        >>> moisture = FuelMoisture(0.06, 0.07, 0.08, live_herb=0.60, live_woody=0.90)
        >>> bed = fuel_bed(table, moisture, rows=table.index(np.array([1, 102])))
        >>> print(bed.ros_no_wind.round(4))  # m/s
        [0.0234 0.0078]
        >>> print(bed.reaction_intensity.round(1))  # kW/m^2
        [156.4 217.5]
    """
    idx = np.arange(len(table)) if rows is None else np.asarray(rows, dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= len(table)):
        raise ValueError("rows must index the fuel model table")
    n = idx.size

    def take(a: FloatArray) -> FloatArray:
        return np.asarray(a, dtype=np.float64)[idx]

    def m(x: float | FloatArray) -> FloatArray:
        return np.broadcast_to(np.asarray(x, dtype=np.float64), (n,)).astype(np.float64)

    m1, m10, m100, mh, mw = (
        m(moisture.one_hour),
        m(moisture.ten_hour),
        m(moisture.hundred_hour),
        m(moisture.live_herb),
        m(moisture.live_woody),
    )
    if any(np.any(x < 0) or not np.all(np.isfinite(x)) for x in (m1, m10, m100, mh, mw)):
        raise ValueError("fuel moisture must be finite and >= 0")

    k = TONS_PER_ACRE_TO_LB_PER_FT2
    herb = take(table.load_live_herb) * k
    cured = np.where(take(table.dynamic).astype(bool), herbaceous_cured_fraction(mh), 0.0)

    # (n, 4) dead classes: 1-h, 10-h, 100-h, cured herbaceous; (n, 2) live: herbaceous, woody.
    load_d = np.stack([take(table.load_1h) * k, take(table.load_10h) * k, take(table.load_100h) * k, herb * cured], 1)
    load_l = np.stack([herb * (1.0 - cured), take(table.load_live_woody) * k], 1)
    sav_d = np.stack([take(table.sav_1h), take(table.sav_10h), take(table.sav_100h), take(table.sav_live_herb)], 1)
    sav_l = np.stack([take(table.sav_live_herb), take(table.sav_live_woody)], 1)
    mst_d = np.stack([m1, m10, m100, m1], 1)  # cured herbaceous takes the 1-h moisture
    mst_l = np.stack([mh, mw], 1)
    # A class with no load contributes nothing; give it a harmless SAV so exp(-138/sav) is finite.
    sav_d = np.where(load_d > 0, sav_d, 1.0)
    sav_l = np.where(load_l > 0, sav_l, 1.0)

    rho_p = take(table.particle_density)
    s_t = take(table.mineral_total)
    s_e = take(table.mineral_effective)
    heat_d = take(table.heat_dead)
    heat_l = take(table.heat_live)
    mx_d = take(table.mx_dead)
    depth = take(table.depth)

    # Surface-area weighting (Rothermel 1972 eqs. 53-57) and Albini's size-class weights.
    f_d, g_d, area_d = _weights(sav_d * load_d / rho_p[:, None], sav_d)
    f_l, g_l, area_l = _weights(sav_l * load_l / rho_p[:, None], sav_l)
    area_t = area_d + area_l
    f_cat_d = area_d / area_t
    f_cat_l = area_l / area_t
    has_live = area_l > 0

    sigma_d = (f_d * sav_d).sum(1)
    sigma_l = (f_l * sav_l).sum(1)
    sigma = f_cat_d * sigma_d + f_cat_l * sigma_l

    # Packing ratio (eqs. 31, 40, 74).
    total_load = load_d.sum(1) + load_l.sum(1)
    rho_b = total_load / depth
    beta = rho_b / rho_p
    beta_op = 3.348 * sigma ** (-0.8189)
    rel_beta = beta / beta_op

    # Live fuel moisture of extinction (Albini 1976; Andrews 2018 eqs. 88-91).
    fine_d = (load_d * np.exp(-138.0 / sav_d)).sum(1)
    fine_l = (load_l * np.exp(-500.0 / sav_l)).sum(1)
    fine_m = np.divide((load_d * np.exp(-138.0 / sav_d) * mst_d).sum(1), fine_d)
    w_ratio = np.divide(fine_d, fine_l, out=np.zeros(n), where=fine_l > 0)
    mx_l = np.where(has_live, np.maximum(2.9 * w_ratio * (1.0 - fine_m / mx_d) - 0.226, mx_d), 0.0)

    # Category means weighted by surface area (eqs. 59-62) and size-class weighted net load (eq. 24 revised).
    def category(f, g, load, sav, mst, heat, mx):
        h = heat  # every class of a category has the same heat content
        m_c = (f * mst).sum(1)
        wn = (g * load).sum(1) * (1.0 - s_t)
        eta_s = np.minimum(0.174 * s_e ** (-0.19), 1.0)
        eta_m = _eta_m(m_c, mx)
        return wn * h * eta_m * eta_s

    src_d = category(f_d, g_d, load_d, sav_d, mst_d, heat_d, mx_d)
    src_l = np.where(has_live, category(f_l, g_l, load_l, sav_l, mst_l, heat_l, np.where(has_live, mx_l, 1.0)), 0.0)

    # Optimum reaction velocity (eqs. 36-39, A from Albini 1976), Btu/ft^2/min.
    s15 = sigma**1.5
    gamma_max = s15 / (495.0 + 0.0594 * s15)
    a = 133.0 * sigma ** (-0.7913)
    gamma = gamma_max * rel_beta**a * np.exp(a * (1.0 - rel_beta))
    i_r = gamma * (src_d + src_l)

    # Propagating flux ratio (eq. 42) and heat sink (eqs. 12, 14, 77).
    xi = np.exp((0.792 + 0.681 * np.sqrt(sigma)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma)
    q_d = (f_d * np.exp(-138.0 / sav_d) * (250.0 + 1116.0 * mst_d)).sum(1)
    q_l = (f_l * np.exp(-138.0 / sav_l) * (250.0 + 1116.0 * mst_l)).sum(1)
    heat_sink = rho_b * (f_cat_d * q_d + f_cat_l * q_l)
    r0 = i_r * xi / heat_sink  # ft/min

    t_r = 384.0 / sigma  # min (Anderson 1969)
    return FuelBed(
        ros_no_wind=r0 * FT_PER_MIN_TO_M_PER_S,
        reaction_intensity=i_r * BTU_PER_FT2_MIN_TO_KW_PER_M2,
        heat_per_area=i_r * t_r * BTU_PER_FT2_TO_KJ_PER_M2,
        residence_time=t_r * 60.0,
        sigma=sigma,
        packing_ratio=beta,
        relative_packing_ratio=rel_beta,
        mx_live=mx_l,
        wind_c=7.47 * np.exp(-0.133 * sigma**0.55),
        wind_b=0.02526 * sigma**0.54,
        wind_e=0.715 * np.exp(-3.59e-4 * sigma),
    )


# ----------------------------------------------------------------------------- head fire


@dataclass(frozen=True)
class SurfaceFire:
    """Head-fire behaviour per cell. Every array has the shape of the fuel model input.

    Cells whose fuel model is not in the table (non-burnable, 0) hold zeros everywhere.

    Attributes:
        ros: Head rate of spread, m/s.
        intensity: Byram fireline intensity at the head, ``heat_per_area * ros``, kW/m.
        flame_length: Head flame length from ``intensity`` (Byram 1959), m.
        heat_per_area: Heat per unit area of the flaming front, kJ/m^2.
        reaction_intensity: ``I_R``, kW/m^2.
        residence_time: Flame residence time, s.
        ros_no_wind: No-wind, no-slope rate ``R_0``, m/s.
        phi_wind: Wind factor, dimensionless.
        phi_slope: Slope factor, dimensionless.
        effective_wind: Effective midflame wind speed of ``phi_wind + phi_slope``, m/s.
        wind_limited: True where the wind limit capped the combined factor.
    """

    ros: FloatArray
    intensity: FloatArray
    flame_length: FloatArray
    heat_per_area: FloatArray
    reaction_intensity: FloatArray
    residence_time: FloatArray
    ros_no_wind: FloatArray
    phi_wind: FloatArray
    phi_slope: FloatArray
    effective_wind: FloatArray
    wind_limited: np.ndarray


def wind_limit(reaction_intensity: FloatArray) -> FloatArray:
    """Rothermel's (1972) maximum reliable effective wind, ``0.9 I_R`` ft/min, returned in m/s.

    Args:
        reaction_intensity: ``I_R``, kW/m^2.
    """
    i_r = np.asarray(reaction_intensity, dtype=np.float64) / BTU_PER_FT2_MIN_TO_KW_PER_M2
    return 0.9 * i_r * FT_PER_MIN_TO_M_PER_S


def surface_fire(
    fuel_model: IntArray,
    moisture: FuelMoisture,
    midflame_wind: float | FloatArray,
    slope: float | FloatArray,
    *,
    table: FuelModelTable | None = None,
    flame: FlameLengthModel | None = None,
    wind_limit_on: bool = False,
) -> SurfaceFire:
    """Head-fire behaviour of every cell, with the wind blowing upslope.

    The fuel-bed part of the model (:func:`fuel_bed`) is evaluated once per distinct fuel model
    present and mapped back to the cells; the wind factor ``phi_w`` (Rothermel 1972, eqs.
    47-50) and slope factor ``phi_s`` (eq. 51) are then computed per cell. With the wind
    blowing straight upslope (or either one zero) the two factors add:
    ``R = R_0 (1 + phi_w + phi_s)`` (eq. 52). The fireline intensity is ``I = H_A R`` and the
    flame length ``L = 0.0775 I^0.46`` (Byram 1959). :class:`RothermelRule` instead combines
    the factors as vectors when wind and slope point different ways and calls
    :func:`head_fire` with the result.

    Cells whose fuel model is not in ``table`` (0, or the non-burnable models 91-99) do not
    burn: every output field is 0 there (``False`` for ``wind_limited``).

    Args:
        fuel_model: Integer fuel model numbers, any shape (``(H, W)`` for a grid).
        moisture: Fuel moisture contents, fractions of oven-dry weight, as scalars: one
            moisture for every cell. Per-cell moisture is not supported; an array field would
            be matched to the distinct fuel models present, not to cells.
        midflame_wind: Midflame wind speed, m/s, >= 0; a scalar or an array broadcasting to
            ``fuel_model``'s shape.
        slope: Slope as rise over run (``tan`` of the slope angle, dimensionless), >= 0; a
            scalar or an array broadcasting to ``fuel_model``'s shape.
        table: Fuel model parameters. None (the default) loads ``configs/fuel_models.csv``.
        flame: Byram's flame length constants. None (the default) loads
            ``configs/flame_length.yaml``.
        wind_limit_on: If True, cap the effective wind at Rothermel's limit
            (:func:`wind_limit`), as BehavePlus does by default. Off by default, following
            Andrews et al. (2013).

    Returns:
        A :class:`SurfaceFire` whose arrays all have ``fuel_model``'s shape: ``ros`` (m/s),
        ``intensity`` (kW/m), ``flame_length`` (m), ``heat_per_area`` (kJ/m^2),
        ``reaction_intensity`` (kW/m^2), ``residence_time`` (s), ``ros_no_wind`` (m/s), the
        dimensionless ``phi_wind`` and ``phi_slope``, ``effective_wind`` (m/s) and the bool
        ``wind_limited``.

    Raises:
        ValueError: ``midflame_wind`` or ``slope`` is negative, NaN or infinite, or does not
            broadcast to ``fuel_model``'s shape; a moisture value is invalid (see
            :func:`fuel_bed`); or, when ``table`` or ``flame`` is None, the default config
            file fails validation.
        FileNotFoundError: ``table`` or ``flame`` is None and the default config file is
            missing.

    Examples:
        A 2 x 2 grid with short grass (1), GR2 (102), a non-burnable cell (0) and timber
        understory TU5 (165), under a 2 m/s midflame wind on flat ground:

        >>> import numpy as np
        >>> fuel = np.array([[1, 102], [0, 165]])
        >>> moisture = FuelMoisture(0.06, 0.07, 0.08, live_herb=0.60, live_woody=0.90)
        >>> fire = surface_fire(fuel, moisture, midflame_wind=2.0, slope=0.0)
        >>> print(fire.ros.round(3))  # head rate of spread, m/s
        [[0.422 0.167]
         [0.    0.046]]
        >>> print(fire.flame_length.round(2))  # m
        [[1.27 1.3 ]
         [0.   2.13]]
    """
    table = load_fuel_models() if table is None else table
    fm = np.asarray(fuel_model)
    shape = fm.shape
    rows = table.index(fm).reshape(-1)
    burnable = rows >= 0
    present, inverse = np.unique(rows[burnable], return_inverse=True)
    bed = fuel_bed(table, moisture, present)

    def per_cell(values: FloatArray) -> FloatArray:
        out = np.zeros(rows.size, dtype=np.float64)
        out[burnable] = values[inverse]
        return out

    def field(x: float | FloatArray, name: str) -> FloatArray:
        a = np.broadcast_to(np.asarray(x, dtype=np.float64), shape).reshape(-1)
        if np.any(a < 0) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite and >= 0")
        return a

    wind = field(midflame_wind, "midflame_wind")[burnable]
    tan = field(slope, "slope")[burnable]
    cell_bed = FuelBed(**{k: v[inverse] for k, v in bed.__dict__.items()})
    phi_w = cell_bed.wind_factor(wind)
    phi_s = cell_bed.slope_factor(tan)
    fire = head_fire(cell_bed, phi_w + phi_s, flame=flame, wind_limit_on=wind_limit_on)

    def out(values: FloatArray) -> FloatArray:
        full = np.zeros(rows.size, dtype=np.float64)
        full[burnable] = values
        return full.reshape(shape)

    limited = np.zeros(rows.size, dtype=bool)
    limited[burnable] = fire.wind_limited
    return SurfaceFire(
        ros=out(fire.ros),
        intensity=out(fire.intensity),
        flame_length=out(fire.flame_length),
        heat_per_area=per_cell(bed.heat_per_area).reshape(shape),
        reaction_intensity=per_cell(bed.reaction_intensity).reshape(shape),
        residence_time=per_cell(bed.residence_time).reshape(shape),
        ros_no_wind=per_cell(bed.ros_no_wind).reshape(shape),
        phi_wind=out(phi_w),
        phi_slope=out(phi_s),
        effective_wind=out(fire.effective_wind),
        wind_limited=limited.reshape(shape),
    )


@dataclass(frozen=True)
class HeadFire:
    """Head-fire values for a combined wind-and-slope factor (output of :func:`head_fire`).

    Attributes:
        ros: Head rate of spread, m/s.
        intensity: Byram fireline intensity, kW/m.
        flame_length: Flame length, m.
        effective_wind: Effective midflame wind of the factor actually used, m/s.
        wind_limited: True where the wind limit capped the factor.
    """

    ros: FloatArray
    intensity: FloatArray
    flame_length: FloatArray
    effective_wind: FloatArray
    wind_limited: np.ndarray


def head_fire(
    bed: FuelBed,
    phi: float | FloatArray,
    *,
    flame: FlameLengthModel | None = None,
    wind_limit_on: bool = False,
) -> HeadFire:
    """Head rate, intensity and flame length for a combined wind-and-slope factor.

    ``R = R_0 (1 + phi)`` (Rothermel 1972, eq. 52), ``I = H_A R`` (Byram 1959) and
    ``L = 0.0775 I^0.46`` from ``configs/flame_length.yaml``. With ``wind_limit_on`` the
    effective wind of ``phi`` is capped at ``0.9 I_R`` ft/min and ``phi`` recomputed from it.

    Args:
        bed: Fuel bed values, arrays of shape ``(n,)``.
        phi: Combined factor (dimensionless, >= 0), broadcasting to ``(n,)``.
        flame: Byram's constants; None loads ``configs/flame_length.yaml``.
        wind_limit_on: Apply Rothermel's wind limit.
    """
    phi = np.broadcast_to(np.maximum(np.asarray(phi, dtype=np.float64), 0.0), bed.ros_no_wind.shape)
    u_eff = bed.effective_wind(phi)
    limited = np.zeros(phi.shape, dtype=bool)
    if wind_limit_on:
        u_max = wind_limit(bed.reaction_intensity)
        limited = u_eff > u_max
        u_eff = np.where(limited, u_max, u_eff)
        phi = np.where(limited, bed.wind_factor(u_max), phi)
    ros = bed.ros_no_wind * (1.0 + phi)
    intensity = bed.heat_per_area * ros
    model = load_flame_length_model() if flame is None else flame
    return HeadFire(
        ros=ros,
        intensity=intensity,
        flame_length=flame_length_from_intensity(intensity, model),
        effective_wind=u_eff,
        wind_limited=limited,
    )


def residence_times(table: FuelModelTable | None = None, moisture: FuelMoisture | None = None) -> dict[int, float]:
    """Flame residence time ``t_r = 384 / sigma`` min (Anderson 1969) of every fuel model, in s.

    ``sigma`` is the characteristic SAV of the fuel bed, which for a dynamic model depends on
    how much herbaceous load has cured. The values in ``configs/residence_time.yaml`` are
    evaluated at the default moisture of :class:`~wildfire_sim.grid.Weather` (Scott & Burgan's
    D2L2 scenario: 6/7/8 % dead, 60 % live herbaceous, 90 % live woody).

    Args:
        table: Fuel models; None loads ``configs/fuel_models.csv``.
        moisture: None uses the defaults of :class:`~wildfire_sim.grid.Weather`.

    Returns:
        Fuel model number -> residence time, s.
    """
    table = load_fuel_models() if table is None else table
    moisture = FuelMoisture.from_weather(Weather()) if moisture is None else moisture
    bed = fuel_bed(table, moisture)
    return {int(n): float(t) for n, t in zip(table.numbers, bed.residence_time, strict=True)}


# ----------------------------------------------------------------------------- spread rule


LatticeRates = Literal["fitted", "sampled"]


class RothermelParams(BaseModel):
    """Schema of ``configs/rothermel.yaml``.

    Attributes:
        ten_m_to_twenty_ft: Ratio of the 10 m to the 20-ft wind (dimensionless).
        sheltered_crown_fill: Crown fill fraction above which fuel is sheltered by the canopy.
        canopy_height: Canopy height, m, for the sheltered wind adjustment factor.
        crown_ratio: Crown ratio (fraction) for the sheltered wind adjustment factor.
        lb_max: Largest length-to-breadth ratio of the fire ellipse (BehavePlus's cap).
        apply_wind_limit: Apply Rothermel's wind limit ``0.9 I_R`` (an open decision).
        wind_limit_note: Why ``apply_wind_limit`` has its value.
        lattice_rates: ``"fitted"`` or ``"sampled"`` directional rates (WP 2.1d).
        lattice_cap_fitted: Stencil size -> largest ratio the lattice represents with fitted rates.
        lattice_cap_sampled: The same with sampled rates.
    """

    model_config = ConfigDict(extra="forbid")

    ten_m_to_twenty_ft: CitedValue
    sheltered_crown_fill: CitedValue
    canopy_height: CitedValue
    crown_ratio: CitedValue
    lb_max: CitedValue
    apply_wind_limit: bool = False
    wind_limit_note: str = ""
    lattice_rates: LatticeRates = "fitted"
    lattice_cap_fitted: dict[int, CitedValue]
    lattice_cap_sampled: dict[int, CitedValue]

    @model_validator(mode="after")
    def _in_range(self) -> RothermelParams:
        if not self.ten_m_to_twenty_ft.value > 0:
            raise ValueError("ten_m_to_twenty_ft must be > 0")
        if not 0 <= self.sheltered_crown_fill.value <= 1:
            raise ValueError("sheltered_crown_fill must lie in [0, 1]")
        if not self.canopy_height.value > 0:
            raise ValueError("canopy_height must be > 0 m")
        if not 0 <= self.crown_ratio.value <= 1:
            raise ValueError("crown_ratio must lie in [0, 1]")
        if not self.lb_max.value >= 1:
            raise ValueError("lb_max must be >= 1")
        for name, caps in (
            ("lattice_cap_fitted", self.lattice_cap_fitted),
            ("lattice_cap_sampled", self.lattice_cap_sampled),
        ):
            if sorted(caps) != [8, 16]:
                raise ValueError(f"{name} needs entries for stencils 8 and 16")
            if any(not c.value >= 1 for c in caps.values()):
                raise ValueError(f"{name} values must be >= 1")
        return self

    def lattice_cap(self, rates: LatticeRates, stencil_size: int) -> float:
        """Largest length-to-breadth ratio for a stencil and rate kind."""
        caps = self.lattice_cap_fitted if rates == "fitted" else self.lattice_cap_sampled
        if stencil_size not in caps:
            raise ValueError(f"no lattice cap for a {stencil_size}-neighbour stencil")
        return float(caps[stencil_size].value)


def load_rothermel_params(path: str | Path | None = None) -> RothermelParams:
    """Load the rule parameters; default ``configs_dir() / "rothermel.yaml"``."""
    return load_yaml(configs_dir() / "rothermel.yaml" if path is None else path, RothermelParams)


def lattice_fit_path(stencil_size: int) -> Path:
    """The shipped fitted-rate table for a stencil, ``configs_dir() / f"lattice_fit_{stencil_size}.csv"``."""
    return configs_dir() / f"lattice_fit_{stencil_size}.csv"


@dataclass(frozen=True)
class RothermelFire:
    """Everything the ``"rothermel"`` rule computes for a grid (the diagnostic form of its output).

    ``(H, W)`` float64 arrays unless stated; zeros on cells that cannot burn. Values are before
    the retardant multiplier.

    Attributes:
        midflame_wind: Midflame wind, m/s.
        phi_wind: Wind factor (dimensionless).
        phi_slope: Slope factor (dimensionless).
        phi_effective: Vector sum of the two, as used (after the wind limit if applied).
        head_bearing: Bearing of maximum spread, degrees.
        effective_wind: Effective midflame wind of ``phi_effective``, m/s.
        length_to_breadth: Length-to-breadth ratio of the fire ellipse as used, after
            ``lb_scale`` and the caps (1 on unburnable cells).
        lb_capped: Bool, burnable cells whose ratio the lattice cap reduced.
        ros_head: Head rate of spread, m/s, times ``ros_multiplier`` (not changed by the cap).
        intensity: Head fireline intensity, kW/m, of that rate.
        heat_per_area: Heat per unit area, kJ/m^2.
        wind_limited: Bool, where the wind limit capped ``phi_effective``.
        burnable: Bool, cells the rule spreads from.
    """

    midflame_wind: FloatArray
    phi_wind: FloatArray
    phi_slope: FloatArray
    phi_effective: FloatArray
    head_bearing: FloatArray
    effective_wind: FloatArray
    length_to_breadth: FloatArray
    lb_capped: np.ndarray
    ros_head: FloatArray
    intensity: FloatArray
    heat_per_area: FloatArray
    wind_limited: np.ndarray
    burnable: np.ndarray


@register("rothermel")
class RothermelRule:
    """Rothermel surface fire spread with an elliptical fire shape (docs/plan.md section 2.2).

    For every burnable cell:

    1. Midflame wind: the 10 m wind divided by ``ten_m_to_twenty_ft`` (20-ft wind), times the
       wind adjustment factor of the cell's fuel bed depth and canopy cover
       (:func:`~wildfire_sim.spread.ellipse.wind_adjustment_factor`).
    2. Fuel bed values for the weather's moisture (:func:`fuel_bed`, once per fuel model), the
       wind factor of the midflame wind and the slope factor of the cell's slope.
    3. Wind and slope added as vectors (toward ``weather.wind_to_bearing`` and up the slope,
       ``aspect + 180``): the head bearing and ``phi_e``; the head rate ``R_0 (1 + phi_e)``,
       with Rothermel's wind limit if ``apply_wind_limit``.
    4. The calibration factors of WP 3.1 (both 1 by default, which leaves every output as it
       was): the head rate times ``ros_multiplier``, with Byram's intensity ``I = H_A R`` and the
       flame length following from the adjusted rate (heat per unit area unchanged), as
       FARSITE's rate-of-spread adjustment factors do (Finney 1998).
    5. The ellipse: ``LB`` from the effective wind, its elongation scaled to
       ``1 + lb_scale (LB - 1)``, then capped at ``lb_max`` and at the lattice cap of the stencil
       (WP 2.1d), which keeps the head rate and draws the fire rounder.
    6. Directional rates: ``R_head (1 - e) / (1 - e cos(b_k - head))`` sampled from the
       ellipse, times the lattice-fitted multipliers if ``lattice_rates`` is ``"fitted"``
       (:class:`~wildfire_sim.spread.ellipse.LatticeFitTable`; 1 for a round fire).
    7. Rates, head intensity and flame length times ``(1 - retardant)``.

    ``ros_multiplier`` differs from :class:`wildfire_sim.ensemble.ScaledRateRule`, the
    ensemble's random rate factor, which scales the directional rates only and leaves the head
    intensity as it was. Here the fire really spreads faster, so it is also more intense.

    After each :meth:`rate_of_spread` the rule keeps :attr:`lb_capped`, the ``(H, W)`` cells
    whose ratio the lattice cap reduced; the simulator reads it to count the cell-hours the cap
    applied to burning cells (``provenance["lb_capped_cell_hours"]``).

    The fuel models come from ``configs/fuel_models.csv``; a cell whose fuel model is not in it
    (non-burnable 91-99, or 0) does not spread, like a cell whose land use cannot burn.

    Args:
        params_path: Rule parameters; default ``configs_dir() / "rothermel.yaml"``.
        fuel_models_path: Fuel model table; default ``configs_dir() / "fuel_models.csv"``.
        apply_wind_limit: Overrides the YAML value when not None.
        canopy_height: Canopy height, m; overrides the YAML value when not None.
        crown_ratio: Crown ratio, fraction; overrides the YAML value when not None.
        lattice_rates: ``"fitted"`` or ``"sampled"``; overrides the YAML value when not None.
        ros_multiplier: Rate-of-spread adjustment factor, dimensionless, finite and > 0
            (calibrated in WP 3.1; ``spread.params`` in a scenario).
        lb_scale: Elongation factor of the fire ellipse, dimensionless, finite and > 0
            (calibrated in WP 3.1; ``spread.params`` in a scenario).

    References:
        Finney, M. A. (1998). FARSITE: Fire Area Simulator - model development and evaluation.
        USDA For. Serv. Res. Pap. RMRS-RP-4 (rate-of-spread adjustment factors).
    """

    name = "rothermel"

    def __init__(
        self,
        params_path: str | Path | None = None,
        fuel_models_path: str | Path | None = None,
        apply_wind_limit: bool | None = None,
        canopy_height: float | None = None,
        crown_ratio: float | None = None,
        lattice_rates: LatticeRates | None = None,
        ros_multiplier: float = 1.0,
        lb_scale: float = 1.0,
    ) -> None:
        self.params = load_rothermel_params(params_path)
        self.table = load_fuel_models(fuel_models_path)
        self.flame = load_flame_length_model()
        p = self.params
        self.apply_wind_limit = p.apply_wind_limit if apply_wind_limit is None else bool(apply_wind_limit)
        self.canopy_height = float(p.canopy_height.value if canopy_height is None else canopy_height)
        self.crown_ratio = float(p.crown_ratio.value if crown_ratio is None else crown_ratio)
        self.lattice_rates: LatticeRates = p.lattice_rates if lattice_rates is None else lattice_rates
        if not self.canopy_height > 0:
            raise ValueError("canopy_height must be > 0 m")
        if not 0 <= self.crown_ratio <= 1:
            raise ValueError("crown_ratio must lie in [0, 1]")
        if self.lattice_rates not in ("fitted", "sampled"):
            raise ValueError("lattice_rates must be 'fitted' or 'sampled'")
        self.ros_multiplier = float(ros_multiplier)
        self.lb_scale = float(lb_scale)
        for label, value in (("ros_multiplier", self.ros_multiplier), ("lb_scale", self.lb_scale)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{label} must be finite and > 0, got {value!r}")
        self._fit_tables: dict[int, LatticeFitTable] = {}
        self.lb_capped: np.ndarray | None = None

    def fit_table(self, stencil_size: int) -> LatticeFitTable:
        """The fitted-rate table of a stencil, read once from ``configs/``."""
        if stencil_size not in self._fit_tables:
            self._fit_tables[stencil_size] = load_fit_table(lattice_fit_path(stencil_size), stencil_size)
        return self._fit_tables[stencil_size]

    def lb_cap(self, stencil: Stencil | None) -> float:
        """Ratio cap in force: BehavePlus's ``lb_max`` and, with a stencil, the lattice cap."""
        cap = float(self.params.lb_max.value)
        if stencil is not None:
            cap = min(cap, self.params.lattice_cap(self.lattice_rates, stencil.size))
        return cap

    def fire(self, grid: Grid, weather: Weather, stencil: Stencil | None = None) -> RothermelFire:
        """Head fire, direction and ellipse of every cell, before the retardant multiplier.

        Args:
            grid: Reads ``fuel_model``, ``canopy_cover`` (fraction), ``slope`` (rise/run),
                ``aspect`` (downhill bearing, degrees) and the burnable mask. Not modified.
            weather: ``wind_speed`` (10 m, m/s), ``wind_to_bearing`` (degrees) and the fuel
                moistures.
            stencil: When given, the ratio is also capped at the stencil's lattice cap.
        """
        h, w = grid.shape
        rows = self.table.index(grid.fuel_model).reshape(-1)
        burnable = (rows >= 0) & grid.burnable_mask().reshape(-1)
        present, inverse = np.unique(rows[burnable], return_inverse=True)
        bed_models = fuel_bed(self.table, FuelMoisture.from_weather(weather), present)
        bed = FuelBed(**{k: v[inverse] for k, v in bed_models.__dict__.items()})
        cells = rows[burnable]

        p = self.params
        waf = wind_adjustment_factor(
            self.table.depth[cells] * FT,
            grid.canopy_cover.reshape(-1)[burnable],
            self.canopy_height,
            self.crown_ratio,
            p.sheltered_crown_fill.value,
        )
        midflame = twenty_foot_wind(float(weather.wind_speed), p.ten_m_to_twenty_ft.value) * waf
        phi_w = bed.wind_factor(midflame)
        phi_s = bed.slope_factor(grid.slope.reshape(-1)[burnable].astype(np.float64))
        upslope = (grid.aspect.reshape(-1)[burnable].astype(np.float64) + 180.0) % 360.0
        phi_e, head = combine_wind_and_slope(phi_w, weather.wind_to_bearing, phi_s, upslope)
        hf = head_fire(bed, phi_e, flame=self.flame, wind_limit_on=self.apply_wind_limit)
        phi_used = np.where(hf.wind_limited, bed.wind_factor(hf.effective_wind), phi_e)
        lb_free = length_to_breadth(hf.effective_wind, p.lb_max.value, self.lb_scale)
        cap = self.lb_cap(stencil)
        lb = np.minimum(lb_free, cap)

        def full(values: FloatArray, fill: float = 0.0) -> FloatArray:
            out = np.full(h * w, fill, dtype=np.float64)
            out[burnable] = values
            return out.reshape(h, w)

        def mask(values: np.ndarray) -> np.ndarray:
            out = np.zeros(h * w, dtype=bool)
            out[burnable] = values
            return out.reshape(h, w)

        return RothermelFire(
            midflame_wind=full(midflame),
            phi_wind=full(phi_w),
            phi_slope=full(phi_s),
            phi_effective=full(phi_used),
            head_bearing=full(head),
            effective_wind=full(hf.effective_wind),
            length_to_breadth=full(lb, 1.0),
            lb_capped=mask(lb_free > cap * (1.0 + 1e-9)),  # rounding noise at the cap is not a cap
            ros_head=full(hf.ros * self.ros_multiplier),
            intensity=full(hf.intensity * self.ros_multiplier),
            heat_per_area=full(bed.heat_per_area),
            wind_limited=mask(hf.wind_limited),
            burnable=burnable.reshape(h, w),
        )

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        """Directional rate of spread for every cell (see the class docstring).

        Returns:
            :class:`SpreadOutput`: ``ros`` ``(K, H, W)`` m/s; head ``intensity`` (kW/m) and
            ``flame_length`` (m); ``heat_per_area`` (kJ/m^2) for the front-normal intensity of
            WP 1.12. Rates, intensity and flame length include ``(1 - retardant)``.
        """
        f = self.fire(grid, weather, stencil)
        keep = 1.0 - np.clip(grid.retardant.astype(np.float64), 0.0, 1.0)
        ros = directional_rates(f.ros_head * keep, f.head_bearing, f.length_to_breadth, stencil.bearings_deg)
        if self.lattice_rates == "fitted":
            ros = ros * self.fit_table(stencil.size).multipliers(f.length_to_breadth, f.head_bearing)
        self.lb_capped = f.lb_capped
        intensity = f.intensity * keep
        return SpreadOutput(
            ros=ros,
            intensity=intensity,
            flame_length=flame_length_from_intensity(intensity, self.flame),
            heat_per_area=f.heat_per_area,
        )
