"""Elliptical fire shape: midflame wind, wind-slope vector combination and directional rates.

A fire spreading from a point under uniform conditions grows as an ellipse with the ignition
point at its rear focus (Anderson 1983). Given the head rate of spread, its direction and the
length-to-breadth ratio ``LB``, the rate at an angle ``theta`` from the head is

.. math::

    e = \\frac{\\sqrt{LB^2 - 1}}{LB}, \\qquad R(\\theta) = R_{head} \\frac{1 - e}{1 - e \\cos\\theta}

so the back spreads at ``R_head (1 - e) / (1 + e)`` (docs/plan.md section 2.2).

This module holds the pieces the ``"rothermel"`` rule (:mod:`wildfire_sim.spread.rothermel`)
needs between the fuel physics and the stencil, as functions of plain arrays:

1. :func:`twenty_foot_wind` and :func:`wind_adjustment_factor`: the 10 m wind of the weather
   sources reduced to the 20-ft wind (Turner and Lawson 1978) and then to midflame height
   (Albini and Baughman 1979; Andrews 2012).
2. :func:`combine_wind_and_slope`: the wind and slope factors added as vectors, giving the
   direction of maximum spread and the combined factor ``phi_e`` (Rothermel 1972; Albini 1976).
3. :func:`length_to_breadth` and :func:`directional_rates`: the ellipse from the effective
   midflame wind (Anderson 1983, the form BehavePlus uses), capped at ``lb_max``.

Bearings are degrees clockwise from north; a bearing ``b`` points east by ``sin b`` and north
by ``cos b``.

References:
    Albini, F. A. & Baughman, R. G. (1979). Estimating windspeeds for predicting wildland fire
    behavior. USDA For. Serv. Res. Pap. INT-221.
    Anderson, H. E. (1983). Predicting wind-driven wild land fire size and shape. USDA For.
    Serv. Res. Pap. INT-305.
    Andrews, P. L. (2012). Modeling wind adjustment factor and midflame wind speed for
    Rothermel's surface fire spread model. USDA For. Serv. Gen. Tech. Rep. RMRS-GTR-266.
    Turner, J. A. & Lawson, B. D. (1978). Weather in the Canadian Forest Fire Danger Rating
    System. Environment Canada, Pacific Forest Research Centre, BC-X-177.
"""

from __future__ import annotations

import numpy as np

from wildfire_sim.types import FloatArray

FT = 0.3048
"""Metres per foot."""
MPH = 0.44704
"""Metres per second per mile per hour."""


def twenty_foot_wind(wind_10m: float | FloatArray, factor: float) -> FloatArray:
    """20-ft (6.1 m) wind from the open 10 m wind: ``U_20ft = U_10m / factor``, m/s.

    Args:
        wind_10m: Open wind at 10 m, m/s, >= 0.
        factor: Ratio of the 10 m to the 20-ft wind; 1.15 (Turner and Lawson 1978, as used by
            BehavePlus; Andrews 2012), from ``configs/rothermel.yaml``.
    """
    if factor <= 0:
        raise ValueError("factor must be > 0")
    return np.asarray(wind_10m, dtype=np.float64) / factor


def wind_adjustment_factor(
    fuel_depth_m: float | FloatArray,
    canopy_cover: float | FloatArray,
    canopy_height_m: float | FloatArray,
    crown_ratio: float | FloatArray,
    sheltered_crown_fill: float = 0.05,
) -> FloatArray:
    """Ratio of the midflame wind to the 20-ft wind (dimensionless), unsheltered or sheltered.

    The crown fill portion is ``f = canopy_cover * crown_ratio / 3``. Where ``f`` is at most
    ``sheltered_crown_fill`` the fuel is unsheltered (Albini and Baughman 1979; Andrews 2012)
    and, with the fuel bed depth ``H`` in ft,

    .. math:: WAF = \\frac{1.83}{\\ln\\left(\\frac{20 + 0.36 H}{0.13 H}\\right)}

    otherwise it is sheltered by the canopy (Andrews 2012), with the canopy height
    ``H_c`` in ft,

    .. math:: WAF = \\frac{0.555}{\\sqrt{f H_c}\\,\\ln\\left(\\frac{20 + 0.36 H_c}{0.13 H_c}\\right)}

    These are the BehavePlus forms; ``tests/test_spread_ellipse.py`` checks them against its
    engine.

    Args:
        fuel_depth_m: Fuel bed depth, m, > 0.
        canopy_cover: Canopy cover, fraction in [0, 1].
        canopy_height_m: Canopy height, m, > 0 where the fuel is sheltered.
        crown_ratio: Fraction of the canopy height occupied by crowns, in [0, 1].
        sheltered_crown_fill: Crown fill above which the fuel counts as sheltered.

    Returns:
        WAF, float64, broadcast shape of the inputs.
    """
    depth_ft = np.asarray(fuel_depth_m, dtype=np.float64) / FT
    cc = np.asarray(canopy_cover, dtype=np.float64)
    ch_ft = np.asarray(canopy_height_m, dtype=np.float64) / FT
    cr = np.asarray(crown_ratio, dtype=np.float64)
    if np.any(depth_ft <= 0):
        raise ValueError("fuel_depth_m must be > 0")
    fill = cc * cr / 3.0
    sheltered = fill > sheltered_crown_fill
    if np.any(sheltered & (ch_ft <= 0)):
        raise ValueError("canopy_height_m must be > 0 where the fuel is sheltered")
    unsheltered_waf = 1.83 / np.log((20.0 + 0.36 * depth_ft) / (0.13 * depth_ft))
    safe_ch = np.where(sheltered, ch_ft, 1.0)
    safe_fill = np.where(sheltered, fill, 1.0)
    sheltered_waf = 0.555 / (np.sqrt(safe_fill * safe_ch) * np.log((20.0 + 0.36 * safe_ch) / (0.13 * safe_ch)))
    return np.where(sheltered, sheltered_waf, unsheltered_waf)


def bearing_vector(bearing_deg: float | FloatArray) -> tuple[FloatArray, FloatArray]:
    """``(east, north)`` components of a unit vector pointing along ``bearing_deg``."""
    b = np.radians(np.asarray(bearing_deg, dtype=np.float64))
    return np.sin(b), np.cos(b)


def combine_wind_and_slope(
    phi_wind: float | FloatArray,
    wind_to_bearing: float | FloatArray,
    phi_slope: float | FloatArray,
    upslope_bearing: float | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Add the wind and slope factors as vectors.

    The rate increments ``R_0 phi_w`` (toward the wind) and ``R_0 phi_s`` (upslope) are vectors;
    their sum gives the direction of maximum spread and the combined factor ``phi_e`` with
    ``R_head = R_0 (1 + phi_e)`` (Rothermel 1972; Albini 1976; Andrews 2018).
    Where wind and slope agree this is ``phi_w + phi_s``; where they point different ways the
    head lies between them, closer to the stronger.

    Args:
        phi_wind: Wind factor (dimensionless, >= 0).
        wind_to_bearing: Bearing the wind blows toward, degrees.
        phi_slope: Slope factor (dimensionless, >= 0).
        upslope_bearing: Bearing straight up the slope (aspect + 180), degrees.

    Returns:
        ``(phi_e, head_bearing)``: the combined factor and the bearing of maximum spread in
        [0, 360) degrees (0 where ``phi_e`` is 0, where every direction is equal).
    """
    pw = np.asarray(phi_wind, dtype=np.float64)
    ps = np.asarray(phi_slope, dtype=np.float64)
    we, wn = bearing_vector(wind_to_bearing)
    se, sn = bearing_vector(upslope_bearing)
    east = pw * we + ps * se
    north = pw * wn + ps * sn
    phi = np.hypot(east, north)
    # Equal and opposite factors cancel to rounding error, whose direction is meaningless.
    phi = np.where(phi > 1e-12 * (pw + ps), phi, 0.0)
    head = np.where(phi > 0, np.degrees(np.arctan2(east, north)) % 360.0, 0.0)
    return phi, head


def length_to_breadth(effective_wind: float | FloatArray, lb_max: float = 8.0) -> FloatArray:
    """Length-to-breadth ratio of the fire ellipse from the effective midflame wind, m/s.

    Anderson (1983) in the form of BehavePlus, with ``U`` in m/s (the mph coefficients 0.1147
    and 0.0692 times 2.2369):

    .. math:: LB = 0.936\\,e^{0.2566 U} + 0.461\\,e^{-0.1548 U} - 0.397

    capped at ``lb_max`` (8 in BehavePlus, ``configs/rothermel.yaml``). ``LB = 1`` at ``U = 0``.
    """
    if lb_max < 1:
        raise ValueError("lb_max must be >= 1")
    u = np.maximum(np.asarray(effective_wind, dtype=np.float64), 0.0)
    lb = 0.936 * np.exp(0.1147 / MPH * u) + 0.461 * np.exp(-0.0692 / MPH * u) - 0.397
    return np.clip(lb, 1.0, lb_max)


def eccentricity(lb: float | FloatArray) -> FloatArray:
    """Eccentricity ``e = sqrt(LB^2 - 1) / LB`` of an ellipse with length-to-breadth ratio ``LB >= 1``."""
    lb = np.asarray(lb, dtype=np.float64)
    if np.any(lb < 1):
        raise ValueError("lb must be >= 1")
    return np.sqrt(lb**2 - 1.0) / lb


def directional_rates(
    ros_head: float | FloatArray,
    head_bearing: float | FloatArray,
    lb: float | FloatArray,
    bearings_deg: FloatArray,
) -> FloatArray:
    """Rate of spread toward each stencil direction from the fire ellipse, m/s.

    ``R_k = R_head (1 - e) / (1 - e cos(b_k - head))`` with the ignition point at the rear
    focus (Anderson 1983; docs/plan.md section 2.2).

    Args:
        ros_head: Head rate of spread, m/s, shape ``S`` (``(H, W)`` for a grid).
        head_bearing: Bearing of maximum spread, degrees, broadcasting to ``S``.
        lb: Length-to-breadth ratio, >= 1, broadcasting to ``S``.
        bearings_deg: ``(K,)`` stencil bearings, degrees.

    Returns:
        ``(K, *S)`` float64 rates.
    """
    r = np.asarray(ros_head, dtype=np.float64)
    e = eccentricity(lb)
    b = np.asarray(bearings_deg, dtype=np.float64).reshape((-1,) + (1,) * r.ndim)
    cos = np.cos(np.radians(b - np.asarray(head_bearing, dtype=np.float64)))
    return r * (1.0 - e) / (1.0 - e * cos)
