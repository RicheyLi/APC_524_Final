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

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from wildfire_sim.types import BoolArray, FloatArray, IntArray

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


def length_to_breadth(effective_wind: float | FloatArray, lb_max: float = 8.0, lb_scale: float = 1.0) -> FloatArray:
    """Length-to-breadth ratio of the fire ellipse from the effective midflame wind, m/s.

    Anderson (1983) in the form of BehavePlus, with ``U`` in m/s (the mph coefficients 0.1147
    and 0.0692 times 2.2369):

    .. math:: LB = 0.936\\,e^{0.2566 U} + 0.461\\,e^{-0.1548 U} - 0.397

    With ``lb_scale`` ``s`` the elongation is scaled, ``LB' = 1 + s (LB - 1)``: the calibrated
    shape parameter of WP 3.1. ``LB' >= 1`` for every ``s > 0`` and a fire without wind or slope
    stays round. The result is then capped at ``lb_max`` (8 in BehavePlus,
    ``configs/rothermel.yaml``). ``LB = 1`` at ``U = 0``.

    Args:
        effective_wind: Effective midflame wind, m/s.
        lb_max: Cap on the ratio, >= 1.
        lb_scale: Elongation factor ``s``, dimensionless, finite and > 0; 1 is Anderson's ratio.

    Returns:
        The ratio (dimensionless), shaped like ``effective_wind``.
    """
    if lb_max < 1:
        raise ValueError("lb_max must be >= 1")
    if not (math.isfinite(lb_scale) and lb_scale > 0):
        raise ValueError(f"lb_scale must be finite and > 0, got {lb_scale!r}")
    u = np.maximum(np.asarray(effective_wind, dtype=np.float64), 0.0)
    lb = 0.936 * np.exp(0.1147 / MPH * u) + 0.461 * np.exp(-0.0692 / MPH * u) - 0.397
    if lb_scale != 1.0:
        lb = 1.0 + lb_scale * (lb - 1.0)
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


# ----------------------------------------------------------------------------- lattice-fitted rates (WP 2.1d)
#
# Without kernel noise the burn of a fire with directional rates r_k is T times the convex hull
# of the points r_k u_k (u_k the unit vector of stencil direction k), a polygon (docs/
# lattice_study.md section 2). Rates sampled from the ellipse put its vertices on the ellipse,
# so the polygon lies inside it and is short between stencil directions. Fitted rates move the
# vertices along their rays to minimize the largest relative radial error between the polygon
# and the ellipse, measured from the ignition point (the rear focus).


def polygon_radius(rates: FloatArray, bearings_deg: FloatArray, theta_deg: FloatArray) -> FloatArray:
    """Distance from the origin to the polygon through the points ``rates[k] u_k``, along ``theta_deg``.

    The polygon joins the points in order of bearing (a star polygon). When it is convex, which
    holds for sampled and fitted rates, it is the convex hull that the lattice burns.

    Args:
        rates: ``(K,)`` positive rates (any unit).
        bearings_deg: ``(K,)`` stencil bearings, degrees.
        theta_deg: Directions to evaluate, degrees, any shape.

    Returns:
        Radii, same unit as ``rates``, shape of ``theta_deg``.
    """
    order = np.argsort(np.asarray(bearings_deg, dtype=np.float64) % 360.0)
    b = np.radians(np.asarray(bearings_deg, dtype=np.float64)[order] % 360.0)
    r = np.asarray(rates, dtype=np.float64)[order]
    th = np.radians(np.asarray(theta_deg, dtype=np.float64) % 360.0)
    k = b.size
    i0 = (np.searchsorted(b, th, side="right") - 1) % k
    i1 = (i0 + 1) % k
    gap = (b[i1] - b[i0]) % (2 * np.pi)
    a0 = (th - b[i0]) % (2 * np.pi)
    r0, r1 = r[i0], r[i1]
    return r0 * r1 * np.sin(gap) / (r0 * np.sin(a0) + r1 * np.sin(gap - a0))


def is_convex(rates: FloatArray, bearings_deg: FloatArray) -> bool:
    """Whether the star polygon of :func:`polygon_radius` is convex (every vertex on the hull)."""
    order = np.argsort(np.asarray(bearings_deg, dtype=np.float64) % 360.0)
    b = np.radians(np.asarray(bearings_deg, dtype=np.float64)[order])
    r = np.asarray(rates, dtype=np.float64)[order]
    p = np.stack([r * np.sin(b), r * np.cos(b)], axis=1)
    e1 = np.roll(p, -1, axis=0) - p
    e2 = np.roll(p, -2, axis=0) - np.roll(p, -1, axis=0)
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return bool(np.all(cross <= 1e-12) or np.all(cross >= -1e-12))


def ellipse_radius(theta_deg: FloatArray, lb: float, head_bearing: float) -> FloatArray:
    """Distance from the rear focus to an ellipse of head distance 1, along ``theta_deg`` (the ellipse rates)."""
    e = float(eccentricity(lb))
    return (1.0 - e) / (1.0 - e * np.cos(np.radians(np.asarray(theta_deg, dtype=np.float64) - head_bearing)))


def shape_errors(rates: FloatArray, bearings_deg: FloatArray, lb: float, head_bearing: float) -> dict[str, float]:
    """Errors of the lattice polygon of ``rates`` (head rate 1) against the analytic ellipse.

    Returns:
        ``radial``: largest relative radial error over all directions; ``head``, ``flank``,
        ``back``: relative errors of the polygon's extent along the head, across it (half the
        width) and backwards, as measured in the lattice study.
    """
    b = np.radians(np.asarray(bearings_deg, dtype=np.float64))
    h = math.radians(head_bearing)
    r = np.asarray(rates, dtype=np.float64)
    east, north = r * np.sin(b), r * np.cos(b)
    along = east * math.sin(h) + north * math.cos(h)
    across = east * math.cos(h) - north * math.sin(h)
    e = float(eccentricity(lb))
    back_exact = (1 - e) / (1 + e)
    half_width_exact = (1 + back_exact) / 2 * math.sqrt(1 - e * e)
    theta = np.linspace(0.0, 360.0, 1440, endpoint=False)
    radial = polygon_radius(r, bearings_deg, theta) / ellipse_radius(theta, lb, head_bearing) - 1.0
    return {
        "radial": float(np.max(np.abs(radial))),
        "head": float(along.max() - 1.0),
        "flank": float(np.ptp(across) / 2 / half_width_exact - 1.0),
        "back": float(-along.min() / back_exact - 1.0),
    }


def fit_rates(
    lb: float, head_bearing: float, bearings_deg: FloatArray, n_theta: int = 720, regularization: float = 1e-4
) -> FloatArray:
    """Directional rates (head rate 1) whose lattice polygon best matches the ellipse.

    Minimizes the largest relative radial error ``|rho(theta) / r_E(theta) - 1|`` over
    ``n_theta`` directions, where ``rho`` is the polygon of :func:`polygon_radius` and ``r_E``
    the ellipse seen from its rear focus (docs/plan.md WP 2.1d). Solved as a smooth constrained
    problem (minimize ``t`` subject to ``-t <= error <= t``) by SLSQP on the logarithms of the
    rates relative to the sampled ones; a small penalty on those logarithms picks one solution
    where several directions do not affect the largest error.

    Args:
        lb: Length-to-breadth ratio, >= 1.
        head_bearing: Head bearing, degrees.
        bearings_deg: ``(K,)`` stencil bearings, degrees.
        n_theta: Directions at which the error is evaluated.
        regularization: Weight of the penalty on the log rates.

    Returns:
        ``(K,)`` rates in the order of ``bearings_deg``.

    Raises:
        RuntimeError: The optimizer fails or returns a polygon that is not convex.
    """
    bearings = np.asarray(bearings_deg, dtype=np.float64)
    k = bearings.size
    sampled = ellipse_radius(bearings, lb, head_bearing)
    theta = np.linspace(0.0, 360.0, n_theta, endpoint=False)
    target = ellipse_radius(theta, lb, head_bearing)

    def error(x: FloatArray) -> FloatArray:
        return polygon_radius(sampled * np.exp(x), bearings, theta) / target - 1.0

    sampled_error = float(np.max(np.abs(error(np.zeros(k)))))
    if sampled_error < 1e-12:
        return sampled
    scale = 1.0 / sampled_error  # the optimizer works on errors of order 1 at every ratio
    z0 = np.zeros(k + 1)
    z0[k] = 1.0
    result = minimize(
        lambda z: z[k] + regularization * scale * float(np.sum(z[:k] ** 2)),
        z0,
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": lambda z: z[k] - scale * error(z[:k])},
            {"type": "ineq", "fun": lambda z: z[k] + scale * error(z[:k])},
        ],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    rates = sampled * np.exp(result.x[:k])
    fitted_error = float(np.max(np.abs(error(result.x[:k]))))
    # SLSQP can stop at the optimum with "positive directional derivative": accept any convex
    # result that beats the sampled rates, whatever its status.
    if not (fitted_error <= sampled_error and is_convex(rates, bearings)):
        raise RuntimeError(f"rate fit failed at LB {lb}, head {head_bearing} deg: {result.message}")
    return rates


# The square lattice is unchanged by the eight rotations and reflections of the square, so a
# fit for any head bearing is a fit for a bearing in [0, 45] degrees with the directions
# permuted. Symmetry operation (q, reflect): rotate by -90 q degrees, then, if reflect, mirror
# about the 45 degree line (b -> 90 - b).


def _apply_symmetry(bearing: FloatArray, q: IntArray | int, reflect: BoolArray | bool) -> FloatArray:
    b = (np.asarray(bearing, dtype=np.float64) - 90.0 * np.asarray(q)) % 360.0
    return np.where(reflect, (90.0 - b) % 360.0, b)


def canonical_head(head_bearing: float | FloatArray) -> tuple[FloatArray, IntArray, BoolArray]:
    """Map head bearings into [0, 45] degrees by a symmetry of the square lattice.

    Returns:
        ``(head, q, reflect)``: the canonical bearing and the operation that produced it.
    """
    h = np.asarray(head_bearing, dtype=np.float64) % 360.0
    q = np.floor(h / 90.0).astype(np.int64) % 4
    r = h - 90.0 * q
    reflect = r > 45.0
    return np.where(reflect, 90.0 - r, r), q, reflect


def symmetry_permutations(bearings_deg: FloatArray) -> IntArray:
    """``(8, K)`` index table: ``perm[op, k]`` is the direction that direction ``k`` maps to under ``op``.

    ``op = 2 q + reflect``. Raises ValueError if the stencil is not symmetric under the square's
    symmetries.
    """
    b = np.asarray(bearings_deg, dtype=np.float64) % 360.0
    perm = np.empty((8, b.size), dtype=np.int64)
    for q in range(4):
        for reflect in (0, 1):
            mapped = _apply_symmetry(b, q, bool(reflect))
            diff = np.abs(((mapped[:, None] - b[None, :]) + 180.0) % 360.0 - 180.0)
            idx = np.argmin(diff, axis=1)
            if np.any(diff[np.arange(b.size), idx] > 1e-6):
                raise ValueError("stencil is not symmetric under the square's rotations and reflections")
            perm[2 * q + reflect] = idx
    return perm


@dataclass(frozen=True)
class LatticeFitTable:
    """Fitted-rate multipliers for one stencil on a grid of ratios and canonical head bearings.

    ``log_multiplier[i, j, k]`` is ``log(fitted / sampled)`` for direction ``k`` (stencil order)
    at ``lb[i]`` and head bearing ``head_deg[j]`` in [0, 45], divided by the same quantity of
    the round fire (``lb = 1``), so a round fire keeps the sampled (equal) rates whose lattice
    error the front-speed correction already removes (docs/lattice_study.md section 7).

    Attributes:
        stencil_size: 8 or 16.
        bearings_deg: ``(K,)`` stencil bearings the columns refer to.
        lb: ``(I,)`` increasing ratios, ``lb[0] == 1``.
        head_deg: ``(J,)`` increasing bearings from 0 to 45.
        log_multiplier: ``(I, J, K)``.
    """

    stencil_size: int
    bearings_deg: FloatArray
    lb: FloatArray
    head_deg: FloatArray
    log_multiplier: FloatArray

    def multipliers(self, lb: FloatArray, head_bearing: FloatArray) -> FloatArray:
        """``(K, *S)`` multipliers of the sampled rates for ratios ``lb`` and head bearings, shape ``S``.

        Bilinear interpolation of the log multipliers in ratio and canonical bearing; ratios
        beyond the table are clamped to its ends.
        """
        lb = np.asarray(lb, dtype=np.float64)
        shape = np.broadcast_shapes(lb.shape, np.shape(head_bearing))
        lbf = np.broadcast_to(lb, shape).reshape(-1)
        head, q, reflect = canonical_head(np.broadcast_to(np.asarray(head_bearing, dtype=np.float64), shape))
        head, q, reflect = head.reshape(-1), q.reshape(-1), reflect.reshape(-1)

        def bracket(grid: FloatArray, x: FloatArray) -> tuple[IntArray, FloatArray]:
            x = np.clip(x, grid[0], grid[-1])
            i = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
            return i, (x - grid[i]) / (grid[i + 1] - grid[i])

        i, fi = bracket(self.lb, lbf)
        j, fj = bracket(self.head_deg, head)
        t = self.log_multiplier
        canon = (
            t[i, j] * ((1 - fi) * (1 - fj))[:, None]
            + t[i + 1, j] * (fi * (1 - fj))[:, None]
            + t[i, j + 1] * ((1 - fi) * fj)[:, None]
            + t[i + 1, j + 1] * (fi * fj)[:, None]
        )  # (N, K) in canonical direction order
        perm = symmetry_permutations(self.bearings_deg)[2 * q + reflect.astype(np.int64)]  # (N, K)
        cell = np.take_along_axis(canon, perm, axis=1)
        return np.exp(cell).T.reshape((self.bearings_deg.size, *shape))


def build_fit_table(
    bearings_deg: FloatArray, stencil_size: int, lbs: FloatArray, heads_deg: FloatArray
) -> LatticeFitTable:
    """Fit the rates at every (ratio, canonical bearing) and normalize by the round fire.

    Args:
        bearings_deg: ``(K,)`` stencil bearings.
        stencil_size: 8 or 16 (recorded only).
        lbs: Increasing ratios starting at 1.
        heads_deg: Increasing bearings from 0 to 45.
    """
    lbs = np.asarray(lbs, dtype=np.float64)
    heads = np.asarray(heads_deg, dtype=np.float64)
    if lbs[0] != 1.0 or heads[0] != 0.0 or heads[-1] != 45.0:
        raise ValueError("the table must start at LB 1 and cover head bearings 0 to 45 degrees")
    b = np.asarray(bearings_deg, dtype=np.float64)
    circle = np.log(fit_rates(1.0, 0.0, b) / ellipse_radius(b, 1.0, 0.0))
    table = np.empty((lbs.size, heads.size, b.size))
    for i, lb in enumerate(lbs):
        for j, h in enumerate(heads):
            if lb == 1.0:
                table[i, j] = 0.0
                continue
            fitted = fit_rates(float(lb), float(h), b)
            table[i, j] = np.log(fitted / ellipse_radius(b, float(lb), float(h))) - circle
    return LatticeFitTable(stencil_size, b, lbs, heads, table)


def save_fit_table(table: LatticeFitTable, path: str | Path, source: str) -> None:
    """Write a :class:`LatticeFitTable` as CSV: one row per (ratio, bearing), one column per direction."""
    k = table.bearings_deg.size
    header = ["lb", "head_deg", *[f"m{i}" for i in range(k)]]
    lines = [
        f"# Lattice-fitted rate multipliers, {table.stencil_size}-neighbour stencil (WP 2.1d).",
        f"# source: {source}",
        "# Columns m0.. are fitted / sampled rate per stencil direction (order of",
        "# wildfire_sim.types.get_stencil), divided by the same ratio of the round fire, for the",
        "# canonical head bearing head_deg in [0, 45]; other bearings follow by symmetry.",
        "# bearings_deg: " + " ".join(f"{x:.6f}" for x in table.bearings_deg),
        ",".join(header),
    ]
    for i, lb in enumerate(table.lb):
        for j, h in enumerate(table.head_deg):
            m = np.exp(table.log_multiplier[i, j])
            lines.append(",".join([f"{lb:.4f}", f"{h:.4f}", *[f"{v:.6f}" for v in m]]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_fit_table(path: str | Path, stencil_size: int) -> LatticeFitTable:
    """Read a table written by :func:`save_fit_table`.

    Raises:
        ValueError: The file is malformed, not on a full (ratio, bearing) grid, or its bearings
            line is missing.
    """
    p = Path(path)
    bearings: FloatArray | None = None
    rows: list[list[float]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("# bearings_deg:"):
            bearings = np.array([float(x) for x in line.split(":", 1)[1].split()])
        elif line.startswith(("#", "lb,")) or not line.strip():
            continue
        else:
            rows.append([float(x) for x in line.split(",")])
    if bearings is None or not rows:
        raise ValueError(f"{p}: missing bearings line or data")
    data = np.array(rows)
    if data.shape[1] != 2 + bearings.size:
        raise ValueError(f"{p}: expected {2 + bearings.size} columns")
    lbs, heads = np.unique(data[:, 0]), np.unique(data[:, 1])
    if lbs.size * heads.size != data.shape[0]:
        raise ValueError(f"{p}: rows do not form a full (lb, head_deg) grid")
    order = np.lexsort((data[:, 1], data[:, 0]))
    m = data[order, 2:].reshape(lbs.size, heads.size, bearings.size)
    if np.any(m <= 0):
        raise ValueError(f"{p}: multipliers must be > 0")
    return LatticeFitTable(stencil_size, bearings, lbs, heads, np.log(m))
