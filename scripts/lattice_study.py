"""Lattice study (WP 1.4): how well each stencil represents an elliptical fire.

Regenerates the figures and the CSV of ``docs/lattice_study.md`` under ``docs/img/lattice/``.

For stencils 8 and 16, length-to-breadth ratios LB in {1, 1.5, 3, 6} and head directions
{0, 22.5, 45} degrees (bearing, clockwise from north), the simulator (:func:`wildfire_sim.sim.run`
with the shipped front-speed correction, ``arrival_shape = 20``) burns flat uniform fuel with an
elliptical spread rule

    R(theta) = R_head (1 - e) / (1 - e cos(theta)),   e = sqrt(LB^2 - 1) / LB,

where theta is the angle between a stencil direction and the head direction. The continuum
burn from a point at time T is the ellipse with the ignition at its rear focus: head distance
``R_head T``, back distance ``R_head T (1 - e) / (1 + e)``, half-width ``b = a sqrt(1 - e^2)``
with ``a = (head + back) / 2``. The simulated distances are the largest projections of burned
cell centres (``arrival_time <= T``) on the head direction, its opposite and its normal.

The script also computes the *deterministic lattice limit*: with every crossing threshold equal
to 1 the burn at large T is ``T`` times the convex hull of the link points ``R(theta_k) u_k``
(``u_k`` the unit vector of stencil direction k), so its head, back and half-width are support
functions of that polygon. That is computed for 8, 16 and hypothetical 32- and 48-neighbour
stencils (all primitive offsets with ``max(|dr|, |dc|) <= 3`` or ``4``), which the simulator
does not have.

Every simulated case is run at the default ``arrival_shape = 20`` and again at
:data:`ALT_SHAPE` (200), each with its shipped correction, and the uncorrected head speed of an
axis-aligned ellipse is printed by LB (the kernel-noise speed-up that the correction assumes).

Outputs (``docs/img/lattice/``; ``_shape200`` files for the alternative shape):

* ``ellipse_errors.csv``: simulated head / flank / back distances and relative errors.
* ``ellipse_errors.png``: relative error of head, flank and back distance against LB.
* ``outline_stencil8.png``, ``outline_stencil16.png``: burned cells over the analytic ellipse.
* ``hull_limit.png``: worst-case deterministic-limit error over head directions vs LB.
* ``isotropic_profile.png``: front speed / R against direction with no wind, after correction.

WP 2.1d adds lattice-fitted rates (``wildfire_sim.spread.ellipse.fit_rates``): ``--fit-tables``
fits them on a grid of ratios and canonical head bearings and writes
``configs/lattice_fit_8.csv`` and ``configs/lattice_fit_16.csv``; ``--fitted`` simulates sampled
and fitted rates at head bearings 0, 13.3, 22.5 and 45 degrees (shape 200), reports the
largest ratio within 10 % for each, and writes ``fitted_errors.csv`` and ``fitted_limit.png``.

Usage::

    uv run python scripts/lattice_study.py                 # WP 1.4 study
    uv run python scripts/lattice_study.py --fit-tables    # write the fitted-rate tables (about 3 min)
    uv run python scripts/lattice_study.py --fitted        # WP 2.1d study (about 3 min)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from wildfire_sim.config import ScenarioConfig, configs_dir
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import run
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.spread.ellipse import build_fit_table, load_fit_table, save_fit_table
from wildfire_sim.spread.rothermel import lattice_fit_path
from wildfire_sim.types import FloatArray, Stencil, StencilSize, get_stencil

OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "lattice"
DX = 30.0  # cell size, m
R_HEAD = 0.2  # head rate of spread, m/s
HEAD_CELLS = 40  # nominal head distance at the measurement time, cells
T_S = HEAD_CELLS * DX / R_HEAD  # measurement time, s
DT = 60.0  # time step, s (arrival times do not depend on it for constant rates)
N_SEEDS = 5
ALT_SHAPE = 200.0  # alternative arrival_shape compared with the default 20
STENCILS: tuple[StencilSize, ...] = (8, 16)
LBS: tuple[float, ...] = (1.0, 1.5, 3.0, 6.0)
HEADS: tuple[float, ...] = (0.0, 22.5, 45.0)


# ----------------------------------------------------------------------------- the elliptical rule


def eccentricity(lb: float) -> float:
    """Eccentricity of an ellipse with length-to-breadth ratio ``lb`` (>= 1)."""
    return math.sqrt(lb * lb - 1.0) / lb


class EllipticalRule:
    """``R(theta) = R_head (1 - e) / (1 - e cos theta)`` on burnable cells, zero elsewhere.

    The same rule is defined in ``tests/test_physics.py``.

    Args:
        ros_head: Head rate of spread, m/s.
        lb: Length-to-breadth ratio (>= 1).
        head_bearing_deg: Bearing of the head direction, degrees clockwise from north.
    """

    name = "elliptical"

    def __init__(self, ros_head: float, lb: float, head_bearing_deg: float) -> None:
        self.ros_head, self.lb, self.head = ros_head, lb, head_bearing_deg

    def directional_rates(self, stencil: Stencil) -> FloatArray:
        """``(K,)`` rate of spread, m/s, in each stencil direction."""
        e = eccentricity(self.lb)
        theta = np.radians(np.asarray(stencil.bearings_deg, dtype=np.float64) - self.head)
        return self.ros_head * (1.0 - e) / (1.0 - e * np.cos(theta))

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        """``(K, H, W)`` rates, m/s; weather is ignored."""
        h, w = grid.shape
        burnable = grid.burnable_mask().astype(np.float64)
        ros = self.directional_rates(stencil)[:, None, None] * burnable[None]
        zeros = np.zeros((h, w), dtype=np.float64)
        return SpreadOutput(ros=ros, intensity=zeros, flame_length=zeros.copy())


@dataclass(frozen=True)
class Ellipse:
    """Analytic burn at time T, m: distances from the ignition point."""

    head: float
    back: float
    half_width: float

    @classmethod
    def of(cls, lb: float, t_s: float, ros_head: float = R_HEAD) -> Ellipse:
        e = eccentricity(lb)
        head = ros_head * t_s
        back = head * (1.0 - e) / (1.0 + e)
        a = (head + back) / 2.0
        return cls(head=head, back=back, half_width=a * math.sqrt(1.0 - e * e))


def burn_distances(arrival: FloatArray, origin: tuple[int, int], head_deg: float, t_s: float, dx: float) -> Ellipse:
    """Head, back and half-width (m) of the cells burned by ``t_s``, measured from ``origin``.

    Args:
        arrival: ``(H, W)`` arrival times, s (NaN where never burned).
        origin: ``(row, col)`` of the ignition cell.
        head_deg: Head bearing, degrees clockwise from north.
        t_s: Time, s.
        dx: Cell size, m.

    Returns:
        Largest projection of burned cell centres on the head direction (head), on its opposite
        (back), and half the spread of projections on its normal (half-width), all in m.
    """
    rows, cols = np.nonzero(np.nan_to_num(arrival, nan=np.inf) <= t_s)
    east = (cols - origin[1]) * dx
    north = -(rows - origin[0]) * dx
    phi = math.radians(head_deg)
    along = east * math.sin(phi) + north * math.cos(phi)
    across = east * math.cos(phi) - north * math.sin(phi)
    return Ellipse(head=float(along.max()), back=float(-along.min()), half_width=float(np.ptp(across) / 2))


def ellipse_config(n: int, stencil: StencilSize, duration_s: float, shape: float = 20.0) -> ScenarioConfig:
    """Flat uniform ``n x n`` grid at :data:`DX` m, centre ignition, gamma shape ``shape``."""
    return ScenarioConfig.model_validate(
        {
            "name": "lattice_study",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [n, n], "dx": DX},
            "spread": {"rule": "constant", "params": {"ros": R_HEAD}, "arrival_shape": shape},
            "ignition": {"center": True},
            "dt_s": DT,
            "duration_s": duration_s,
            "stencil": stencil,
        }
    )


@dataclass(frozen=True)
class CaseResult:
    """Simulated distances (mean over seeds, m), their standard errors, one arrival map."""

    stencil: StencilSize
    shape: float
    lb: float
    head_deg: float
    sim: Ellipse
    se: Ellipse
    exact: Ellipse
    arrival: FloatArray

    def error(self, which: str) -> float:
        """Relative error (sim - exact) / exact of ``which`` in {head, back, half_width}."""
        return getattr(self.sim, which) / getattr(self.exact, which) - 1.0


def run_case(stencil: StencilSize, lb: float, head_deg: float, seeds: Sequence[int], shape: float = 20.0) -> CaseResult:
    """Simulate one (stencil, LB, head, shape) case for every seed and average the distances."""
    n = 2 * math.ceil(1.35 * HEAD_CELLS) + 1
    cfg = ellipse_config(n, stencil, T_S, shape)
    rule = EllipticalRule(R_HEAD, lb, head_deg)
    origin = (n // 2, n // 2)
    samples = []
    arrival = np.empty(0)
    for seed in seeds:
        arrival = run(cfg, seed=seed, rule=rule, record_events=False).arrival_time
        rows, cols = np.nonzero(~np.isnan(arrival))
        if rows.min() == 0 or cols.min() == 0 or rows.max() == n - 1 or cols.max() == n - 1:
            raise RuntimeError("burn reached the grid edge")
        samples.append(burn_distances(arrival, origin, head_deg, T_S, DX))
    arr = np.array([[s.head, s.back, s.half_width] for s in samples])
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / math.sqrt(len(samples)) if len(samples) > 1 else np.zeros(3)
    return CaseResult(
        stencil=stencil,
        shape=shape,
        lb=lb,
        head_deg=head_deg,
        sim=Ellipse(*map(float, mean)),
        se=Ellipse(*map(float, se)),
        exact=Ellipse.of(lb, T_S),
        arrival=arrival,
    )


# ----------------------------------------------------------------------------- deterministic limit


def primitive_bearings(reach: int) -> FloatArray:
    """Bearings (deg) of all primitive offsets with ``max(|dr|, |dc|) <= reach``.

    ``reach`` 1, 2, 3, 4 give 8, 16, 32 and 48 directions. Only 8 and 16 exist in the simulator.
    """
    offs = [
        (dr, dc)
        for dr in range(-reach, reach + 1)
        for dc in range(-reach, reach + 1)
        if (dr, dc) != (0, 0) and math.gcd(abs(dr), abs(dc)) == 1
    ]
    off = np.array(offs, dtype=np.float64)
    return (np.degrees(np.arctan2(off[:, 1], -off[:, 0])) + 360.0) % 360.0


def hull_distances(bearings_deg: FloatArray, lb: float, head_deg: float) -> Ellipse:
    """Deterministic-limit head, back and half-width per unit time and unit ``R_head``.

    The support function of ``conv{R(theta_k) u_k}`` in the head direction, its opposite and
    its normal (half the width).
    """
    e = eccentricity(lb)
    theta = np.radians(np.asarray(bearings_deg, dtype=np.float64) - head_deg)
    r = (1.0 - e) / (1.0 - e * np.cos(theta))
    along, across = r * np.cos(theta), r * np.sin(theta)
    return Ellipse(head=float(along.max()), back=float(-along.min()), half_width=float(np.ptp(across) / 2))


def hull_errors(bearings_deg: FloatArray, lb: float, heads: FloatArray) -> FloatArray:
    """``(3, len(heads))`` relative errors of head, flank, back in the deterministic limit."""
    exact = Ellipse.of(lb, 1.0, 1.0)
    out = np.empty((3, heads.size))
    for i, h in enumerate(heads):
        hull = hull_distances(bearings_deg, lb, float(h))
        out[:, i] = [
            hull.head / exact.head - 1,
            hull.half_width / exact.half_width - 1,
            hull.back / exact.back - 1,
        ]
    return out


# ----------------------------------------------------------------------------- figures


def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def plot_errors(results: Sequence[CaseResult], path: Path) -> None:
    """Relative error of head / flank / back distance vs LB, one panel per (stencil, measure)."""
    fig = _figure(11, 6.5)
    axes = fig.subplots(len(STENCILS), 3, sharex=True, sharey="col", squeeze=False)
    measures = (("head", "head"), ("half_width", "flank half-width"), ("back", "back"))
    markers = {0.0: "o", 22.5: "s", 45.0: "^"}
    for i, stencil in enumerate(STENCILS):
        for j, (key, label) in enumerate(measures):
            ax = axes[i, j]
            ax.axhspan(-0.1, 0.1, color="0.9", zorder=0, label="target +-10 %")
            for head in HEADS:
                cases = [r for r in results if r.stencil == stencil and r.head_deg == head]
                ax.plot(
                    [c.lb for c in cases],
                    [100 * c.error(key) for c in cases],
                    marker=markers[head],
                    label=f"head {head:g} deg",
                )
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{stencil} neighbours: {label} (shape {results[0].shape:g})")
            if j == 0:
                ax.set_ylabel("relative error, %")
            if i == len(STENCILS) - 1:
                ax.set_xlabel("length-to-breadth ratio LB")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=110)


def plot_outlines(results: Sequence[CaseResult], stencil: StencilSize, path: Path) -> None:
    """Burned cells at T (last seed) over the analytic ellipse, rows LB, columns head direction."""
    fig = _figure(10, 12.5)
    axes = fig.subplots(len(LBS), len(HEADS), squeeze=False)
    for i, lb in enumerate(LBS):
        for j, head in enumerate(HEADS):
            ax = axes[i, j]
            case = next(r for r in results if r.stencil == stencil and r.lb == lb and r.head_deg == head)
            arr = case.arrival
            h, w = arr.shape
            burned = (np.nan_to_num(arr, nan=np.inf) <= T_S).astype(float)
            half = w // 2
            ext_km = (np.array([-half - 0.5, half + 0.5]) * DX) / 1000.0
            ax.imshow(
                burned,
                cmap="Greys",
                vmin=0,
                vmax=1.6,
                extent=(ext_km[0], ext_km[1], ext_km[0], ext_km[1]),
                interpolation="nearest",
            )
            exact = case.exact
            e = eccentricity(lb)
            a = (exact.head + exact.back) / 2
            t = np.linspace(0, 2 * np.pi, 400)
            along = a * np.cos(t) + (exact.head - a)  # centre sits a e ahead of the focus
            across = a * math.sqrt(1 - e * e) * np.sin(t)
            phi = math.radians(head)
            east = along * math.sin(phi) + across * math.cos(phi)
            north = along * math.cos(phi) - across * math.sin(phi)
            ax.plot(east / 1000, north / 1000, color="#D55E00", lw=1.4)
            ax.plot(0, 0, marker="+", color="#0072B2")
            ax.set_xlim(-1.45, 1.45)
            ax.set_ylim(-1.45, 1.45)
            ax.set_aspect("equal")
            ax.set_title(
                f"LB {lb:g}, head {head:g} deg\nhead {100 * case.error('head'):+.0f} %, "
                f"flank {100 * case.error('half_width'):+.0f} %, back {100 * case.error('back'):+.0f} %",
                fontsize=9,
            )
            ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{stencil}-neighbour stencil, arrival_shape {results[0].shape:g}: "
        f"burned cells at T = {T_S / 3600:.2f} h (grey), analytic ellipse"
    )
    fig.supxlabel("east, km")
    fig.supylabel("north, km")
    fig.savefig(path, dpi=100)


HULL_LBS = np.linspace(1.0, 8.0, 141)
HULL_STENCILS = ("8", "16", "32 (hypothetical)", "48 (hypothetical)")


def plot_hull_limit(path: Path) -> FloatArray:
    """Worst-case (over head directions 0-45 deg) deterministic-limit error vs LB.

    For the 8 and 16 stencils of the simulator and hypothetical 32 and 48 stencils.

    Returns:
        ``(4 stencils, 3 measures, len(HULL_LBS))`` worst-case relative errors (signed, largest
        magnitude), measures head, flank half-width, back.
    """
    lbs = HULL_LBS
    heads = np.linspace(0.0, 45.0, 181)
    assert np.allclose(np.sort(primitive_bearings(1)), np.sort(get_stencil(8).bearings_deg))
    assert np.allclose(np.sort(primitive_bearings(2)), np.sort(get_stencil(16).bearings_deg))
    stencils = dict(zip(HULL_STENCILS, (primitive_bearings(r) for r in (1, 2, 3, 4)), strict=True))
    worst = np.zeros((len(stencils), 3, lbs.size))
    for s, bearings in enumerate(stencils.values()):
        for i, lb in enumerate(lbs):
            err = hull_errors(bearings, float(lb), heads)
            idx = np.abs(err).argmax(axis=1)
            worst[s, :, i] = err[np.arange(3), idx]
    fig = _figure(11, 3.6)
    axes = fig.subplots(1, 3, sharey=True)
    for j, label in enumerate(("head", "flank half-width", "back")):
        ax = axes[j]
        ax.axhspan(-10, 10, color="0.9", zorder=0)
        for s, name in enumerate(stencils):
            ax.plot(lbs, 100 * worst[s, j], label=f"{name} neighbours")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{label} (worst over head 0-45 deg)")
        ax.set_xlabel("LB")
    axes[0].set_ylabel("deterministic-limit error, %")
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=110)
    return worst


def plot_isotropic_profile(path: Path, seeds: Sequence[int]) -> dict[tuple[int, float], tuple[FloatArray, FloatArray]]:
    """Front speed / R vs direction (folded into 0-45 deg) with the shipped correction, no wind.

    Speed of a cell at radius 30-45 cells is ``radius / arrival_time / R``; cells are binned by
    the angle from the nearest grid axis.

    Returns:
        (stencil, arrival_shape) -> (bin centres in degrees, mean speed / R per bin).
    """
    n = 2 * 62 + 1
    c = n // 2
    rows, cols = np.mgrid[0:n, 0:n]
    dr, dc = rows - c, cols - c
    radius = np.hypot(dr, dc)
    ang = np.degrees(np.arctan2(np.abs(dr), np.abs(dc)))
    ang = np.minimum(ang, 90.0 - ang)  # fold into [0, 45]
    ring = (radius >= 30) & (radius <= 45)
    edges = np.linspace(0, 45, 10)
    centres = (edges[:-1] + edges[1:]) / 2
    out: dict[tuple[int, float], tuple[FloatArray, FloatArray]] = {}
    fig = _figure(6.5, 4.2)
    ax = fig.add_subplot()
    bins = list(zip(edges[:-1], edges[1:], strict=True))
    for stencil in STENCILS:
        for shape in (20.0, ALT_SHAPE):
            cfg = ellipse_config(n, stencil, 60 * DX / R_HEAD, shape)
            speeds = []
            for seed in seeds:
                arr = run(cfg, seed=seed, record_events=False).arrival_time
                with np.errstate(divide="ignore", invalid="ignore"):
                    speed = radius * DX / arr / R_HEAD  # NaN at the ignition cell (0 / 0)
                speeds.append([np.nanmean(speed[ring & (ang >= lo) & (ang < hi)]) for lo, hi in bins])
            mean = np.asarray(speeds).mean(axis=0)
            out[(stencil, shape)] = (centres, mean)
            style = "-" if shape == 20.0 else "--"
            ax.plot(centres, mean, style, marker="o", label=f"{stencil} neighbours, shape {shape:g}")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("angle from nearest grid axis, deg")
    ax.set_ylabel("front speed / R (corrected)")
    ax.set_title("No wind: front speed by direction, radius 30-45 cells")
    ax.legend()
    fig.savefig(path, dpi=110)
    return out


@contextmanager
def neutral_correction() -> Generator[None]:
    """Run with an empty front-speed correction table (temporary copy of ``configs/``)."""
    old = os.environ.get("WILDFIRE_SIM_CONFIGS")
    with tempfile.TemporaryDirectory(prefix="wildfire_configs_") as tmp:
        for path in configs_dir().glob("*.yaml"):
            shutil.copy(path, Path(tmp) / path.name)
        (Path(tmp) / "kernel_correction.yaml").write_text("entries: []\n", encoding="utf-8")
        os.environ["WILDFIRE_SIM_CONFIGS"] = tmp
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("WILDFIRE_SIM_CONFIGS", None)
            else:
                os.environ["WILDFIRE_SIM_CONFIGS"] = old


def noise_speedup(
    seeds: Sequence[int], shapes: Sequence[float] = (20.0, 50.0, 200.0)
) -> dict[tuple[int, float, float], float]:
    """Uncorrected head speed / R for a head along the grid axis, by stencil, shape and LB.

    With no correction, the head of an aligned ellipse moves along one stencil direction, so
    any departure from 1 is the speed-up the kernel noise gives through competing paths. The
    front-speed correction assumes the isotropic (LB = 1) value for every LB.

    Returns:
        ``(stencil, shape, lb)`` -> mean head distance / (R_head T) over ``seeds``.
    """
    n = 2 * math.ceil(1.6 * HEAD_CELLS) + 1
    out: dict[tuple[int, float, float], float] = {}
    with neutral_correction():
        for stencil in STENCILS:
            for shape in shapes:
                cfg = ellipse_config(n, stencil, T_S, shape)
                for lb in LBS:
                    rule = EllipticalRule(R_HEAD, lb, 0.0)
                    heads = []
                    for seed in seeds:
                        arr = run(cfg, seed=seed, rule=rule, record_events=False).arrival_time
                        heads.append(burn_distances(arr, (n // 2, n // 2), 0.0, T_S, DX).head)
                    out[(stencil, shape, lb)] = float(np.mean(heads)) / (R_HEAD * T_S)
    return out


# ----------------------------------------------------------------------------- main


def write_csv(results: Sequence[CaseResult], path: Path) -> None:
    """Write simulated and analytic distances (m) and relative errors of ``results`` to ``path``."""
    keys = ("head", "half_width", "back")
    header = ["stencil", "arrival_shape", "lb", "head_deg"]
    for key in keys:
        header += [f"{key}_m", f"{key}_se_m", f"{key}_exact_m", f"{key}_err"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in results:
            row: list[object] = [r.stencil, r.shape, r.lb, r.head_deg]
            for key in keys:
                row += [
                    f"{getattr(r.sim, key):.1f}",
                    f"{getattr(r.se, key):.1f}",
                    f"{getattr(r.exact, key):.1f}",
                    f"{r.error(key):+.3f}",
                ]
            writer.writerow(row)


def print_results(results: Sequence[CaseResult]) -> None:
    """Print the relative errors of ``results`` as a Markdown table."""
    print(f"\nsimulated vs analytic ellipse, T = {T_S:g} s, R_head = {R_HEAD} m/s, {N_SEEDS} seeds")
    print("| stencil | shape | LB | head deg | head err | flank err | back err | back exact (cells) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(
            f"| {r.stencil} | {r.shape:g} | {r.lb:g} | {r.head_deg:g} | {100 * r.error('head'):+.1f} % | "
            f"{100 * r.error('half_width'):+.1f} % | {100 * r.error('back'):+.1f} % | {r.exact.back / DX:.1f} |"
        )


# ----------------------------------------------------------------------------- WP 2.1d: lattice-fitted rates

FIT_LBS = np.round(np.arange(1.0, 2.6001, 0.05), 4)  # table ratios
FIT_HEADS = np.arange(0.0, 45.001, 1.0)  # table canonical head bearings, deg
FITTED_HEADS = (0.0, 13.3, 22.5, 45.0)  # plan: on an axis, midway between 16-neighbour directions, 22.5, diagonal
FITTED_LBS: dict[StencilSize, tuple[float, ...]] = {
    8: (1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6),
    16: (1.0, 1.25, 1.5, 1.75, 2.0, 2.05, 2.1, 2.15, 2.2, 2.3),
}
FITTED_SHAPE = 200.0  # the shipped arrival_shape
TOLERANCE = 0.10  # head and flank within 10 % of the analytic ellipse (plan WP 2.1d)
MARGIN = 0.25
"""Fitted rates ship only if they raise the cap of the default (16-neighbour) stencil by at least
this much over sampled rates (plan WP 2.1d: "if the fitted rates gain less than the study's
stated margin over sampled rates ... only the cap ships")."""


class FittedEllipticalRule(EllipticalRule):
    """:class:`EllipticalRule` times the lattice-fitted multipliers of the shipped table."""

    name = "elliptical_fitted"

    def directional_rates(self, stencil: Stencil) -> FloatArray:
        table = load_fit_table(lattice_fit_path(stencil.size), stencil.size)
        return super().directional_rates(stencil) * table.multipliers(np.array(self.lb), np.array(self.head))


def write_fit_tables() -> None:
    """Fit the rates on the (ratio, bearing) grid for both stencils and write ``configs/lattice_fit_*.csv``."""
    for size in STENCILS:
        bearings = get_stencil(size).bearings_deg
        table = build_fit_table(bearings, size, FIT_LBS, FIT_HEADS)
        source = (
            "scripts/lattice_study.py --fit-tables (wildfire_sim.spread.ellipse.build_fit_table): "
            f"minimax relative radial error of the lattice polygon, ratios {FIT_LBS[0]:g}-{FIT_LBS[-1]:g} "
            f"step 0.05, canonical head bearings 0-45 deg step 1"
        )
        path = lattice_fit_path(size)
        save_fit_table(table, path, source)
        print(f"wrote {path}")


def run_fitted_case(stencil: StencilSize, lb: float, head_deg: float, fitted: bool, seeds: Sequence[int]) -> CaseResult:
    """Like :func:`run_case` at the shipped shape, with sampled or fitted rates."""
    n = 2 * math.ceil(1.35 * HEAD_CELLS) + 1
    cfg = ellipse_config(n, stencil, T_S, FITTED_SHAPE)
    rule = FittedEllipticalRule(R_HEAD, lb, head_deg) if fitted else EllipticalRule(R_HEAD, lb, head_deg)
    origin = (n // 2, n // 2)
    samples = []
    arrival = np.empty(0)
    for seed in seeds:
        arrival = run(cfg, seed=seed, rule=rule, record_events=False).arrival_time
        rows, cols = np.nonzero(~np.isnan(arrival))
        if rows.min() == 0 or cols.min() == 0 or rows.max() == n - 1 or cols.max() == n - 1:
            raise RuntimeError("burn reached the grid edge")
        samples.append(burn_distances(arrival, origin, head_deg, T_S, DX))
    arr = np.array([[s.head, s.back, s.half_width] for s in samples])
    se = arr.std(axis=0, ddof=1) / math.sqrt(len(samples)) if len(samples) > 1 else np.zeros(3)
    return CaseResult(
        stencil,
        FITTED_SHAPE,
        lb,
        head_deg,
        Ellipse(*map(float, arr.mean(axis=0))),
        Ellipse(*map(float, se)),
        Ellipse.of(lb, T_S),
        arrival,
    )


def largest_within(results: Sequence[CaseResult], stencil: StencilSize) -> float:
    """Largest tested ratio up to which head and flank stay within :data:`TOLERANCE` at every bearing."""
    limit = 1.0
    for lb in FITTED_LBS[stencil]:
        cases = [r for r in results if r.stencil == stencil and r.lb == lb]
        if all(abs(r.error("head")) <= TOLERANCE and abs(r.error("half_width")) <= TOLERANCE for r in cases):
            limit = lb
        else:
            break
    return limit


def fitted_study() -> dict[tuple[StencilSize, str], float]:
    """Sampled vs fitted rates in the simulator: errors, the caps, and whether fitting meets the margin.

    Writes ``docs/img/lattice/fitted_errors.csv`` and ``fitted_limit.png`` and returns the cap per
    (stencil, rate kind).
    """
    seeds = range(N_SEEDS)
    results: dict[str, list[CaseResult]] = {"sampled": [], "fitted": []}
    for kind in results:
        for s in STENCILS:
            for lb in FITTED_LBS[s]:
                for h in FITTED_HEADS:
                    results[kind].append(run_fitted_case(s, lb, h, kind == "fitted", seeds))
    caps = {(s, kind): largest_within(rs, s) for kind, rs in results.items() for s in STENCILS}
    with (OUT / "fitted_errors.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rates", "stencil", "lb", "head_deg", "head_err", "flank_err", "back_err"])
        for kind, rs in results.items():
            for r in rs:
                writer.writerow(
                    [
                        kind,
                        r.stencil,
                        r.lb,
                        r.head_deg,
                        f"{r.error('head'):+.3f}",
                        f"{r.error('half_width'):+.3f}",
                        f"{r.error('back'):+.3f}",
                    ]
                )
    fig = _figure(9.0, 3.6)
    for i, s in enumerate(STENCILS):
        ax = fig.add_subplot(1, 2, i + 1)
        for kind, style in (("sampled", "--"), ("fitted", "-")):
            worst = [
                max(
                    max(abs(r.error("head")), abs(r.error("half_width")))
                    for r in results[kind]
                    if r.stencil == s and r.lb == lb
                )
                for lb in FITTED_LBS[s]
            ]
            ax.plot(FITTED_LBS[s], 100 * np.array(worst), style, marker="o", ms=3, color="#1f4e79", label=kind)
        ax.axhline(100 * TOLERANCE, color="#a33", lw=0.8)
        ax.set(title=f"{s} neighbours", xlabel="length-to-breadth ratio", ylabel="worst head or flank error, %")
        ax.legend(frameon=False)
    FigureCanvasAgg(fig).print_png(str(OUT / "fitted_limit.png"))
    print(f"\nsampled vs fitted rates, shape {FITTED_SHAPE:g}, {N_SEEDS} seeds, head bearings {FITTED_HEADS}")
    print("| rates | stencil | LB | head deg | head err | flank err | back err |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for kind, rs in results.items():
        for r in rs:
            print(
                f"| {kind} | {r.stencil} | {r.lb:g} | {r.head_deg:g} | {100 * r.error('head'):+.1f} % | "
                f"{100 * r.error('half_width'):+.1f} % | {100 * r.error('back'):+.1f} % |"
            )
    for s in STENCILS:
        print(f"stencil {s}: cap sampled {caps[(s, 'sampled')]:g}, fitted {caps[(s, 'fitted')]:g}")
    gain = caps[(16, "fitted")] - caps[(16, "sampled")]
    verdict = "ship fitted rates" if gain >= MARGIN else "ship the cap only"
    print(f"gain on 16 neighbours: {gain:+.2f} (margin {MARGIN}): {verdict}")
    return caps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit-tables", action="store_true", help="fit the rates and write configs/lattice_fit_*.csv")
    parser.add_argument("--fitted", action="store_true", help="WP 2.1d study: sampled vs fitted rates, the caps")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fit_tables:
        write_fit_tables()
    if args.fitted:
        fitted_study()
    if args.fit_tables or args.fitted:
        return
    seeds = range(N_SEEDS)
    for shape, suffix in ((20.0, ""), (ALT_SHAPE, f"_shape{ALT_SHAPE:g}")):
        results = [run_case(s, lb, h, seeds, shape) for s in STENCILS for lb in LBS for h in HEADS]
        write_csv(results, OUT / f"ellipse_errors{suffix}.csv")
        print_results(results)
        plot_errors(results, OUT / f"ellipse_errors{suffix}.png")
        for s in STENCILS:
            plot_outlines(results, s, OUT / f"outline_stencil{s}{suffix}.png")

    worst = plot_hull_limit(OUT / "hull_limit.png")
    print("\ndeterministic-limit worst-case error over head 0-45 deg (head, flank, back):")
    for lb in (1.5, 3.0, 6.0):
        i = int(np.argmin(np.abs(HULL_LBS - lb)))
        for s, name in enumerate(HULL_STENCILS):
            vals = ", ".join(f"{100 * v:+.0f} %" for v in worst[s, :, i])
            print(f"  LB {lb:g}, stencil {name}: {vals}")
    print("largest LB with head and flank deterministic-limit error within 10 % for every head direction:")
    for s, name in enumerate(HULL_STENCILS):
        ok = np.all(np.abs(worst[s, :2]) <= 0.10, axis=0)
        bad = np.nonzero(~ok)[0]
        limit = HULL_LBS[bad[0] - 1] if bad.size else HULL_LBS[-1]
        print(f"  stencil {name}: LB <= {limit:.2f}")
    speedup = noise_speedup(range(N_SEEDS))
    print("\nuncorrected head speed / R, head along the grid axis (noise speed-up) by LB:")
    for (stencil, shape, lb), v in speedup.items():
        print(f"  stencil {stencil}, shape {shape:g}, LB {lb:g}: {v:.3f}")
    profile = plot_isotropic_profile(OUT / "isotropic_profile.png", range(N_SEEDS))
    print("\nno-wind corrected front speed / R by angle from axis (deg):")
    for s, (centres, mean) in profile.items():
        print(
            f"  stencil {s[0]}, shape {s[1]:g}: "
            + ", ".join(f"{c:.0f}: {m:.3f}" for c, m in zip(centres, mean, strict=True))
        )
    print(f"\nwrote figures and CSV to {OUT}")


if __name__ == "__main__":
    main()
