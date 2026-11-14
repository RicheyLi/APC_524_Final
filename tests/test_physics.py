"""Physics tests of the simulator (WP 1.4, docs/plan.md sections 2 and 7).

Every test uses fixed seeds, builds its scenario in code (``ScenarioConfig.model_validate``) on
a grid sized so the fire never reaches the edge (asserted), and states its threshold and the
evidence for it in its docstring. Thresholds that differ from the plan's targets are flagged
there and in ``docs/lattice_study.md``; the measurements behind them come from
``scripts/kernel_front_speed.py`` and ``scripts/lattice_study.py``.

Shared definitions (lengths in m, times in s, ratios dimensionless):

* Burned set at time ``T``: cells with ``arrival_time <= T``.
* Axis extent: ``(ptp(col) + ptp(row)) / 4 * dx`` over the burned set; diagonal extent:
  ``(ptp(row + col) + ptp(row - col)) / 4 / sqrt(2) * dx``. Both are mean distances from the
  centre to the farthest burned cell centre in the four axis / diagonal directions.
"""

from __future__ import annotations

import functools
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import brentq

from wildfire_sim.config import ScenarioConfig, load_scenario
from wildfire_sim.data.synthetic import flat_uniform
from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import load_kernel_corrections, run
from wildfire_sim.spread.base import SpreadOutput
from wildfire_sim.spread.rothermel import RothermelRule
from wildfire_sim.types import FloatArray, Stencil, get_stencil

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
SCENARIOS = CONFIGS / "scenarios"
DX = 30.0  # cell size, m
ROS = 0.2  # rate of spread, m/s


# ----------------------------------------------------------------------------- helpers


def scenario(
    *,
    shape: tuple[int, int],
    rule: str = "constant",
    params: dict[str, Any] | None = None,
    arrival_shape: float = 20.0,
    cells: list[tuple[int, int]] | None = None,
    dt: float = 60.0,
    duration: float,
    stencil: int = 8,
    weather: list[dict[str, float]] | None = None,
) -> ScenarioConfig:
    """Scenario on a flat uniform 30 m grid; centre ignition unless ``cells`` is given."""
    data: dict[str, Any] = {
        "name": "physics",
        "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": list(shape), "dx": DX},
        "spread": {"rule": rule, "params": params or {"ros": ROS}, "arrival_shape": arrival_shape},
        "ignition": {"cells": [list(c) for c in cells or []], "center": not cells},
        "dt_s": dt,
        "duration_s": duration,
        "stencil": stencil,
    }
    if weather is not None:
        data["weather"] = {"series": weather}
    return ScenarioConfig.model_validate(data)


def burned_by(arrival: FloatArray, t_s: float) -> np.ndarray:
    """``(H, W)`` bool: cells whose arrival time (s) is at most ``t_s``."""
    return np.nan_to_num(arrival, nan=np.inf) <= t_s


def assert_inside(mask: np.ndarray) -> None:
    """The burned ``(H, W)`` mask must not touch the grid edge."""
    assert not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()), "fire reached the edge"


def extents(mask: np.ndarray) -> tuple[float, float]:
    """Axis and diagonal extent (m) of a burned ``(H, W)`` mask (see the module docstring)."""
    rows, cols = np.nonzero(mask)
    axis = (np.ptp(cols) + np.ptp(rows)) / 4 * DX
    diag = (np.ptp(rows + cols) + np.ptp(rows - cols)) / 4 / math.sqrt(2) * DX
    return float(axis), float(diag)


@pytest.fixture
def neutral_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy of ``configs/`` with an empty front-speed correction table, made active."""
    for path in CONFIGS.glob("*.yaml"):
        shutil.copy(path, tmp_path / path.name)
    (tmp_path / "kernel_correction.yaml").write_text("entries: []\n", encoding="utf-8")
    monkeypatch.setenv("WILDFIRE_SIM_CONFIGS", str(tmp_path))
    return tmp_path


def _output(ros: FloatArray) -> SpreadOutput:
    h, w = ros.shape[1:]
    return SpreadOutput(ros=ros, intensity=np.zeros((h, w)), flame_length=np.zeros((h, w)))


class EllipticalRule:
    """``R(theta) = R_head (1 - e) / (1 - e cos theta)``, ``e = sqrt(LB^2 - 1) / LB``.

    ``theta`` is the angle between a stencil direction and the head bearing. Burnable cells only;
    weather is ignored. Same rule as ``scripts/lattice_study.py``.

    Args:
        ros_head: Head rate of spread, m/s.
        lb: Length-to-breadth ratio, >= 1.
        head_bearing_deg: Head direction, degrees clockwise from north.
    """

    name = "elliptical"

    def __init__(self, ros_head: float, lb: float, head_bearing_deg: float) -> None:
        self.ros_head, self.lb, self.head = ros_head, lb, head_bearing_deg

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        """``(K, H, W)`` rates, m/s."""
        e = math.sqrt(self.lb**2 - 1.0) / self.lb
        theta = np.radians(np.asarray(stencil.bearings_deg, dtype=np.float64) - self.head)
        rates = self.ros_head * (1.0 - e) / (1.0 - e * np.cos(theta))
        return _output(rates[:, None, None] * grid.burnable_mask()[None].astype(np.float64))


class StripRule:
    """``ros`` everywhere except columns ``[c0, c1)``, where it is ``ros / factor`` (m/s)."""

    name = "strip"

    def __init__(self, ros: float, c0: int, c1: int, factor: float) -> None:
        self.ros, self.c0, self.c1, self.factor = ros, c0, c1, factor

    def rate_of_spread(self, grid: Grid, weather: Weather, stencil: Stencil) -> SpreadOutput:
        """``(K, H, W)`` rates, m/s, equal in all directions."""
        cell = np.where(grid.burnable_mask(), self.ros, 0.0)
        cell[:, self.c0 : self.c1] /= self.factor
        return _output(np.broadcast_to(cell, (stencil.size, *cell.shape)).copy())


# ----------------------------------------------------------------------------- (a) circular burn


@pytest.mark.parametrize("stencil", [8, 16])
def test_no_wind_burn_is_round(stencil: int) -> None:
    """(a) No wind, constant rule, 20 runs of ``flat_no_wind.yaml``: the burn is round.

    Thresholds (plan targets, kept): mean diagonal extent / axis extent in [0.95, 1.05], and
    for every run the boundary-radius standard deviation / mean < 0.1 (boundary cells: burned
    cells with an unburned 4-neighbour; radius from the ignition cell).

    Evidence (``scripts/kernel_front_speed.py``, radius 40 cells, 20 seeds): diagonal / axis
    = 1.033 on 8 neighbours and 1.042 on 16. The 16-neighbour stencil therefore passes with
    a margin of under 0.01: its diagonals run about 4 % faster than its axes, the residual
    anisotropy of the lattice (docs/lattice_study.md). The radius scatter is about 0.03.
    """
    cfg = load_scenario(SCENARIOS / "flat_no_wind.yaml").model_copy(update={"stencil": stencil})
    ratios, scatters = [], []
    for seed in range(20):
        result = run(cfg, seed=seed, record_events=False)
        mask = result.burned_mask
        assert_inside(mask)
        axis, diag = extents(mask)
        ratios.append(diag / axis)
        padded = np.pad(mask, 1)
        interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        rows, cols = np.nonzero(mask & ~interior)
        c = mask.shape[0] // 2
        radius = np.hypot(rows - c, cols - c)
        scatters.append(radius.std() / radius.mean())
    assert 0.95 <= float(np.mean(ratios)) <= 1.05, np.mean(ratios)
    assert max(scatters) < 0.1, max(scatters)


# ----------------------------------------------------------------------------- (b) ellipse

ELLIPSE_HEAD_CELLS = 40  # nominal head distance at the measurement time, cells
ELLIPSE_T_S = ELLIPSE_HEAD_CELLS * DX / ROS  # 6000 s
ELLIPSE_HEADS = (0.0, 22.5, 45.0)  # head bearings, deg: on an axis, between axis and diagonal, diagonal

# Lower bounds on the relative error of (head, flank half-width), from the lattice study
# (docs/lattice_study.md, 5 seeds, arrival_shape 20) with a margin of 5-10 points below the
# worst measured value over the three head directions (worst measured in the comment).
ELLIPSE_LOWER = {
    (8, 1.5): (-0.25, -0.15),  # -17.7 % (head 22.5 deg), -7.5 %
    (8, 3.0): (-0.65, -0.45),  # -55.8 %, -36.1 %
    (8, 6.0): (-0.92, -0.80),  # -85.0 %, -70.2 %
    (16, 1.5): (-0.22, -0.18),  # -15.5 % (head 0 deg), -11.1 %
    (16, 3.0): (-0.30, -0.30),  # -21.0 %, -22.8 %
    (16, 6.0): (-0.45, -0.68),  # -36.0 %, -58.3 %
}


@functools.cache
def ellipse_errors(
    stencil: int, lb: float, head_deg: float, arrival_shape: float = 20.0
) -> tuple[float, float, float, float]:
    """Simulated vs analytic ellipse for :class:`EllipticalRule` with ``R_head = 0.2`` m/s.

    109 x 109 grid, centre ignition, ``arrival_shape`` 20 unless given, shipped correction, T = 6000 s,
    seeds 0-4 (same set-up as ``scripts/lattice_study.py``). Distances are the largest
    projections of burned cell centres on the head direction, its opposite and its normal
    (half the spread), averaged over seeds.

    Returns:
        Relative error of the head distance, relative error of the flank half-width, simulated
        back distance (m) and analytic back distance (m).
    """
    n = 2 * math.ceil(1.35 * ELLIPSE_HEAD_CELLS) + 1
    c = n // 2
    cfg = scenario(shape=(n, n), duration=ELLIPSE_T_S, stencil=stencil, arrival_shape=arrival_shape)
    rule = EllipticalRule(ROS, lb, head_deg)
    phi = math.radians(head_deg)
    dists = []
    for seed in range(5):
        mask = run(cfg, seed=seed, rule=rule, record_events=False).burned_mask
        assert_inside(mask)
        rows, cols = np.nonzero(mask)
        east, north = (cols - c) * DX, -(rows - c) * DX
        along = east * math.sin(phi) + north * math.cos(phi)
        across = east * math.cos(phi) - north * math.sin(phi)
        dists.append((along.max(), -along.min(), np.ptp(across) / 2))
    head, back, half_width = np.mean(dists, axis=0)
    e = math.sqrt(lb**2 - 1.0) / lb
    head_exact = ROS * ELLIPSE_T_S
    back_exact = head_exact * (1 - e) / (1 + e)
    b_exact = (head_exact + back_exact) / 2 * math.sqrt(1 - e**2)
    return float(head / head_exact - 1), float(half_width / b_exact - 1), float(back), back_exact


@pytest.mark.parametrize("head_deg", ELLIPSE_HEADS)
@pytest.mark.parametrize("lb", [1.5, 3.0, 6.0])
@pytest.mark.parametrize("stencil", [8, 16])
def test_steady_wind_ellipse(stencil: int, lb: float, head_deg: float) -> None:
    """(b) Elliptical rule: head, flank and back distances against the analytic ellipse.

    **The plan's target (each within 10 % for LB 1.5, 3 and 6) is not met by either stencil**
    (docs/lattice_study.md; :func:`test_ellipse_plan_target` records that as a strict xfail).
    Two causes, both measured: (1) geometry - a stencil reaches directions between its link
    directions only by zig-zag paths, so the burn is a polygon inside the ellipse (worst
    deterministic-limit head / flank error over head directions: 8 neighbours -24 % / -14 % at
    LB 1.5, 16 neighbours -10 % / -4 %); (2) the front-speed correction is measured on a round
    fire, where kernel noise speeds the front up through competing paths, but a narrow
    ellipse has few competing paths, so its head is slowed by nearly the full correction (-11 %
    on 8, -22 % on 16 at LB >= 3).

    Thresholds set from that evidence, as regression guards: head and flank error between the
    lower bounds in :data:`ELLIPSE_LOWER` and +5 % (no case overshoots); back distance within
    10 % at LB 1.5 (plan target, met: -2 % to -8 %), and at LB 3 and 6, where the analytic back
    distance is under 1.2 cells and cannot be resolved, within one cell (30 m) of it.
    """
    head_err, flank_err, back, back_exact = ellipse_errors(stencil, lb, head_deg)
    head_lo, flank_lo = ELLIPSE_LOWER[(stencil, lb)]
    assert head_lo <= head_err <= 0.05, head_err
    assert flank_lo <= flank_err <= 0.05, flank_err
    if lb == 1.5:
        assert back == pytest.approx(back_exact, rel=0.10)
    else:
        assert abs(back - back_exact) <= DX


@pytest.mark.xfail(strict=True, reason="lattice limit: no stencil meets 10 % on the ellipse (docs/lattice_study.md)")
@pytest.mark.parametrize("lb", [1.5, 3.0, 6.0])
@pytest.mark.parametrize("stencil", [8, 16])
def test_ellipse_plan_target(stencil: int, lb: float) -> None:
    """(b) The plan's target: head and flank within 10 % for every head direction. Not met.

    Strict xfail: if a change to the lattice (a larger stencil, a different correction) makes
    this pass, the xfail turns into a failure and this record must be updated. Worst cases
    measured (head, flank): 8 neighbours LB 1.5 -17.7 % / -7.5 %, LB 3 -55.8 % / -36.1 %,
    LB 6 -85 % / -70 %; 16 neighbours LB 1.5 -15.5 % / -11.1 %, LB 3 -21 % / -23 %,
    LB 6 -36 % / -58 %.
    """
    for head_deg in ELLIPSE_HEADS:
        head_err, flank_err, _, _ = ellipse_errors(stencil, lb, head_deg)
        assert abs(head_err) <= 0.10 and abs(flank_err) <= 0.10, (head_deg, head_err, flank_err)


# ----------------------------------------------------------------------------- (c) dt halving


@pytest.mark.slow
def test_halving_dt_keeps_the_arrival_time_distribution() -> None:
    """(c) dt 15 s vs 7.5 s, 50 runs each with disjoint seeds: same arrival-time distribution.

    Thresholds: the mean arrival time along an east transect (1-15 cells from the ignition,
    averaged over the transect and the runs) within 2 % (plan target), and a two-sample KS test
    ``p > 0.01`` (plan target) on the arrival times at the far end of the transect and on the
    per-run transect means. Per transect cell the means must agree within 3 standard errors
    of their difference.

    Changed from the plan's reading "within 2 % at every cell": the two dt use different seeds,
    so the per-cell difference is sampling noise of size ``sqrt(2 / 50) * CV``. One cell from
    the ignition the arrival time is close to a single gamma(20) crossing (CV about 0.2), so the
    standard error of the difference is a few percent: measured 2.1 % at that cell (176.0 s vs
    172.3 s, 0.9 standard errors), 0.1-1.9 % farther out, 0.13 % over the whole transect; KS
    p = 0.72 and 0.27. A 2 % bound per cell would test the seeds, not dt.
    Also asserted: the same seed gives the same arrival times at both dt (rtol 1e-9), because
    the in-step cascade is exact for constant rates.
    """
    c = 25
    cfg15 = scenario(shape=(51, 51), dt=15.0, duration=3000.0)
    cfg75 = cfg15.model_copy(update={"dt_s": 7.5})
    same15 = run(cfg15, seed=7, record_events=False).arrival_time
    same75 = run(cfg75, seed=7, record_events=False).arrival_time
    np.testing.assert_array_equal(np.isnan(same15), np.isnan(same75))
    np.testing.assert_allclose(same15, same75, rtol=1e-9, atol=1e-9)

    transect = (c, slice(c + 1, c + 16))
    a15, a75 = [], []
    for seed in range(50):
        r15 = run(cfg15, seed=seed, record_events=False)
        r75 = run(cfg75, seed=1000 + seed, record_events=False)
        for res, out in ((r15, a15), (r75, a75)):
            assert_inside(res.burned_mask)
            out.append(res.arrival_time[transect])
    t15, t75 = np.array(a15), np.array(a75)  # (50, 15), s
    assert not (np.isnan(t15).any() or np.isnan(t75).any())
    assert t75.mean() == pytest.approx(t15.mean(), rel=0.02)
    diff = t75.mean(axis=0) - t15.mean(axis=0)
    se = np.sqrt(t15.var(axis=0, ddof=1) / len(t15) + t75.var(axis=0, ddof=1) / len(t75))
    assert np.all(np.abs(diff) < 3 * se), (diff, se)
    assert stats.ks_2samp(t15[:, -1], t75[:, -1]).pvalue > 0.01
    assert stats.ks_2samp(t15.mean(axis=1), t75.mean(axis=1)).pvalue > 0.01


# ----------------------------------------------------------------------------- (d), (e) barriers


@pytest.mark.parametrize("stencil", [8, 16])
@pytest.mark.parametrize("ring", ["square", "circle"])
def test_complete_fireline_ring_stops_the_fire(stencil: int, ring: str) -> None:
    """(d) A complete fireline ring 3 cells wide around the ignition: no cell outside ignites.

    Threshold (plan target, kept): zero burned cells outside the ring, for 3 seeds. The fire
    must also reach the ring (cells just inside it burn), so the test is not passed vacuously.
    """
    n, c = 61, 30
    rows, cols = np.mgrid[0:n, 0:n]
    if ring == "square":
        dist = np.maximum(np.abs(rows - c), np.abs(cols - c))
        line = (dist >= 10) & (dist <= 12)
    else:
        dist = np.hypot(rows - c, cols - c)
        line = (dist >= 10) & (dist < 13)
    grid = flat_uniform((n, n), DX)
    grid.fireline[line] = True
    outside = (dist > 12) & ~line
    cfg = scenario(shape=(n, n), duration=6000.0, stencil=stencil)
    for seed in range(3):
        result = run(cfg, seed=seed, grid=grid, record_events=False)
        assert not result.burned_mask[outside].any()
        assert not result.burned_mask[line].any()
        assert result.burned_mask[(dist >= 9) & (dist < 10)].all()


@pytest.mark.parametrize("stencil", [8, 16])
def test_river_scenario_stops_the_fire(stencil: int) -> None:
    """(e) ``river_barrier.yaml`` (2-cell river at columns 60-61): no cell east of it ignites.

    Threshold (plan target, kept): zero burned cells in columns >= 60, and the fire reaches the
    west bank (column 59).
    """
    cfg = load_scenario(SCENARIOS / "river_barrier.yaml").model_copy(update={"stencil": stencil})
    result = run(cfg, record_events=False)
    assert result.burned_mask[:, 59].any()
    assert not result.burned_mask[:, 60:].any()


def test_one_cell_river_stops_knight_moves() -> None:
    """(e, 16 neighbours) A river one cell wide also holds: knight moves over it are blocked."""
    cfg = load_scenario(SCENARIOS / "river_barrier.yaml")
    data = cfg.model_dump()
    data["grid"]["params"] = {"col": 60, "width": 1}
    data["stencil"] = 16
    result = run(ScenarioConfig.model_validate(data), record_events=False)
    assert result.burned_mask[:, 59].any()
    assert not result.burned_mask[:, 60:].any()


# ----------------------------------------------------------------------------- (f) front speed


@pytest.mark.parametrize("stencil", [8, 16])
def test_corrected_front_speed_matches_r(stencil: int) -> None:
    """(f) With the shipped correction the front moves at R: axis extent / (R T) in [0.97, 1.03].

    Threshold (plan target, kept), on the mean over 10 seeds at T = 6000 s (nominal radius 40
    cells). Also asserted: the area-equivalent radius / (R T), the measure that defines the
    correction, in [0.98, 1.02]. Expected from the correction table's own measurement: axis
    0.990 (8) and 0.986 (16), area 1.000. The table must have entries for this stencil.
    """
    assert load_kernel_corrections().entry_for(stencil, 20.0) is not None
    t_s = 6000.0
    cfg = scenario(shape=(111, 111), duration=t_s, stencil=stencil)
    axis_r, area_r = [], []
    for seed in range(10):
        mask = run(cfg, seed=seed, record_events=False).burned_mask
        assert_inside(mask)
        axis_r.append(extents(mask)[0] / (ROS * t_s))
        area_r.append(math.sqrt(mask.sum() / math.pi) * DX / (ROS * t_s))
    assert 0.97 <= float(np.mean(axis_r)) <= 1.03, np.mean(axis_r)
    assert 0.98 <= float(np.mean(area_r)) <= 1.02, np.mean(area_r)


@pytest.mark.parametrize("stencil", [8, 16])
def test_memoryless_kernel_runs_far_too_fast(stencil: int, neutral_configs: Path) -> None:
    """(f) Uncorrected, ``arrival_shape = 1``: axis extent / (R T) > 2 (plan target, kept).

    T = 600 s (nominal radius 4 cells), 3 seeds, every run. Measured ratios grow with the size
    of the burn (4.7 on 8 neighbours and 10.8 on 16 at an area radius of 40 cells), which is
    why shape 1 has no correction entry.
    """
    t_s = 600.0
    cfg = scenario(shape=(161, 161), arrival_shape=1.0, duration=t_s, stencil=stencil)
    for seed in range(3):
        result = run(cfg, seed=seed, record_events=False)
        assert result.provenance["kernel_correction"] == 1.0
        mask = burned_by(result.arrival_time, t_s)
        assert_inside(mask)
        assert extents(mask)[0] / (ROS * t_s) > 2.0


# ----------------------------------------------------------------------------- (g) slow strip


STRIP_C0, STRIP_WIDTH = 20, 5  # slow strip occupies columns [20, 25)
STRIP_PREDICTED_S = 3 * STRIP_WIDTH * DX / ROS  # 2250 s


def strip_delay(stencil: int, arrival_shape: float, seeds: range) -> float:
    """Mean delay (s) of a planar front 15 cells beyond a 5-cell strip at R / 4.

    21 x 60 grid, line ignition along column 0. For each seed, the arrival-time difference at
    column 40 between a run with the strip and the same seed without it (common random
    numbers), averaged over rows; then averaged over seeds.
    """
    h, w = 21, 60
    cells = [(r, 0) for r in range(h)]
    cfg = scenario(shape=(h, w), cells=cells, duration=16000.0, stencil=stencil, arrival_shape=arrival_shape)
    col = STRIP_C0 + STRIP_WIDTH + 15
    delays = []
    for seed in seeds:
        slow = run(cfg, seed=seed, rule=StripRule(ROS, STRIP_C0, STRIP_C0 + STRIP_WIDTH, 4.0), record_events=False)
        base = run(cfg, seed=seed, rule=StripRule(ROS, STRIP_C0, STRIP_C0 + STRIP_WIDTH, 1.0), record_events=False)
        assert not np.isnan(slow.arrival_time[:, col]).any()
        delays.append(float(np.mean(slow.arrival_time[:, col] - base.arrival_time[:, col])))
    return float(np.mean(delays))


@pytest.mark.parametrize("stencil", [8, 16])
def test_slow_fuel_strip_delays_the_front(stencil: int) -> None:
    """(g) A strip 5 cells wide at R / 4 delays a planar front by ``3 w dx / R`` within 10 %.

    Prediction: with harmonic-mean link rates, crossing the strip costs ``(4 w + 1) dx / R``
    instead of ``(w + 1) dx / R``, a delay of ``3 w dx / R = 2250 s``. Threshold (plan target,
    kept): within 10 %, 5 seeds (see :func:`strip_delay`).

    Measured at ``arrival_shape = 20``: 8 neighbours -2.5 %, 16 neighbours -9.3 % (kernel noise
    lets the front pick faster-than-average links through the strip; the margin on 16
    neighbours is small). The deterministic limit is exact, see
    :func:`test_slow_fuel_strip_is_exact_without_noise`.
    """
    assert strip_delay(stencil, 20.0, range(5)) == pytest.approx(STRIP_PREDICTED_S, rel=0.10)


@pytest.mark.parametrize("stencil", [8, 16])
def test_slow_fuel_strip_is_exact_without_noise(stencil: int) -> None:
    """(g) Deterministic limit (``arrival_shape = 1e6``): the delay is 2250 s within 0.1 %.

    Regression test for knight links: a knight link's rate is the harmonic mean over the four
    cells its segment crosses. When it used only its two end cells, fire tunnelled through one
    slow cell at each edge of the strip and the 16-neighbour delay was 1977 s (-12 %).
    Measured now: 2250.1 s on both stencils.
    """
    assert strip_delay(stencil, 1e6, range(1)) == pytest.approx(STRIP_PREDICTED_S, rel=1e-3)


# ----------------------------------------------------------------------------- (h) wind shift


def test_wind_shift_mid_run_does_not_depend_on_dt() -> None:
    """(h) Wind shifts from west to north at 1800 s: same arrival times for dt 15 s and 7.5 s.

    Alexandridis rule (``ros_base = 0.4`` m/s, 5 m/s wind). The breakpoint is a multiple of
    both step sizes, since weather is looked up at step starts. Thresholds: the same seed gives
    the same arrival times (rtol 1e-9; exact by construction), and over 10 disjoint seeds per
    dt the mean arrival time of the cells burned in every run agrees within 2 % (plan target).
    The shift must matter: the burn after it reaches farther south than at the shift time.
    """
    weather = [
        {"time_s": 0.0, "wind_speed": 5.0, "wind_direction": 270.0},
        {"time_s": 1800.0, "wind_speed": 5.0, "wind_direction": 0.0},
    ]
    cfg15 = scenario(
        shape=(81, 81),
        rule="alexandridis",
        params={"ros_base": 0.4},
        dt=15.0,
        duration=3600.0,
        weather=weather,
    )
    cfg75 = cfg15.model_copy(update={"dt_s": 7.5})
    same15 = run(cfg15, seed=3, record_events=False).arrival_time
    same75 = run(cfg75, seed=3, record_events=False).arrival_time
    np.testing.assert_allclose(same15, same75, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(np.isnan(same15), np.isnan(same75))
    rows = np.nonzero(burned_by(same15, 3600.0))[0]
    rows_at_shift = np.nonzero(burned_by(same15, 1800.0))[0]
    assert rows.max() > rows_at_shift.max() + 3

    arr15 = np.array([run(cfg15, seed=s, record_events=False).arrival_time for s in range(10)])
    arr75 = np.array([run(cfg75, seed=100 + s, record_events=False).arrival_time for s in range(10)])
    for arr in (arr15, arr75):
        assert_inside(~np.isnan(arr).all(axis=0))
    common = ~np.isnan(arr15).any(axis=0) & ~np.isnan(arr75).any(axis=0)
    assert common.sum() > 200
    assert arr75[:, common].mean() == pytest.approx(arr15[:, common].mean(), rel=0.02)


# ----------------------------------------------------------------------------- shipped defaults

DEFAULT_STENCIL = 16
DEFAULT_SHAPE = 200.0


def test_shipped_defaults_are_the_ones_tested_here() -> None:
    """The tests below pin 16 neighbours and ``arrival_shape = 200``; fail if the defaults move."""
    cfg = ScenarioConfig.model_validate(
        {"name": "d", "grid": {"kind": "synthetic", "generator": "flat_uniform"}, "ignition": {"center": True}}
    )
    assert cfg.stencil == DEFAULT_STENCIL
    assert cfg.spread.arrival_shape == DEFAULT_SHAPE
    assert load_kernel_corrections().entry_for(DEFAULT_STENCIL, DEFAULT_SHAPE) is not None


def test_default_kernel_front_speed_matches_r() -> None:
    """(f) at the shipped defaults: axis extent / (R T) in [0.97, 1.03], area radius in [0.98, 1.02].

    Measured (10 seeds, T = 6000 s): axis 0.976, area 1.000, diagonal / axis 1.03. The axis
    value sits near the lower bound because the correction is defined on the area-equivalent
    radius and the 16-neighbour front is slightly slower along the axes than along diagonals.
    """
    t_s = 6000.0
    cfg = scenario(shape=(111, 111), duration=t_s, stencil=DEFAULT_STENCIL, arrival_shape=DEFAULT_SHAPE)
    axis_r, area_r = [], []
    for seed in range(10):
        mask = run(cfg, seed=seed, record_events=False).burned_mask
        assert_inside(mask)
        axis_r.append(extents(mask)[0] / (ROS * t_s))
        area_r.append(math.sqrt(mask.sum() / math.pi) * DX / (ROS * t_s))
    assert 0.97 <= float(np.mean(axis_r)) <= 1.03, np.mean(axis_r)
    assert 0.98 <= float(np.mean(area_r)) <= 1.02, np.mean(area_r)


@pytest.mark.parametrize("head_deg", ELLIPSE_HEADS)
@pytest.mark.parametrize(("lb", "tolerance"), [(1.5, 0.10), (3.0, 0.15)])
def test_default_kernel_ellipse(lb: float, tolerance: float, head_deg: float) -> None:
    """(b) at the shipped defaults: head and flank within 10 % at LB 1.5 and 15 % at LB 3.

    Measured worst case over the three head directions (5 seeds): LB 1.5 head -4.5 %, flank
    -6.5 % (the plan's 10 % target is met); LB 3 head -10.3 %, flank -12.6 % (target missed by
    under 3 points; threshold 15 %). LB 6 is outside what the lattice can represent (flank -32 %
    to -46 %) and is not asserted here; see docs/lattice_study.md and the open decision in
    docs/plan.md section 13.
    """
    head_err, flank_err, _back, _back_exact = ellipse_errors(DEFAULT_STENCIL, lb, head_deg, DEFAULT_SHAPE)
    assert -tolerance <= head_err <= 0.05, head_err
    assert -tolerance <= flank_err <= 0.05, flank_err


def test_default_kernel_slow_fuel_strip() -> None:
    """(g) at the shipped defaults: the strip delay is 2250 s within 5 % (measured +0.4 %, 5 seeds)."""
    assert strip_delay(DEFAULT_STENCIL, DEFAULT_SHAPE, range(5)) == pytest.approx(STRIP_PREDICTED_S, rel=0.05)


# ----------------------------------------------------------------------------- the Rothermel rule (WP 2.1b)

# FM1 (short grass, the fuel of the flat_uniform generator) at the default Weather moisture.
ROTHERMEL_PARAMS = {"apply_wind_limit": False}


@functools.cache
def rothermel_head(lb: float) -> tuple[float, float]:
    """10 m wind (m/s) that gives FM1 an ellipse of ratio ``lb``, and the head rate (m/s) it gives."""
    rule = RothermelRule(apply_wind_limit=False)
    g = Grid.empty((1, 1), DX)
    g.fuel_model[:] = 1

    def fire(u: float) -> Any:
        return rule.fire(g, Weather(wind_speed=u))

    u = 0.0 if lb == 1.0 else float(brentq(lambda x: fire(x).length_to_breadth[0, 0] - lb, 0.01, 40.0))
    return u, float(fire(u).ros_head[0, 0])


@pytest.mark.parametrize("head_deg", [*ELLIPSE_HEADS, 13.3, 200.0])
@pytest.mark.parametrize("lb", [1.0, 1.5, 3.0])
def test_rothermel_rule_on_uniform_ground_is_the_elliptical_rule(lb: float, head_deg: float) -> None:
    """WP 2.1b acceptance (physics tests 1.4 pass with this rule): on flat uniform fuel and uniform
    weather the ``"rothermel"`` rule gives every cell exactly :class:`EllipticalRule`'s rates for
    its head rate, ratio and head bearing, so the ellipse tests above hold for it unchanged. The
    two runs below check it in the simulator at the shipped defaults."""
    wind, ros_head = rothermel_head(lb)
    grid = flat_uniform((5, 6), DX, fuel_model=1)
    weather = Weather(wind_speed=wind, wind_direction=(head_deg + 180.0) % 360.0)
    for k in (8, 16):
        stencil = get_stencil(k)
        mine = RothermelRule(apply_wind_limit=False).rate_of_spread(grid, weather, stencil).ros
        theirs = EllipticalRule(ros_head, lb, head_deg).rate_of_spread(grid, weather, stencil).ros
        np.testing.assert_allclose(mine, theirs, rtol=1e-9)


def test_rothermel_no_wind_burn_is_round() -> None:
    """(a) and (f) with the Rothermel rule at the shipped defaults: FM1, no wind, flat.

    Axis extent / (R_0 T) in [0.97, 1.03] and area radius / (R_0 T) in [0.98, 1.02], the bounds
    of :func:`test_default_kernel_front_speed_matches_r`. Measured (5 seeds): 0.975 and 0.999,
    the same as the constant rule, as the rates are the same up to scale.
    """
    _, r0 = rothermel_head(1.0)
    t_s = 40 * DX / r0
    cfg = scenario(
        shape=(109, 109),
        rule="rothermel",
        params=ROTHERMEL_PARAMS,
        duration=t_s,
        stencil=DEFAULT_STENCIL,
        arrival_shape=DEFAULT_SHAPE,
    )
    axis_r, area_r = [], []
    for seed in range(5):
        mask = run(cfg, seed=seed, record_events=False).burned_mask
        assert_inside(mask)
        axis_r.append(extents(mask)[0] / (r0 * t_s))
        area_r.append(math.sqrt(mask.sum() / math.pi) * DX / (r0 * t_s))
    assert 0.97 <= float(np.mean(axis_r)) <= 1.03, np.mean(axis_r)
    assert 0.98 <= float(np.mean(area_r)) <= 1.02, np.mean(area_r)


@pytest.mark.parametrize("head_deg", ELLIPSE_HEADS)
def test_rothermel_wind_ellipse(head_deg: float) -> None:
    """(b) with the Rothermel rule at the shipped defaults: a 10 m wind giving FM1 a ratio of 1.5.

    Head and flank within 10 % of the analytic ellipse of the rule's own head rate, the bound of
    :func:`test_default_kernel_ellipse`. Measured (5 seeds): head -4.5 %, -4.1 %, -4.2 % and
    flank -4.4 %, -2.6 %, -6.5 % at head bearings 0, 22.5 and 45 deg.
    """
    lb = 1.5
    wind, ros_head = rothermel_head(lb)
    t_s = ELLIPSE_HEAD_CELLS * DX / ros_head
    n = 2 * math.ceil(1.35 * ELLIPSE_HEAD_CELLS) + 1
    c = n // 2
    cfg = scenario(
        shape=(n, n),
        rule="rothermel",
        params=ROTHERMEL_PARAMS,
        duration=t_s,
        stencil=DEFAULT_STENCIL,
        arrival_shape=DEFAULT_SHAPE,
        weather=[{"wind_speed": wind, "wind_direction": (head_deg + 180.0) % 360.0}],
    )
    phi = math.radians(head_deg)
    dists = []
    for seed in range(5):
        mask = run(cfg, seed=seed, record_events=False).burned_mask
        assert_inside(mask)
        rows, cols = np.nonzero(mask)
        east, north = (cols - c) * DX, -(rows - c) * DX
        along = east * math.sin(phi) + north * math.cos(phi)
        across = east * math.cos(phi) - north * math.sin(phi)
        dists.append((along.max(), np.ptp(across) / 2))
    head, half_width = np.mean(dists, axis=0)
    e = math.sqrt(lb**2 - 1.0) / lb
    head_exact = ros_head * t_s
    b_exact = (head_exact + head_exact * (1 - e) / (1 + e)) / 2 * math.sqrt(1 - e**2)
    assert -0.10 <= head / head_exact - 1 <= 0.05, head / head_exact - 1
    assert -0.10 <= half_width / b_exact - 1 <= 0.05, half_width / b_exact - 1
