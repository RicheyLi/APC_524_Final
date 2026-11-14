"""Measure the front speed of the real simulator relative to the nominal rate of spread.

Produces the front-speed correction table ``configs/kernel_correction.yaml`` (docs/plan.md
section 2.1, WP 1.4). Each run burns flat uniform fuel (dx = 30 m) with the constant rule
(R = 0.2 m/s), no wind, from one central ignition, through :func:`wildfire_sim.sim.run` with a
*neutral* correction (an empty table in a temporary copy of ``configs/``, selected with
``WILDFIRE_SIM_CONFIGS``), so the measured speed is the lattice's own.

Measures (all dimensionless, front speed / R), evaluated on the burned set
``B(T) = {cells with arrival_time <= T}``:

* ``area``: area-equivalent radius ``sqrt(|B| / pi) dx / T``. **This defines the correction.**
  It averages the front speed over all directions, and counting cell centres inside the
  front estimates the burned area without the half-cell bias of an extent.
* ``axis``: mean extent along the four grid axes, ``(ptp(col) + ptp(row)) / 4 * dx / T``.
* ``diag``: mean extent along the four diagonals, ``(ptp(row + col) + ptp(row - col)) / 4 / sqrt(2) * dx / T``.

The three differ by a few percent: that is the lattice's residual anisotropy, which one scalar
cannot remove. The area measure sits between the axis and the diagonal, so after the
correction neither is off by more than about 3 %; calibrating on the axis would push the
diagonal (and every direction in between) further off.

The time ``T`` is not fixed: it is the arrival time at which ``|B|`` first reaches
``pi * r_ref^2`` cells, i.e. the moment the burn's area-equivalent radius is ``r_ref`` cells.
Every shape is therefore measured at the same spatial scale, which matters because the
speed of a first-passage front still grows slowly with distance (``--scale`` prints it).
Arrival times do not depend on ``dt`` for constant rates (the in-step cascade is exact), so
one run gives ``B(T)`` for every ``T``.

Usage::

    uv run python scripts/kernel_front_speed.py              # print the table
    uv run python scripts/kernel_front_speed.py --scale      # also the speed vs. radius table
    uv run python scripts/kernel_front_speed.py --write      # write configs/kernel_correction.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wildfire_sim.config import ScenarioConfig, configs_dir
from wildfire_sim.sim import run
from wildfire_sim.types import FloatArray, StencilSize

ROS = 0.2  # nominal rate of spread, m/s
DX = 30.0  # cell size, m
DT = 60.0  # time step, s (arrival times are dt-independent for constant rates)
R_REF = 40  # reference area-equivalent radius, cells (1200 m)
N_SEEDS = 20
STENCILS: tuple[StencilSize, ...] = (8, 16)
SHAPES: tuple[float, ...] = (1.0, 5.0, 20.0, 50.0, 200.0)
WRITE_SHAPES: tuple[float, ...] = (5.0, 20.0, 50.0, 200.0)  # shape 1 is not written: see the docstring of write_table
SCALE_RADII: tuple[int, ...] = (10, 20, 40, 80, 120)
SCALE_SEEDS = 10


@contextmanager
def neutral_correction() -> Generator[Path]:
    """Point ``WILDFIRE_SIM_CONFIGS`` at a copy of ``configs/`` whose correction table is empty.

    Yields:
        The temporary configs directory. The environment variable is restored on exit.
    """
    source = configs_dir()
    old = os.environ.get("WILDFIRE_SIM_CONFIGS")
    with tempfile.TemporaryDirectory(prefix="wildfire_configs_") as tmp:
        tmp_path = Path(tmp)
        for path in source.glob("*.yaml"):
            shutil.copy(path, tmp_path / path.name)
        (tmp_path / "kernel_correction.yaml").write_text("entries: []\n", encoding="utf-8")
        os.environ["WILDFIRE_SIM_CONFIGS"] = str(tmp_path)
        try:
            yield tmp_path
        finally:
            if old is None:
                os.environ.pop("WILDFIRE_SIM_CONFIGS", None)
            else:
                os.environ["WILDFIRE_SIM_CONFIGS"] = old


def flat_config(n: int, stencil: StencilSize, shape: float, duration_s: float) -> ScenarioConfig:
    """Flat uniform ``n x n`` grid (dx = :data:`DX` m), constant rule at :data:`ROS` m/s, centre ignition."""
    return ScenarioConfig.model_validate(
        {
            "name": "kernel_front_speed",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [n, n], "dx": DX},
            "spread": {"rule": "constant", "params": {"ros": ROS}, "arrival_shape": shape},
            "ignition": {"center": True},
            "dt_s": DT,
            "duration_s": duration_s,
            "stencil": stencil,
        }
    )


@dataclass(frozen=True)
class FrontSpeed:
    """Front speed / R (dimensionless) of one run at one reference radius."""

    area: float
    axis: float
    diag: float
    t_s: float


def speeds_at_radius(arrival: FloatArray, radius_cells: float) -> FrontSpeed:
    """Front speeds / R when the burn's area-equivalent radius reaches ``radius_cells``.

    Args:
        arrival: ``(H, W)`` arrival times, s (NaN where never burned).
        radius_cells: Reference area-equivalent radius, cells.

    Raises:
        RuntimeError: Too few cells burned, or the burned set touches the grid edge at ``T``.
    """
    h, w = arrival.shape
    n_ref = round(math.pi * radius_cells**2)
    times = np.sort(arrival[~np.isnan(arrival)])
    if times.size < n_ref:
        raise RuntimeError(f"only {times.size} cells burned, need {n_ref}; raise the duration")
    t_ref = float(times[n_ref - 1])
    rows, cols = np.nonzero(np.nan_to_num(arrival, nan=np.inf) <= t_ref)
    if rows.min() == 0 or cols.min() == 0 or rows.max() == h - 1 or cols.max() == w - 1:
        raise RuntimeError("the burn reached the grid edge before the reference radius; enlarge the grid")
    nominal = ROS * t_ref
    axis = (np.ptp(cols) + np.ptp(rows)) / 4 * DX
    diag = (np.ptp(rows + cols) + np.ptp(rows - cols)) / 4 / math.sqrt(2) * DX
    area = math.sqrt(rows.size / math.pi) * DX
    return FrontSpeed(area=area / nominal, axis=float(axis / nominal), diag=float(diag / nominal), t_s=t_ref)


def measure(
    stencil: StencilSize, shape: float, seeds: Sequence[int], radii: Sequence[int] = (R_REF,)
) -> dict[int, list[FrontSpeed]]:
    """Run one burn per seed (neutral correction must be active) and measure it at every radius.

    Returns:
        Radius (cells) -> one :class:`FrontSpeed` per seed.
    """
    r_max = max(radii)
    n = 2 * math.ceil(1.45 * r_max) + 1
    duration = 2.0 * r_max * DX / ROS  # enough for any speed ratio >= 0.5
    cfg = flat_config(n, stencil, shape, duration)
    out: dict[int, list[FrontSpeed]] = {r: [] for r in radii}
    for seed in seeds:
        arrival = run(cfg, seed=seed, record_events=False).arrival_time
        for r in radii:
            out[r].append(speeds_at_radius(arrival, r))
    return out


def summarize(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error of the mean (ddof = 1)."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))


@dataclass(frozen=True)
class TableRow:
    """Means and standard errors over seeds for one (stencil, shape)."""

    stencil: StencilSize
    shape: float
    area: tuple[float, float]
    axis: tuple[float, float]
    diag: tuple[float, float]
    n_seeds: int


def correction_table(seeds: Sequence[int]) -> list[TableRow]:
    """Measure every (stencil, shape) in :data:`STENCILS` x :data:`SHAPES` at :data:`R_REF`."""
    rows = []
    with neutral_correction():
        for stencil in STENCILS:
            for shape in SHAPES:
                runs = measure(stencil, shape, seeds)[R_REF]
                rows.append(
                    TableRow(
                        stencil=stencil,
                        shape=shape,
                        area=summarize([s.area for s in runs]),
                        axis=summarize([s.axis for s in runs]),
                        diag=summarize([s.diag for s in runs]),
                        n_seeds=len(seeds),
                    )
                )
    return rows


def print_table(rows: Sequence[TableRow]) -> None:
    """Print ``mean +- standard error`` of each measure."""
    print(
        f"front speed / R, neutral correction; R = {ROS} m/s, dx = {DX:g} m, no wind, "
        f"reference area-equivalent radius {R_REF} cells; mean +- s.e. over {rows[0].n_seeds} seeds"
    )
    print(f"{'stencil':>7} {'shape':>5} {'area (correction)':>18} {'axis':>14} {'diag':>14} {'diag/axis':>9}")
    for r in rows:
        print(
            f"{r.stencil:>7} {r.shape:>5g} {r.area[0]:>10.3f} +- {r.area[1]:.3f} "
            f"{r.axis[0]:>6.3f} +- {r.axis[1]:.3f} {r.diag[0]:>6.3f} +- {r.diag[1]:.3f} {r.diag[0] / r.axis[0]:>9.3f}"
        )


def print_scale_table() -> None:
    """Print the area-measure speed / R at several radii (shape 20), to show the scale drift."""
    seeds = range(SCALE_SEEDS)
    print(f"\nscale dependence, shape 20, area measure, mean +- s.e. over {SCALE_SEEDS} seeds")
    print(f"{'stencil':>7} " + " ".join(f"{f'r={r}':>15}" for r in SCALE_RADII))
    with neutral_correction():
        for stencil in STENCILS:
            res = measure(stencil, 20.0, seeds, SCALE_RADII)
            cells = []
            for r in SCALE_RADII:
                mean, se = summarize([s.area for s in res[r]])
                cells.append(f"{mean:>7.3f} +- {se:.3f}")
            print(f"{stencil:>7} " + " ".join(f"{c:>15}" for c in cells))


def write_table(rows: Sequence[TableRow], path: Path) -> None:
    """Write the area-measure values for :data:`WRITE_SHAPES` to ``path`` as the correction table.

    Shape 1 (memoryless) is measured and printed but not written: its speed ratio is large and
    keeps growing with the size of the burn, so no constant correction is meaningful for it.
    """
    lines = [
        "# Front-speed correction of the ignition kernel per (stencil, arrival_shape), dimensionless",
        "# (docs/plan.md section 2.1). The simulator divides every link rate of spread by `value`, so",
        "# the rate R of a spread rule means the same front speed on every stencil. Read by",
        "# wildfire_sim.sim (KernelCorrectionTable); a (stencil, arrival_shape) pair with no entry",
        "# uses 1.0. The match on arrival_shape is exact.",
        "#",
        "# Generated by `uv run python scripts/kernel_front_speed.py --write` (WP 1.4); see",
        "# docs/lattice_study.md. `value` is the area-equivalent-radius front speed / R of the",
        "# uncorrected simulator at a burn radius of 40 cells; `note` gives the axis and diagonal",
        "# speeds, whose spread around `value` is the residual anisotropy one scalar cannot remove.",
        "entries:",
    ]
    for r in rows:
        if r.shape not in WRITE_SHAPES:
            continue
        source = (
            f"scripts/kernel_front_speed.py: area-equivalent-radius front speed / R, constant rule "
            f"R = {ROS} m/s, dx = {DX:g} m, no wind, flat uniform fuel, reference radius {R_REF} cells, "
            f"{r.n_seeds} seeds, standard error {r.area[1]:.4f}"
        )
        note = (
            f"uncorrected axis {r.axis[0]:.3f} +- {r.axis[1]:.3f}, "
            f"diagonal {r.diag[0]:.3f} +- {r.diag[1]:.3f} (front speed / R)"
        )
        lines += [
            f"  - stencil: {r.stencil}",
            f"    arrival_shape: {r.shape:.1f}",
            f"    value: {r.area[0]:.3f}",
            f'    source: "{source}"',
            f'    note: "{note}"',
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--write", action="store_true", help="write configs/kernel_correction.yaml")
    parser.add_argument("--scale", action="store_true", help="also print speed vs. burn radius (shape 20)")
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help=f"seeds per entry (default {N_SEEDS})")
    args = parser.parse_args(argv)
    target = configs_dir() / "kernel_correction.yaml"  # resolved before the neutral override
    rows = correction_table(range(args.seeds))
    print_table(rows)
    if args.scale:
        print_scale_table()
    if args.write:
        write_table(rows, target)


if __name__ == "__main__":
    main()
