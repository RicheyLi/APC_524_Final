"""Measure how fast the ignition kernel moves a fire front relative to the nominal rate of spread.

Evidence for the choice of ``spread.arrival_shape`` (see the ignition-kernel section of
docs/plan.md). Burns flat uniform fuel with no wind from a single central ignition, using
:func:`wildfire_sim.spread.base.ignition_probability` directly, and prints the front speed
divided by the nominal rate ``R`` along the grid axes, along the diagonals, and as an
area-equivalent radius. A value of 1.00 means the front moves at exactly ``R``.

Usage::

    uv run python scripts/kernel_front_speed.py
"""

from __future__ import annotations

import numpy as np

from wildfire_sim.spread.base import ignition_probability, neighbour_ignition_probability
from wildfire_sim.types import StencilSize, get_stencil

ROS = 0.2  # nominal rate of spread, m/s
DX = 30.0  # cell size, m
N = 201  # grid is N x N cells
DURATION = 2400.0  # simulated time, s
N_SEEDS = 6
_FUEL, _BURNING = 1, 2


def front_speed_ratios(shape: float, dt: float, stencil_size: StencilSize, seed: int) -> tuple[float, float, float]:
    """Run one burn and return (axis, diagonal, area-radius) front speed divided by ``ROS``.

    Args:
        shape: Gamma shape of the crossing time (dimensionless, >= 1).
        dt: Time step, s.
        stencil_size: 8 or 16 neighbours.
        seed: Seed for the run's random generator.

    Raises:
        RuntimeError: The fire reached the grid edge, so the extents are truncated.
    """
    stencil = get_stencil(stencil_size)
    rng = np.random.default_rng(seed)
    ros = np.full((stencil.size, N, N), ROS)
    state = np.full((N, N), _FUEL, dtype=np.int8)
    arrival = np.full((N, N), np.nan)
    centre = N // 2
    state[centre, centre] = _BURNING
    arrival[centre, centre] = 0.0
    t = 0.0
    for _ in range(round(DURATION / dt)):
        burning = state == _BURNING
        age = np.where(burning, t - arrival, 0.0)
        p = ignition_probability(ros, dt, DX, stencil, age=age, shape=shape)
        ignite = (state == _FUEL) & (rng.random((N, N)) < neighbour_ignition_probability(p, burning, stencil))
        t += dt
        state[ignite] = _BURNING
        arrival[ignite] = t
    rows, cols = np.nonzero(state == _BURNING)
    if rows.min() == 0 or cols.min() == 0 or rows.max() == N - 1 or cols.max() == N - 1:
        raise RuntimeError("fire reached the grid edge")
    axis = (np.ptp(cols) + np.ptp(rows)) / 4 * DX
    diag = (np.ptp(rows + cols) + np.ptp(rows - cols)) / 4 / np.sqrt(2) * DX
    area_radius = np.sqrt(rows.size / np.pi) * DX
    nominal = ROS * DURATION
    return float(axis / nominal), float(diag / nominal), float(area_radius / nominal)


def main() -> None:
    print(f"front speed / R (R = {ROS} m/s, dx = {DX:g} m, {DURATION:g} s, mean of {N_SEEDS} seeds)")
    print(f"{'stencil':>7} {'shape':>5} {'dt_s':>5} {'axis':>6} {'diag':>6} {'area':>6}")
    stencil_sizes: tuple[StencilSize, ...] = (8, 16)
    for stencil_size in stencil_sizes:
        for shape in (1.0, 5.0, 20.0, 50.0):
            for dt in (30.0, 15.0, 7.5):
                try:
                    runs = [front_speed_ratios(shape, dt, stencil_size, seed) for seed in range(N_SEEDS)]
                except RuntimeError:
                    print(f"{stencil_size:>7} {shape:>5g} {dt:>5g}   off grid")
                    continue
                axis, diag, area = np.mean(runs, axis=0)
                print(f"{stencil_size:>7} {shape:>5g} {dt:>5g} {axis:>6.2f} {diag:>6.2f} {area:>6.2f}")


if __name__ == "__main__":
    main()
