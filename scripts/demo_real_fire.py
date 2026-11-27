"""Demo: simulate a validation fire on its real grid and compare it with what was observed.

Not calibrated (Phase 3 does that): a first look at what the model does with real data. Needs a
built region and its observations (``scripts/download_data.py <region> --observations``).

Usage::

    uv run python scripts/demo_real_fire.py lake_2024 --hours 24
    uv run python scripts/demo_real_fire.py silverado_2020 --hours 12

Writes ``runs/demo_<region>.png`` (simulated arrival time, observed perimeter, VIIRS detections)
and prints the overlap with the observed burned area and the arrival-interval errors. The run is
ignited at the recorded point and time; the comparison is with the daily perimeters posted by the
end of the run when the fire has them, else with the final MTBS perimeter. With the
observations cached it needs no network.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wildfire_sim.calibration import real_fire
from wildfire_sim.data.cache import load_regions
from wildfire_sim.metrics import arrival_interval_error, burned_area_ha, jaccard, sorensen
from wildfire_sim.sim import run


def main(argv: list[str] | None = None) -> int:
    regions = load_regions().regions
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("region", choices=sorted(regions))
    ap.add_argument("--hours", type=float, default=24.0, help="hours simulated after the ignition")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dt", type=float, default=60.0, help="time step, s")
    ap.add_argument("--cache-root", type=Path, default=Path("data") / "cache")
    args = ap.parse_args(argv)

    region = regions[args.region]
    try:
        fire = real_fire(args.region, args.hours, cache_root=args.cache_root, dt_s=args.dt)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    config, grid, obs = fire.config, fire.grid, fire.observations
    ignition_s, end_s, engine = fire.ignition_s, fire.end_s, config.engine
    print(
        f"{args.region}: {grid.shape[0]} x {grid.shape[1]} cells, ignition {obs.ignition.time_utc:%Y-%m-%d %H:%M} UTC, "
        f"{args.hours:g} h, Rothermel, {engine} engine"
    )
    t0 = time.perf_counter()
    result = run(config, seed=args.seed)
    print(f"  ran in {time.perf_counter() - t0:.0f} s")

    until = fire.until
    observed = obs.observed_burn_mask(grid, until) if obs.daily else obs.observed_burn_mask(grid)
    label = f"daily perimeter posted by {until:%m-%d %H:%M} UTC" if obs.daily else "final MTBS perimeter"
    sim = result.burned_mask
    print(
        f"  simulated burned area {burned_area_ha(result):,.0f} ha; "
        f"observed ({label}) {observed.sum() * grid.dx**2 / 1e4:,.0f} ha"
    )
    print(f"  overlap: Jaccard {jaccard(sim, observed):.2f}, Sorensen {sorensen(sim, observed):.2f}")
    dets = [d for d in obs.detections(grid) if d.t_detect <= end_s]
    if dets:
        err = arrival_interval_error(result.arrival_time, dets, 375.0, end_s, dx=grid.dx) / 3600
        print(
            f"  {len(dets)} VIIRS detections in the window: {np.mean(err == 0):.0%} with the simulated arrival inside "
            f"its interval; median error {np.median(err):.1f} h"
        )
    edge = result.provenance.get("reached_edge_s")
    edge_text = "no" if edge is None else f"yes, {(edge - ignition_s) / 3600:.1f} h after ignition"
    print(
        f"  reached the grid edge: {edge_text}; "
        f"elongation cap applied for {result.provenance['lb_capped_cell_hours']:.0f} cell-hours; "
        f"breaks breached {result.provenance['breaks']['breaches']}"
    )

    hours = (result.arrival_time - ignition_s) / 3600
    fig, ax = plt.subplots(figsize=(9, 8))
    background = np.where(grid.burnable_mask(), 0.85, 0.55)
    ax.imshow(background, cmap="gray", vmin=0, vmax=1)
    im = ax.imshow(np.ma.masked_invalid(hours), cmap="inferno_r", vmin=0, vmax=args.hours)
    ax.contour(observed.astype(float), levels=[0.5], colors="deepskyblue", linewidths=1.5)
    if dets:
        ax.scatter([d.col for d in dets], [d.row for d in dets], s=4, c="lime", label="VIIRS first detections")
    r, c = obs.ignition_cell(grid)
    ax.plot(c, r, marker="*", color="white", markeredgecolor="black", markersize=14, label="point of origin")
    ax.plot([], [], color="deepskyblue", label=f"observed ({label})")
    ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=ax, label="simulated arrival, hours after ignition", shrink=0.8)
    ax.set_title(f"{region.description}\nuncalibrated Rothermel run, {args.hours:g} h (seed {args.seed})", fontsize=10)
    ax.set_axis_off()
    out = Path("runs") / f"demo_{args.region}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  map: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
