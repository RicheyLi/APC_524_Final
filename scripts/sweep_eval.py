"""Score one model configuration against a built real fire (sweep and diagnosis harness).

Loads the region once, applies the requested changes (spread parameters, weather transforms,
parameter-file edits, terrain and break switches), runs the fire over the calibration window for
each ``ros_multiplier`` in ``--multipliers`` and each seed, and writes one JSON record with the
calibration terms (``1 - J``, normalized arrival error, objective), the burned area, the edge
time, and a direction diagnostic: the share of the observed cells reached in each of eight
compass sectors around the ignition, and the centroid offset of the simulated fire from the
observed one. A run that reaches the grid edge is scored anyway and flagged (unlike the
calibration, which rejects it), so a sweep can still rank configurations that are too fast.

Examples (fire A must be built: ``scripts/download_data.py lake_2024 --observations``):

    uv run --extra fast --extra geo python scripts/sweep_eval.py lake_2024 \\
        --param lb_scale=0.641 --multipliers 0.383 --seeds 2 --out runs/sweep/baseline.json

    uv run --extra fast --extra geo python scripts/sweep_eval.py lake_2024 \\
        --wind-dir-offset 180 --multipliers 0.25,0.383,0.6 --out runs/sweep/wind_180.json

    uv run --extra fast --extra geo python scripts/sweep_eval.py lake_2024 \\
        --edit rothermel.yaml:crown_ratio.value=0.6 --edit breach.yaml:min_intensity_mw_m.value=0.09
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_sim.calibration import load_calibration_settings, observed_target, real_fire
from wildfire_sim.config import ScenarioConfig
from wildfire_sim.grid import Grid
from wildfire_sim.metrics import burned_area_ha, sorensen
from wildfire_sim.sensitivity import parameter_files
from wildfire_sim.sim import run
from wildfire_sim.spread import make_spread_rule

SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _parse_kv(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        if not key or not raw:
            raise SystemExit(f"expected key=value, got {item!r}")
        try:
            out[key] = float(raw) if raw.lower() not in ("true", "false") else raw.lower() == "true"
        except ValueError:
            out[key] = raw
    return out


def _parse_edits(items: list[str]) -> dict[str, dict[str, Any]]:
    """``file.yaml:dotted.key=value`` -> ``{file: {key: value}}``."""
    edits: dict[str, dict[str, Any]] = {}
    for item in items:
        file, _, rest = item.partition(":")
        key, _, raw = rest.partition("=")
        if not (file and key and raw):
            raise SystemExit(f"expected file.yaml:dotted.key=value, got {item!r}")
        try:
            value: Any = float(raw)
        except ValueError:
            value = {"true": True, "false": False}.get(raw.lower(), raw)
        edits.setdefault(file, {})[key] = value
    return edits


def transform_weather(config: ScenarioConfig, args: argparse.Namespace) -> ScenarioConfig:
    data = config.model_dump()
    for p in data["weather"]["series"]:
        p["wind_speed"] = p["wind_speed"] * args.wind_scale
        p["wind_direction"] = (p["wind_direction"] + args.wind_dir_offset) % 360.0
        if args.wind_dir_fixed is not None:
            p["wind_direction"] = args.wind_dir_fixed % 360.0
        for k in ("moisture_1h", "moisture_10h", "moisture_100h"):
            p[k] = p[k] * args.dead_moisture_scale
        if args.live_herb is not None:
            p["moisture_live_herb"] = args.live_herb
        if args.live_woody is not None:
            p["moisture_live_woody"] = args.live_woody
    return ScenarioConfig.model_validate(data)


def transform_grid(grid: Grid, args: argparse.Namespace) -> Grid:
    changes: dict[str, Any] = {}
    if args.no_slope:
        changes["slope"] = np.zeros_like(grid.slope)
        changes["aspect"] = np.zeros_like(grid.aspect)
    if args.slope_scale != 1.0:
        changes["slope"] = (grid.slope * args.slope_scale).astype(grid.slope.dtype)
    if args.no_breaks:
        changes["break_width_m"] = np.zeros_like(grid.break_width_m)
        changes["fireline"] = np.zeros_like(grid.fireline)
    if not changes:
        return grid
    g = dataclasses.replace(grid, **changes)
    g.validate()
    return g


def sector_recall(sim: np.ndarray, obs: np.ndarray, ignition: tuple[int, int]) -> dict[str, dict[str, float]]:
    """Per compass sector around the ignition: observed cells, the share the simulation burned, and
    the simulated cells outside the observed fire (false alarms)."""
    rows, cols = np.indices(obs.shape)
    north = -(rows - ignition[0]).astype(np.float64)
    east = (cols - ignition[1]).astype(np.float64)
    bearing = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    idx = np.floor(((bearing + 22.5) % 360.0) / 45.0).astype(int)
    out: dict[str, dict[str, float]] = {}
    for k, name in enumerate(SECTORS):
        sel = idx == k
        n_obs = int(np.sum(obs & sel))
        hit = int(np.sum(obs & sim & sel))
        false = int(np.sum(sim & ~obs & sel))
        out[name] = {
            "observed_cells": n_obs,
            "recall": hit / n_obs if n_obs else math.nan,
            "false_alarm_cells": false,
        }
    return out


def centroid_offset_m(sim: np.ndarray, obs: np.ndarray, dx: float) -> dict[str, float]:
    def c(mask: np.ndarray) -> tuple[float, float]:
        r, cc = np.nonzero(mask)
        return float(r.mean()), float(cc.mean())

    (rs, cs), (ro, co) = c(sim), c(obs)
    return {"east_m": (cs - co) * dx, "north_m": -(rs - ro) * dx}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("region")
    ap.add_argument("--window-hours", type=float, default=None, help="default: configs/calibration.yaml")
    ap.add_argument("--rule", default="rothermel")
    ap.add_argument("--param", action="append", default=[], help="spread param key=value (repeatable)")
    ap.add_argument("--multipliers", default="1.0", help="comma-separated ros_multiplier values to scan")
    ap.add_argument("--seeds", type=int, default=1, help="number of calibration seeds to run (same seeds as WP 3.1)")
    ap.add_argument(
        "--edit", action="append", default=[], help="file.yaml:dotted.key=value, applied via WILDFIRE_SIM_CONFIGS"
    )
    ap.add_argument("--wind-scale", type=float, default=1.0)
    ap.add_argument("--wind-dir-offset", type=float, default=0.0, help="degrees added to the 'from' direction")
    ap.add_argument("--wind-dir-fixed", type=float, default=None, help="replace every hour's 'from' direction")
    ap.add_argument("--dead-moisture-scale", type=float, default=1.0)
    ap.add_argument("--live-herb", type=float, default=None, help="fraction, replaces every hour")
    ap.add_argument("--live-woody", type=float, default=None, help="fraction, replaces every hour")
    ap.add_argument("--no-slope", action="store_true")
    ap.add_argument("--slope-scale", type=float, default=1.0)
    ap.add_argument("--no-breaks", action="store_true")
    ap.add_argument("--stencil", type=int, default=None, choices=(8, 16))
    ap.add_argument("--arrival-shape", type=float, default=None)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--cache-root", default="data/cache")
    ap.add_argument("--label", default="", help="free text stored in the record")
    ap.add_argument("--out", type=Path, default=None, help="JSON record; default runs/sweep/<timestamp>.json")
    ap.add_argument("--save-masks", action="store_true", help="also save burned masks beside the JSON (npz)")
    args = ap.parse_args(argv)

    settings = load_calibration_settings()
    hours = args.window_hours if args.window_hours is not None else settings.window_hours.value
    weights = settings.weights
    seeds = settings.seeds(args.seeds)
    multipliers = [float(x) for x in args.multipliers.split(",") if x]
    params = _parse_kv(args.param)
    edits = _parse_edits(args.edit)
    out = args.out or Path("runs/sweep") / f"{args.region}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with parameter_files(edits):
        fire = real_fire(args.region, hours, cache_root=args.cache_root, dt_s=args.dt, rule=args.rule)
        target = observed_target(fire, settings.footprint_m.value)
        config = transform_weather(fire.config, args)
        data = config.model_dump()
        if args.stencil is not None:
            data["stencil"] = args.stencil
        if args.arrival_shape is not None:
            data["spread"]["arrival_shape"] = args.arrival_shape
        config = ScenarioConfig.model_validate(data)
        grid = transform_grid(fire.grid, args)
        ignition = tuple(config.ignition.cells[0])
        obs_ha = float(burned_area_ha(target.observed, dx=grid.dx))

        print(
            f"{args.region}: {grid.shape[0]} x {grid.shape[1]} cells, window {hours:g} h, observed {obs_ha:.0f} ha, "
            f"{len(target.detections)} detections, seeds {list(seeds)}, multipliers {multipliers}",
            flush=True,
        )
        records: list[dict[str, Any]] = []
        for m in multipliers:
            p = {**params, "ros_multiplier": m}
            rule = make_spread_rule(args.rule, **p)
            per_seed: list[dict[str, Any]] = []
            for seed in seeds:
                t0 = time.perf_counter()
                spread = {**data["spread"], "params": p}
                scenario = ScenarioConfig.model_validate({**config.model_dump(), "spread": spread})
                result = run(scenario, seed=seed, rule=rule, grid=grid, record_events=False)
                o, a = target.terms(result)
                sim = result.burned_mask
                rec = {
                    "seed": seed,
                    "one_minus_jaccard": o,
                    "jaccard": 1.0 - o,
                    "sorensen": sorensen(sim, target.observed),
                    "arrival_error": a,
                    "objective": weights[0] * o + weights[1] * a,
                    "burned_ha": float(burned_area_ha(result)),
                    "area_ratio": float(burned_area_ha(result)) / obs_ha,
                    "reached_edge_s": result.provenance["reached_edge_s"],
                    "reached_edge_h_after_ignition": None
                    if result.provenance["reached_edge_s"] is None
                    else (result.provenance["reached_edge_s"] - target.ignition_s) / 3600.0,
                    "lb_capped_cell_hours": result.provenance["lb_capped_cell_hours"],
                    "breaches": result.provenance["breaks"].get("breaches", 0),
                    "sectors": sector_recall(sim, target.observed, ignition),
                    "centroid_offset_m": centroid_offset_m(sim, target.observed, grid.dx),
                    "wall_s": time.perf_counter() - t0,
                }
                per_seed.append(rec)
                edge_h = rec["reached_edge_h_after_ignition"]
                edge = "" if edge_h is None else f"  EDGE at {edge_h:.1f} h"
                print(
                    f"  m={m:g} seed={seed}: J={rec['jaccard']:.3f} arr={a:.3f} obj={rec['objective']:.3f} "
                    f"area={rec['burned_ha']:.0f} ha (x{rec['area_ratio']:.2f}) breaches={rec['breaches']} "
                    f"{rec['wall_s']:.0f}s{edge}",
                    flush=True,
                )
                if args.save_masks:
                    mask_path = out.with_suffix(f".m{m:g}.s{seed}.npz")
                    np.savez_compressed(mask_path, burned=sim, arrival=result.arrival_time)
            keys = ("jaccard", "arrival_error", "objective", "burned_ha", "area_ratio")
            mean = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}
            records.append({"ros_multiplier": m, "params": p, "mean": mean, "runs": per_seed})
            sectors = per_seed[0]["sectors"].items()
            rec_s = " ".join(f"{k}={v['recall']:.2f}" for k, v in sectors if not math.isnan(v["recall"]))
            print(f"  m={m:g} mean: J={mean['jaccard']:.3f} obj={mean['objective']:.3f}  sectors: {rec_s}", flush=True)

    best = min(records, key=lambda r: r["mean"]["objective"])
    record = {
        "region": args.region,
        "label": args.label,
        "argv": sys.argv[1:] if argv is None else argv,
        "window_hours": hours,
        "weights": weights,
        "seeds": list(seeds),
        "params": params,
        "edits": edits,
        "observed_ha": obs_ha,
        "n_detections": len(target.detections),
        "observed_sectors": {k: v["observed_cells"] for k, v in records[0]["runs"][0]["sectors"].items()},
        "best": {"ros_multiplier": best["ros_multiplier"], **best["mean"]},
        "scan": records,
    }
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    b = best["mean"]
    print(f"best: m={best['ros_multiplier']:g} J={b['jaccard']:.3f} obj={b['objective']:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
