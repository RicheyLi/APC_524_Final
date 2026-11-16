"""Calibrate the spread rule on a fire's early growth with an Optuna study (WP 3.1).

The objective, the edge rule and the parameters are described in :mod:`wildfire_sim.calibration`
and docs/calibration.md; defaults come from ``configs/calibration.yaml``. Needs the ``opt``
extra (optuna), and the ``fast`` extra for the Numba engine.

Two targets:

* a built region and its observations (``scripts/download_data.py <region> --observations``):
  the run starts at the recorded ignition and is scored against the daily perimeters posted by
  the end of the window and the VIIRS detections in it. Writes ``configs/calibrated.yaml``.
  With the observations cached it needs no network.
* ``--synthetic SCENARIO``: a synthetic "observed" fire, one run of the scenario with the known
  parameters ``--truth`` (seed ``truth_seed``), with detections sampled from its arrival times
  at synthetic overpasses. A test of the method; writes ``runs/calibration_synthetic.yaml``.
  ``--weather`` replaces the scenario's weather (e.g. the committed weather of a committed crop).

Usage::

    uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024
    uv run --extra fast --extra opt python scripts/calibrate.py --synthetic scenario.yaml \\
        --truth ros_multiplier=1.2 --truth lb_scale=1.1 --window-hours 1 --trials 3 --seeds 2

Prints the best parameters, the two objective terms, the trials rejected by the edge rule and the
objective surface around the optimum.
"""

from __future__ import annotations

import argparse
import importlib
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from wildfire_sim.calibration import (
    Evaluation,
    Target,
    calibrated_params,
    load_calibration_settings,
    objective_surface,
    observed_target,
    real_fire,
    run_study,
    synthetic_target,
    write_calibrated,
)
from wildfire_sim.config import ScenarioConfig, WeatherConfig, load_scenario, load_yaml
from wildfire_sim.data import build_grid
from wildfire_sim.data.cache import load_regions


def _truth(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--truth takes NAME=VALUE, got {item!r}")
        out[name.strip()] = float(value)
    return out


def _format(ev: Evaluation) -> str:
    params = ", ".join(f"{k} {v:.4g}" for k, v in ev.params.items())
    if ev.objective is None:
        return f"{params}: rejected (a run reached the grid edge at t = {ev.reached_edge_s:.0f} s)"
    return (
        f"{params}: objective {ev.objective:.4f} (1 - Jaccard {np.mean(ev.overlap):.3f}, "
        f"arrival {np.mean(ev.arrival):.3f})"
    )


def main(argv: list[str] | None = None) -> int:
    regions = load_regions().regions
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("region", nargs="?", choices=sorted(regions), help="built region to calibrate on")
    ap.add_argument("--synthetic", type=Path, metavar="SCENARIO", help="scenario YAML of a synthetic observed fire")
    ap.add_argument("--weather", type=Path, help="weather YAML replacing the synthetic scenario's weather")
    ap.add_argument("--truth", action="append", default=[], metavar="NAME=VALUE", help="known parameter (synthetic)")
    ap.add_argument("--rule", default="rothermel", help="spread rule for a region (default rothermel)")
    ap.add_argument("--window-hours", type=float, help="early-growth window after the ignition, h")
    ap.add_argument("--trials", type=int, help="Optuna trials")
    ap.add_argument("--seeds", type=int, help="fixed run seeds per trial")
    ap.add_argument("--dt", type=float, default=60.0, help="time step for a region, s")
    ap.add_argument("--cache-root", type=Path, default=Path("data") / "cache")
    ap.add_argument("--settings", type=Path, help="calibration settings (default configs/calibration.yaml)")
    ap.add_argument("--storage", help="Optuna storage URL, e.g. sqlite:///runs/calibration.db")
    ap.add_argument("--study-name", help="study name in --storage")
    ap.add_argument("--no-surface", action="store_true", help="skip the objective surface around the optimum")
    ap.add_argument("--out", type=Path, help="output YAML")
    args = ap.parse_args(argv)
    if (args.region is None) == (args.synthetic is None):
        ap.error("give either a region or --synthetic SCENARIO")
    if args.synthetic is None and (args.truth or args.weather):
        ap.error("--truth and --weather apply to --synthetic only")

    settings = load_calibration_settings(args.settings)
    hours = settings.window_hours.value if args.window_hours is None else args.window_hours
    n_trials = int(settings.n_trials.value) if args.trials is None else args.trials
    seeds = settings.seeds(args.seeds)
    footprint = float(settings.footprint_m.value)
    if args.region is not None and args.rule not in settings.parameters:
        ap.error(f"no calibration parameters for rule {args.rule!r}; available: {sorted(settings.parameters)}")

    target: Target
    if args.synthetic is not None:
        try:
            truth = _truth(args.truth)
        except ValueError as exc:
            ap.error(str(exc))
        base = load_scenario(args.synthetic)
        data = base.model_dump()
        if args.weather is not None:
            data["weather"] = load_yaml(args.weather, WeatherConfig).model_dump()
        data["duration_s"] = base.ignition.time_s + hours * 3600.0
        config = ScenarioConfig.model_validate(data)
        if config.spread.rule not in settings.parameters:
            ap.error(
                f"no calibration parameters for rule {config.spread.rule!r}; available: {sorted(settings.parameters)}"
            )
        space = settings.space(config.spread.rule)
        if set(truth) != set(space):
            ap.error(f"--truth must give exactly {sorted(space)} for rule {config.spread.rule!r}")
        truth_seed = int(settings.truth_seed.value)
        if truth_seed in seeds:
            ap.error("truth_seed must not be one of the calibration seeds")
        grid = build_grid(config.grid)
        try:
            target, _ = synthetic_target(
                config,
                truth,
                seed=truth_seed,
                overpass_interval_s=float(settings.overpass_interval_s.value),
                footprint_m=footprint,
                grid=grid,
            )
        except ValueError as exc:
            print(f"calibration failed: {exc}", file=sys.stderr)
            return 1
        status, out = "synthetic", args.out or Path("runs") / "calibration_synthetic.yaml"
    else:
        try:
            fire = real_fire(args.region, hours, cache_root=args.cache_root, dt_s=args.dt, rule=args.rule)
        except FileNotFoundError as exc:
            print(exc)
            return 1
        config, grid = fire.config, fire.grid
        space = settings.space(config.spread.rule)
        try:
            target = observed_target(fire, footprint)
        except ValueError as exc:
            print(f"calibration failed: {exc}", file=sys.stderr)
            return 1
        status, out = "calibrated", args.out or Path("configs") / "calibrated.yaml"

    print(
        f"{target.description}: {grid.shape[0]} x {grid.shape[1]} cells, rule {config.spread.rule}, "
        f"{config.engine} engine; {np.count_nonzero(target.observed)} cells observed burned, "
        f"{len(target.detections)} detections; {n_trials} trials x {len(seeds)} seeds"
    )
    try:
        optuna = importlib.import_module("optuna")
    except ImportError:
        print("calibration needs optuna: install the opt extra (uv sync --extra opt)", file=sys.stderr)
        return 1
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # one line per trial is printed below
    start = time.perf_counter()

    def progress(number: int, ev: Evaluation) -> None:
        print(f"  trial {number:3d}  {_format(ev)}  [{time.perf_counter() - start:.0f} s]", flush=True)

    try:
        study = run_study(
            config,
            target,
            space,
            n_trials=n_trials,
            seeds=seeds,
            weights=settings.weights,
            sampler_seed=int(settings.sampler_seed.value),
            grid=grid,
            storage=args.storage,
            study_name=args.study_name,
            progress=progress,
        )
    except RuntimeError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1
    print(f"best of {len(study.trials)} trials ({study.n_rejected} rejected by the edge rule): {_format(study.best)}")

    surface: list[Evaluation] = []
    if not args.no_surface:
        surface = objective_surface(
            config, target, study.best.params, space, settings.surface_steps, seeds, settings.weights, grid=grid
        )
        print("objective surface around the optimum:")
        for ev in surface:
            print(f"  {_format(ev)}")
    record = calibrated_params(
        config,
        target,
        study,
        surface,
        space,
        settings.weights,
        status=status,
        command=shlex.join(["scripts/calibrate.py", *(sys.argv[1:] if argv is None else argv)]),
    )
    write_calibrated(record, out)
    print(f"wrote {out} ({time.perf_counter() - start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
