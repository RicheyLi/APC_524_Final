"""Validate the calibrated model on fire A and, without retuning, on fire B (WP 3.3).

Runs an ensemble of unsuppressed runs of a fire from its recorded ignition to the end of a
comparison window and scores it against the observations (method: :mod:`wildfire_sim.evaluation`,
docs/validation.md; settings: ``configs/validation.yaml``). The parameters are those of a
calibrated file (``configs/calibrated.yaml`` by default); nothing is tuned here, and fire B is
run with fire A's parameters unchanged. Needs a built region and its observations
(``scripts/download_data.py <region> --observations``); listing the VIIRS overpasses queries
NASA's public catalogue, so it needs an internet connection even when every overpass is cached.

Writes, under ``--out`` (default ``docs/img/validation/``), with the prefix ``<region>_<rule>``
(``synthetic_<scenario>_<rule>`` for ``--synthetic``):

* ``_summary.json``: target, parameters (status, file and its SHA-256), ensemble, the runs that
  reached the grid edge, the metrics and the provenance (git commit, package version, command);
* ``_metrics.csv``: the metrics table with intervals;
* ``_runs.csv``: every run's seed, perturbation, edge time and scores;
* ``_detections.csv``: per VIIRS detection, its interval and the simulated arrival across runs;
* ``_convergence.csv``: the running mean of the burned area with its interval;
* ``_maps.npz``: burn probability, consensus and observed masks;
* ``_surface.csv``: the objective surface of the calibration (not for a placeholder);
* figures ``_burn_probability.png``, ``_arrival.png``, ``_convergence.png``, ``_surface.png``.

Every output says whether the parameters were ``calibrated``, ``placeholder`` or ``synthetic``;
anything but ``calibrated`` is printed as a warning and stamped on every figure. The exit code is
1 if the region is not built, the parameters do not fit, or every run reached the grid edge.

Modes::

    # fire A (calibrated on it) and fire B (not retuned)
    uv run --extra fast --extra geo python scripts/validate.py lake_2024 --workers 8
    uv run --extra fast --extra geo python scripts/validate.py silverado_2020 --workers 8
    # the Alexandridis rule, with a calibration of it on fire A
    uv run --extra fast --extra geo python scripts/validate.py lake_2024 --rule alexandridis \\
        --calibrated configs/calibrated_alexandridis.yaml
    # the two rules side by side, from their summaries
    uv run python scripts/validate.py --compare docs/img/validation/lake_2024_rothermel_summary.json \\
        docs/img/validation/lake_2024_alexandridis_summary.json
    # a synthetic observed fire (a test of the pipeline, labelled synthetic everywhere)
    uv run python scripts/validate.py --synthetic crop.yaml --weather tests/fixtures/lake_2024_weather.yaml \\
        --truth ros_multiplier=1.2 --truth lb_scale=1.1 --window-hours 2 --runs 8 --out /tmp/val
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from wildfire_sim import __version__
from wildfire_sim.calibration import CalibratedParams, load_calibrated, load_calibration_settings, real_fire
from wildfire_sim.calibration import synthetic_target as make_synthetic_target
from wildfire_sim.config import ScenarioConfig, WeatherConfig, configs_dir, load_scenario, load_yaml
from wildfire_sim.data import build_grid
from wildfire_sim.data.cache import load_regions
from wildfire_sim.ensemble import mean_ci, running_mean
from wildfire_sim.evaluation import (
    DetectionSummary,
    MetricRow,
    Observed,
    ParameterRecord,
    ValidationRuns,
    check_not_tuned_on,
    check_seeds_disjoint,
    detection_summaries,
    load_validation_settings,
    metric_rows,
    observed_fire,
    run_validation,
    synthetic_observed,
)
from wildfire_sim.sensitivity import available_engine, ignition_cell
from wildfire_sim.sim import model_provenance
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.viz import PROBABILITY_CMAP

OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "validation"
SYNTHETIC_STAMP = "SYNTHETIC OBSERVED FIRE - NOT A VALIDATION RESULT"
# Okabe & Ito (2008) colour-blind-safe palette, as in wildfire_sim.viz.
VERMILLION, SKY_BLUE, PURPLE, GREEN, BLUE = "#D55E00", "#56B4E9", "#CC79A7", "#009E73", "#0072B2"


# ----------------------------------------------------------------------------- helpers


def _truth(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--truth takes NAME=VALUE, got {item!r}")
        out[name.strip()] = float(value)
    return out


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else ("inf" if math.isinf(value) else f"{value:.6g}")


def _json_value(value: Any) -> Any:
    """NaN as null and infinities as strings, so the file is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else str(value)
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _hours(t_s: float, observed: Observed) -> float:
    return (t_s - observed.ignition_s) / 3600.0


def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _title(fig: Figure, ax: Axes, title: str, stamps: Sequence[str]) -> None:
    """Title the axes (wrapped); the stamps, one per line, are the figure's title, in bold vermillion."""
    ax.set_title(textwrap.fill(title, 90), fontsize=9)
    if stamps:
        fig.suptitle("\n".join(stamps), fontsize=10, weight="bold", color=VERMILLION)


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=110, metadata={"Software": None})


# ----------------------------------------------------------------------------- tables


def write_metrics(rows: Sequence[MetricRow], path: Path) -> None:
    """Write the metrics table (value and interval in ``units``)."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["metric", "comparison", "statistic", "value", "lower", "upper", "n_runs", "units", "note"])
        for r in rows:
            writer.writerow(
                [
                    r.metric,
                    r.comparison,
                    r.statistic,
                    _fmt(r.value),
                    _fmt(r.lower),
                    _fmt(r.upper),
                    r.n_runs,
                    r.units,
                    r.note,
                ]
            )


def write_runs(result: ValidationRuns, observed: Observed, path: Path) -> None:
    """Write every run: seed, perturbation, edge time (s and h after the ignition) and scores."""
    days = [f"area_ratio_{_hours(t, observed):g}h" for t, _ in observed.daily]
    keys = list(result.runs[0].perturbation)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            ["run", "seed", *keys, "reached_edge_s", "reached_edge_h", "burned_area_ha", "jaccard_window"]
            + ["sorensen_window", "jaccard_final", "sorensen_final", "arrival_inside_fraction"]
            + ["arrival_error_median_h", "arrival_error_mean_h", *days, "lb_capped_cell_hours", "breaches"]
        )
        for i, r in enumerate(result.runs):
            writer.writerow(
                [i, r.seed, *(_fmt(r.perturbation[k]) for k in keys), _fmt(r.reached_edge_s)]
                + [_fmt(_hours(r.reached_edge_s, observed)), _fmt(r.burned_area_ha), _fmt(r.jaccard_window)]
                + [_fmt(r.sorensen_window), _fmt(r.jaccard_final), _fmt(r.sorensen_final), _fmt(r.inside_fraction)]
                + [_fmt(r.median_error_h), _fmt(r.mean_error_h), *(_fmt(a) for a in r.area_ratio)]
                + [_fmt(r.lb_capped_cell_hours), r.breaches]
            )


def write_detections(summaries: Sequence[DetectionSummary], observed: Observed, path: Path) -> None:
    """Write each detection's interval and the simulated arrival across runs, h after the ignition."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            ["row", "col", "t_clear_h", "t_detect_h", "sim_q05_h", "sim_median_h", "sim_q95_h"]
            + ["reached_fraction", "median_error_h"]
        )
        for s in summaries:
            d = s.detection
            writer.writerow(
                [d.row, d.col, _fmt(_hours(d.t_clear, observed)), _fmt(_hours(d.t_detect, observed))]
                + [_fmt(_hours(s.q05, observed)), _fmt(_hours(s.median, observed)), _fmt(_hours(s.q95, observed))]
                + [_fmt(s.reached_fraction), _fmt(s.median_error_s / 3600.0)]
            )


def convergence(areas: Sequence[float], level: float) -> list[tuple[int, float, float, float, float]]:
    """``(runs, area, running mean, lower, upper)`` per run, ha; the interval (Student t) from 2 runs on."""
    means = running_mean(np.asarray(areas))
    out = []
    for i, (area, mean) in enumerate(zip(areas, means, strict=True)):
        lo, hi = (math.nan, math.nan) if i == 0 else mean_ci(np.asarray(areas[: i + 1]), level)[1:]
        out.append((i + 1, float(area), float(mean), lo, hi))
    return out


def write_convergence(rows: Sequence[tuple[int, float, float, float, float]], path: Path) -> None:
    """Write :func:`convergence` (runs inside the grid, in seed order)."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["runs", "burned_area_ha", "running_mean_ha", "lower_ha", "upper_ha"])
        for n, area, mean, lo, hi in rows:
            writer.writerow([n, _fmt(area), _fmt(mean), _fmt(lo), _fmt(hi)])


def surface_rows(record: CalibratedParams) -> list[dict[str, Any]] | None:
    """The calibration's objective surface as rows, or None if the file has no surface."""
    if record.provenance is None or not record.provenance.surface:
        return None
    return [
        {
            **p.params,
            "objective": p.objective,
            "overlap": p.overlap,
            "arrival": p.arrival,
            "rejected": p.objective is None,
        }
        for p in record.provenance.surface
    ]


def write_surface(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write :func:`surface_rows` (parameters, objective and its two terms, dimensionless; empty if rejected)."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) if isinstance(v, float) else ("" if v is None else v) for k, v in row.items()})


# ----------------------------------------------------------------------------- figures


def _cell_xy(rows: Sequence[int], cols: Sequence[int], shape: tuple[int, int], dx: float) -> tuple[Any, Any]:
    """Cell centres in km from the south-west corner."""
    return (np.asarray(cols) + 0.5) * dx / 1000.0, (shape[0] - np.asarray(rows) - 0.5) * dx / 1000.0


def plot_burn_probability(prob: np.ndarray, observed: Observed, title: str, stamps: Sequence[str], path: Path) -> None:
    """Burn probability of the runs inside the grid with the observed perimeters, detections and ignition."""
    h, w = prob.shape
    dx = observed.dx
    fig = _figure(8.0, 7.5)
    ax = fig.add_subplot()
    image = ax.imshow(
        prob,
        cmap=PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        extent=(0.0, w * dx / 1000.0, 0.0, h * dx / 1000.0),
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, shrink=0.8, label="burn probability (fraction of runs inside the grid)")
    xs, ys = (np.arange(w) + 0.5) * dx / 1000.0, (h - np.arange(h) - 0.5) * dx / 1000.0
    handles = []
    if observed.window_mask is not None:
        ax.contour(xs, ys, observed.window_mask.astype(float), levels=[0.5], colors=VERMILLION, linewidths=1.6)
        label = f"observed burned by the window end ({observed.window_hours:g} h)"
        handles.append(Line2D([], [], color=VERMILLION, linewidth=1.6, label=label))
    ax.contour(
        xs, ys, observed.final_mask.astype(float), levels=[0.5], colors=SKY_BLUE, linewidths=1.4, linestyles="--"
    )
    handles.append(Line2D([], [], color=SKY_BLUE, linewidth=1.4, linestyle="--", label=observed.final_label))
    if observed.detections:
        px, py = _cell_xy([d.row for d in observed.detections], [d.col for d in observed.detections], (h, w), dx)
        ax.scatter(px, py, s=14, facecolors="none", edgecolors=PURPLE, linewidths=0.9)
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor=PURPLE,
                label="VIIRS first detections",
            )
        )
    ix, iy = _cell_xy([observed.ignition_cell[0]], [observed.ignition_cell[1]], (h, w), dx)
    ax.plot(ix, iy, marker="*", markersize=14, color="white", markeredgecolor="black", linestyle="none")
    handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=12,
            color="white",
            markeredgecolor="black",
            label="ignition",
        )
    )
    fig.legend(handles=handles, loc="outside lower center", fontsize=8, ncols=2)
    ax.set_xlabel("easting from west edge (km)")
    ax.set_ylabel("northing from south edge (km)")
    ax.set_aspect("equal")
    _title(fig, ax, title, stamps)
    _save(fig, path)


def plot_arrival(
    summaries: Sequence[DetectionSummary], observed: Observed, title: str, stamps: Sequence[str], path: Path
) -> None:
    """Simulated arrival against each detection's interval ``(t_clear, t_detect]``, h after the ignition.

    Each detection is a horizontal segment spanning its interval at the height of the median
    simulated arrival, with a vertical bar from the 5 % to the 95 % quantile over the runs; the
    simulated arrival lies in the interval where the segment crosses the diagonal. A median later
    than the window end is drawn at the window end as a triangle.
    """
    end = observed.window_hours
    fig = _figure(7.5, 7.0)
    ax = fig.add_subplot()
    ax.plot([0, end], [0, end], color="0.5", linewidth=1, linestyle="--")
    for s in summaries:
        d = s.detection
        x0, x1 = _hours(d.t_clear, observed), _hours(d.t_detect, observed)
        reached = math.isfinite(s.median)
        y = min(_hours(s.median, observed), end) if reached else end
        colour = GREEN if s.median_error_s == 0 else VERMILLION
        ax.plot([x0, x1], [y, y], color=colour, linewidth=1.4, alpha=0.8)
        lo = _hours(s.q05, observed) if math.isfinite(s.q05) else end
        hi = _hours(s.q95, observed) if math.isfinite(s.q95) else end
        ax.plot([x1, x1], [min(lo, end), min(hi, end)], color=colour, linewidth=0.7, alpha=0.6)
        ax.plot(x1, y, marker="o" if reached else "^", markersize=4, color=colour, linestyle="none")
    handles = [
        Line2D([], [], color=GREEN, marker="o", markersize=4, label="median simulated arrival inside the interval"),
        Line2D([], [], color=VERMILLION, marker="o", markersize=4, label="median simulated arrival outside it"),
        Line2D(
            [],
            [],
            color="black",
            marker="^",
            markersize=5,
            linestyle="none",
            label="median run not there by the window end (drawn at it)",
        ),
        Line2D([], [], color="0.5", linestyle="--", label="simulated = observed"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_xlim(0, end * 1.02)
    ax.set_ylim(0, end * 1.05)
    ax.set_xlabel("observed arrival interval, from last clear to detecting overpass (h after ignition)")
    ax.set_ylabel("simulated arrival in the footprint, median and 5-95 % of runs (h after ignition)")
    ax.grid(color="0.9", linewidth=0.6)
    _title(fig, ax, title, stamps)
    _save(fig, path)


def plot_convergence(
    rows: Sequence[tuple[int, float, float, float, float]],
    rel: float,
    needed: int | None,
    title: str,
    stamps: Sequence[str],
    path: Path,
) -> None:
    """Running mean of the burned area with its interval and the target precision band."""
    n = np.array([r[0] for r in rows])
    mean, lo, hi = (np.array([r[k] for r in rows]) for k in (2, 3, 4))
    fig = _figure(8.0, 4.8)
    ax = fig.add_subplot()
    ax.fill_between(n, lo, hi, color=SKY_BLUE, alpha=0.4, linewidth=0, label="interval of the running mean")
    ax.plot(n, mean, color=BLUE, linewidth=1.5, label="running mean")
    final = float(mean[-1])
    ax.axhspan(final * (1 - rel), final * (1 + rel), color="0.85", zorder=0, label=f"final mean +- {rel:.0%}")
    ax.set_xlabel("runs inside the grid")
    ax.set_ylabel("burned area at the window end (ha)")
    ax.grid(color="0.9", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8)
    note = "" if needed is None else f"; runs needed for +- {rel:.0%}: {needed}"
    _title(fig, ax, f"{title}{note}", stamps)
    _save(fig, path)


def plot_surface(record: CalibratedParams, rows: Sequence[dict[str, Any]], stamps: Sequence[str], path: Path) -> bool:
    """The calibration's objective on its grid around the optimum (False, nothing drawn, unless two parameters)."""
    assert record.provenance is not None
    names = list(record.params)
    if len(names) != 2:
        return False
    fig = _figure(7.0, 5.8)
    ax = fig.add_subplot()
    scored = [r for r in rows if not r["rejected"]]
    rejected = [r for r in rows if r["rejected"]]
    if scored:
        points = ax.scatter(
            [r[names[0]] for r in scored],
            [r[names[1]] for r in scored],
            c=[r["objective"] for r in scored],
            cmap="viridis",
            marker="s",
            s=260,
        )
        fig.colorbar(points, ax=ax, label="objective (dimensionless)")
    if rejected:
        ax.scatter(
            [r[names[0]] for r in rejected],
            [r[names[1]] for r in rejected],
            marker="x",
            color="black",
            s=60,
            label="rejected (a run reached the edge)",
        )
    ax.plot(
        record.params[names[0]],
        record.params[names[1]],
        marker="*",
        markersize=16,
        color=VERMILLION,
        markeredgecolor="black",
        linestyle="none",
        label="optimum",
    )
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    fig.legend(loc="outside lower center", fontsize=8, ncols=2)
    p = record.provenance
    _title(fig, ax, f"Calibration objective around the optimum ({p.rule}): {p.target}, {len(p.seeds)} seeds", stamps)
    _save(fig, path)
    return True


def parameter_problem(record: CalibratedParams, path: Path, rule: str, test_region: str | None) -> str | None:
    """Why ``record`` cannot be used for a run with ``rule`` (None if it can).

    ``test_region`` is the region of a fire tested without retuning (role B): parameters
    calibrated on it are refused.
    """
    if record.rule != rule:
        return (
            f"{path} holds parameters of rule {record.rule!r}, the run uses {rule!r}. Calibrate that rule on fire A "
            "first, e.g.\n  uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024 "
            f"--rule {rule} --out configs/calibrated_{rule}.yaml\nand pass --calibrated configs/calibrated_{rule}.yaml"
        )
    if test_region is not None:
        try:
            check_not_tuned_on(record, test_region)
        except ValueError as exc:
            return str(exc)
    return None


# ----------------------------------------------------------------------------- comparison of two rules


def compare(paths: Sequence[Path], out: Path) -> int:
    """Side-by-side metrics of two validations of the same target with different rules.

    Writes ``<name>_rule_comparison.csv`` (one row per metric present in both, value and interval
    per rule) and ``<name>_rule_comparison.json`` (both summaries' target, parameters and edge
    counts). Returns 1 if the summaries are not of the same target or use the same rule.
    """
    summaries = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    a, b = summaries
    if a["target"] != b["target"]:
        print("the two summaries are not of the same target and window: nothing compared", file=sys.stderr)
        return 1
    rules = [s["parameters"]["rule"] for s in summaries]
    if rules[0] == rules[1]:
        print(f"both summaries use the rule {rules[0]!r}: nothing to compare", file=sys.stderr)
        return 1
    key = ("metric", "comparison", "statistic")
    tables = [{tuple(r[k] for k in key): r for r in s["metrics"]} for s in summaries]
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a['name']}_rule_comparison"
    header = [*key, "units"] + [f"{rule}_{col}" for rule in rules for col in ("value", "lower", "upper", "n_runs")]
    lines = []
    for k, row in tables[0].items():
        if k in tables[1]:
            cells = [*k, row["units"]]
            for t in tables:
                cells += ["nan" if t[k][c] is None else t[k][c] for c in ("value", "lower", "upper")] + [t[k]["n_runs"]]
            lines.append(cells)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(lines)
    record = {
        "target": a["target"],
        "synthetic": a["synthetic"],
        "rules": {
            s["parameters"]["rule"]: {"summary": str(p), "parameters": s["parameters"], "edge": s["edge"]}
            for s, p in zip(summaries, paths, strict=True)
        },
    }
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if a["synthetic"]:
        print(SYNTHETIC_STAMP)
    for s in summaries:
        p = s["parameters"]
        print(f"{p['rule']}: parameters {p['status']} ({p['path']}); {s['edge']['n_reached']} runs reached the edge")
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |" * len(header))
    for cells in lines:
        print("| " + " | ".join(str(c) for c in cells) + " |")
    print(f"wrote {stem.with_suffix('.csv')} and {stem.with_suffix('.json')}")
    return 0


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    regions = load_regions().regions
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("region", nargs="?", choices=sorted(regions), help="built region to validate on")
    ap.add_argument("--synthetic", type=Path, metavar="SCENARIO", help="scenario YAML of a synthetic observed fire")
    ap.add_argument("--weather", type=Path, help="weather YAML replacing the synthetic scenario's weather")
    ap.add_argument("--truth", action="append", default=[], metavar="NAME=VALUE", help="known parameter (synthetic)")
    ap.add_argument("--compare", type=Path, nargs=2, metavar="SUMMARY", help="compare two rules' summary JSON files")
    ap.add_argument("--rule", default="rothermel", help="spread rule for a region (default rothermel)")
    ap.add_argument("--calibrated", type=Path, help="calibrated parameters (default configs/calibrated.yaml)")
    ap.add_argument("--window-hours", type=float, help="comparison window after the ignition, h")
    ap.add_argument("--runs", type=int, help="ensemble runs (default from configs/validation.yaml)")
    ap.add_argument("--workers", type=int, default=1, help="processes; results do not depend on it")
    ap.add_argument("--dt", type=float, help="time step, s (default from configs/validation.yaml)")
    ap.add_argument("--cache-root", type=Path, default=Path("data") / "cache")
    ap.add_argument("--settings", type=Path, help="validation settings (default configs/validation.yaml)")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = ap.parse_args(argv)
    if sum([args.region is not None, args.synthetic is not None, args.compare is not None]) != 1:
        ap.error("give exactly one of a region, --synthetic SCENARIO or --compare SUMMARY SUMMARY")
    if args.synthetic is None and (args.truth or args.weather):
        ap.error("--truth and --weather apply to --synthetic only")
    if args.compare is not None:
        return compare(args.compare, args.out)
    if (args.runs is not None and args.runs < 1) or args.workers < 1:
        ap.error("--runs and --workers must be >= 1")

    settings = load_validation_settings(args.settings)
    cal = load_calibration_settings()
    footprint = float(cal.footprint_m.value)
    n_runs = int(settings.n_runs.value) if args.runs is None else args.runs
    calibrated_path = args.calibrated or configs_dir() / "calibrated.yaml"
    record = load_calibrated(calibrated_path)
    params = ParameterRecord.read(record, calibrated_path)

    used = {*cal.seeds(), int(cal.truth_seed.value), *(record.provenance.seeds if record.provenance else [])}
    try:
        check_seeds_disjoint(settings.seeds(n_runs), used)
    except ValueError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    role: str | None = None
    observed: Observed
    if args.synthetic is not None:
        if args.window_hours is None:
            ap.error("--synthetic needs --window-hours")
        try:
            truth = _truth(args.truth)
        except ValueError as exc:
            ap.error(str(exc))
        base = load_scenario(args.synthetic)
        data = base.model_dump()
        if args.weather is not None:
            data["weather"] = load_yaml(args.weather, WeatherConfig).model_dump()
        data["duration_s"] = base.ignition.time_s + args.window_hours * 3600.0
        data["engine"] = available_engine()
        if args.dt is not None:
            data["dt_s"] = args.dt
        config = ScenarioConfig.model_validate(data)
        rule = config.spread.rule
        if rule not in cal.parameters or set(truth) != set(cal.space(rule)):
            ap.error(f"--truth must give exactly the calibrated parameters of rule {rule!r}")
        problem = parameter_problem(record, calibrated_path, rule, None)
        if problem:
            print(f"validation failed: {problem}", file=sys.stderr)
            return 1
        grid = build_grid(config.grid)
        try:
            target, truth_run = make_synthetic_target(
                config,
                truth,
                seed=int(cal.truth_seed.value),
                overpass_interval_s=float(cal.overpass_interval_s.value),
                footprint_m=footprint,
                grid=grid,
            )
        except ValueError as exc:
            print(f"validation failed: {exc}", file=sys.stderr)
            return 1
        observed = synthetic_observed(target, truth_run.arrival_time, ignition_cell(config, grid.shape), grid.dx)
        name = "synthetic_" + re.sub(r"[^a-z0-9]+", "_", config.name.lower()).strip("_")
    else:
        rule, name = args.rule, args.region
        role = regions[name].role
        if name not in settings.window_hours and args.window_hours is None:
            ap.error(f"no window_hours for {name} in the settings: give --window-hours")
        problem = parameter_problem(record, calibrated_path, rule, name if role == "B" else None)
        if problem:
            print(f"validation failed: {problem}", file=sys.stderr)
            return 1
        hours = settings.window_hours[name].value if args.window_hours is None else args.window_hours
        dt_s = float(settings.dt_s.value) if args.dt is None else args.dt
        try:
            fire = real_fire(name, hours, cache_root=args.cache_root, dt_s=dt_s, rule=rule)
        except FileNotFoundError as exc:
            print(exc)
            return 1
        config = fire.config
        observed = observed_fire(fire, footprint)

    config = settings.with_ensemble(record.apply(config), n_runs)
    rule_caps = hasattr(make_spread_rule(rule, **config.spread.params), "lb_capped")

    stamps = [SYNTHETIC_STAMP] if observed.synthetic else []
    warning = params.warning()
    if warning is not None:
        stamps.append(warning)
        print(f"WARNING: {warning}: {params.source}", file=sys.stderr)
    if observed.synthetic:
        print(f"WARNING: {SYNTHETIC_STAMP}", file=sys.stderr)
    retune = None
    if role == "B":
        retune = (
            f"fire B is run with the parameters of {calibrated_path} unchanged (calibrated on: "
            f"{params.calibrated_on}); nothing is tuned"
        )
        print(retune)
    elif role == "A" and params.status == "calibrated" and not str(params.calibrated_on).startswith(f"region {name},"):
        print(
            f"WARNING: fire A's parameters were calibrated on {params.calibrated_on!r}, not on {name}", file=sys.stderr
        )
    for note in observed.notes:
        print(f"note: {note}")
    prefix = f"{name}_{rule}"
    print(
        f"{observed.description}: rule {rule} ({params.status} parameters {params.params}), {config.engine} engine, "
        f"{n_runs} runs, {len(observed.detections)} detections"
    )

    base_seed = int(settings.base_seed.value)
    result = run_validation(config, observed, n_runs=n_runs, base_seed=base_seed, n_workers=args.workers)
    rows = metric_rows(result, observed, settings, rule_caps=rule_caps)
    dets = detection_summaries(result, observed)
    edge = result.edge
    print(
        f"EDGE: {len(edge)} of {len(result.runs)} runs reached the grid edge (lower bounds, left out of every "
        "metric; milestone M2 requires 0)",
        file=sys.stderr if edge else sys.stdout,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    files = [f"{prefix}_summary.json"]

    def path(suffix: str) -> Path:
        files.append(f"{prefix}_{suffix}")
        return args.out / f"{prefix}_{suffix}"

    write_metrics(rows, path("metrics.csv"))
    write_runs(result, observed, path("runs.csv"))
    maps: dict[str, Any] = {"final_mask": observed.final_mask}
    if observed.window_mask is not None:
        maps["window_mask"] = observed.window_mask
    prob = result.burn_probability
    level = float(settings.ci_level.value)
    head = f"{observed.description}; {rule}"
    if prob is not None:
        maps["burn_probability"] = prob
        maps["consensus"] = prob >= float(settings.consensus_threshold.value)
        title = f"Burn probability, {head}, {len(result.inside)} runs inside the grid"
        plot_burn_probability(prob, observed, title, stamps, path("burn_probability.png"))
    np.savez_compressed(path("maps.npz"), **maps)
    if dets:
        write_detections(dets, observed, path("detections.csv"))
        plot_arrival(dets, observed, f"Arrival at VIIRS detections, {head}", stamps, path("arrival.png"))
    needed = next((int(r.value) for r in rows if r.metric == "runs_for_precision"), None)
    areas = [r.burned_area_ha for r in result.inside]
    if len(areas) >= 2:
        conv = convergence(areas, level)
        write_convergence(conv, path("convergence.csv"))
        rel = float(settings.rel_half_width.value)
        plot_convergence(conv, rel, needed, f"Convergence of the burned area, {head}", stamps, path("convergence.png"))
    surface = surface_rows(record)
    surface_note = None
    if surface is None:
        surface_note = f"no objective surface: {calibrated_path} is a placeholder (produced by no study)"
    else:
        write_surface(surface, path("surface.csv"))
        if plot_surface(record, surface, stamps, args.out / f"{prefix}_surface.png"):
            files.append(f"{prefix}_surface.png")
        else:
            surface_note = "objective surface not drawn: it is drawn for two calibrated parameters only"
    if surface_note:
        print(f"note: {surface_note}")

    dx2_ha = observed.dx**2 / 1e4
    summary = {
        "name": name,
        "synthetic": observed.synthetic,
        "label": SYNTHETIC_STAMP if observed.synthetic else "validation",
        "warnings": stamps,
        "target": {
            "description": observed.description,
            "region": None if observed.synthetic else name,
            "role": role,
            "window_hours": observed.window_hours,
            "ignition_s": observed.ignition_s,
            "window_end_s": observed.window_end_s,
            "start_time_utc": None if config.start_time_utc is None else config.start_time_utc.isoformat(),
            "ignition_cell": list(observed.ignition_cell),
            "footprint_m": observed.footprint_m,
            "n_detections": len(observed.detections),
            "observed_window_ha": None if observed.window_mask is None else float(observed.window_mask.sum()) * dx2_ha,
            "observed_final_ha": float(observed.final_mask.sum()) * dx2_ha,
            "final_label": observed.final_label,
            "notes": list(observed.notes),
        },
        "parameters": {**params.__dict__, "retuned": False, "note": retune},
        "ensemble": {
            "n_runs": n_runs,
            "base_seed": base_seed,
            "seeds": list(settings.seeds(n_runs)),
            "sigmas": settings.ensemble.values(),
            "ignition_perturbed": False,
            "consensus_threshold": float(settings.consensus_threshold.value),
            "ci_level": level,
            "engine": config.engine,
            "dt_s": config.dt_s,
        },
        "edge": {
            "n_reached": len(edge),
            "n_runs": len(result.runs),
            "m2_no_run_reaches_edge": not edge,
            "runs": [
                {
                    "run": i,
                    "seed": r.seed,
                    "reached_edge_s": r.reached_edge_s,
                    "hours_after_ignition": _hours(r.reached_edge_s, observed),
                }
                for i, r in edge
            ],
        },
        "metrics": [r.__dict__ for r in rows],
        "surface_note": surface_note,
        "files": sorted(files),
        "provenance": {
            "package_version": __version__,
            "command": shlex.join(["scripts/validate.py", *(sys.argv[1:] if argv is None else argv)]),
            "model": model_provenance(config),
        },
    }
    (args.out / f"{prefix}_summary.json").write_text(
        json.dumps(_json_value(summary), indent=2) + "\n", encoding="utf-8"
    )

    if observed.synthetic:
        print(SYNTHETIC_STAMP)
    print("| metric | comparison | statistic | value | interval | runs | units |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        interval = "" if math.isnan(r.lower) else f"[{r.lower:.4g}, {r.upper:.4g}]"
        print(f"| {r.metric} | {r.comparison} | {r.statistic} | {r.value:.4g} | {interval} | {r.n_runs} | {r.units} |")
    print(f"wrote {len(files)} files to {args.out} ({result.elapsed_s:.0f} s)")
    if not result.inside:
        print("validation failed: every run reached the grid edge, so nothing could be scored", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
