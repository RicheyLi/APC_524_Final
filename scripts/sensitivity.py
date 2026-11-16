"""Morris screening (WP 3.2): which inputs drive burned area and head arrival time.

Regenerates the CSVs and figures of ``docs/sensitivity.md`` under ``docs/img/sensitivity/``
from the study in ``configs/sensitivity.yaml`` (method: :mod:`wildfire_sim.sensitivity`). For
every scenario and run seed of the study it writes

* ``<scenario>_seed<s>_indices.csv``: ``mu``, ``mu*``, ``sigma`` (ha or s per unit of the
  normalised input), the number of valid elementary effects and the rank by ``mu*``, per output
  and input;
* ``<scenario>_seed<s>_runs.csv``: every run of the design, its input values and outputs, and
  whether its fire reached the grid edge within the window;

and, per scenario, ``<scenario>_mu_star.png`` (``mu*`` by input, bars for the first run seed,
ticks for the others) and ``<scenario>_mu_sigma.png`` (the Morris ``mu*``-``sigma`` plane of the
first run seed). It prints the indices as Markdown tables, the edge and undefined-output counts
and the run time. The Numba engine is used when it is installed, else NumPy.

Usage::

    uv run python scripts/sensitivity.py --workers 4     # the committed study
    uv run python scripts/sensitivity.py --scenario synthetic --trajectories 4 --out /tmp/sens
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from wildfire_sim.config import configs_dir
from wildfire_sim.sensitivity import (
    OUTPUTS,
    ScreeningResult,
    available_engine,
    flat_records,
    load_study,
    run_screening,
    summary_rows,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "sensitivity"
OUTPUT_LABELS = {"burned_area_ha": "burned area (ha)", "head_arrival_s": "head arrival time (s)"}
OUTPUT_UNITS = {"burned_area_ha": "ha", "head_arrival_s": "s"}
# Okabe & Ito (2008) colour-blind-safe palette, as in wildfire_sim.viz.
BAR_COLOUR = "#0072B2"  # blue
SEED_COLOUR = "#000000"


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.6g}"


def write_indices(result: ScreeningResult, path: Path) -> None:
    """Write :func:`~wildfire_sim.sensitivity.summary_rows` of ``result`` to ``path``."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["output", "input", "mu", "mu_star", "sigma", "n_valid", "rank", "n_edge"])
        for row in summary_rows(result):
            writer.writerow(
                [
                    row["output"],
                    row["input"],
                    _fmt(row["mu"]),
                    _fmt(row["mu_star"]),
                    _fmt(row["sigma"]),
                    row["n_valid"],
                    row["rank"],
                    row["n_edge"],
                ]
            )


def write_runs(result: ScreeningResult, path: Path) -> None:
    """Write every run of ``result``: indices, input values, outputs (ha, s) and edge flags."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            [
                "trajectory",
                "point",
                *result.input_names,
                *OUTPUTS,
                "reached_edge_s",
                "edge_in_window",
                "lb_capped_cell_hours",
                "breaches",
            ]
        )
        for t, j, values, rec in flat_records(result):
            inputs = [values[n] if isinstance(values[n], bool) else _fmt(float(values[n])) for n in result.input_names]
            writer.writerow(
                [
                    t,
                    j,
                    *inputs,
                    _fmt(rec.burned_area_ha),
                    _fmt(rec.head_arrival_s),
                    _fmt(rec.reached_edge_s),
                    rec.edge_in_window,
                    _fmt(rec.lb_capped_cell_hours),
                    rec.breaches,
                ]
            )


def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def plot_mu_star(results: list[ScreeningResult], inert: set[str], path: Path) -> None:
    """``mu*`` per input and output: bars for the first run seed, ticks for the other seeds."""
    first = results[0]
    k = len(first.input_names)
    fig = _figure(11, 0.32 * k + 1.6)
    axes = fig.subplots(1, len(OUTPUTS), squeeze=False)[0]
    for ax, out in zip(axes, OUTPUTS, strict=True):
        mu_star = first.indices[out].mu_star
        order = np.argsort(np.nan_to_num(mu_star, nan=-1.0), kind="stable")  # largest at the top
        names = [
            f"{first.input_names[i]} (inert)" if first.input_names[i] in inert else first.input_names[i] for i in order
        ]
        ypos = np.arange(k)
        ax.barh(ypos, mu_star[order], color=BAR_COLOUR, height=0.7)
        handles: list[Artist] = [Patch(color=BAR_COLOUR, label=f"seed {first.run_seed}")]
        for other in results[1:]:
            (tick,) = ax.plot(
                other.indices[out].mu_star[order], ypos, linestyle="none", marker="|", markersize=12, color=SEED_COLOUR
            )
            tick.set_label(f"seed {other.run_seed}")
            handles.append(tick)
        ax.set_yticks(ypos, names)
        ax.set_xlabel(f"mu* ({OUTPUT_UNITS[out]} per unit of normalised input)")
        ax.set_title(OUTPUT_LABELS[out])
        ax.grid(axis="x", color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.suptitle(f"Morris screening, {first.scenario}: mu* (r = {len(first.records)} trajectories)")
    fig.savefig(path, dpi=110, metadata={"Software": None})


def plot_mu_sigma(result: ScreeningResult, inert: set[str], path: Path) -> None:
    """The Morris plane: ``sigma`` against ``mu*`` per input on log axes, one panel per output.

    Inputs with ``mu* = 0`` cannot be drawn on log axes and are listed in the panel instead.
    """
    fig = _figure(11, 5.5)
    axes = fig.subplots(1, len(OUTPUTS), squeeze=False)[0]
    for ax, out in zip(axes, OUTPUTS, strict=True):
        ind = result.indices[out]
        shown = [i for i in range(len(result.input_names)) if ind.mu_star[i] > 0 and ind.sigma[i] > 0]
        zero = [result.input_names[i] for i in range(len(result.input_names)) if i not in shown]
        both = np.concatenate([ind.mu_star[shown], ind.sigma[shown]]) if shown else np.array([1.0])
        lo, hi = 0.5 * float(both.min()), 2.0 * float(both.max())
        ax.plot([lo, hi], [lo, hi], color="0.6", linewidth=1, linestyle="--", label="sigma = mu*")
        for i in shown:
            name = result.input_names[i]
            ax.plot(ind.mu_star[i], ind.sigma[i], marker="o", markersize=8, linestyle="none", color=BAR_COLOUR)
            ax.annotate(name, (ind.mu_star[i], ind.sigma[i]), xytext=(5, -3), textcoords="offset points", fontsize=7)
        if zero:
            ax.text(
                0.98,
                0.02,
                "mu* = 0:\n" + "\n".join(f"{n} (inert control)" if n in inert else n for n in zero),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="0.3",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"mu* ({OUTPUT_UNITS[out]})")
        ax.set_ylabel(f"sigma ({OUTPUT_UNITS[out]})")
        ax.set_title(OUTPUT_LABELS[out])
        ax.grid(color="0.9", linewidth=0.6, which="major")
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Morris screening, {result.scenario}, seed {result.run_seed}: mu* and sigma")
    fig.savefig(path, dpi=110, metadata={"Software": None})


def print_tables(results: list[ScreeningResult]) -> None:
    """Print ``mu*``, ``sigma`` and rank of every input as Markdown, the seeds side by side."""
    first = results[0]
    for out in OUTPUTS:
        print(f"\n{first.scenario}, {OUTPUT_LABELS[out]}")
        head = ["input"]
        for res in results:
            head += [f"mu* (seed {res.run_seed})", f"sigma (seed {res.run_seed})", f"rank (seed {res.run_seed})"]
        print("| " + " | ".join(head) + " |")
        print("|" + " --- |" * len(head))
        order = np.argsort(-np.nan_to_num(first.indices[out].mu_star, nan=-1.0), kind="stable")
        for i in order:
            cells = [first.input_names[i]]
            for res in results:
                ind = res.indices[out]
                cells += [f"{ind.mu_star[i]:.4g}", f"{ind.sigma[i]:.4g}", str(int(ind.rank[i]))]
            print("| " + " | ".join(cells) + " |")
        for res in results:
            n_valid = res.indices[out].n_valid
            if int(n_valid.min()) < len(res.records):
                missing = {
                    res.input_names[i]: len(res.records) - int(n) for i, n in enumerate(n_valid) if n < len(res.records)
                }
                print(f"seed {res.run_seed}: elementary effects left out (undefined output): {missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=configs_dir() / "sensitivity.yaml", help="study file")
    parser.add_argument("--scenario", action="append", help="scenario to run (repeatable); default all")
    parser.add_argument("--trajectories", type=int, help="override the study's number of trajectories")
    parser.add_argument("--workers", type=int, default=1, help="processes")
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args(argv)

    study = load_study(args.config)
    names = args.scenario or list(study.scenarios)
    unknown = [n for n in names if n not in study.scenarios]
    if unknown:
        parser.error(f"unknown scenario(s) {unknown}; the study has {sorted(study.scenarios)}")
    inert = {n for n, spec in study.inputs.items() if spec.inert}
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(f"engine {available_engine()}, p = {study.levels}, design seed {study.design_seed}")
    for name in names:
        results = []
        for seed in study.run_seeds:
            res = run_screening(study, name, seed, trajectories=args.trajectories, n_workers=args.workers)
            results.append(res)
            write_indices(res, args.out / f"{name}_seed{seed}_indices.csv")
            write_runs(res, args.out / f"{name}_seed{seed}_runs.csv")
            n_runs = sum(len(row) for row in res.records)
            undefined = int(np.isnan(res.output("head_arrival_s")).sum())
            print(
                f"{name}, seed {seed}: {n_runs} runs in {res.elapsed_s:.0f} s; "
                f"{res.n_edge_in_window} reached the grid edge within the window; "
                f"{undefined} never reached the head distance"
            )
        plot_mu_star(results, inert, args.out / f"{name}_mu_star.png")
        plot_mu_sigma(results[0], inert, args.out / f"{name}_mu_sigma.png")
        print_tables(results)
    print(f"\ntotal {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
