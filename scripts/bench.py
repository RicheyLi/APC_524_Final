"""Benchmark the fire engines: seconds per run, per engine and stencil (WP 1.7).

Workloads (flat uniform fuel, dx = 30 m, centre ignition, ``record_events=False``):

* ``reference``: 333 x 333 cells (about 10 km square), constant rule 0.2 m/s, dt = 30 s,
  1440 steps (12 h). The fire reaches the grid edge and goes out after about 1140 steps.
* ``wind``: the same grid with the Alexandridis rule (ros_base 0.2 m/s) and a 5 m/s west wind.
* ``small``: 101 x 101 cells, constant rule 0.2 m/s, dt = 30 s, 180 steps (``flat_no_wind``).

Each (workload, engine, stencil) is run ``--repeats`` times after one warm-up run; min and
median wall times are printed. The numba engine's JIT compile time is excluded from those
numbers and reported separately (first call on a tiny grid minus a warm call: compile, or
load from numba's on-disk cache when it is warm).

**Machine-independent regression check.** Absolute times differ between machines, so
``--check`` compares *normalised* times: each engine's median divided by a calibration loop
timed in the same process just before. The calibration matches the kind of work each engine
does, so the ratio is roughly stable across CPUs:

* ``numpy`` engine: :func:`calibrate_numpy`, a fixed loop of small-array NumPy operations (fancy
  indexing, divmod, comparisons, ``lexsort`` on ~1000-element arrays) dominated, like the
  engine, by per-call overhead rather than memory bandwidth;
* ``numba`` engine: :func:`calibrate_numba`, a compiled binary-heap sort of 200,000 floats
  (branchy scalar code with indirect loads, like the kernel's Dijkstra cascade).

A slowdown of the code under test raises its normalised time; a slower machine raises both
numerator and denominator. The numba/numpy ratio was rejected as the check quantity because
it hides a slowdown of the NumPy engine (the ratio improves) and couples two independent
engines. The check fails when ``normalised / baseline_normalised > 1 + tolerance`` for any
entry present in both. Regenerate the baseline (``--json scripts/bench_baseline.json``) when
a change is meant to alter speed, preferably from the CI job's uploaded ``bench.json``.

**A real fire over a window** (WP 3.1 sizes the calibration study from it). With
``--region NAME`` (a region built in ``--cache-root``, ignited at its recorded cell and time) or
``--grid FILE --weather FILE`` (a saved grid and its weather series, ignited at the centre at
``--ignition-time-s``), the ``"rothermel"`` rule with its default parameters is run for
``--hours`` after the ignition (16 neighbours, ``--dt`` 60 s) instead of the workloads above.
The engines' repeats are interleaved, so drift in the machine's speed affects each alike. The
time of one evaluation of the rule over the whole grid is reported too: it is paid every time
the hourly weather changes.

Usage::

    uv run --extra fast python scripts/bench.py                        # print the table
    uv run --extra fast python scripts/bench.py --json bench.json      # also write the numbers
    uv run --extra fast python scripts/bench.py --workloads reference --check scripts/bench_baseline.json
    uv run --extra fast --extra geo python scripts/bench.py --region lake_2024 --hours 24 --engines numba
    uv run --extra fast python scripts/bench.py --grid tests/fixtures/lake_2024_5km.npz \\
        --weather tests/fixtures/lake_2024_weather.yaml --ignition-time-s 56880 --hours 1
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_sim import sim
from wildfire_sim.config import CachedGridConfig, ScenarioConfig, WeatherConfig, load_yaml
from wildfire_sim.grid import Grid
from wildfire_sim.spread import make_spread_rule
from wildfire_sim.types import get_stencil

WORKLOADS: dict[str, dict[str, Any]] = {
    "reference": {"shape": (333, 333), "duration_s": 43200.0, "wind": False},
    "wind": {"shape": (333, 333), "duration_s": 43200.0, "wind": True},
    "small": {"shape": (101, 101), "duration_s": 5400.0, "wind": False},
}
ENGINES = ("numpy", "numba")
STENCILS = (8, 16)


def workload_config(name: str, engine: str, stencil: int) -> ScenarioConfig:
    """The scenario of workload ``name`` for ``engine`` and ``stencil`` (dt = 30 s, dx = 30 m)."""
    spec = WORKLOADS[name]
    data: dict[str, Any] = {
        "name": f"bench_{name}",
        "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": list(spec["shape"]), "dx": 30.0},
        "spread": {"rule": "constant", "params": {"ros": 0.2}},
        "ignition": {"center": True},
        "dt_s": 30.0,
        "duration_s": spec["duration_s"],
        "stencil": stencil,
        "engine": engine,
    }
    if spec["wind"]:
        data["spread"] = {"rule": "alexandridis", "params": {"ros_base": 0.2}}
        data["weather"] = {"series": [{"time_s": 0.0, "wind_speed": 5.0, "wind_direction": 270.0}]}
    return ScenarioConfig.model_validate(data)


def time_runs(config: ScenarioConfig, repeats: int) -> list[float]:
    """Wall time, s, of ``repeats`` runs of ``config`` after one untimed warm-up run."""
    sim.run(config, record_events=False)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        sim.run(config, record_events=False)
        times.append(time.perf_counter() - start)
    return times


def _median_time(func: Callable[[], object], repeats: int) -> float:
    func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def calibrate_numpy(repeats: int = 7) -> float:
    """Median time, s, of a fixed small-array NumPy loop (see module docstring)."""
    rng = np.random.default_rng(0)
    w = 333
    src = rng.integers(0, w * w, size=1000)
    state = rng.integers(0, 4, size=(w, w)).astype(np.int8)
    rate = rng.random((8, w * w))

    def loop() -> None:
        for k in range(2000):
            rows, cols = np.divmod(src, w)
            tr, tc = rows + (k % 3 - 1), cols + (k % 5 - 2)
            ok = (tr >= 0) & (tr < w) & (tc >= 0) & (tc < w) & (rate[k % 8, src] > 0)
            ok[ok] = state[tr[ok], tc[ok]] == 1
            idx = tr[ok] * w + tc[ok]
            np.lexsort((rate[0, idx], idx))

    return _median_time(loop, repeats)


def _heap_sort(values: np.ndarray) -> float:
    """Binary-heap sort of a copy of ``values``; returns a checksum (compiled by :func:`calibrate_numba`)."""
    a = values.copy()
    n = a.shape[0]
    for start in range(n // 2 - 1, -1, -1):
        i = start
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            if child + 1 < n and a[child + 1] > a[child]:
                child += 1
            if a[child] <= a[i]:
                break
            a[i], a[child] = a[child], a[i]
            i = child
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        i = 0
        while True:
            child = 2 * i + 1
            if child >= end:
                break
            if child + 1 < end and a[child + 1] > a[child]:
                child += 1
            if a[child] <= a[i]:
                break
            a[i], a[child] = a[child], a[i]
            i = child
    return float(a[0] + a[n - 1])


def calibrate_numba(repeats: int = 7) -> float:
    """Median time, s, of a compiled heap sort of 200,000 floats (compile excluded)."""
    numba = importlib.import_module("numba")
    kernel = numba.njit(cache=False)(_heap_sort)
    values = np.random.default_rng(0).random(200_000)
    return _median_time(lambda: kernel(values), repeats)


def numba_compile_time() -> float:
    """JIT compile (or cache load) time of the numba engine, s: first tiny run minus a warm one."""
    data = workload_config("small", "numba", 8).model_dump()
    data["grid"]["shape"] = [15, 15]
    data["duration_s"] = 300.0
    tiny = ScenarioConfig.model_validate(data)
    start = time.perf_counter()
    sim.run(tiny, record_events=False)
    first = time.perf_counter() - start
    start = time.perf_counter()
    sim.run(tiny, record_events=False)
    return first - (time.perf_counter() - start)


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        info["numba"] = importlib.import_module("numba").__version__
    except ImportError:
        info["numba"] = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["processor"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def run_benchmarks(workloads: Sequence[str], engines: Sequence[str], stencils: Sequence[int], repeats: int) -> dict:
    """Time every combination; returns the JSON-serializable report."""
    report: dict[str, Any] = {"machine": machine_info(), "repeats": repeats, "calibration_s": {}, "results": []}
    if "numba" in engines:
        report["numba_compile_s"] = numba_compile_time()
    for engine in engines:
        calibration = calibrate_numpy() if engine == "numpy" else calibrate_numba()
        report["calibration_s"][engine] = calibration
        for workload in workloads:
            for stencil in stencils:
                times = time_runs(workload_config(workload, engine, stencil), repeats)
                median = statistics.median(times)
                report["results"].append(
                    {
                        "workload": workload,
                        "engine": engine,
                        "stencil": stencil,
                        "min_s": min(times),
                        "median_s": median,
                        "normalised": median / calibration,
                    }
                )
    return report


def fire_config(
    grid_path: Path, weather: WeatherConfig, cell: tuple[int, int] | None, ignition_s: float, hours: float, dt_s: float
) -> ScenarioConfig:
    """A fire on a saved grid for ``hours`` (h) after its ignition at ``ignition_s`` (s) at ``cell``
    (None: the grid centre); ``"rothermel"`` rule, 16 neighbours, time step ``dt_s`` (s)."""
    ignition: dict[str, Any] = {"center": True} if cell is None else {"cells": [list(cell)]}
    return ScenarioConfig.model_validate(
        {
            "name": f"bench_fire_{grid_path.stem}",
            "grid": {"kind": "cached", "path": str(grid_path)},
            "weather": weather.model_dump(),
            "spread": {"rule": "rothermel"},
            "ignition": {**ignition, "time_s": ignition_s},
            "dt_s": dt_s,
            "duration_s": ignition_s + hours * 3600.0,
            "stencil": 16,
        }
    )


def run_fire_benchmark(config: ScenarioConfig, engines: Sequence[str], repeats: int) -> dict:
    """Wall time, s, of ``config`` per engine (one warm-up each, then ``repeats`` interleaved rounds),
    and of one evaluation of the rule over the grid with the weather at the ignition."""
    assert isinstance(config.grid, CachedGridConfig)
    grid = Grid.load(config.grid.path)
    configs = {e: config.model_copy(update={"engine": e}) for e in engines}
    times: dict[str, list[float]] = {e: [] for e in engines}
    last: dict[str, sim.RunResult] = {}
    for e in engines:
        sim.run(configs[e], grid=grid, record_events=False)
    for _ in range(repeats):
        for e in engines:
            start = time.perf_counter()
            last[e] = sim.run(configs[e], grid=grid, record_events=False)
            times[e].append(time.perf_counter() - start)
    rule = make_spread_rule("rothermel")
    weather = config.weather.to_series().at(config.ignition.time_s)
    rule_s = _median_time(lambda: rule.rate_of_spread(grid, weather, get_stencil(16)), repeats)
    report: dict[str, Any] = {
        "machine": machine_info(),
        "repeats": repeats,
        "fire": {
            "grid": str(config.grid.path),
            "shape": list(grid.shape),
            "hours": (config.duration_s - config.ignition.time_s) / 3600.0,
            "dt_s": config.dt_s,
            "rule_evaluation_s": rule_s,
        },
        "results": [],
    }
    for e in engines:
        edge = last[e].provenance["reached_edge_s"]
        report["results"].append(
            {
                "workload": "fire",
                "engine": e,
                "stencil": 16,
                "min_s": min(times[e]),
                "median_s": statistics.median(times[e]),
                "burned_cells": int(last[e].burned_mask.sum()),
                "reached_edge_h": None if edge is None else (edge - config.ignition.time_s) / 3600.0,
            }
        )
    return report


def print_fire_report(report: dict) -> None:
    m, f = report["machine"], report["fire"]
    print(f"machine: {m['processor']} | {m['platform']}")
    print(f"python {m['python']}, numpy {m['numpy']}, numba {m['numba']}; {report['repeats']} interleaved repeats")
    print(
        f"{f['grid']}: {f['shape'][0]} x {f['shape'][1]} cells, {f['hours']:g} h after the ignition, dt {f['dt_s']:g} s"
    )
    print(f"one evaluation of the rothermel rule over the grid: {f['rule_evaluation_s']:.3f} s")
    print(f"{'engine':6s} {'min s':>8s} {'median s':>9s} {'burned cells':>13s} {'reached edge':>13s}")
    for r in report["results"]:
        edge = "no" if r["reached_edge_h"] is None else f"{r['reached_edge_h']:.1f} h"
        print(f"{r['engine']:6s} {r['min_s']:8.3f} {r['median_s']:9.3f} {r['burned_cells']:13d} {edge:>13s}")


def print_report(report: dict) -> None:
    m = report["machine"]
    print(f"machine: {m['processor']} | {m['platform']}")
    print(f"python {m['python']}, numpy {m['numpy']}, numba {m['numba']}; repeats {report['repeats']} (+1 warm-up)")
    for engine, cal in report["calibration_s"].items():
        print(f"calibration loop ({engine}): {cal * 1e3:.2f} ms")
    if "numba_compile_s" in report:
        print(f"numba engine compile / cache load: {report['numba_compile_s']:.2f} s (excluded below)")
    print(f"{'workload':10s} {'engine':6s} {'stencil':>7s} {'min s':>8s} {'median s':>9s} {'normalised':>11s}")
    by_key = {}
    for r in report["results"]:
        by_key[(r["workload"], r["engine"], r["stencil"])] = r
        print(
            f"{r['workload']:10s} {r['engine']:6s} {r['stencil']:7d} {r['min_s']:8.3f} {r['median_s']:9.3f} "
            f"{r['normalised']:11.2f}"
        )
    for (workload, engine, stencil), r in by_key.items():
        if engine == "numba" and (workload, "numpy", stencil) in by_key:
            speedup = by_key[(workload, "numpy", stencil)]["median_s"] / r["median_s"]
            print(f"speed-up numba vs numpy, {workload}, stencil {stencil}: {speedup:.1f}x")


def check(report: dict, baseline: dict, tolerance: float) -> list[str]:
    """Entries whose normalised median exceeds the baseline's by more than ``tolerance`` (fraction)."""
    base = {(r["workload"], r["engine"], r["stencil"]): r for r in baseline["results"]}
    failures = []
    compared = 0
    for r in report["results"]:
        key = (r["workload"], r["engine"], r["stencil"])
        if key not in base:
            continue
        compared += 1
        ratio = r["normalised"] / base[key]["normalised"]
        status = "FAIL" if ratio > 1.0 + tolerance else "ok"
        print(
            f"check {key}: normalised {r['normalised']:.2f} vs baseline "
            f"{base[key]['normalised']:.2f} -> {ratio:.2f} {status}"
        )
        if status == "FAIL":
            failures.append(f"{key}: {ratio:.2f}x the baseline (limit {1 + tolerance:.2f}x)")
    if compared == 0:
        failures.append("no benchmark entry matches the baseline")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workloads", default="reference,small", help=f"comma-separated, from {sorted(WORKLOADS)}")
    parser.add_argument("--engines", default=",".join(ENGINES), help="comma-separated; unavailable ones are skipped")
    parser.add_argument("--stencils", default="8,16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    parser.add_argument("--check", type=Path, help="baseline JSON; exit 1 on a slowdown beyond --tolerance")
    parser.add_argument("--tolerance", type=float, default=0.2, help="allowed slowdown, fraction (default 0.2)")
    parser.add_argument("--region", help="time a built region's fire instead of the workloads")
    parser.add_argument("--cache-root", type=Path, default=Path("data") / "cache", help="built regions")
    parser.add_argument("--grid", type=Path, help="time a fire on this saved grid instead of the workloads")
    parser.add_argument("--weather", type=Path, help="weather YAML of --grid")
    parser.add_argument("--ignition-time-s", type=float, default=0.0, help="ignition time on --grid, s")
    parser.add_argument("--hours", type=float, default=24.0, help="hours after the ignition for a fire")
    parser.add_argument("--dt", type=float, default=60.0, help="time step for a fire, s")
    args = parser.parse_args(argv)
    if args.region is not None and args.grid is not None:
        parser.error("give --region or --grid, not both")
    if (args.grid is None) != (args.weather is None):
        parser.error("--grid and --weather go together")

    workloads = [w for w in args.workloads.split(",") if w]
    unknown = set(workloads) - set(WORKLOADS)
    if unknown:
        parser.error(f"unknown workloads {sorted(unknown)}")
    engines = [e for e in args.engines.split(",") if e]
    if "numba" in engines and importlib.util.find_spec("numba") is None:
        print("numba is not installed: skipping the numba engine (uv sync --extra fast)", file=sys.stderr)
        engines.remove("numba")
    stencils = [int(s) for s in args.stencils.split(",") if s]

    if args.region is not None:
        from wildfire_sim.calibration import real_fire

        try:
            fire = real_fire(args.region, args.hours, cache_root=args.cache_root, dt_s=args.dt, viirs=False)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        assert isinstance(fire.config.grid, CachedGridConfig)
        cell = fire.config.ignition.cells[0]
        config = fire_config(fire.config.grid.path, fire.config.weather, cell, fire.ignition_s, args.hours, args.dt)
        report = run_fire_benchmark(config, engines, args.repeats)
        print_fire_report(report)
    elif args.grid is not None:
        weather = load_yaml(args.weather, WeatherConfig)
        config = fire_config(args.grid, weather, None, args.ignition_time_s, args.hours, args.dt)
        report = run_fire_benchmark(config, engines, args.repeats)
        print_fire_report(report)
    else:
        report = run_benchmarks(workloads, engines, stencils, args.repeats)
        print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.check:
        failures = check(report, json.loads(args.check.read_text(encoding="utf-8")), args.tolerance)
        for failure in failures:
            print(f"benchmark regression: {failure}", file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
