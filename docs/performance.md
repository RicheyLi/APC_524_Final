# Performance (WP 1.7)

Where a simulation run spends its time, the compiled (Numba) engine, measured seconds per run
for both engines, the M1 performance target, and what ensembles and the strategy search cost
at these speeds.

Regenerate the numbers with

```sh
uv run --extra fast python scripts/bench.py --workloads reference,wind,small --json bench.json
```

Units: times in s (wall clock), grid sizes in cells of 30 m; speed-ups and normalised times
are dimensionless.

## Summary

- **Reference run** (333 x 333 cells, 12 h at dt = 30 s): NumPy engine **1.1 s** (8 neighbours)
  and **2.5 s** (16); Numba engine **0.14 s** and **0.23 s**. Speed-up **8-11x** on the
  reference grid, 6-9x on the small 101 x 101 grid.
- **Same results.** For the same scenario and seed the Numba engine reproduces the NumPy
  engine **bit for bit** on every case tried (arrival times, final grid, events; 22 scenario /
  seed / stencil combinations plus the parity tests). The tests demand only 1e-9 relative.
- **Where the NumPy engine's time goes:** about 95 % in the per-step front update, of which
  link crossings 35-40 %, burn-out 25-28 % and the in-step cascade 16-19 %. These are many
  small NumPy calls on arrays of a few thousand front cells, so the time is per-call overhead,
  not arithmetic; there is no whole-grid work per step left to remove. Setup (threshold draw,
  YAML tables, rule evaluation) is 3-5 %.
- **Proposed M1 target** (section *M1 performance target*): reference run under 2 s (8
  neighbours) / 4 s (16) with the NumPy engine and under 0.3 s / 0.5 s with Numba; CI fails on
  a slowdown of more than 20 % in calibration-normalised time.
- **Cost:** an ensemble of 200 runs takes about 4 min (NumPy, 1 core) or 30 s (Numba, 1 core);
  the 15,000-run optimisation about 4.8 h (NumPy, 1 core) or 7 min (Numba, 8 workers).

## Machine

All numbers on one laptop: Intel Core Ultra 7 256V (4 performance + 4 efficiency cores, 8
logical CPUs), Windows 11 with WSL 2 (Ubuntu 26.04, Linux 6.18), the repository on the
Windows-mounted disk (`/mnt/c`). Python 3.12.16, NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.0.
Run-to-run noise on this machine is about 5-10 % (min and median below).

## Workloads

Flat uniform fuel model 1, dx = 30 m, ignition at the centre, `record_events=False`:

| Name | Grid | Rule | Steps | Note |
| --- | --- | --- | --- | --- |
| `reference` | 333 x 333 | constant, 0.2 m/s | 1440 x 30 s (12 h) | the plan's 10 km area; the whole grid burns and the fire goes out after ~1140 steps |
| `wind` | 333 x 333 | Alexandridis, ros_base 0.2 m/s, 5 m/s west wind | 1440 x 30 s | an elongated fire, ~28,000 cells burned |
| `small` | 101 x 101 | constant, 0.2 m/s | 180 x 30 s | `flat_no_wind` |

## Profile of the NumPy engine

`cProfile` and a hand-timed copy of `_FireEngine.advance` (a timer between each block), on
the reference workload. Percent of the whole run:

| Part of a run | constant, 8 | constant, 16 | wind, 8 | wind, 16 |
| --- | --- | --- | --- | --- |
| total run time | 0.91 s | 1.94 s | 0.86 s | 1.83 s |
| crossings from cells burning at the step start (`_crossings`) | 35.9 % | 39.8 % | 34.7 % | 39.0 % |
| burn-out (`_burn_out`; 85 % of it is the live-link test `_has_live_link`) | 26.1 % | 28.1 % | 25.4 % | 27.6 % |
| in-step cascade (`_cascade`, label correction) | 18.5 % | 15.8 % | 19.4 % | 17.4 % |
| progress update of the sources | 5.1 % | 6.6 % | 3.8 % | 4.9 % |
| ignition of new cells (`ignite`) | 4.6 % | 2.8 % | 4.3 % | 2.5 % |
| filter extinguished cells, time windows | 2.2 % | 1.3 % | 2.3 % | 1.4 % |
| new-cell progress, burning set | 1.8 % | 1.3 % | 1.5 % | 1.0 % |
| spread-rule evaluation + link rates (`refresh`; computed once, then a cache check per step) | 1.5 % | 1.1 % | 2.4 % | 2.2 % |
| threshold draw (`draw_crossing_thresholds`) | 1.4 % | 1.5 % | 1.9 % | 1.8 % |
| YAML tables (kernel correction, residence times), after the change below | 1.2 % | 0.6 % | 1.5 % | 0.6 % |
| rule construction, grid build | 0.2 % | 0.1 % | 1.2 % | 0.6 % |
| event recording (with `record_events=True`: +4-5 %) | 0.1 % | 0.1 % | 0.1 % | 0.1 % |

`cProfile` attributes about half of the run to one helper, `_targets` (the per-direction
neighbour lookup used by the crossings, the cascade and the live-link test): 27,000-68,000
calls of about 18 us each, on arrays of a few hundred to a few thousand cells. The engine is
already front-only, so the cost is the number of NumPy calls (K directions x a few calls x
several passes per step), not the grid size; vectorising further across directions would
trade it for larger temporaries. That is why a compiled per-cell loop is the right next step.

**Changes made to the NumPy engine** (results unchanged bit for bit):

1. `run` loaded `configs/kernel_correction.yaml` and `configs/residence_time.yaml` twice per
   run: once for the run and again inside `model_provenance` for the provenance record. The
   provenance is now built from the tables the run already loaded (private helper
   `_model_provenance`; the public `model_provenance` is unchanged). This removes 2 of 4 YAML
   parses per run, 10-20 ms on this disk: about 1.5 % of a reference run and 10 % of a
   `small` run, which matters for ensembles of small runs.

Proof that the results did not change: 22 cases (flat at 3 seeds, steady wind, river,
a fast fire crossing ten cells per step, delayed ignition, a mid-run weather change, a
fireline hook, `max_source_age_s`, wind on a slope; each for 8 and 16 neighbours) were run
with the unmodified `sim.py` from git `HEAD` and with the changed one. Arrival times, final
`state`, `burn_timer`, `intensity`, the grid's `arrival_time` and the event lists are
identical (`np.array_equal`, dtypes equal), with and without event recording. Hoisting the
repeated `divmod` out of `_targets` was also tried; it was bit-identical too, but its gain was
below the noise, so it was not kept.

## The Numba engine

`engine: numba` in a scenario (field `ScenarioConfig.engine`, default `"numpy"`) selects
`wildfire_sim.sim_fast.NumbaFireEngine`. It needs the `fast` extra (`uv sync --extra fast`);
without numba, `import wildfire_sim`, the NumPy engine and the CLI work as before, and
`engine: numba` raises an `ImportError` saying how to install it. The engine used is recorded
in the run provenance (`provenance["engine"]`, plus `versions["numba"]`).

**Design.** `sim.run` does all the setup for both engines: grid, thresholds from the same seed
streams, kernel correction, residence times, weather, hooks and the initial ignition. The
Numba engine subclasses the NumPy engine and keeps its spread-rule cache (`refresh`: the rule
output, link rates, the correction and the blocked-link mask, computed by the NumPy code). It
replaces only `advance`, the per-step update, with one `@numba.njit(cache=True)` kernel on the
same flat arrays: drop extinguished cells; crossings from the cells burning at the step start;
the in-step cascade as a Dijkstra search with an indexed binary heap over the cells reached in
the step (the NumPy engine finds the same shortest-path fixed point by label correction); ignite
the cells reached; burn-out. Every arithmetic expression is the NumPy engine's, element by
element, which is why the results come out bit-identical. The per-link arrays are stored
cell-major, `(H * W, K)`, so one cell's links share a cache line (this alone took 16-neighbour
runs from 0.31 s to 0.24 s). The kernel draws no random numbers.

**JIT compile time.** The first `engine: numba` run in a process compiles the kernel: about
2.4-3.4 s with a cold cache, 0.5-0.6 s when numba's on-disk cache is warm. `bench.py` excludes
it from the per-run numbers. The cache lives in `src/wildfire_sim/__pycache__` or, if
`NUMBA_CACHE_DIR` is set, there; when no location is writable numba compiles in memory.

**What is left in a Numba run** (reference, 8 neighbours, 0.14-0.17 s): the kernel is about
60 %; the rest is Python per step (weather lookup, the rule-cache check, the kernel call, about
25-35 us per step) and setup (threshold draw 14-27 ms, link rates 8-13 ms).

## Measured seconds per run

`scripts/bench.py --workloads reference,wind,small --repeats 5` (one warm-up run each; also
the committed baseline `scripts/bench_baseline.json`):

| Workload | Engine | Stencil | min (s) | median (s) | Speed-up (median) |
| --- | --- | --- | --- | --- | --- |
| reference | NumPy | 8 | 1.098 | 1.108 | |
| reference | NumPy | 16 | 2.378 | 2.501 | |
| reference | Numba | 8 | 0.127 | 0.137 | 8.1x |
| reference | Numba | 16 | 0.217 | 0.234 | 10.7x |
| wind | NumPy | 8 | 1.060 | 1.105 | |
| wind | NumPy | 16 | 2.257 | 2.355 | |
| wind | Numba | 8 | 0.112 | 0.119 | 9.3x |
| wind | Numba | 16 | 0.196 | 0.199 | 11.8x |
| small | NumPy | 8 | 0.132 | 0.133 | |
| small | NumPy | 16 | 0.219 | 0.222 | |
| small | Numba | 8 | 0.021 | 0.023 | 5.8x |
| small | Numba | 16 | 0.026 | 0.026 | 8.5x |

Numba compile (cold cache) 2.4 s, excluded. Across separate runs of the script the NumPy
reference times ranged 0.95-1.24 s (8) and 1.96-2.72 s (16) with the machine's load.

## M1 performance target

Proposed, on a machine comparable to the one above, excluding JIT compile time, for the
reference workload (333 x 333 cells, 1440 steps of 30 s, `record_events=False`):

| Engine | 8 neighbours | 16 neighbours |
| --- | --- | --- |
| NumPy (reference engine) | **< 2 s** per run | **< 4 s** per run |
| Numba | **< 0.3 s** per run | **< 0.5 s** per run |

The limits are about twice the measured medians, which absorbs machine-to-machine variation.
Both are met today. In CI the target is enforced relatively (next section): any change that
makes either engine more than 20 % slower fails the build.

## Regression check in CI

The `fast-engine` job in `.github/workflows/ci.yml` installs the `fast` extra, runs the parity
tests (`tests/test_sim_fast.py`) and then

```sh
uv run --extra fast python scripts/bench.py --workloads reference --repeats 5 \
    --json bench.json --check scripts/bench_baseline.json --tolerance 0.2
```

which exits non-zero when any (engine, stencil) entry is more than 20 % slower than the
baseline. **Absolute times cannot be compared** between this laptop and a CI runner, so the
compared quantity is the median run time divided by a *calibration loop* timed in the same
process just before:

- NumPy engine: a fixed loop of small-array NumPy operations (divmod, fancy indexing,
  comparisons, `lexsort` on 1000-element arrays) that, like the engine, is dominated by
  per-call overhead (~80-105 ms here);
- Numba engine: a compiled binary-heap sort of 200,000 floats, branchy scalar code with
  indirect loads like the kernel's cascade (~21 ms here).

A slower machine (or a busy one) slows numerator and denominator together; a slower engine
moves only the numerator. On this laptop the normalised times of two separate runs differed by
at most 7 % while the raw NumPy times differed by 12 %. The numba/numpy time ratio was
considered and rejected: it cannot see a slowdown of the NumPy engine (the ratio improves) and
it couples two independent engines. The calibration is not perfect across CPU generations, so
the committed baseline should be refreshed from the CI job's uploaded `bench.json` artifact
after the first run on GitHub's runners, and whenever a change is meant to alter speed.

## Cost of the Phase 1 additions (WPs 1.10-1.12)

Measured on the reference workload by alternating runs of the code before and after (the
machine's timing drifts by 10-20 % between separate invocations, so only interleaved runs are
compared; minimum of 3 runs each, 3-4 rounds):

| What | NumPy engine | Numba engine |
| --- | --- | --- |
| Dormant sources and edge detection, rule without intensity (the benchmark's constant rule) | within noise | +5 to +9 % |
| Front-normal intensity, rule that reports intensity (uniform 0.2 m/s, 8 / 16 neighbours) | +8 % / +5 % | +2 % / +2 % |

Two guards keep the ordinary case cheap: a burned-out cell is checked for stalled links only if
it has a zero-rate link at all (`_FireEngine.may_stall`, recomputed with the link rates), and
the plane fit is skipped where the rule's head intensity is zero. Without them the NumPy
engine was 18-28 % slower on this workload. The CI baseline (`scripts/bench_baseline.json`)
was not regenerated here; it should be refreshed from a CI run once these are merged.

## Cost of ensembles and of the strategy search

This replaces the order-of-magnitude table of the plan's section 6 (10 km area, 12-hour fire,
dt = 30 s, 1440 steps). Per-run times are the measured serial ensemble times, stencil 8
(NumPy 1.14 s, Numba 0.145 s per run inside `run_ensemble`) and 16 (2.5 s, 0.23 s, from the
benchmark). "8 workers" is `run_ensemble(n_workers=8)` on the 8 logical CPUs of the laptop
above, measured: steady-state throughput is 3.5x serial for the NumPy engine (0.33 s per run)
and 4.9x for Numba (0.029 s per run), well short of 8x because half the cores are efficiency
cores; starting the `spawn` pool costs a fixed 9-17 s (worker imports from the Windows disk,
plus each worker's numba compile or cache load). Stencil-16 times on 8 workers use the same
scaling factors.

| Workload | Runs | NumPy, 1 core | NumPy, 8 workers | Numba, 1 core | Numba, 8 workers |
| --- | --- | --- | --- | --- | --- |
| One scenario | 1 | 1.1 s / 2.5 s | - | 0.14 s / 0.23 s | - |
| One ensemble | 200 | 3.8 min / 8.3 min | 1.3 min / 2.5 min | 29 s / 47 s | 20 s / 25 s (start-up dominated) |
| One optimisation | 300 candidates x 50 runs = 15,000 | 4.8 h / 10.4 h | 1.4 h / 3.0 h | 36 min / 59 min | 7 min / 12 min |

(Each cell: 8 neighbours / 16 neighbours. Optimisation times on 8 workers assume one worker
pool for the whole search: calling `run_ensemble(n_workers=8)` once per candidate would add
300 pool start-ups, another 45-85 min.)

Not included: suppression hooks (Phase 4) run in Python once per step and will add to every
run; Rothermel (Phase 2) changes only the cached rule evaluation, which is under 3 % of a run
unless the weather changes every step.

## Numbers for the plan

For docs/plan.md section 6 (not edited by this work package): replace the estimates with

| Workload | Runs | NumPy (1.1-2.5 s/run) | Numba (0.14-0.23 s/run) |
| --- | --- | --- | --- |
| One scenario | 1 | 1.1-2.5 s | 0.14-0.23 s |
| One ensemble | 200 | 4-8 min (1 core), 1.3-2.5 min (8 workers) | 30-50 s (1 core), 20-25 s (8 workers) |
| One optimisation | 15,000 | 4.8-10.4 h (1 core), 1.4-3 h (8 workers) | 36-59 min (1 core), 7-12 min (8 workers) |

## Real grids: break crossing and the hourly rule (WP 2.9 follow-up)

Measured 2026-11-16 on the developer laptop with the Numba engine, the `"rothermel"` rule and
fire A's region (`lake_2024`: 815 x 978 cells, 36,576 break cells), ignition at the recorded
time, `dt` 60 s.

| Run | Before | After |
| --- | --- | --- |
| 5 km crop of fire A, 4 h | 5.7 s | 0.6 s |
| Fire A, 3 h after ignition | 124 s | 8.8 s |
| Fire A, 24 h after ignition (18,700 ha burned, 4,081 breaches) | over 50 min | 60 s |

Three costs were removed, each without changing any result (the runs are bit-identical, tested):

- **A full refresh after every breach.** WP 2.9 re-ran the spread rule over the whole grid
  (1.2-1.4 s on this grid) in every step with a breach. A breach only stops a cell being a
  barrier, so the engine now updates the blocked links, link rates and stall flags of the cells
  within `max_offset` of it (`_FireEngine.update_barriers`); only a breached constructed line,
  which changes the rule's input, still triggers the full refresh.
- **Rule evaluations before the ignition.** The weather series of a region starts at local
  midnight, about 16 h before fire A's ignition, so the hourly weather changes cost 16 full
  rule evaluations before anything burned. The engine is now first refreshed in the ignition
  step, with any grid edits made before it.
- **Checking every held break cell every step.** Only held cells next to a newly ignited cell
  can be breached, so only those are checked (5.6 ms per step on this grid before).

What remains on the real grid is the spread rule, re-evaluated over the whole grid each time
the hourly weather changes: about 1.4 s per hour of fire on fire A's grid (the lattice-fitted
multipliers 0.6 s, the directional rates 0.2 s, the link rates 0.3 s), so a 24-hour run spends
about 35 of its 60 s there. Evaluating the rule only near the fire would remove most of it; WP
3.1 measures whether calibration needs that.

## Sizing the calibration study (WP 3.1)

The plan asks for `scripts/bench.py` on fire A's grid over the early-growth window before the
trial and seed counts of the calibration study are chosen. **Fire A's built region was not
available on the machine where WP 3.1 was written** (building it needs a LANDFIRE e-mail and a
NASA Earthdata token), so the full-grid measurement is still to be made; what is below is the
committed 5 km crop of fire A, the first full-grid measurement quoted above, and what they imply.
Once the region is built, run

```sh
uv run --extra fast --extra geo python scripts/bench.py --region lake_2024 --hours 48 --engines numba --repeats 3
```

and revise the counts in `configs/calibration.yaml` if the time per run differs much from the
estimate below.

**Measured on the crop** (2026-11-16, the laptop above, Python 3.12.16, NumPy 2.5.4, Numba
0.68.1): `tests/fixtures/lake_2024_5km.npz` (167 x 167 cells) with its hourly HRRR weather, the
`"rothermel"` rule, 16 neighbours, dt 60 s, ignited at the centre at fire A's ignition time
(56,880 s). Each invocation ran 7 interleaved repeats per engine after a warm-up; medians of
separate invocations (this machine drifts by 10-20 % between invocations):

```sh
uv run --extra fast python scripts/bench.py --grid tests/fixtures/lake_2024_5km.npz \
    --weather tests/fixtures/lake_2024_weather.yaml --ignition-time-s 56880 --hours 2 --repeats 7
```

| Window after ignition | Cells burned | NumPy engine | Numba engine |
| --- | --- | --- | --- |
| 1 h | 542 | 0.35-0.39 s | 0.24-0.25 s |
| 2 h | 3,055 | 0.72-0.93 s | 0.37-0.49 s |

One evaluation of the rule over the crop took 0.046-0.084 s, 1.6-3.0 us per cell; on fire A's
815 x 978 cells that is the 1.4 s per hour of fire measured above (1.8 us per cell). After 2.4 h
the crop's fire reaches the crop's edge, so longer windows cannot be timed on it. On small grids
a fixed cost dominates: each run re-reads its YAML tables, about 60 ms of an 85 ms run on the
101 x 101 grid of the recovery tests (cProfile); on fire A's grid it is negligible.

**Fire A's grid, estimated.** The first measurement (previous section) is 60 s per 24-hour run
with the Numba engine, 35 s of it in the hourly re-evaluation of the rule. The calibration
window is 48 h (`configs/calibration.yaml`, a placeholder: the first two burning periods). The
rule part doubles to about 70 s and the front part grows with the burned area, so a run of the
window takes **about 2 min or more** (an estimate, not a measurement).

**Seeds.** The objective is averaged over fixed seeds (common random numbers), so the seeds only
need to average out the kernel's noise. On the crop (2-hour window, a synthetic observed fire),
the objective at fixed parameters varied between 8 seeds with a standard deviation of 0.002-0.007,
while a 10 % change of either parameter raised it by 0.05-0.12. Two seeds are enough:
`n_seeds: 2`.

**Trials.** On the synthetic recovery problem of `tests/test_calibration.py` (two parameters,
the plan's bounds, sampler seed 0) the largest error of the recovered parameters was 19 % after
40 trials, 6.4 % after 80 and 2.5 % after 150 (`docs/calibration.md`). `n_trials: 100`.

**Budget.** 100 trials x 2 seeds x about 2 min is about 7 h on one core, plus the objective
surface (25 points x 2 seeds, about 1.7 h): an overnight run. Trials rejected by the edge rule
are cheaper (the remaining seeds are skipped). The rule re-evaluation is about half of each run,
so evaluating it only near the fire, the open question of the previous section, would save up
to about half of the study; it is not needed to run it. Runs of one trial are independent and
could use the cores of the machine in parallel; `scripts/calibrate.py` runs them one after another.
