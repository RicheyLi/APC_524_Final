# Project plan

The single plan for wildfire-sim: model design and rationale (sections 1-7), then the schedule,
gates and work packages (WPs) that implement it (sections 8-13). Each WP is one branch and one
pull request (see `CONTRIBUTING.md`).

History: design written Oct 8, 2026 (Richey Li); schedule and work packages added Oct 27 with
Phase 0; the two documents were merged, checked against the Phase 0 code and re-baselined on
**Nov 10, 2026**. Section 14 lists what that review changed.

Submission deadline: **early December 2026 (Dec 1-5)**.

## 1. Summary

A cellular automaton (CA) in which every ignition probability comes from a physical rate of
spread, inputs come from fire-science GIS datasets, and "optimization" is a search over reactive
suppression policies scored on Monte Carlo ensembles.

- **Ignition probability comes from physics, not hand-picked weights.** Each neighbour's
  ignition chance is computed from a directional rate of spread, so cells and time steps have
  real meaning.
- **Time step is tied to cell size.** The step is chosen from the cell size and the fastest
  spread, and the model is tested to give the same answer when the step is halved.
- **Burning lasts a physical duration.** A cell is a fire source for as long as it takes the
  front to cross it, then flames for its fuel's residence time.
- **Data comes from LANDFIRE, USGS, gridMET and OpenStreetMap.** A map is only how the user
  picks the area.
- **Calibration against a real fire is a required milestone.** Without it the model is not
  predictive.
- **Firefighting is modelled as resources** with arrival delays, production rates and hourly
  costs, acting on the grid.
- **The optimizer tunes parameterized reactive policies,** for example "build a line 200 m
  ahead of the head when it nears a town". Reinforcement learning is a stretch goal.
- **Every result is an ensemble** of runs, reported as distributions and burn-probability maps.
- **The real-world pipeline is limited to the US,** where 30 m fuel maps exist, starting with
  one cached study region.

### Why the design looks like this

Four problems with the original idea (a Drossel-Schwabl forest-fire model fed from Google Maps)
shaped the design:

1. **Google Maps is the wrong data source.** It shows what an area looks like, not fuel type,
   fuel load, canopy, moisture or usable elevation, and its terms of service forbid extracting
   map content. A map is used only to pick a bounding box.
2. **Parameters cannot be calculated from data alone.** Data gives physical inputs (fuel model,
   slope, wind, moisture). Turning those into a probability needs either a physical spread
   model (Rothermel) or coefficients fitted to a real fire, as Alexandridis et al. did for the
   1990 Spetses Island fire. Either way a calibration step is needed.
3. **The Drossel-Schwabl baseline has no clock.** A step is not a unit of time, so "the fire
   reached town in 3 hours" has no meaning until the step is linked to a spread rate.
4. **Square grids distort fire shape.** With no wind a fire spreads as a diamond or octagon
   instead of a circle. The model uses the true diagonal distance, keeps ignition random and
   tests that a no-wind fire on flat uniform fuel comes out nearly circular.

## 2. Core model: a CA with physical scale

Cells are 30 m, the native resolution of the US fuel maps.

**Grid.** A 10 km x 10 km study area at 30 m is about 333 x 333 = 111,000 cells. Each cell
stores (see `wildfire_sim.grid.Grid`):

- Static layers: fuel model, canopy cover, elevation, slope, aspect, land use (wildland, urban,
  water, road) and asset value.
- State: unburnable, unburned fuel, burning (with a timer) or burned out; arrival time;
  intensity.
- Suppression layers: fireline present, and retardant strength.

Weather (wind speed and direction, dead and live fuel moisture) is one value per time for the
whole grid, because the sources (about 4 km) are coarser than the study area.

### Ignition kernel

Spread from a burning cell i to an unburned neighbour j is a random crossing time with mean
equal to the travel distance over the spread rate, d\_ij / R\_ij. R\_ij is the rate of spread
from i toward j (m/s) and d\_ij is the centre-to-centre distance (dx, or sqrt(2) dx on a
diagonal). The crossing time is **gamma-distributed with shape k** (coefficient of variation
1 / sqrt(k)). The per-step ignition probability is the chance the crossing completes during
the step given that it has not yet, where a\_i is the time since cell i ignited and Q is the
regularized upper incomplete gamma function:

```latex
p_{ij} = 1 - \frac{S(a_i + \Delta t)}{S(a_i)}, \qquad
S(t) = Q\!\left(k,\; k\,\frac{R_{ij}\,t}{d_{ij}}\right)
```

With several burning neighbours the cell ignites with probability

```latex
P_j = 1 - \prod_{i \in \text{burning neighbours}} \left(1 - p_{ij}\right)
```

This keeps the "more burning neighbours means more likely" idea, and because step survival
probabilities multiply to S, the arrival-time distribution does not depend on the time step.

**Why not the memoryless form.** The first version of this plan used
p = 1 - exp(-R dt / d), which is the k = 1 case. Its mean crossing time for one pair of cells is
d / R, but on a lattice the front follows the fastest of many random paths, so the fire runs
several times faster than R and gets faster as the step shrinks. Measured with
`scripts/kernel_front_speed.py` (flat uniform fuel, no wind, R = 0.2 m/s, dx = 30 m, 40
simulated minutes, mean of 6 seeds; values are front speed / R along the grid axes):

| Shape k | 8 neighbours, dt 30 s | dt 15 s | dt 7.5 s | 16 neighbours, dt 30 s | dt 7.5 s |
| --- | --- | --- | --- | --- | --- |
| 1 (memoryless) | 3.07 | 3.85 | 4.47 | 5.80 | off grid |
| 5 | 1.27 | 1.38 | 1.44 | 1.72 | 1.90 |
| **20 (default)** | **1.01** | **1.03** | **1.07** | 1.15 | 1.21 |
| 50 | 0.93 | 0.96 | 0.99 | 1.02 | 1.07 |

The diagonal speed is within 6 % of the axis speed in every row, so the burn stays round. The
default is `spread.arrival_shape = 20`: within 10 % of R on the 8-neighbour stencil, weakly
dependent on dt, and still visibly stochastic. The residual bias (and the larger one on the
16-neighbour stencil) is absorbed by the calibrated spread-rate multiplier (WP 3.1). The kernel
uses the current R with the source cell's age, so a weather change during a crossing is
approximated rather than integrated.

**Rate of spread.** The head-fire rate comes from a spread rule behind one interface
(`SpreadRule`). Spread in other directions follows an ellipse whose length-to-breadth ratio
grows with wind speed.

- *Constant rule*: one rate in every direction; the Drossel-Schwabl-style baseline used by the
  kernel tests.
- *Alexandridis rule*: a base rate times vegetation, density, wind and slope factors
  (Alexandridis et al. 2008). Quick to build, calibrated on real fires, and the fallback if
  Rothermel is cut.
- *Rothermel rule*: surface fire spread from fuel model, fuel moisture, wind and slope, via the
  `pyrothermel` library.

**Where temperature and humidity enter.** They act mainly through dead fuel moisture, which
feeds Rothermel. They are a moisture field, not separate multipliers on probability.

**Burn duration.** A cell has two clocks. It is an *ignition source* (state BURNING) until none
of its neighbours is unburned fuel it can still reach, which with the gamma kernel happens
within a few multiples of d / R. It is *flaming* (contributes `intensity`, blocks crews) for
the time the front takes to cross it plus its fuel's residence time (seconds for grass, longer
for timber). Burning out a cell after the residence time alone would stop the fire: grass
flames for under 10 s while a 30 m cell takes 150 s to cross at 0.2 m/s (measured with the memoryless kernel: with a 30 s
burn the fire dies after about 10 cells).

**Time step.** dt is chosen so the fastest crossing takes several steps:
`max_stable_dt(ros_max, dx)` gives about 21 s for a 1 m/s grass fire on 30 m cells.

**Barriers and stencils.** With the 8-neighbour stencil, fire passes diagonally between two
barrier cells that touch only at a corner, so rasterized lines must be 4-connected. With the
16-neighbour stencil a knight move jumps a barrier one cell wide, so barriers must be at least
two cells wide or the simulator must block knight moves whose intermediate cells are
unburnable (WP 1.3 does the latter).

**Stretch: spotting.** Embers thrown ahead of the front, which let fire jump firelines.
Alexandridis et al. include a simple version.

## 3. Real-world data pipeline

Every model input has a free, public US source. The pipeline is: user picks a bounding box,
download each layer, reproject everything onto one 30 m grid, cache to disk, run.

| Input | Source | Resolution | Notes |
| --- | --- | --- | --- |
| Fuel model, canopy cover, canopy height | LANDFIRE (Scott & Burgan 40 fuel models) | 30 m | US only; this is why the project is limited to the US |
| Elevation, slope, aspect | USGS 3DEP | 10-30 m | Slope and aspect computed here |
| Wind, temperature, humidity | gridMET (daily) | about 4 km | One hourly scalar series per run; HRRR is out of scope |
| Dead fuel moisture | gridMET 100-h and 1000-h fuel moisture | about 4 km | Live fuel moisture: assumed seasonal value |
| Roads, rivers, lakes, buildings | OpenStreetMap (osmnx) | Vector | Natural firebreaks, crew access, assets |
| Asset value | Building footprints x an assumed USD per building | Vector | A team assumption, stated in the README |
| Historical fires (validation) | MTBS burn perimeters; NASA FIRMS VIIRS hotspots | Perimeter; 375 m detections | Final burned area and timing of spread |

Libraries: rasterio and xarray for rasters, geopandas and osmnx for vectors. Downloaded data is
never committed; a download script and a local cache are. Each source's access method is
verified when its WP starts and the date recorded in `docs/data.md`, because download services
change.

## 4. Firefighting tactics

Every tactic is a resource that arrives after a dispatch delay, works at a limited rate, costs
money per hour, and changes cells on the grid. More crews can always be called, but no line is
built instantly.

| Tactic | Effect on the grid | Key parameters | Cost drivers |
| --- | --- | --- | --- |
| Indirect attack (control line far from the fire) | Cells along a planned path become unburnable, one segment at a time as it is built. An unfinished segment does not stop fire. | Path, line width, build rate (m/h per crew or dozer), start time | Crew and dozer hours |
| Parallel attack (line close to the fire) | Same as indirect, but the path is the current fire edge pushed outward by an offset d, recomputed as the fire moves | Offset d, which flank, build rate | Crew and dozer hours |
| Direct attack: ground crews | Crews move along the burning edge and put out burning cells next to them with some probability per step | Crew speed along the edge, extinguish rate, maximum fire intensity crews can face | Crew hours |
| Direct attack: aircraft | A drop covers a strip of cells and multiplies their spread rate by a factor below 1, which fades over hours (retardant) or minutes (water) | Drop footprint, reduction factor, decay time, turnaround time | Flight hours, litres dropped |

**Line breaching.** A fire can cross a line with a probability that rises with fire intensity
relative to line width. Without this any line is perfect and the optimizer will exploit it. A
breach turns the line cell back into fuel (`fireline = False`, state FUEL).

**Crew safety limit.** Ground crews cannot work an edge whose flame lengths are too high; NWCG
guidance sets such limits by resource type. With the limit enforced the optimizer learns when
direct attack is not an option.

Parameter values (build rates, drop sizes, decay times, costs) come from fire-management
references and live in `configs/` with citations.

## 5. Strategy search

The code tunes a family of reactive rules rather than searching over raw line placements. A
rule family has 5-15 numbers, which a black-box optimizer can handle in a few thousand
ensemble runs.

**What a reactive policy is.** At each decision point (every 30-60 simulated minutes) the
policy looks at the fire's current state and decides what to deploy. The four rules of the
parameterized policy (`ReactivePolicy`, WP 4.4), each controlled by numbers the optimizer
tunes:

1. Dispatch k crews, m dozers and n aircraft at first report.
2. Build a parallel line at offset d on the flank nearest the assets.
3. When the head's spread rate passes a threshold, switch from direct attack to an indirect
   line L metres ahead, downwind.
4. Send aircraft to the head of the fire unless an asset is within a set distance, then protect
   the asset.

The policy reads only things a real incident commander would know: perimeter, head direction,
spread rate, intensity and distance to assets. A user-defined strategy is a policy whose
numbers the optimizer tunes.

**Cost function.** The optimizer minimizes the expected total cost over an ensemble of runs:

```latex
J(\theta) = \mathbb{E}\left[\sum_{r} c_r\,T_r(\theta) + c_w V_w(\theta) + c_{rt} V_{rt}(\theta) + \sum_{k\,\in\,\text{burned}} v_k\right]
```

theta is the policy's parameter vector; T\_r is hours worked by resource type r at hourly cost
c\_r; V\_w and V\_rt are litres of water and retardant at unit costs c\_w and c\_rt; v\_k is the
value of burned cell k. The 90th-percentile cost is reported as well, because an average can
hide the rare runs where a town burns.

**Optimizer.** Optuna: it handles a mix of whole-number parameters (crew counts) and continuous
ones (offsets), tolerates noisy scores, and can stop unpromising candidates early.

**Baselines.** No suppression, direct attack only and indirect attack only. The optimized
policy has to beat all three for the result to mean anything.

**Stretch: reinforcement learning.** Only if everything else is done.

## 6. Stochastic runs: cost

One run is cheap; the cost comes from multiplying runs per ensemble by candidate strategies.

Rough estimates for the 10 km area, a 12-hour fire at dt = 30 s (1,440 steps). These are
order-of-magnitude guesses to be replaced by `scripts/bench.py` numbers (WP 1.7).

| Workload | Runs | NumPy (1-2 s/run) | Numba, burning front only (0.1 s/run) |
| --- | --- | --- | --- |
| One scenario | 1 | 1-2 s | 0.1 s |
| One ensemble | 200 | 3-7 min | 20 s |
| One optimization | 300 candidates x 50 runs = 15,000 | 4-8 h | 25 min |

The gamma kernel evaluates an incomplete gamma function per source cell and direction, so even
the NumPy engine must evaluate it only for burning cells, never for the whole grid.

Runs are independent and split across CPU cores. The uncertainty of an average shrinks with
the square root of the run count; plot the running mean of burned area against run count and
stop when it settles (50-200 runs is typical).

**Keeping it cheap:**

- Update only cells near the burning front.
- Score all candidate strategies on the same set of random seeds, so differences reflect the
  strategy, not luck.
- Give every candidate a few runs first and spend more only on the promising ones (Optuna
  pruning).
- Record every seed so any run can be reproduced exactly.

## 7. Validation

Calibrate on one well-documented historical US fire; test on a second if time allows.

**Fire A: the 2016 Chimney Fire**, San Luis Obispo County, CA. It has an MTBS perimeter, VIIRS
hotspots showing daily growth, pre-fire LANDFIRE fuels, and a published CA simulation to
compare with (arXiv 2507.00761). Fire B is a stretch goal; the report states the single-fire
limitation and the suppression confound (the real fire was fought, the calibration runs are
not).

**Metrics:**

- Overlap between simulated and real burned area (Jaccard or Sorensen index), using cells
  burned in at least half the runs.
- Arrival-time error: when the simulated fire reaches each VIIRS detection versus when it was
  detected.
- Ratio of simulated to real burned area.

**Calibration.** Tune only a few parameters (an overall spread-rate multiplier, the ellipse
shape) with the same optimizer.

**Physics tests in the test suite** (WP 1.4):

- No wind, flat, uniform fuel: a near-circular burn whose front speed matches R.
- Steady wind: a burn stretched downwind.
- Halving dt: the same distribution of arrival times.
- A complete line: fire stops.
- Water and urban cells never burn.

## 8. Timeline and gates

Phase 0 was finished on Oct 27. No Phase 1 work was merged in weeks 1-2, so **Gate 1 (Nov 9)
was missed** and the remaining work is re-baselined into three weeks.

| Week | Dates | Work | Gate |
| --- | --- | --- | --- |
| 1-2 | Oct 27 - Nov 9 | Phase 0 (done Oct 27). Planned Phase 1 work did not start. | Gate 1 missed |
| 3 | Nov 10 - 16 | Phase 0 review fixes (done Nov 10); Phase 1 WPs 1.1-1.6 and 1.8; Phase 4 WPs 4.1-4.2 in parallel | **Gate 1 (Nov 16)** |
| 4 | Nov 17 - 23 | Phase 2 WPs 2.2-2.8 (Chimney Fire region); Phase 4 WPs 4.3, 4.4, 4.6; WP 1.7 if runs are too slow | |
| 5 | Nov 24 - Dec 1 | Phase 3 calibration; WP 4.5 optimizer; WP 4.7 experiments; Phase 5 deliverables | **Gate 2 (Nov 27)**, submit |

**Gate 1 - simulator works.** The Alexandridis rule passes every physics test in
`tests/test_physics.py` on synthetic landscapes; an ensemble and a GIF are reproducible from a
YAML file and a seed; one run on the 333 x 333 reference grid takes under 2 s.

**Gate 2 - model is validated.** Calibrated on the 2016 Chimney Fire with a reported Jaccard
index and arrival-time error.

**Fallback rule (triggered).** The plan has always said: if the first gate slips, cut the
Rothermel step and finish on the Alexandridis rule; the rest of the pipeline works with either.
The gate slipped, so WP 2.1 is now a stretch goal, to be started only if Gate 1 is met by
Nov 16 with time to spare. This needs team confirmation (see *Open decisions*). Other stretch
goals (spotting, reinforcement learning, non-US data, fire B) start only after Gate 2.

## 9. Scope decisions

Fixed on Oct 27, 2026 unless dated otherwise:

| Decision | Choice |
| --- | --- |
| Rothermel implementation | `pyrothermel` library behind the `SpreadRule` interface (optional extra `rothermel`). Hand-written equations are out. **Risk:** only a Windows wheel is published; Linux/macOS need a C++ compiler to build it. The wrapper imports lazily so the package works without it. Stretch since Nov 10 (see the fallback rule). |
| Ignition kernel (Nov 10) | Gamma-distributed crossing time, `spread.arrival_shape = 20`; the memoryless kernel is kept only as the `arrival_shape = 1` special case. |
| Package name | `wildfire_sim` (CLI `wildfire-sim`). |
| Validation fire A | 2016 Chimney Fire, San Luis Obispo County, CA. |
| Validation fire B | **Stretch.** |
| Weather | Hourly scalar series per run (gridMET daily, optionally disaggregated). HRRR is out. |
| Primary Python | 3.12. 3.13 tested in CI; 3.14 allowed to fail. |
| Out of scope unless everything else is done | Spotting, reinforcement learning, non-US data, 16-neighbour stencil as default (the stencil exists in `types.py` for the circularity experiment). |
| Minimum suppression set | Ground crews and control lines. Aircraft may slip to week 5. |
| File paths in scenario files (Nov 10) | Relative paths resolve against the working directory; commands are run from the repository root. Parameter tables live in `configs/` and are read from there, so the package is used from a checkout, not from an installed wheel. |

## 10. Module map

```text
src/wildfire_sim/
  types.py            CellState, LandUse, Stencil (8/16)           [frozen, Phase 0]
  grid.py             Grid layers, Weather, WeatherSeries          [frozen, Phase 0]
  config.py           Pydantic schemas, YAML loaders               [frozen, Phase 0]
  spread/base.py      SpreadRule protocol, ignition kernel         [frozen, Phase 0]
  spread/__init__.py  rule registry (register, make_spread_rule)   [Phase 0]
  spread/constant.py                                               [WP 1.1]
  spread/alexandridis.py                                           [WP 1.2]
  spread/rothermel.py                                              [WP 2.1, stretch]
  sim.py              time loop, event log, RunResult              [WP 1.3]
  viz.py              maps, burn-probability plots, GIF            [WP 1.5]
  ensemble.py         parallel seeded runs                         [WP 1.6]
  sim_fast.py         Numba front-only kernel                      [WP 1.7]
  cli.py              validate (Phase 0) / run / ensemble / plot   [WP 1.8]
  data/synthetic.py   test landscapes                              [Phase 0]
  data/landfire.py, terrain.py, weather.py, osm.py, cache.py, validation.py  [WP 2.2-2.7]
  metrics.py          burned area, overlap, cost                   [WP 2.8]
  suppression/geometry.py, resources.py, tactics.py, policies.py  [WP 4.1-4.4]
  optimize.py         Optuna study                                 [WP 4.5]
configs/              scenarios/*.yaml, costs.yaml, resources.yaml, alexandridis.yaml,
                      residence_time.yaml, fuel_models.csv, calibrated.yaml
scripts/              kernel_front_speed.py, bench.py, download_data.py, calibrate.py,
                      run_experiments.py
tests/                offline, deterministic; fixtures/ holds committed .npz grids
```

Frozen modules are the contracts every other WP builds on. Changing them needs a PR that
updates all callers and explicit sign-off. Adding an optional config field with a default is
allowed.

A spread rule registered as `x` must live in `spread/x.py`; `make_spread_rule("x")` imports that
module on first use, so a scenario file can select a rule by name.

## 11. Work packages

Each row is self-contained: a model or teammate handed one row plus the frozen modules should
be able to complete it without further context. "Consumes" lists the interfaces the WP relies
on; "Acceptance" is what the PR's tests must show.

### Phase 0 - Foundations (complete Oct 27; review fixes Nov 10)

| WP | Deliverable | Status |
| --- | --- | --- |
| 0.1 | Repo hygiene: LF line endings, rename to `wildfire_sim`, correct URLs, 3.12 primary, `AGENTS.md`, data dirs ignored | done |
| 0.2 | Dependency groups (`fast`, `geo`, `rothermel`, `opt`, `all`); pyrothermel API verified | done |
| 0.3 | `types.py` (`CellState`, `LandUse`, `Stencil`, `STENCIL_8`, `STENCIL_16`), `grid.py` (`Grid`, `Weather`, `WeatherSeries`, save/load) | done |
| 0.4 | `spread/base.py`: `SpreadOutput`, `SpreadRule`, `ignition_probability`, `neighbour_ignition_probability`, `max_stable_dt`; registry in `spread/__init__.py` | done |
| 0.5 | `config.py`: `ScenarioConfig`, `CostConfig` with `CitedValue`, loaders; three scenario YAMLs; `configs/costs.yaml` placeholders | done |
| 0.6 | `data/synthetic.py` generators, `build_grid`, `tests/conftest.py`, committed fixture `tests/fixtures/flat_21x21.npz` | done |
| 0.7 | This page | done |
| 0.8 | Review fixes (section 14): gamma ignition kernel and `spread.arrival_shape`; lazy rule import; `n_steps` rounding; ignition-cell bounds; `engine_usd_per_h`; finite-value grid validation; CLI error handling; `scripts/kernel_front_speed.py` | done Nov 10 |

### Phase 1 - Simulator core

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 1.1 | `spread/constant.py` | 0.4 | `ConstantRule(ros: float)` registered as `"constant"`: same `ros` (m/s) in every direction for burnable cells, zero elsewhere, times `(1 - retardant)`; `intensity`/`flame_length` zero | Output validates; non-burnable cells zero; retardant 0.5 halves ros; `make_spread_rule("constant", ros=0.2)` works without importing the module first |
| 1.2 | `spread/alexandridis.py`, `configs/alexandridis.yaml`, `configs/scenarios/steady_wind.yaml` | 0.4 | Alexandridis et al. 2008 rule expressed as a rate: `ros_k = ros_base (1 + p_veg)(1 + p_den) p_w(theta_k) p_s(k)` where `ros_base` (m/s) replaces the paper's base probability `p_h` and is the calibration parameter; wind factor `p_w = exp(c1 V) exp(c2 V (cos(theta) - 1))` with `theta` the angle between spread direction k and the wind-to bearing; slope factor `p_s = exp(a * theta_s)` with `theta_s = atan((elev_neighbour - elev_cell) / d_k)` in degrees; fuel-model-to-(veg, density) mapping table and constants (paper values `a = 0.078`, `c1 = 0.045`, `c2 = 0.131`; check against the paper) cited in the YAML; switch `steady_wind.yaml` to this rule | Downwind ros > upwind; upslope > downslope; constants loaded from YAML, not hard-coded; edge cells use zero slope toward missing neighbours |
| 1.3 | `sim.py`, `configs/residence_time.yaml` | 0.3-0.6, any rule | `run(config, seed, rule=None) -> RunResult` with `RunResult(grid, arrival_time, burned_mask, events, elapsed_s)`. Per step: weather lookup; `rate_of_spread`; `age = t - arrival_time` for burning cells; `ignition_probability(..., age=age, shape=config.spread.arrival_shape)` **evaluated only for burning cells**; `neighbour_ignition_probability`; draw with `rng.random`; ignite FUEL cells (set `arrival_time`). A BURNING cell becomes BURNED once it has no FUEL neighbour it can reach with `ros > 0` (or its age exceeds an optional `max_source_age_s`); `burn_timer` holds the remaining flaming time `d / ros_head + residence_time(fuel_model)` used for `intensity`. Knight moves (16-stencil) are blocked when both intermediate cells are unburnable. `events` is a list of `(t, kind, row, col)`; ignition from `IgnitionConfig`, erroring if a cell is outside the grid or unburnable. Residence times are `CitedValue`s keyed by fuel model with a default | Water/urban never burn; identical output for identical seed; fire ends when no cell is burning; `arrival_time` NaN exactly where never burned; burning set stays a thin front on `flat_no_wind` |
| 1.4 | `tests/test_physics.py` | 1.1, 1.2, 1.3 | (a) no wind, constant rule, 20 runs, seed 0: diagonal extent / axis extent in [0.9, 1.1] and boundary-radius standard deviation / mean < 0.1 (an edge-counting isoperimetric quotient cannot be used: it is 0.62 for a perfect raster disc); (b) wind from the west, Alexandridis rule: east extent / west extent `>= 1.5`; (c) dt halving (15 s vs 7.5 s), 50 runs each: mean arrival time along a transect within 8 % and KS `p > 0.01`; (d) complete 3-cell-wide fireline ring: no cell outside ignites; (e) river scenario: no cell east of the river ignites, for stencils 8 and 16; (f) front speed: axis extent / (R T) in [0.9, 1.2] for the constant rule with `arrival_shape = 20`, and > 2 with `arrival_shape = 1` (documents why the default is not 1) | All six pass with thresholds stated in docstrings |
| 1.5 | `viz.py` | `RunResult`, `Grid` | `plot_state(grid, ax)`, `plot_arrival(result)`, `plot_burn_probability(stack)`, `animate(results_frames, path.gif)` using matplotlib + imageio; colour map keyed on `CellState` | Smoke tests write PNG/GIF to `tmp_path`; no `plt.show()`; Agg backend |
| 1.6 | `ensemble.py` | 1.3 | `run_ensemble(config, n_runs, base_seed, n_workers) -> EnsembleResult(burn_probability, arrival_times (n,H,W), burned_area_m2 (n,), seeds)`; seeds from `np.random.SeedSequence(base_seed).spawn(n)`; `multiprocessing` pool with the `spawn` start method (works on Windows); `running_mean(burned_area)` helper | n=4 parallel equals n=4 serial bit-for-bit; seeds recorded |
| 1.7 | `sim_fast.py`, `scripts/bench.py` | 1.3 | Numba kernel updating only cells within the stencil halo of burning cells; selected by `config.engine = "numba"` (add optional field with default `"numpy"`); benchmark prints s/run on 333 x 333, 1440 steps. Do this only if the NumPy engine misses the 2 s Gate 1 target or ensembles are too slow for WP 4.5 | Same result as NumPy engine for same seed; `<= 0.5 s/run` with numba, documented numbers in `docs/performance.md` |
| 1.8 | `cli.py` | 0.5, 1.3, 1.5, 1.6 | `run SCENARIO --seed --out DIR` (writes `grid.npz`, `events.csv`, `summary.json`, `final.png`); `ensemble SCENARIO --n --out`; `plot DIR` | `CliRunner` tests for each command on `flat_no_wind` |

### Phase 2 - Real data (and Rothermel as a stretch)

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 2.1 (stretch) | `spread/rothermel.py`, `configs/fuel_models.csv` | 0.4, pyrothermel | `RothermelRule` registered as `"rothermel"`: head ros from `pyrothermel.PyrothermelRun.run_surface_fire_in_direction_of_max_spread` with `FuelModel.from_existing(number)` and `MoistureScenario` built from `Weather`; midflame wind via pyrothermel's `wind_input_mode`; directional ros from the Anderson (1983) ellipse with length-to-breadth `LB = 0.936 exp(0.2566 U) + 0.461 exp(-0.1548 U) - 0.397` (U = midflame wind in m/s, capped at 8); Byram intensity and flame length from pyrothermel outputs; results cached per (fuel, moisture, wind, slope) tuple; lazy import so the package works without pyrothermel | Head ros for fuel models GR1, SH2, TL3 at a documented moisture/wind within 10 % of BehavePlus values recorded in the test; physics tests 1.4 pass unchanged; `pytest.importorskip("pyrothermel")` |
| 2.2 | `data/landfire.py` | 0.3 | Fetch LANDFIRE fuel model (FBFM40), canopy cover and height for a lat/lon bbox via the LANDFIRE Product Service; write GeoTIFFs under `data/cache/<bbox-hash>/` | Request builder unit-tested with mocked HTTP; live test marked `network` |
| 2.3 | `data/terrain.py` | 0.3 | 3DEP DEM via `py3dep` for the bbox; reproject/resample onto the LANDFIRE grid with rasterio; slope (rise/run) and aspect (downhill bearing) from `np.gradient` | Slope and aspect of `synthetic.slope_ramp` elevation recovered to 1e-3 / 0.5 deg |
| 2.4 | `data/weather.py` | 0.3 | gridMET daily wind speed/direction, RH, T, 100-h and 1000-h fuel moisture for the bbox centroid and date range (THREDDS/OPeNDAP via xarray); map to `WeatherConfig.series` with 1-h and 10-h moisture from the 100-h value by documented offsets | Mocked dataset converts to a valid `WeatherConfig`; live test marked `network` |
| 2.5 | `data/osm.py` | 0.3 | `osmnx` roads, water, buildings for the bbox; rasterize to `land_use`; `asset_value = buildings per cell * usd_per_building` from `costs.yaml` | Known polygon rasterizes to the expected cells |
| 2.6 | `scripts/download_data.py`, `data/cache.py`, `docs/data.md` | 2.2-2.5 | One command builds `data/cache/chimney_2016/grid.npz` and `weather.yaml` for the Chimney Fire bbox and Aug 13 - Sep 6 2016; also writes a 5 km x 5 km crop to `tests/fixtures/chimney_crop.npz` (committed); `docs/data.md` records each source's access method and the date it was verified | Script runs end to end; cached grid validates; fixture under 2 MB |
| 2.7 | `data/validation.py` | 0.3 | MTBS perimeter and VIIRS detections loaders for the Chimney Fire; ignition point and time from CAL FIRE/InciWeb with citation; `observed_burn_mask(grid) -> bool array`, `detections(grid) -> list[(t, row, col)]` | Mocked inputs rasterize correctly; live test marked `network` |
| 2.8 | `metrics.py` | `RunResult`, `EnsembleResult` | `burned_area_ha`, `jaccard(sim_mask, obs_mask)`, `sorensen`, `arrival_time_error(arrival, detections)`, `area_ratio`, `suppression_cost(hours_by_type, litres, costs)`, `asset_loss(grid)` (urban cell counts as lost if any 8-neighbour burned) | Hand-computed cases |

### Phase 3 - Calibration

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 3.1 | `scripts/calibrate.py`, `configs/calibrated.yaml` | 1.2 (or 2.1), 2.6-2.8, optuna | Optuna study over the spread-rate multiplier (`ros_base` for Alexandridis; `ros_multiplier` in [0.3, 3] for Rothermel) and the directional shape (`c2` for Alexandridis; ellipse `lb_scale` in [0.5, 2] for Rothermel) maximizing mean Jaccard over 20 fixed seeds on the Chimney grid; writes best parameters | Runs on the committed crop in CI with 3 trials x 2 seeds (marked `slow`) |
| 3.2 | `docs/validation.md`, `notebooks/validation.ipynb` | 3.1 | Burn-probability map vs. MTBS perimeter, arrival-time scatter, metrics table, discussion of the suppression confound; Rothermel vs. Alexandridis comparison if 2.1 was done | Figures regenerate from a script |

### Phase 4 - Suppression and strategy search

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 4.1 | `suppression/geometry.py` | 0.3 | `perimeter_cells(burned_mask)`, `rasterize_polyline(points, width_cells, shape)` producing **4-connected** lines (no diagonal gaps), `offset_perimeter(burned_mask, offset_m, dx)` via `scipy.ndimage.distance_transform_edt`, `head_direction(arrival_time, window_s)` from recent ignition centroid drift, `distance_to_assets(grid)` | Hand-drawn shapes; offset of a filled disc is a ring at the right radius; a diagonal line has no corner-only contacts |
| 4.2 | `suppression/resources.py` | 0.5 | `Resource` dataclass from `ResourceConfig`; `Fleet` tracks dispatch time, arrival, hours worked, litres dropped; cost accrual per step, using the `CostConfig` rate for the type when `hourly_cost_usd` is 0 | Hours and cost match hand calculation |
| 4.3 | `suppression/tactics.py` | 4.1, 4.2, 1.3 hooks | `IndirectLine(path)`, `ParallelLine(offset, flank)`, `DirectAttack(crews)`, `AirDrop(footprint, factor, decay_s)`; line segments become `fireline = True` at the build rate (then `grid.apply_unburnable()`); breach: each step a line cell next to a burning cell reverts to FUEL with probability `p_b = sigmoid((intensity - I_crit(width)) / s)`; crews skip cells with `flame_length > max_flame_length_m`; `sim.py` gains a `tactics` hook called each step | Complete wide line stops fire; breach probability rises with intensity; crews refuse unsafe edges; retardant decays to zero |
| 4.4 | `suppression/policies.py` | 4.1-4.3 | `Observation` dataclass (perimeter length, head bearing, head ros, max intensity, distance to nearest asset, burned area, elapsed time); `Policy` protocol `act(obs, fleet, t) -> list[Action]`; baselines `NoSuppression`, `DirectOnly`, `IndirectOnly`; `ReactivePolicy(params)` implementing the four rules in section 5 | Each baseline runs end to end on `with_town` |
| 4.5 | `optimize.py` | 4.4, 1.6, 2.8, optuna | `optimize_policy(config, n_trials, runs_per_trial, seeds)`: objective is mean total cost with P90 reported; common random numbers; pruning after the first 5 runs; study saved to SQLite | Toy problem recovers a known optimum |
| 4.6 | `configs/costs.yaml`, `configs/resources.yaml` | 0.5 | Replace every PLACEHOLDER with a cited value (NWCG, USFS cost guides); resource build rates and safety limits | Validators pass; no `PLACEHOLDER` string remains |
| 4.7 | `scripts/run_experiments.py`, `docs/results.md` | everything | Optimized policy vs. three baselines on the Chimney grid with a synthetic town; cost distributions, burn-probability maps, convergence plot of running mean vs. run count | Figures regenerate from committed configs and seeds |

Intensity-dependent pieces (breaching, crew safety) need a rule that reports `intensity` and
`flame_length`. The constant and Alexandridis rules return zeros, so if Rothermel stays cut,
WP 4.3 takes intensity from Byram's relation `I = h w R` with a per-fuel-model heat yield and
load table in `configs/` (cited), and flame length from Byram's `L = 0.0775 I^0.46`.

### Phase 5 - Deliverables

README example run with a GIF, `docs/` page per module, report draft assembled from
`docs/validation.md` and `docs/results.md`, final changelog, tagged release.

## 12. Dispatch batches

These can run concurrently, one branch each:

- **Batch A (Nov 10-13):** 1.1, 1.2, 1.3, 1.5, 4.1, 4.2
- **Batch B (after 1.3 merges, by Nov 16):** 1.4, 1.6, 1.8, 4.3, 4.4
- **Batch C (Nov 17-23):** 2.2-2.8, 4.6, 1.7 if needed, 2.1 if un-cut
- **Batch D (Nov 24 - Dec 1):** 3.1, 3.2, 4.5, 4.7, Phase 5

A handoff prompt for one WP contains: the WP row above, the source of the frozen modules it
consumes (pasted, not linked), the rules from `CONTRIBUTING.md`, and the command
`just check`.

## 13. Open decisions

- [ ] **Confirm cutting Rothermel (WP 2.1) to a stretch goal** now that Gate 1 has slipped.
- [ ] Whether burned wildland carries a dollar value (`usd_per_ha_wildland`, currently 0).
- [ ] Dollar value per building (currently a 400 000 USD placeholder).
- [ ] Residence times per fuel model and their source (needed by WP 1.3). Candidate: Anderson
      (1969), `t_r = 384 / sigma` minutes with `sigma` the surface-area-to-volume ratio in 1/ft.
- [ ] Cost figures with sources: crew, dozer, engine and aircraft hourly rates; water and
      retardant per litre (WP 4.6).
- [ ] Module owners for the three team members.
- [ ] Validation fire B, only if Gate 2 lands early.

Resolved: study region and fire A (Chimney Fire); Rothermel via library, not hand-written;
package name; weather source.

## 14. Plan review, Nov 10, 2026

The two plan documents were merged and each claim checked against the Phase 0 code.

**Implemented as planned:** WPs 0.1-0.7 all exist and their tests pass (types and stencils,
grid container with save/load, spread contract and registry, config schemas and loaders,
synthetic generators, three scenarios, cost placeholders, `validate` CLI command).

**Problems found and fixed in Phase 0 code (WP 0.8):**

| Problem | Fix |
| --- | --- |
| The memoryless ignition kernel made the front run 3 to 4.5 times faster than R and depend on dt, contradicting the plan's central claim; physics tests (c) and Gate 1 could not have passed | Gamma crossing-time kernel: `ignition_probability(..., age=, shape=)`; new optional `spread.arrival_shape` (default 20); evidence script `scripts/kernel_front_speed.py` |
| `make_spread_rule("constant")` would fail unless the caller had imported the rule module | Registry imports `wildfire_sim.spread.<name>` on first use; `register` keeps the decorated class's type |
| `ScenarioConfig.n_steps` lost a step to float rounding (3600 s / 0.1 s gave 35999) | Rounding-tolerant floor |
| Ignition cells outside a synthetic grid passed validation | Checked in `ScenarioConfig` |
| `ResourceConfig` has an `engine` type but `CostConfig` had no engine rate | `engine_usd_per_h` added (placeholder in `costs.yaml`) |
| `Grid.validate` accepted NaN in slope, elevation and asset value, and negative asset values | Finite and non-negative checks |
| `wildfire-sim validate` printed a traceback for malformed YAML, bad generator arguments or a missing cached grid | Reported as errors with exit code 1 |

**Problems found and fixed in the plan itself:**

- Burn duration equal to the residence time alone extinguishes the fire on 30 m cells
  (section 2, WP 1.3).
- Physics test (a) used an isoperimetric quotient that a raster disc cannot satisfy; test (b)
  needed a wind-aware rule but WP 1.4 consumed only the constant rule; a front-speed test (f)
  was missing.
- The Alexandridis rule's conversion from probability to rate was unspecified (WP 1.2).
- `configs/residence_time.yaml` was referenced but owned by no WP (now WP 1.3).
- Diagonal leaks through 8-connected lines and knight-move jumps over one-cell barriers
  (section 2, WPs 1.3 and 4.1).
- Breaching was undefined for cells that are already unburnable (WP 4.3).
- Intensity for breaching and crew safety had no source without Rothermel (Phase 4 note).
- The NumPy engine must not evaluate the kernel over the whole grid (section 6, WP 1.3).
- Gate 1 was missed; the schedule is re-baselined and the Rothermel fallback rule applies
  (section 8).

## 15. Sources

- [Alexandridis et al. 2008, CA model of the Spetses Island fire](https://dspace.lib.ntua.gr/xmlui/handle/123456789/18520)
- [Alexandridis et al. 2011, CA spread with suppression tactics (Int. J. Wildland Fire)](https://publish.csiro.au/wf/ExportCitation/wf09119)
- [Russo et al. 2013, same CA applied to the 2008 Rhodes fire without retuning](https://cetjournal.it/index.php/cet/article/view/6182)
- [CA wildfire simulator applied to the 2016 Chimney Fire (arXiv 2507.00761)](https://arxiv.org/pdf/2507.00761)
- [pyretechnics: Python fire behavior equations](https://ost.ecosyste.ms/projects/351112)
- [pyrothermel: Python bindings for Rothermel models](https://pypi.org/project/pyrothermel/)
- [Forest-fire model (the original baseline)](https://en.wikipedia.org/wiki/Forest-fire_model)
