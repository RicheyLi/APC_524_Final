# Project plan

Dated work plan for wildfire-sim. The model design and rationale are in
[revised_plan.md](revised_plan.md); this page is the schedule, the gates, and the work
packages (WPs) that implement it. Each WP is one branch and one pull request (see
`CONTRIBUTING.md`).

Submission deadline: **early December 2026 (Dec 1-5)**. That leaves five weeks from the
start of implementation (Oct 27), so the original eight-week plan is compressed and some
items are cut (see *Scope decisions*).

## Timeline and gates

| Week | Dates | Work | Gate |
| --- | --- | --- | --- |
| 1 | Oct 27 - Nov 2 | Phase 0 (done), Phase 1 WPs 1.1-1.4 | |
| 2 | Nov 3 - 9 | Phase 1 WPs 1.5-1.8; Phase 4 WPs 4.1-4.3 in parallel | **Gate 1 (Nov 9)** |
| 3 | Nov 10 - 16 | Phase 2 WPs 2.1-2.6 (Chimney Fire region); Phase 4 WPs 4.4-4.6 | |
| 4 | Nov 17 - 23 | Phase 3 calibration; Phase 4 WP 4.5 optimizer | **Gate 2 (Nov 23)** |
| 5 | Nov 24 - Dec 1 | WP 4.7 experiments; Phase 5 deliverables; buffer | Submit |

**Gate 1 - simulator works.** The Alexandridis rule passes every physics test in
`tests/test_physics.py` on synthetic landscapes; an ensemble and a GIF are reproducible from a
YAML file and a seed; one run on the 333 x 333 reference grid takes under 2 s.

**Gate 2 - model is validated.** Calibrated on the 2016 Chimney Fire with a reported Jaccard
index and arrival-time error; decision recorded on whether Rothermel is kept or the project
finishes on the Alexandridis rule.

## Scope decisions

Fixed on Oct 27, 2026:

| Decision | Choice |
| --- | --- |
| Rothermel implementation | `pyrothermel` library behind the `SpreadRule` interface (optional extra `rothermel`). Hand-written equations are a stretch goal. **Risk:** only a Windows wheel is published; Linux/macOS need a C++ compiler to build it. The wrapper imports lazily so the package works without it. |
| Package name | `wildfire_sim` (CLI `wildfire-sim`). |
| Validation fire A | 2016 Chimney Fire, San Luis Obispo County, CA (published CA comparison: arXiv 2507.00761). |
| Validation fire B | **Stretch.** Picked only if Gate 2 lands early. The report states the single-fire limitation. |
| Weather | Hourly scalar series per run (gridMET daily, optionally disaggregated). HRRR is out. |
| Primary Python | 3.12. 3.13 tested in CI; 3.14 allowed to fail. |
| Out of scope unless everything else is done | Spotting, reinforcement learning, non-US data, 16-neighbour stencil as default (the stencil exists in `types.py` for the circularity experiment). |
| Minimum suppression set | Ground crews and control lines. Aircraft may slip to week 4. |

## Module map

```text
src/wildfire_sim/
  types.py            CellState, LandUse, Stencil (8/16)           [frozen, Phase 0]
  grid.py             Grid layers, Weather, WeatherSeries          [frozen, Phase 0]
  config.py           Pydantic schemas, YAML loaders               [frozen, Phase 0]
  spread/base.py      SpreadRule protocol, ignition kernel         [frozen, Phase 0]
  spread/constant.py                                               [WP 1.1]
  spread/alexandridis.py                                           [WP 1.2]
  spread/rothermel.py                                              [WP 2.1]
  sim.py              time loop, event log, RunResult              [WP 1.3]
  viz.py              maps, burn-probability plots, GIF            [WP 1.5]
  ensemble.py         parallel seeded runs                         [WP 1.6]
  cli.py              run / ensemble / plot / validate             [WP 1.8]
  data/synthetic.py   test landscapes                              [Phase 0]
  data/landfire.py, terrain.py, weather.py, osm.py, validation.py  [WP 2.2-2.7]
  metrics.py          burned area, overlap, cost                   [WP 2.8]
  suppression/geometry.py, resources.py, tactics.py, policies.py  [WP 4.1-4.4]
  optimize.py         Optuna study                                 [WP 4.5]
configs/              scenarios/*.yaml, costs.yaml, fuel_models.csv, alexandridis.yaml
scripts/              bench.py, download_data.py, calibrate.py, run_experiments.py
tests/                offline, deterministic; fixtures/ holds committed .npz grids
```

Frozen modules are the contracts every other WP builds on. Changing them needs a PR that
updates all callers and explicit sign-off.

## Work packages

Each row is self-contained: a model or teammate handed one row plus the frozen modules
should be able to complete it without further context. "Consumes" lists the interfaces the
WP relies on; "Acceptance" is what the PR's tests must show.

### Phase 0 - Foundations (complete, Oct 27)

| WP | Deliverable | Status |
| --- | --- | --- |
| 0.1 | Repo hygiene: LF line endings, rename to `wildfire_sim`, correct URLs, 3.12 primary, `AGENTS.md`, data dirs ignored | done |
| 0.2 | Dependency groups (`fast`, `geo`, `rothermel`, `opt`, `all`); pyrothermel API verified | done |
| 0.3 | `types.py` (`CellState`, `LandUse`, `Stencil`, `STENCIL_8`, `STENCIL_16`), `grid.py` (`Grid`, `Weather`, `WeatherSeries`, save/load) | done |
| 0.4 | `spread/base.py`: `SpreadOutput`, `SpreadRule`, `ignition_probability`, `neighbour_ignition_probability`, `max_stable_dt`; registry in `spread/__init__.py` | done |
| 0.5 | `config.py`: `ScenarioConfig`, `CostConfig` with `CitedValue`, loaders; three scenario YAMLs; `configs/costs.yaml` placeholders | done |
| 0.6 | `data/synthetic.py` generators, `build_grid`, `tests/conftest.py`, committed fixture `tests/fixtures/flat_21x21.npz` | done |
| 0.7 | This page; `docs/revised_plan.md` | done |

### Phase 1 - Simulator core

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 1.1 | `spread/constant.py` | 0.4 | `ConstantRule(ros: float)` registered as `"constant"`: same `ros` (m/s) in every direction for burnable cells, zero elsewhere, times `(1 - retardant)`; `intensity`/`flame_length` zero | Output validates; non-burnable cells zero; retardant halves ros |
| 1.2 | `spread/alexandridis.py`, `configs/alexandridis.yaml` | 0.4 | Alexandridis et al. 2008 rule: `p = p_h (1 + p_veg)(1 + p_den) p_w p_s` converted to an equivalent ros per direction; wind factor `exp(c1 V) exp(c2 V (cos(theta) - 1))`, slope factor `exp(a * slope_angle)`; fuel-model-to-(veg, density) mapping table and constants cited in the YAML | Downwind ros > upwind; upslope > downslope; constants loaded from YAML, not hard-coded |
| 1.3 | `sim.py` | 0.3-0.6, any rule | `run(config, seed, rule=None) -> RunResult` with `RunResult(grid, arrival_time, burned_mask, events, elapsed_s)`; per step: weather lookup, `rate_of_spread`, `ignition_probability`, `neighbour_ignition_probability`, draw with `rng.random`, ignite FUEL cells, decrement `burn_timer` (residence time from `configs/residence_time.yaml` by fuel model), BURNING to BURNED at zero; `events` is a list of `(t, kind, row, col)`; ignition from `IgnitionConfig` | Water/urban never burn; identical output for identical seed; fire ends when no cell is burning; arrival_time NaN exactly where never burned |
| 1.4 | `tests/test_physics.py` | 1.1, 1.3 | (a) no wind: isoperimetric quotient `4 pi A / P^2 >= 0.85` of the burned region after fixed time, 20 runs averaged, seed 0; (b) wind from west: east extent / west extent `>= 1.5`; (c) dt halving: KS test on arrival times along a transect, 50 runs each, `p > 0.01`; (d) complete 3-cell-wide fireline ring: no cell outside ignites; (e) river scenario: no cell east of river ignites | All five pass with thresholds stated in docstrings |
| 1.5 | `viz.py` | `RunResult`, `Grid` | `plot_state(grid, ax)`, `plot_arrival(result)`, `plot_burn_probability(stack)`, `animate(results_frames, path.gif)` using matplotlib + imageio; colour map keyed on `CellState` | Smoke tests write PNG/GIF to `tmp_path`; no `plt.show()` |
| 1.6 | `ensemble.py` | 1.3 | `run_ensemble(config, n_runs, base_seed, n_workers) -> EnsembleResult(burn_probability, arrival_times (n,H,W), burned_area_m2 (n,), seeds)`; seeds from `np.random.SeedSequence(base_seed).spawn(n)`; `multiprocessing` pool; `running_mean(burned_area)` helper | n=4 parallel equals n=4 serial bit-for-bit; seeds recorded |
| 1.7 | `sim_fast.py`, `scripts/bench.py` | 1.3 | Numba kernel updating only cells within the stencil halo of burning cells; selected by `config.engine = "numba"` (add optional field with default `"numpy"`); benchmark prints s/run on 333 x 333, 1440 steps | Same result as NumPy engine for same seed; `<= 0.5 s/run` with numba, documented numbers in `docs/performance.md` |
| 1.8 | `cli.py` | 0.5, 1.3, 1.5, 1.6 | `run SCENARIO --seed --out DIR` (writes `grid.npz`, `events.csv`, `summary.json`, `final.png`); `ensemble SCENARIO --n --out`; `plot DIR` | `CliRunner` tests for each command on `flat_no_wind` |

### Phase 2 - Rothermel and real data

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 2.1 | `spread/rothermel.py`, `configs/fuel_models.csv` | 0.4, pyrothermel | `RothermelRule` registered as `"rothermel"`: head ros from `pyrothermel.PyrothermelRun.run_surface_fire_in_direction_of_max_spread` with `FuelModel.from_existing(number)` and `MoistureScenario` built from `Weather`; midflame wind via pyrothermel's `wind_input_mode`; directional ros from Anderson (1983) ellipse with length-to-breadth `LB = 0.936 exp(0.2566 U) + 0.461 exp(-0.1548 U) - 0.397` (U in m/s, capped at 8); Byram intensity and flame length from pyrothermel outputs; results cached per (fuel, moisture, wind, slope) tuple; lazy import so the package works without pyrothermel | Head ros for fuel models GR1, SH2, TL3 at a documented moisture/wind within 10% of BehavePlus values recorded in the test; physics tests 1.4 pass unchanged; `pytest.importorskip("pyrothermel")` |
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
| 3.1 | `scripts/calibrate.py`, `configs/calibrated.yaml` | 2.1, 2.6-2.8, optuna | Optuna study over `ros_multiplier` in [0.3, 3] and ellipse `lb_scale` in [0.5, 2] maximizing mean Jaccard over 20 fixed seeds on the Chimney grid; writes best parameters | Runs on the committed crop in CI with 3 trials x 2 seeds (marked `slow`) |
| 3.2 | `docs/validation.md`, `notebooks/validation.ipynb` | 3.1 | Burn-probability map vs. MTBS perimeter, arrival-time scatter, metrics table, discussion of the suppression confound; Rothermel vs. Alexandridis comparison | Figures regenerate from a script |

### Phase 4 - Suppression and strategy search

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 4.1 | `suppression/geometry.py` | 0.3 | `perimeter_cells(burned_mask)`, `rasterize_polyline(points, width_cells, shape)`, `offset_perimeter(burned_mask, offset_m, dx)` via `scipy.ndimage.distance_transform_edt`, `head_direction(arrival_time, window_s)` from recent ignition centroid drift, `distance_to_assets(grid)` | Hand-drawn shapes; offset of a filled disc is a ring at the right radius |
| 4.2 | `suppression/resources.py` | 0.5 | `Resource` dataclass from `ResourceConfig`; `Fleet` tracks dispatch time, arrival, hours worked, litres dropped; cost accrual per step | Hours and cost match hand calculation |
| 4.3 | `suppression/tactics.py` | 4.1, 4.2, 1.3 hooks | `IndirectLine(path)`, `ParallelLine(offset, flank)`, `DirectAttack(crews)`, `AirDrop(footprint, factor, decay_s)`; line segments become `fireline = True` at the build rate; breach probability `p_b = sigmoid((intensity - I_crit(width)) / s)`; crews skip cells with `flame_length > max_flame_length_m`; `sim.py` gains a `tactics` hook called each step | Complete wide line stops fire; breach probability rises with intensity; crews refuse unsafe edges; retardant decays to zero |
| 4.4 | `suppression/policies.py` | 4.1-4.3 | `Observation` dataclass (perimeter length, head bearing, head ros, max intensity, distance to nearest asset, burned area, elapsed time); `Policy` protocol `act(obs, fleet, t) -> list[Action]`; baselines `NoSuppression`, `DirectOnly`, `IndirectOnly`; `ReactivePolicy(params)` implementing the four rules in revised_plan.md | Each baseline runs end to end on `with_town` |
| 4.5 | `optimize.py` | 4.4, 1.6, 2.8, optuna | `optimize_policy(config, n_trials, runs_per_trial, seeds)`: objective is mean total cost with P90 reported; common random numbers; pruning after the first 5 runs; study saved to SQLite | Toy problem recovers a known optimum |
| 4.6 | `configs/costs.yaml`, `configs/resources.yaml` | 0.5 | Replace every PLACEHOLDER with a cited value (NWCG, USFS cost guides); resource build rates and safety limits | Validators pass; no `PLACEHOLDER` string remains |
| 4.7 | `scripts/run_experiments.py`, `docs/results.md` | everything | Optimized policy vs. three baselines on the Chimney grid with a synthetic town; cost distributions, burn-probability maps, convergence plot of running mean vs. run count | Figures regenerate from committed configs and seeds |

### Phase 5 - Deliverables

README example run with a GIF, `docs/` page per module, report draft assembled from
`docs/validation.md` and `docs/results.md`, final changelog, tagged release.

## Dispatch batches

Once Phase 0 is merged, these can run concurrently, one branch each:

- **Batch A (week 1-2):** 1.1, 1.2, 1.3, 1.5, 4.1, 4.2
- **Batch B (after 1.3 merges):** 1.4, 1.6, 1.7, 1.8, 4.3, 4.4
- **Batch C (week 3):** 2.1-2.8, 4.5, 4.6
- **Batch D (week 4-5):** 3.1, 3.2, 4.7, Phase 5

A handoff prompt for one WP contains: the WP row above, the source of the frozen modules it
consumes (pasted, not linked), the rules from `CONTRIBUTING.md`, and the command
`just check`.

## Open decisions

- [ ] Whether burned wildland carries a dollar value (`usd_per_ha_wildland`, currently 0).
- [ ] Dollar value per building (currently a 400 000 USD placeholder).
- [ ] Residence times per fuel model and their source (needed by WP 1.3).
- [ ] Validation fire B, only if Gate 2 lands early.
