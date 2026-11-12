# Unreleased

These are the changes that will go out in the next release.

## Added

- Phase 0 foundations (see `docs/plan.md`):
  - `wildfire_sim.types`: `CellState`, `LandUse`, `Stencil` with 8- and 16-direction stencils.
  - `wildfire_sim.grid`: `Grid` layer container with validation, masks and `.npz` save/load; `Weather` and piecewise-constant `WeatherSeries`.
  - `wildfire_sim.spread.base`: `SpreadRule` protocol, `SpreadOutput`, the `p = 1 - exp(-R dt / d)` ignition kernel, neighbour combination and `max_stable_dt`; rule registry.
  - `wildfire_sim.config`: Pydantic schemas (`ScenarioConfig`, `CostConfig` with cited values) and YAML loaders.
  - `wildfire_sim.data.synthetic`: `flat_uniform`, `slope_ramp`, `with_river`, `with_town` generators and `build_grid`.
  - `wildfire-sim validate` CLI command.
  - Scenario files `configs/scenarios/*.yaml`, `configs/costs.yaml` (placeholders), committed test fixture `tests/fixtures/flat_21x21.npz`.
  - `docs/plan.md` (model design, milestones, work packages).
- Optional dependency extras: `fast` (numba), `geo`, `rothermel`, `opt`, `all`.
- Phase 0 review (WP 0.8, see `docs/plan.md` section 14):
  - `ignition_probability` takes `age` and `shape`: cell-to-cell crossing times are gamma-distributed with mean `d / R`. New optional scenario field `spread.arrival_shape` (default 20).
  - `scripts/kernel_front_speed.py`: measures front speed against the nominal rate of spread.
  - `CostConfig.engine_usd_per_h` (placeholder value in `configs/costs.yaml`).
  - `make_spread_rule` imports `wildfire_sim.spread.<name>` on first use.
  - `link_rate_of_spread`: harmonic mean of source and target cell rates per link, so fire does not enter slow fuel or retardant at the fast cell's speed.
  - `draw_crossing_thresholds`: one gamma threshold per link per run, the form the simulator will use (exact under changing rates, common random numbers across policies).
- Phase 1 (simulator core, see `docs/plan.md` section 12):
  - `wildfire_sim.config.configs_dir()` locates the cited parameter tables in `configs/`.
  - WP 1.1: `wildfire_sim.spread.constant.ConstantRule`, registered as `constant`.
  - WP 1.2: `wildfire_sim.spread.alexandridis.AlexandridisRule`, registered as `alexandridis` (wind and slope factors of Alexandridis et al. 2008 as a rate), with cited constants and the fuel-model mapping in `configs/alexandridis.yaml`; `steady_wind.yaml` now uses it.
  - WP 1.5: `wildfire_sim.viz`: state, arrival-time and burn-probability maps and GIF animation, off-screen, with a colour-blind-safe palette.
  - WP 1.3: `wildfire_sim.sim.run` and `RunResult`: per-link threshold kernel with interpolated arrival times and an exact in-step cascade, blocked links through barriers, two-clock burn-out, step hooks on a separate random stream, run provenance; `configs/residence_time.yaml` and `configs/kernel_correction.yaml` (placeholders).
  - WP 1.6: `wildfire_sim.ensemble.run_ensemble` (seeded, optionally parallel, bit-identical serial and parallel) with per-run input perturbations from the new optional `ensemble` scenario section; `running_mean`, `mean_ci`, `runs_for_precision`.
  - WP 1.8: `wildfire-sim run`, `ensemble` and `plot` commands; `wildfire_sim.io` writes and reads run and ensemble directories (arrays, events, strict-JSON summary and provenance, resolved scenario, figures). `sim.model_provenance` gives the code and kernel provenance shared by all runs of a scenario.
  - WP 1.4: `tests/test_physics.py` (circularity, ellipse shape, dt independence, barriers, front speed, slow-fuel strip, wind shift); measured front-speed corrections in `configs/kernel_correction.yaml` from `scripts/kernel_front_speed.py`; `docs/lattice_study.md` with `scripts/lattice_study.py`. The study shows neither stencil reproduces an ellipse to 10 % beyond a length-to-breadth ratio of about 1.5; the unmet target is recorded as a strict expected failure.
  - WP 1.9: Hypothesis property tests of the kernel, grid, simulator, config and ensemble invariants (`tests/test_properties.py`); golden-run regression tests with committed arrival-time arrays and a regeneration script (`tests/golden/`); coverage floor raised from 50 % to 85 % (measured 94 %).
  - WP 1.7: optional compiled engine `wildfire_sim.sim_fast` selected with the new scenario field `engine: numba` (needs the `fast` extra; bit-identical to the NumPy engine in every tested case, about 8 to 10 times faster); `scripts/bench.py` with a machine-normalised regression check; profile and measured costs in `docs/performance.md`; a CI job for engine parity and the benchmark.

## Changed

- **Defaults changed after the lattice study:** `ScenarioConfig.stencil` 8 to 16 and `spread.arrival_shape` 20 to 200. Fires keep their shape better under wind (head and flank within 7 % of the analytic ellipse at a length-to-breadth ratio of 1.5 and 13 % at 3) and the front-speed correction is close to 1; runs take about twice as long. Scenario files that do not set these fields give different results; the shipped scenarios now use 16 neighbours and the golden arrays were regenerated.
- `configs/scenarios/flat_no_wind.yaml`: duration 7200 s to 5400 s so the fire stays inside the grid.
- `docs/revised_plan.md` merged into `docs/plan.md`. The plan has no schedule: work is ordered by dependencies and judged by undated milestones. A second design review (section 14) strengthened the kernel, Rothermel, data, validation, suppression and testing sections.
- `ScenarioConfig` rejects ignition cells outside a synthetic grid.
- `Grid.validate` rejects non-finite static layers and negative asset values.
- `load_yaml` raises `ValueError` for malformed YAML; `wildfire-sim validate` reports grid-build errors instead of a traceback.

- Package renamed from `wildfire_spread_and_suppression_model` to `wildfire_sim`; CLI is `wildfire-sim`.
- Primary Python is 3.12; CI tests 3.12 and 3.13, with 3.14 allowed to fail.
- Repository URLs point at `RicheyLi/APC_524_Final`; PyPI publishing workflow removed.
- `.gitattributes` enforces LF line endings.

- `docs/plan.md`: fourth design review (section 14). New work packages 1.10 (dormant sources: a fire survives a period of zero spread rate), 1.11 (grid-edge detection), 1.12 (front-normal intensity) and 2.1d (lattice-fitted ellipse rates and a length-to-breadth cap); WP 4.3 draws suppression randomness once per cell, not per step; fire grids are sized from the observed perimeter; fire B criteria restated; spotting (6.1) is required by the strategy experiments (4.7). No code changes.

## Fixed

- CodeQL workflow: `init` and `analyze` are on the same `github/codeql-action` version again (a Dependabot bump had moved only `analyze`, which makes the job fail); Dependabot now groups the two.
- `ensemble.mean_ci` returned NaN bounds for finite samples near the float64 limit (found by a property test).
- `link_rate_of_spread`: a knight link (16-neighbour stencil) now uses the harmonic mean over the four cells its segment crosses. Using only the two end cells let fire tunnel through one cell of slow fuel at each edge of a strip (delay 12 % short).
- `Grid.apply_unburnable` no longer overwrites BURNING or BURNED cells (a fireline placed under a burning cell erased it from the burned area).
- The memoryless ignition kernel (`p = 1 - exp(-R dt / d)`) moved the fire front 3 to 4.5 times faster than `R` and depended on the time step. It remains available as `arrival_shape = 1` only.
- `ScenarioConfig.n_steps` lost a step to floating-point rounding (3600 s at 0.1 s gave 35999).

- `pyproject.toml` authors/maintainers arrays were missing commas.

## Scaffold

Generated from [Cookiecutter PyPackage](https://github.com/audreyfeldroy/cookiecutter-pypackage): CI (Ruff, ty, pytest, coverage), CodeQL, Dependabot, Zizmor, Zensical docs, justfile, issue/PR templates, contributing guide, code of conduct, security policy, MIT license.
