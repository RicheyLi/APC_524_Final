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
  - `docs/plan.md` (model design, dated schedule, gates, work packages).
- Optional dependency extras: `fast` (numba), `geo`, `rothermel`, `opt`, `all`.
- Phase 0 review (WP 0.8, see `docs/plan.md` section 14):
  - `ignition_probability` takes `age` and `shape`: cell-to-cell crossing times are gamma-distributed with mean `d / R`. New optional scenario field `spread.arrival_shape` (default 20).
  - `scripts/kernel_front_speed.py`: measures front speed against the nominal rate of spread.
  - `CostConfig.engine_usd_per_h` (placeholder value in `configs/costs.yaml`).
  - `make_spread_rule` imports `wildfire_sim.spread.<name>` on first use.

## Changed

- `docs/revised_plan.md` merged into `docs/plan.md`; the schedule is re-baselined after Gate 1 (Nov 9) was missed.
- `ScenarioConfig` rejects ignition cells outside a synthetic grid.
- `Grid.validate` rejects non-finite static layers and negative asset values.
- `load_yaml` raises `ValueError` for malformed YAML; `wildfire-sim validate` reports grid-build errors instead of a traceback.

- Package renamed from `wildfire_spread_and_suppression_model` to `wildfire_sim`; CLI is `wildfire-sim`.
- Primary Python is 3.12; CI tests 3.12 and 3.13, with 3.14 allowed to fail.
- Repository URLs point at `RicheyLi/APC_524_Final`; PyPI publishing workflow removed.
- `.gitattributes` enforces LF line endings.

## Fixed

- The memoryless ignition kernel (`p = 1 - exp(-R dt / d)`) moved the fire front 3 to 4.5 times faster than `R` and depended on the time step. It remains available as `arrival_shape = 1` only.
- `ScenarioConfig.n_steps` lost a step to floating-point rounding (3600 s at 0.1 s gave 35999).

- `pyproject.toml` authors/maintainers arrays were missing commas.

## Scaffold

Generated from [Cookiecutter PyPackage](https://github.com/audreyfeldroy/cookiecutter-pypackage): CI (Ruff, ty, pytest, coverage), CodeQL, Dependabot, Zizmor, Zensical docs, justfile, issue/PR templates, contributing guide, code of conduct, security policy, MIT license.
