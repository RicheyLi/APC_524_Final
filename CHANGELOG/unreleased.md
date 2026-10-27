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
  - `docs/plan.md` (dated schedule, gates, work packages) and `docs/revised_plan.md` (model design).
- Optional dependency extras: `fast` (numba), `geo`, `rothermel`, `opt`, `all`.

## Changed

- Package renamed from `wildfire_spread_and_suppression_model` to `wildfire_sim`; CLI is `wildfire-sim`.
- Primary Python is 3.12; CI tests 3.12 and 3.13, with 3.14 allowed to fail.
- Repository URLs point at `RicheyLi/APC_524_Final`; PyPI publishing workflow removed.
- `.gitattributes` enforces LF line endings.

## Fixed

- `pyproject.toml` authors/maintainers arrays were missing commas.

## Scaffold

Generated from [Cookiecutter PyPackage](https://github.com/audreyfeldroy/cookiecutter-pypackage): CI (Ruff, ty, pytest, coverage), CodeQL, Dependabot, Zizmor, Zensical docs, justfile, issue/PR templates, contributing guide, code of conduct, security policy, MIT license.
