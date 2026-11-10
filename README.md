# wildfire-sim

APC 524 final project: a cellular-automaton (CA) wildfire spread model whose ignition
probabilities come from a physical rate of spread, with firefighting modelled as costed
resources acting on the grid, and a black-box optimizer that tunes reactive suppression
policies on Monte Carlo ensembles.

* [GitHub](https://github.com/RicheyLi/APC_524_Final) | [Documentation](https://richeyli.github.io/APC_524_Final/)
* Richey Li, Oscar Ansted and Sunny Lin, Princeton University
* MIT License

## What it does

* 30 m grid with static (fuel, terrain, land use, assets), weather, state and suppression layers.
* Swappable spread rules behind one interface: constant rate, Alexandridis et al. (2008), Rothermel.
* Cell-to-cell crossing times drawn from a gamma distribution with mean `d / R`, so the front moves at the physical rate of spread and results do not depend on the time step.
* Suppression tactics (indirect/parallel lines, ground crews, aircraft) with dispatch delays, work rates, costs and line breaching.
* Ensembles of seeded runs reported as burn-probability maps and cost distributions.
* Optuna search over parameterized reactive policies; calibration against the 2016 Chimney Fire and a test on a second fire.

See [docs/plan.md](docs/plan.md) for the model design, the milestones and the work packages.

## Installation

```sh
git clone https://github.com/RicheyLi/APC_524_Final.git
cd APC_524_Final
uv sync                 # core + dev tools
uv sync --extra all     # + numba, geospatial stack, pyrothermel, optuna
```

## Usage

Run from the repository root; outputs go to `runs/` unless you pass `--out`.

```sh
uv run wildfire-sim --help
uv run wildfire-sim validate configs/scenarios/flat_no_wind.yaml
uv run wildfire-sim run configs/scenarios/steady_wind.yaml --seed 1 --out runs/wind --gif
uv run wildfire-sim ensemble configs/scenarios/steady_wind.yaml --n 20 --workers 2
uv run wildfire-sim plot runs/wind
```

`run` writes the final grid, arrival times, an event log, a summary, provenance and maps;
`ensemble` writes a burn-probability map and burned-area statistics; `plot` redraws the
figures of a saved run or ensemble. See [docs/usage.md](docs/usage.md) for every option,
the output files and the Python API.

## Development

```sh
just check   # ruff format/lint, ty type-check, pytest
just test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch/PR workflow and the rules every work
package follows.

## Credits

Built from the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) template.
