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
* Swappable spread rules behind one interface: constant rate, Alexandridis et al. (2008), Rothermel (stretch).
* Per-neighbour ignition probability from a gamma-distributed crossing time with mean `d / R`, so the front moves at the physical rate of spread and results do not depend on the time step.
* Suppression tactics (indirect/parallel lines, ground crews, aircraft) with dispatch delays, work rates, costs and line breaching.
* Ensembles of seeded runs reported as burn-probability maps and cost distributions.
* Optuna search over parameterized reactive policies; calibration against the 2016 Chimney Fire.

See [docs/plan.md](docs/plan.md) for the model design, the dated schedule and the work packages.

## Installation

```sh
git clone https://github.com/RicheyLi/APC_524_Final.git
cd APC_524_Final
uv sync                 # core + dev tools
uv sync --extra all     # + numba, geospatial stack, pyrothermel, optuna
```

## Usage

```sh
uv run wildfire-sim --help
uv run wildfire-sim validate configs/scenarios/flat_no_wind.yaml
```

Running scenarios, ensembles and plots arrives with Phase 1 (see the plan).

## Development

```sh
just check   # ruff format/lint, ty type-check, pytest
just test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch/PR workflow and the rules every work
package follows.

## Credits

Built from the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) template.
