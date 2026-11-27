# Contributing

This is the development workflow for the wildfire-sim team (APC 524 final project).
Work is organised as **work packages (WPs)** defined in [docs/plan.md](docs/plan.md).
Each WP is one branch and one pull request.

## Set up

```sh
git clone https://github.com/RicheyLi/APC_524_Final.git
cd APC_524_Final
uv sync                       # core + dev tools (ruff, ty, pytest, coverage)
uv sync --extra all           # also numba, geospatial stack, pyrothermel, optuna
```

`uv` creates `.venv/` inside the repo. Never commit it.

Install the commit hooks once per clone:

```sh
uvx pre-commit install
```

Besides formatting, they refuse a commit that contains a NASA Earthdata token, either a token
file (even one added with `git add -f`) or a token pasted into any text file. The repository is
public, so a token that reaches it must be revoked at <https://urs.earthdata.nasa.gov>.

On Windows or WSL set `git config core.autocrlf false`; `.gitattributes` enforces LF endings so
checkouts stay clean.

## Daily loop

```sh
git checkout -b wp-1.3-sim-loop   # branch name = WP id + short slug
just fix                          # ruff format + autofix
just check                        # ruff, ty, pytest  -- must pass before you push
git push -u origin wp-1.3-sim-loop
```

Then open a pull request. CI runs lint, type-check and tests on Python 3.12 and 3.13
(3.14 is allowed to fail). The PR must be green and reviewed by one other person before merge.

## Rules for every work package

1. **One WP per branch/PR.** Touch only the files the WP owns, plus its tests, plus
   `CHANGELOG/unreleased.md`. If you need a change in someone else's module, open a
   separate small PR for it.
2. **Tests are offline.** Nothing under `tests/` may touch the network. Code that downloads
   data is tested with mocks; live checks are marked `@pytest.mark.network` and only run with
   `pytest --run-network`.
3. **Parameters live in `configs/`, not in code.** Any physical or cost constant goes in a
   YAML/CSV under `configs/` with a `source:` citation. Code reads it through
   `wildfire_sim.config`.
4. **Units in docstrings.** Every public function states the units of its inputs and outputs
   (m, s, m/s, kW/m, USD/h). Arrays state their shape, e.g. `(8, H, W)`.
5. **Determinism.** Any randomness goes through a `numpy.random.Generator` passed in by the
   caller. Never call `np.random.*` module functions.
6. **Interfaces are frozen.** `wildfire_sim.types`, `wildfire_sim.grid`, `wildfire_sim.spread.base`
   and `wildfire_sim.config` are the contracts other WPs build on. Changing them needs a
   PR that updates every caller and gets explicit sign-off.
7. **Update the changelog.** Add a line under `CHANGELOG/unreleased.md` describing the change.

## Tests

```sh
just test                 # all tests on 3.12
just test tests/test_grid.py -k stencil
just coverage             # html report in htmlcov/
```

Physics tests (`tests/test_physics.py`) use fixed seeds and documented tolerances. If you
change a tolerance, say why in the test docstring.

## Docs

```sh
just docs-serve           # live preview at http://localhost:8000
just docs-build           # strict build, fails on warnings
```

API docs are generated from docstrings, so write them. Longer design notes go in `docs/`.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.
