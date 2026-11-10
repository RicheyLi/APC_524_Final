# Usage

Every run is defined by a YAML scenario file (see `configs/scenarios/`) plus a seed. Run the
commands below from the repository root: relative paths inside scenario files resolve against
the current directory, and outputs go to `runs/` (git-ignored) unless you pass `--out`.

`uv run wildfire-sim --help` lists the commands, and `uv run wildfire-sim COMMAND --help`
shows each command's options with their units and defaults.

## Validate a scenario

```sh
uv run wildfire-sim validate configs/scenarios/flat_no_wind.yaml
```

Prints the grid, spread rule, time step and seed, builds the grid and reports its size.

## Run one simulation

```sh
uv run wildfire-sim run configs/scenarios/flat_no_wind.yaml
uv run wildfire-sim run configs/scenarios/steady_wind.yaml --seed 1 --out runs/wind --gif
```

* `--seed N` overrides the scenario's `seed`; the same seed always gives the same fire.
* `--out DIR` defaults to `runs/<scenario name>-seed<N>`.
* `--gif` also writes an animation (`--frames`, default 40, and `--fps`, default 10 frames/s).
* An output directory that exists and is not empty is refused. `--force` overwrites the files
  this command writes (and removes a stale `fire.gif`), leaving any other file alone:

  ```sh
  uv run wildfire-sim run configs/scenarios/flat_no_wind.yaml --force
  ```

The command prints a summary table. The run directory contains:

| File | Contents |
| --- | --- |
| `grid.npz` | Final grid (`Grid.load` reads it): fuel, terrain, land use, cell state, burn timer, intensity, arrival time. |
| `arrival_time.npy` | `(H, W)` float64, time each cell ignited, s since the start; NaN = never burned. |
| `events.csv` | One row per ignition, in time order: `t_s,kind,row,col`; `kind` is `ignition` (initial) or `spread`. |
| `summary.json` | Scenario name, seed, burned cells, burned area (m^2 and ha), simulated time (s), steps executed, time of last ignition (s), wall-clock time (s), count of cells in each final state. |
| `provenance.json` | Resolved config, seed, git commit, Python/numpy/scipy/package versions, spread rule, stencil, kernel correction, residence times, steps executed. |
| `scenario.yaml` | The resolved scenario with the seed used; `wildfire-sim run DIR/scenario.yaml` reproduces the run. |
| `final.png` | Map of the final cell states. |
| `arrival.png` | Map of arrival time with isochrones. |
| `fire.gif` | Only with `--gif`: the spread replayed from the arrival times. |

All JSON files are strict JSON: missing or undefined numbers are `null`, never `NaN`.

## Run an ensemble

```sh
uv run wildfire-sim ensemble configs/scenarios/steady_wind.yaml --n 20 --workers 2
```

* `--n` and `--seed` (the base seed) default to the scenario's `ensemble.n_runs` and
  `ensemble.base_seed`; the input perturbations (wind, moisture, rate, ignition jitter) come
  from the scenario's `ensemble` section.
* `--workers W` runs in `W` processes; results are identical for any `W`.
* `--out DIR` defaults to `runs/<scenario name>-ensemble`; `--force` works as for `run`.

The ensemble directory contains:

| File | Contents |
| --- | --- |
| `ensemble.npz` | Compressed arrays: `burn_probability` `(H, W)` (fraction of runs, 0-1), `arrival_times` `(n, H, W)` float32 (s, NaN = not burned), `burned_area_m2` `(n,)`, `seeds` `(n,)`, `dx_m` (cell size, m). |
| `runs.csv` | One row per run: `index`, `seed`, `burned_area_m2`, `burned_area_ha` and every perturbation actually applied (wind-speed factor, wind-direction offset in degrees, moisture factor, rate multiplier, ignition offsets in cells). |
| `summary.json` | `n`, base seed, mean burned area (m^2 and ha) with its 95 % confidence interval (Student t; `null` for one run), median, min, max and 90th percentile (m^2), workers, wall-clock time (s). |
| `provenance.json` | Resolved config, base seed, `n`, every run seed, workers, git commit, package versions, spread rule and kernel settings. |
| `scenario.yaml` | The resolved scenario with `ensemble.n_runs` and `ensemble.base_seed` set to the values used. |
| `burn_probability.png` | Map of burn probability. |
| `burned_area.png` | Histogram of burned area (ha) and the running mean against the number of runs. |

## Redraw figures

`plot` regenerates the figures of a run or ensemble directory from its data files alone,
without re-running anything. It recognises the directory from its files.

```sh
uv run wildfire-sim plot runs/wind --gif --frames 20 --fps 5
uv run wildfire-sim plot runs/steady_wind-ensemble
```

A run directory gets `final.png` and `arrival.png` (and `fire.gif` with `--gif`); an ensemble
directory gets `burn_probability.png` and `burned_area.png`.

## Errors

Mistakes in the input (an invalid scenario, an unknown spread rule, an ignition on an
unburnable cell, a missing file, a non-empty output directory without `--force`, a `plot`
directory that is not a run or ensemble directory) print one red line and exit with code 1.
A failed run creates no output directory.

## From Python

```python
from wildfire_sim import ensemble, sim, viz
from wildfire_sim.config import load_scenario

cfg = load_scenario("configs/scenarios/steady_wind.yaml")

# One run: RunResult with grid, arrival_time (s), burned_mask, events, provenance.
result = sim.run(cfg, seed=1)
print(int(result.burned_mask.sum()), "cells burned")
viz.save_figure(viz.plot_state(result.grid), "runs/py/final.png")
viz.save_figure(viz.plot_arrival(result), "runs/py/arrival.png")
viz.animate(result, "runs/py/fire.gif", n_frames=20, fps=5)

# An ensemble: EnsembleResult with burn_probability, arrival_times, burned_area_m2 (m^2), seeds.
ens = ensemble.run_ensemble(cfg, n_runs=10, base_seed=0, n_workers=1)
mean, low, high = ensemble.mean_ci(ens.burned_area_m2)
print(f"mean burned area {mean / 1e4:.1f} ha, 95 % CI {low / 1e4:.1f}-{high / 1e4:.1f} ha")
viz.save_figure(viz.plot_burn_probability(ens.burn_probability, result.grid.dx), "runs/py/burn_probability.png")
```

With `n_workers > 1`, `run_ensemble` starts worker processes with the `spawn` method, so a
script that calls it must put that call under `if __name__ == "__main__":`.

The files the CLI writes are produced by `wildfire_sim.io` (`write_run`, `write_ensemble`,
`load_run`, `plot_run_dir`, `plot_ensemble_dir`), which you can also call directly, for
example `io.write_run(result, cfg, Path("runs/py-run"))` into an existing directory.
