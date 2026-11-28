# Running Phase 3 on the real fires

Phase 3's code (calibration, sensitivity analysis, validation) is tested on synthetic fires and
on the committed 5 km crops. Every step has been run once on an Apple M4 Pro: the calibration
of fire A (`docs/calibration.md`, *Fire A*) and the validation of both fires
(`docs/validation.md`, section 11). This page is
the list of steps that produces them, on a laptop or on a cluster. Milestone M2
(`docs/plan.md` section 9) is met only when these steps have been run and their outputs
committed.

Steps 1-3 need the internet and two credentials. Steps 4-8 are compute only and need neither,
so they can run on a cluster's compute nodes.

| Step | Needs | Rough time | Writes |
| --- | --- | --- | --- |
| 1. Credentials | a browser | 10 min | nothing in the repository |
| 2. Install | internet | 5 min | `.venv/` |
| 3. Build both regions | internet, both credentials | 15-30 min | `data/cache/` |
| 4. Benchmark | 1 core | 10 min | numbers for `docs/performance.md` |
| 5. Calibrate on fire A (two rules) | 1 core each | 1 h 40 min (Rothermel) and 42 min (Alexandridis), measured on an Apple M4 Pro | `configs/calibrated.yaml`, `configs/calibrated_alexandridis.yaml` |
| 6. Validate A and B (two rules) | 8 or more cores | about 1 h per run with 8 workers (estimate) | `docs/img/validation/` |
| 7. Compare the rules | seconds | seconds | `docs/img/validation/*_rule_comparison.*` |
| 8. Write up and commit | | | `docs/validation.md`, `docs/performance.md`, `docs/plan.md` |

The times of steps 5 and 6 are extrapolated from one measurement (60 s for a 24-hour run on
fire A's grid, `docs/performance.md`). Step 4 replaces them with measurements; size the jobs from
its output.

## 1. Credentials

- **LANDFIRE.** The LANDFIRE Product Service wants an e-mail address with every job. No account
  is needed. The address is sent to the service and is not written to disk.
- **NASA Earthdata**, for the VIIRS active-fire data. Create a free account at
  <https://urs.earthdata.nasa.gov>, then *Profile*, *Generate Token*. A token is valid for 60
  days and is sent only to NASA's servers.

```sh
export WILDFIRE_SIM_LANDFIRE_EMAIL="you@example.edu"
mkdir -p data && printf '%s\n' "PASTE-THE-TOKEN" > data/.earthdata_token   # git-ignored
```

`EARTHDATA_TOKEN` in the environment works instead of the file. In PowerShell the first line is
`$env:WILDFIRE_SIM_LANDFIRE_EMAIL = "you@example.edu"`.

The repository is public. Any file named `*earthdata_token*` is git-ignored in every folder, and
the commit hooks (`uvx pre-commit install`, once per clone) also refuse a token file added with
`git add -f` or a token pasted into any other file. If a token is ever pushed, revoke it on the
Earthdata token page and generate a new one: deleting it in a later commit leaves it in the
history.

## 2. Install

From the repository root, on the machine that will download (see `AGENTS.md`: run `uv` from WSL
if the `.venv` was made there):

```sh
uv sync --extra fast --extra opt --extra geo
```

`--extra all` also asks for `pyrothermel`, which needs a C++ toolchain and is not used here.
Every `uv run` below repeats the three extras; without them `uv` removes the packages again.

## 3. Build both regions

```sh
uv run --extra fast --extra opt --extra geo python scripts/download_data.py lake_2024 --observations
uv run --extra fast --extra opt --extra geo python scripts/download_data.py silverado_2020 --observations
```

Each command downloads LANDFIRE fuels, the 3DEP elevation model, OpenStreetMap, hourly HRRR
weather, the MTBS perimeter, the incident record, the daily perimeters where they exist (fire A
only) and every VIIRS overpass, and writes `data/cache/<region>/` plus the caches of each source
beside it. The LANDFIRE job is queued by the service and is the slowest part. Running a command
again reads the caches. `--verify` rebuilds a region from its caches and compares file hashes.

Check what was built:

- the command prints the grid size (fire A: 815 x 978 cells), the number of first detections
  (fire A: 654, fire B: 192 when WP 2.7 was built) and the daily perimeters found;
- `uv run --extra fast --extra opt --extra geo python scripts/demo_real_fire.py lake_2024 --hours 24`
  runs the uncalibrated model once and writes `runs/demo_lake_2024.png`.

After this step nothing needs the network: the list of overpasses is cached with them.

**Regions built before WP 3.4** lack the per-cell HRRR wind field (`wind_field.npz`). Add it
without redoing the LANDFIRE, DEM, OSM or VIIRS steps (public HRRR data, no account; about two
1-2 MB downloads per hour, 10-30 minutes for 14 days):

```sh
uv run --extra fast --extra opt --extra geo python scripts/download_data.py lake_2024 --wind-field
uv run --extra fast --extra opt --extra geo python scripts/download_data.py silverado_2020 --wind-field
```

From then on calibration, validation and `scripts/sweep_eval.py` use the field; each takes
`--no-wind-field` to go back to the single centroid point. If the command rewrites
`tests/fixtures/<region>_5km.npz` or `<region>_weather.yaml` (it should not), restore them with
`git checkout`.

## 4. Benchmark on fire A's grid

```sh
uv run --extra fast --extra opt --extra geo python scripts/bench.py --region lake_2024 --hours 48 --engines numba --repeats 3
```

This is the measurement WP 3.1 asks for before the study is sized. Record it in
`docs/performance.md` ("Sizing the calibration study"), and if a 48-hour run takes much more or
less than about 2 minutes, revise `n_trials` in `configs/calibration.yaml` or the job's time
limit. The calibration runs its trials one after another on one core, so its time is
`n_trials x n_seeds x` the time of one run, plus a quarter of that for the objective surface.

## 5. Calibrate on fire A

```sh
uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024
uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024 --rule alexandridis --out configs/calibrated_alexandridis.yaml
```

The first writes `configs/calibrated.yaml` (Rothermel: `ros_multiplier`, `lb_scale`). The second
needs `--out`: without it it would overwrite the first. The two are independent and can run at
the same time. Each prints one line per trial.

Before trusting the result, check in the output and in the written file:

- **The window.** The default is 48.5 hours after the ignition (`configs/calibration.yaml`),
  chosen for fire A, whose second daily perimeter was posted 48.1 hours after the ignition. The
  command refuses to run if no daily perimeter was posted by the window end.
  A perimeter is posted some hours after the flight that mapped it, so the observed area at the
  window end is somewhat older than the window end; if the first perimeters were posted late,
  choose `--window-hours` so that it ends shortly after a posting.
- **Rejected trials.** A trial in which a run reaches the grid edge is rejected. A few are
  normal; if most are, the window is too long for the grid.
- **The objective surface** (`provenance.surface` in the file): a clear minimum at the optimum
  means the data identify the parameters. A flat valley along `lb_scale` means the fire spent
  the window above the lattice cap, where `lb_scale` has no effect (`docs/calibration.md`).
- **The bounds.** An optimum on a bound (`ros_multiplier` 0.3 or 3, `lb_scale` 0.5 or 2) is not
  a calibration; look for what is wrong before widening them.

## 6. Validate fire A and, without retuning, fire B

```sh
uv run --extra fast --extra opt --extra geo python scripts/validate.py lake_2024 --workers 8
uv run --extra fast --extra opt --extra geo python scripts/validate.py silverado_2020 --workers 8
uv run --extra fast --extra opt --extra geo python scripts/validate.py lake_2024 --workers 8 --rule alexandridis --calibrated configs/calibrated_alexandridis.yaml
uv run --extra fast --extra opt --extra geo python scripts/validate.py silverado_2020 --workers 8 --rule alexandridis --calibrated configs/calibrated_alexandridis.yaml
```

Each runs an ensemble of 200 (`configs/validation.yaml`) and writes tables and figures to
`docs/img/validation/` with the prefix `<region>_<rule>_`. Fire B is run with fire A's
parameters; the script tunes nothing and refuses parameters calibrated on fire B. The result
does not depend on `--workers`.

Read the first lines of each `*_summary.json` or `*_metrics.csv`:

- **`runs_reaching_edge` must be 0** for milestone M2. Runs that reach the edge are left out of
  every metric and counted; if there are any, shorten the window or report the result as a
  lower bound.
- **`runs_for_precision`** says whether 200 runs were enough for the burned area.
- **`lb_capped_cell_hours`** on fire B is the test of the lattice cap (`docs/plan.md` section
  13, "Front propagation above a ratio of 2.1"): a Santa Ana fire is expected to spend its main
  run above it.
- A warning that the parameters are a placeholder means step 5 has not replaced
  `configs/calibrated.yaml`.

## 7. Compare the rules

```sh
uv run --extra fast --extra opt --extra geo python scripts/validate.py --compare docs/img/validation/lake_2024_rothermel_summary.json docs/img/validation/lake_2024_alexandridis_summary.json
uv run --extra fast --extra opt --extra geo python scripts/validate.py --compare docs/img/validation/silverado_2020_rothermel_summary.json docs/img/validation/silverado_2020_alexandridis_summary.json
```

## 8. Write up and commit

On a branch, commit `configs/calibrated.yaml`, `configs/calibrated_alexandridis.yaml` and
`docs/img/validation/`, then:

- fill the Results section of `docs/validation.md` from the tables (it names the file behind
  each number), and browse them with `notebooks/validation.ipynb`;
- put the benchmark of step 4 in `docs/performance.md`;
- update the status of Phase 3 and milestone M2 in `docs/plan.md`;
- consider rerunning the sensitivity analysis: the multiplier's range there is the calibration's
  prior, and it is much narrower after calibration (`docs/sensitivity.md`, section 7).

`data/` is git-ignored and is never committed.

## On a cluster

Compute nodes usually have no internet access, so split the work:

1. **On a machine with internet** (the login node, or a laptop): steps 1-3. If that is not the
   cluster, copy the cache over; it is self-contained:

   ```sh
   rsync -a data/cache/ CLUSTER:/path/to/APC_524_Final/data/cache/
   ```

   The Earthdata token and the e-mail address are not needed on the cluster.
2. **On the login node:** clone the repository and run step 2 once, so that `.venv/` and uv's
   package cache exist before a job starts. Then make jobs run offline with
   `export UV_OFFLINE=1`.
3. **As jobs:** steps 4-7. The calibration is one process on one core per rule; the validation
   uses as many cores as `--workers`. Allow about 2 GB of memory per worker on fire A's grid
   until step 4 and a first job say otherwise (an estimate, not measured).

A Slurm script for the two calibrations, as a two-task array (adjust the time limit from step 4):

```sh
#!/bin/bash
#SBATCH --job-name=wf-calibrate
#SBATCH --array=0-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=runs/calibrate_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_OFFLINE=1 NUMBA_NUM_THREADS=1
RUN="uv run --extra fast --extra opt --extra geo python"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  $RUN scripts/calibrate.py lake_2024 --storage sqlite:///runs/calibration_rothermel.db --study-name rothermel
else
  $RUN scripts/calibrate.py lake_2024 --rule alexandridis --out configs/calibrated_alexandridis.yaml \
      --storage sqlite:///runs/calibration_alexandridis.db --study-name alexandridis
fi
```

`--storage` keeps a record of every trial in case the job is cut off. A study cannot be resumed:
a rerun needs a new `--study-name` or a deleted database file, and starts again. Create `runs/`
before submitting (`mkdir -p runs`).

And for the validations, submitted after the calibrations have finished
(`sbatch --dependency=afterok:<calibration job id> validate.sh`):

```sh
#!/bin/bash
#SBATCH --job-name=wf-validate
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=runs/validate_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_OFFLINE=1 NUMBA_NUM_THREADS=1
RUN="uv run --extra fast --extra opt --extra geo python scripts/validate.py"
W="--workers $SLURM_CPUS_PER_TASK"
case "$SLURM_ARRAY_TASK_ID" in
  0) $RUN lake_2024 $W ;;
  1) $RUN silverado_2020 $W ;;
  2) $RUN lake_2024 $W --rule alexandridis --calibrated configs/calibrated_alexandridis.yaml ;;
  3) $RUN silverado_2020 $W --rule alexandridis --calibrated configs/calibrated_alexandridis.yaml ;;
esac
```

The four tasks write files with different prefixes into `docs/img/validation/`, so they can run
together. Step 7 takes seconds and can run on the login node. Then copy
`configs/calibrated*.yaml` and `docs/img/validation/` back, or commit them from the cluster, and
do step 8.

These scripts have not been run on a cluster; treat the resource requests as starting points.
