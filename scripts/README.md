# scripts

Command-line entry points that are not part of the installed package.

| Script | WP | Purpose |
| --- | --- | --- |
| `release.py` | scaffold | tag and publish a GitHub release |
| `kernel_front_speed.py` | 1.4 | measure the simulator's front speed against the nominal rate of spread per stencil and gamma shape; `--write` regenerates `configs/kernel_correction.yaml` |
| `lattice_study.py` | 1.4 | simulated burn shapes against the analytic ellipse per stencil; regenerates the figures of `docs/lattice_study.md` |
| `bench.py` | 1.7, 3.1 | seconds per run (min / median) on the 333 x 333 reference grid and a small grid, per engine and stencil; `--json` writes the numbers, `--check scripts/bench_baseline.json` fails on a > 20 % slowdown of the calibration-normalised time; `--region NAME` or `--grid FILE --weather FILE` time a real fire with the Rothermel rule over `--hours` (see `docs/performance.md`) |
| `download_data.py` | 2.6 | build and cache a real-world study region |
| `calibrate.py` | 3.1 | Optuna study of the spread multiplier and ellipse shape against a built region's early growth, or a synthetic observed fire (`--synthetic`); writes `configs/calibrated.yaml` (see `docs/calibration.md`) |
| `sensitivity.py` | 3.2 | Morris screening of burned area and head arrival time over the main inputs (`configs/sensitivity.yaml`); regenerates the CSVs and figures of `docs/sensitivity.md` |
| `validate.py` | 3.3 | ensemble of a fire with the calibrated parameters (fire B not retuned) scored against its observations: metrics with intervals, edge count, burn-probability map, arrival scatter, objective surface, convergence; `--rule alexandridis`, `--compare` two rules, `--synthetic` for a synthetic observed fire; writes to `docs/img/validation/` (see `docs/validation.md`) |
| `run_experiments.py` | 4.7 | optimized policy vs. baselines, figures for the report |
