# scripts

Command-line entry points that are not part of the installed package.

| Script | WP | Purpose |
| --- | --- | --- |
| `release.py` | scaffold | tag and publish a GitHub release |
| `kernel_front_speed.py` | 1.4 | measure the simulator's front speed against the nominal rate of spread per stencil and gamma shape; `--write` regenerates `configs/kernel_correction.yaml` |
| `lattice_study.py` | 1.4 | simulated burn shapes against the analytic ellipse per stencil; regenerates the figures of `docs/lattice_study.md` |
| `bench.py` | 1.7 | time one run on the 333 x 333 reference grid |
| `download_data.py` | 2.6 | build and cache a real-world study region |
| `calibrate.py` | 3.1 | tune spread multiplier and ellipse shape against a historical fire |
| `sensitivity.py` | 3.2 | Morris screening of burned area and arrival time over the main inputs |
| `run_experiments.py` | 4.7 | optimized policy vs. baselines, figures for the report |
