# scripts

Command-line entry points that are not part of the installed package.

| Script | WP | Purpose |
| --- | --- | --- |
| `release.py` | scaffold | tag and publish a GitHub release |
| `kernel_front_speed.py` | 0.8, 1.4 | front speed of the ignition kernel vs. nominal rate of spread, by gamma shape, time step and stencil |
| `bench.py` | 1.7 | time one run on the 333 x 333 reference grid |
| `download_data.py` | 2.6 | build and cache a real-world study region |
| `calibrate.py` | 3.1 | tune spread multiplier and ellipse shape against a historical fire |
| `sensitivity.py` | 3.2 | Morris screening of burned area and arrival time over the main inputs |
| `run_experiments.py` | 4.7 | optimized policy vs. baselines, figures for the report |
