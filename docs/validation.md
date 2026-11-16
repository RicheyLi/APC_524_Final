# Validation (WP 3.3)

How the calibrated model is tested against the two validation fires: which fires and why, what
the test can and cannot show, how the suppression of the real fires is kept out of the score,
the ensemble, the edge rule, each metric, the lattice cap, the comparison of the two spread rules,
and the exact commands that produce every table and figure. The design is in `docs/plan.md`
section 7; the code is `wildfire_sim.evaluation` and `scripts/validate.py`; the settings are
`configs/validation.yaml`; `notebooks/validation.ipynb` displays the outputs.

**Status: no real-fire results yet.** No validation of a real fire has been run, and this page
contains no real-fire result. Two things are missing: the built regions of both fires with their
observations (building them needs a LANDFIRE e-mail and a NASA Earthdata token, `docs/data.md`)
and the calibration of fire A (`configs/calibrated.yaml` is a placeholder, `docs/calibration.md`).
This work package delivers the pipeline, tested end to end on synthetic observed fires, and the
protocol below. Section 11 lists the commands that will produce the results and the files they
will write. Until they are run, `docs/img/validation/` does not exist.

Units: times in s (windows and arrival errors in h where stated), areas in ha, lengths in m;
overlaps, ratios and the calibration factors are dimensionless.

## 1. The two fires

**Fire A, calibration: the 2024 Lake Fire**, Santa Barbara County, CA (MTBS
`CA3478712005220240705`, region `lake_2024`), a chaparral and grass fire that grew for about twelve
days in July 2024 in onshore summer winds. It has an MTBS final perimeter, daily perimeters in
NIFC's archive, VIIRS detections from three satellites (Suomi NPP, NOAA-20, NOAA-21), hourly HRRR
analyses and pre-fire LANDFIRE fuels (LF2023). It replaced the 2016 Chimney Fire, whose pre-fire
fuels could not be obtained (`docs/plan.md` section 13).

**Fire B, test: the 2020 Silverado Fire**, Orange County, CA (MTBS `CA3375311768720201026`,
region `silverado_2020`, 5,060 ha): grass-shrub and chaparral like fire A, but driven by offshore
Santa Ana winds in October, in another county; aircraft were grounded by the wind at first, so
early suppression was light. It was chosen against three other candidates for the most VIIRS
overpasses in its first 48 hours (11, from Suomi NPP and NOAA-20) and complete HRRR hours
(`docs/plan.md` section 13). It has the same data as fire A **except daily perimeters**: NIFC's
archive starts in 2022.

## 2. Calibrate on A, test on B without retuning

The spread rule's few free parameters (`ros_multiplier` and `lb_scale` for Rothermel) are fitted
to fire A's early growth (WP 3.1). Fire B is then run with **exactly fire A's parameters**:
`scripts/validate.py` never tunes anything, records `retuned: false` in every summary, and refuses
a calibrated file whose study was run on fire B itself. Its seeds are not the calibration seeds
(checked), so the test does not reuse the link thresholds the parameters were fitted on.

**What fire B can show.** The calibrated multiplier mostly corrects the bias of the fuel models
that dominate fire A; FARSITE's rate-of-spread adjustment factors are specific to a fuel model for
this reason (Finney 1998). Fire B is therefore in the same broad fuel group (shrub and grass) but
a different place, wind direction and season. If the model with fire A's parameters reproduces
fire B within its uncertainty, the calibration transfers between fires of that fuel group and is
not an artefact of fire A's wind, terrain or grid orientation.

**What it cannot show.**

- Transfer across fuel groups: a timber fire, or one dominated by other fuel models, would be
  expected to fail for a reason known in advance (`docs/plan.md` section 7).
- Anything that depends on the fire's growth between its overpasses at a resolution finer than
  the VIIRS interval: without daily perimeters, fire B's growth is known only from the
  detections (section 6) and its final perimeter.
- A clean test of spread in Santa Ana winds: the lattice cap draws such a fire rounder than its
  wind implies (section 7), so a failure on fire B's flanks may be the cap, not the calibration.
- A test of the urban model: 34 % of fire B's grid is urban (5.5 % inside the perimeter, the edge
  of Irvine and Lake Forest), and the urban surrogate's parameters are placeholders
  (`docs/plan.md` section 13). Where the simulated fire runs into town, the comparison mixes the
  two.

One fire B is one test. A pass is evidence, not proof, and a failure needs the diagnostics below
(edge count, cell-hours under the cap, the arrival errors by place) before it is read.

## 3. The suppression confound

Real fires are fought; the model runs are not. Comparing unsuppressed runs with a fought fire's
final perimeter biases a calibrated rate low and makes a validated model look too fast
(`docs/plan.md` section 7). The validation limits this in three ways:

- **The comparison window.** Each fire is compared over a window after its recorded ignition
  (`window_hours` in `configs/validation.yaml`: 48 h for both, the first two burning periods),
  before containment lines held. For fire A the observed burned area at the window end is the
  daily perimeters posted by then, the same target the calibration used.
- **The detections.** The VIIRS detections in the window are scored as intervals (section 6),
  independently of any perimeter.
- **The final perimeter is reported separately** and labelled "includes suppression" in every
  table and figure. It is the only overlap available for fire B, so fire B's overlap carries the
  confound by construction; its detections do not.

The window reduces the confound but does not remove it: initial attack begins within hours of
discovery, and fire B burned to the edge of Irvine (`docs/plan.md` section 13), though its
aircraft were at first grounded by the wind. A run that burns beyond the final perimeter where
the fire was fought is consistent with suppression; a run that has not reached a place by the
time VIIRS saw it burning is not explained by suppression.

## 4. The ensemble

One validation of one fire is an ensemble of `n_runs` unsuppressed runs (200 by default, the top
of the 50-200 range of `docs/plan.md` section 6) from the recorded ignition cell and time to the
window end, with the calibrated parameters. Each run draws, from its own seed
(`derive_seeds(base_seed, n)`, `base_seed` 20261117):

- the kernel's link thresholds (the run's random fire spread);
- a perturbation of the inputs, applied to the whole run (`docs/plan.md` section 6), drawn and
  applied exactly as `wildfire_sim.ensemble.run_ensemble` does (tested bit for bit):

| Input | Distribution | Sigma | Source (`configs/validation.yaml`) |
| --- | --- | --- | --- |
| 10 m wind speed | log-normal factor, median 1 | 0.15 | judgment: the +-30 % range of `configs/sensitivity.yaml` as two standard deviations; no local verification of the HRRR wind has been done for these fires |
| wind direction | normal offset | 15 deg | judgment: the +-30 deg range of `configs/sensitivity.yaml` as two standard deviations |
| all five fuel moistures together | log-normal factor, median 1 | 0.15 | judgment: the +-30 % dead-moisture ranges of `configs/sensitivity.yaml` as two standard deviations |
| rate of spread | log-normal factor on top of the calibrated `ros_multiplier`, median 1 | 0.10 | judgment: of the order of the 10 % recovery tolerance of WP 3.1 |

The ignition point and time are not perturbed: they are the incident record's (IRWIN), and the
uncertainty of the point of origin is not known. The sigmas are judgments, and the width of the
burn-probability map depends on them; the Morris screening (`docs/sensitivity.md`) shows that wind
speed and the rate multiplier matter most.

**How many runs.** The burned area is reported with a 95 % Student-t interval, and the script
computes the number of runs whose interval would be within 5 % of the mean
(`ensemble.runs_for_precision`, the stated precision of section 6), and plots the running mean
with its interval (`_convergence.png`). If the number needed exceeds the runs made, the metrics
table says "more runs needed".

**Cost (estimate, not a measurement).** One 48-hour run on fire A's grid is estimated at about
2 min with the Numba engine (`docs/performance.md`, *Sizing the calibration study*), so 200 runs
are about 7 core-hours: about an hour with 8 worker processes. Fire B's grid has 573 x 543 cells,
fewer than fire A's 815 x 978, but its fire burns faster. Runs are independent; `--workers`
spreads them over processes, and the results do not depend on the number of workers (tested).

## 5. The edge rule

A run that reaches the edge band of its grid is a lower bound on its fire: the model cannot say
what would have burned beyond (`docs/plan.md` section 2). Each run records whether and when it
reached the edge. Such runs are

- **counted** (the first row of every metrics table, `runs_reaching_edge`; printed on every run of
  the script; `edge.n_reached` in the summary), and **listed** with their seed and the time they
  reached the edge (`_runs.csv`, `edge.runs` in the summary);
- **left out of every metric**: the burn probability, the consensus mask, the overlaps, the
  arrival errors, the area ratios and the burned area are computed over the runs that stayed
  inside the grid;
- included in exactly one number, the mean burned area over **all** runs, which is labelled a
  lower bound.

Leaving them out is not neutral either: the runs that stay inside are the slower ones, so the
remaining metrics describe a selection. Milestone M2 requires that **no reported run reaches the
edge**; the summary records whether that holds (`edge.m2_no_run_reaches_edge`). If every run
reaches the edge nothing is scored and the script exits with code 1. A grid that is too small for
the window is a problem of the region build (`buffer_m` in `configs/regions.yaml`) or of the
window, not something to average away.

## 6. Metrics

All from `wildfire_sim.metrics`, over the runs inside the grid; per-run values are averaged over
runs with a 95 % percentile bootstrap interval (10,000 resamples, seeded; Efron & Tibshirani
1993).

| Metric | Definition | Compared with |
| --- | --- | --- |
| Jaccard, consensus | `|A and B| / |A or B|`, `A` the cells burned in at least half the runs (`docs/plan.md` section 7), `B` observed | fire A: daily perimeters posted by the window end; both: final perimeter (includes suppression) |
| Sorensen, consensus | `2 |A and B| / (|A| + |B|)` | as above |
| Jaccard, Sorensen per run | the same with each run's burned cells; mean and interval over runs | as above |
| Fraction of detections inside their interval | per run, the share of VIIRS first detections whose simulated arrival falls in `(t_clear, t_detect]` | VIIRS first detections up to the window end |
| Median and mean arrival-interval error | per run, over detections (h) | as above |
| Burned-area ratio per day | simulated over observed cells burned by the end of each day after the ignition and by the window end | fire A only: daily perimeters posted by then |
| Burned area | mean with Student-t interval (ha); over all runs as a lower bound if any run reached the edge | - |
| Runs for precision | runs needed for a 95 % half-width within 5 % of the mean burned area | - |
| Cell-hours under the lattice cap | `lb_capped_cell_hours` per run (cell-h), and the fraction of runs with any | - (section 7) |
| Breaks breached | per run (WP 2.9) | - |

**Arrival time against an interval** (`docs/plan.md` section 7). A VIIRS detection says only
that the fire had arrived by the detecting overpass, so its arrival lies between the last earlier
overpass that saw the place clear (or the ignition) and the detecting one. The simulated arrival
of a detection is the earliest among the cells of its 375 m footprint (Schroeder et al. 2014;
about 12 x 12 cells). The error is 0 inside the interval and the distance to the nearer end
otherwise; a footprint no simulated fire reached by the window end scores `max(0, window end -
t_detect)`, the smallest error consistent with arriving later. Treating the detection time as the
arrival time would make the model look several hours slow per detection. The scatter
(`_arrival.png`) draws each detection's interval as a horizontal segment at the height of the
median simulated arrival over runs, with the 5-95 % range of the runs; the simulated arrival is in
the interval where the segment crosses the diagonal. A run that never reached the footprint
counts as arriving after the window end in those quantiles.

**Daily ratio caveat.** A daily perimeter is posted after the flight that mapped it, so the
observed area at a given time lags the fire; the ratio is biased high by that lag.

## 7. The lattice cap and what fire B tests

A lattice cannot draw a narrow ellipse: the Rothermel rule caps the length-to-breadth ratio at
2.1 on 16 neighbours, the largest the fitted directional rates represent within 10 % at every
bearing (WP 2.1d, `docs/lattice_study.md` section 7). Above it, about 3.4 m/s effective midflame
wind (a 10 m wind of about 10.7 m/s over open grass), the fire keeps its head rate but is drawn
rounder than it is: its flanks run too wide (`docs/plan.md` section 2.4). Each run records the
burning cell-hours under the cap.

Fire B is a Santa Ana fire: its HRRR 10 m wind reaches 20.3 m/s (`docs/data.md`),
so it will very likely spend its main run above the cap. That makes fire B the test of option
(a) of the open decision "Front propagation above a ratio of 2.1" (`docs/plan.md` section 13):
accept the cap and flag it. What to look for, once the results exist:

- `lb_capped_cell_hours` and the fraction of capped runs, fire B against fire A;
- the overlap: a capped fire is too wide, so the consensus mask spills over the observed flanks
  and the overlap falls even with the right head rate;
- the arrival errors: the head rate is kept, so detections along the fire's axis should fare
  better than those on its flanks (`_detections.csv` has each detection's cell for this).

`lb_scale` is not identified above the cap (`docs/calibration.md`): a calibration on a fire that
grew mostly capped fixes the multiplier but not the shape.

## 8. Rothermel against Alexandridis

The same validation with the Alexandridis et al. (2008) rule, calibrated on fire A over its own
two parameters (`ros_base` and `c2`, `configs/calibration.yaml`), on the same window, seeds and
perturbations. `scripts/validate.py --compare` reads the two summaries of one fire and writes the
metrics present in both side by side. The Alexandridis rule has no lattice cap (its cell-hours
under the cap are "not applicable") and reports no intensity, which is below the breach model's
minimum intensity, so with it no road or stream break is ever breached (`wildfire_sim.breaks`,
WP 2.9): every break is a full barrier. Part of any difference between the rules on a grid with
breaks is this, not the rate of spread. Otherwise both rules share the kernel, the grid and the
weather, so a difference in the metrics is a difference of the spread rule after each was
calibrated on fire A.

`scripts/calibrate.py` writes `configs/calibrated.yaml` unless told otherwise: the Alexandridis
calibration must be written elsewhere (`--out configs/calibrated_alexandridis.yaml`, section 11)
or it replaces the Rothermel one. `scripts/validate.py` refuses a calibrated file of another rule
and prints that command.

## 9. Comparison with published simulations

- **Fire A.** The plan's earlier fire A, the 2016 Chimney Fire, had a published simulation to
  compare with (arXiv 2507.00761); that comparison went with the fire when it was replaced by the
  Lake Fire (`docs/plan.md` sections 7 and 13).
- **Fire B and the Lake Fire.** We have not found, and do not cite, a published simulation of the
  2024 Lake Fire or the 2020 Silverado Fire. No search of the literature has been recorded in this
  repository; before the report is written, the team should search for one (for either fire,
  with the simulator, inputs and metrics it used) and add it here with a full reference, or state
  that none was found. Nothing on this page compares with a published simulation.

## 10. Running a validation

```sh
uv run --extra fast --extra geo python scripts/validate.py lake_2024 --workers 8
```

- Inputs: the built region under `--cache-root` (default `data/cache`), its observations, the
  calibrated file (`--calibrated`, default `configs/calibrated.yaml`) and the settings
  (`--settings`, default `configs/validation.yaml`). `--window-hours`, `--runs` and `--dt`
  override the settings; `--rule alexandridis` selects the other rule.
- Outputs, all under `--out` (default `docs/img/validation/`), with the prefix
  `<region>_<rule>`: `_summary.json` (target, parameters with their status, file and SHA-256,
  ensemble seeds and sigmas, the edge runs, the metrics, the git commit, package and library
  versions, the command), `_metrics.csv`, `_runs.csv`, `_detections.csv`, `_convergence.csv`,
  `_maps.npz` (burn probability, consensus, observed masks), `_surface.csv` (the calibration's
  objective surface), and the figures `_burn_probability.png`, `_arrival.png`,
  `_convergence.png`, `_surface.png`. Every figure is drawn from the files beside it; given the
  seeds, every file except the summary's record of the command is byte for byte the same for any
  number of workers (tested for one and two).
- **Parameter status.** Every output records whether the parameters were `calibrated`,
  `placeholder` or `synthetic`. Anything but `calibrated` is printed as a warning and stamped in
  bold on every figure. With the committed placeholder the script runs, but its results are of
  the uncalibrated model and say so.
- **Without a calibration** there is no objective surface; the script says so.
- **Exit code** 1 if the region is not built (the message says how to build it), the parameters
  are of another rule or were tuned on fire B, or every run reached the edge.
- **Clusters.** Nothing is interactive; every input is a flag and every output goes under `--out`.
  Listing the VIIRS overpasses currently queries NASA's public catalogue even when every overpass
  is cached (`calibration.real_fire`), so a compute node without internet access cannot run it
  yet.
- **Synthetic mode** (`--synthetic SCENARIO --weather FILE --truth NAME=VALUE ... --window-hours
  H`): a run of the scenario with known parameters (seed `truth_seed`) stands in for the
  observations, as in `scripts/calibrate.py --synthetic`; outputs are named `synthetic_...` and
  labelled "SYNTHETIC OBSERVED FIRE - NOT A VALIDATION RESULT" in every table and figure. It is
  how the pipeline is tested (section 12).

## 11. Results

**Status: pending.** No real-fire table or figure exists. They are produced, in this order, by
(each command prints its metrics table and the edge count):

```sh
# 1. build both regions with their observations (LANDFIRE e-mail, NASA Earthdata token; docs/data.md)
uv run --extra geo python scripts/download_data.py lake_2024 --observations
uv run --extra geo python scripts/download_data.py silverado_2020 --observations

# 2. calibrate both rules on fire A (docs/calibration.md); the second must not overwrite the first
uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024
uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024 --rule alexandridis \
    --out configs/calibrated_alexandridis.yaml

# 3. validate: fire A (calibrated on it) and fire B (not retuned), each with both rules
uv run --extra fast --extra geo python scripts/validate.py lake_2024 --workers 8
uv run --extra fast --extra geo python scripts/validate.py silverado_2020 --workers 8
uv run --extra fast --extra geo python scripts/validate.py lake_2024 --rule alexandridis \
    --calibrated configs/calibrated_alexandridis.yaml --workers 8
uv run --extra fast --extra geo python scripts/validate.py silverado_2020 --rule alexandridis \
    --calibrated configs/calibrated_alexandridis.yaml --workers 8

# 4. the two rules side by side
uv run python scripts/validate.py --compare docs/img/validation/lake_2024_rothermel_summary.json \
    docs/img/validation/lake_2024_alexandridis_summary.json
uv run python scripts/validate.py --compare docs/img/validation/silverado_2020_rothermel_summary.json \
    docs/img/validation/silverado_2020_alexandridis_summary.json
```

| Deliverable (WP 3.3) | Written by | File(s) in `docs/img/validation/` |
| --- | --- | --- |
| Burn-probability map against the observed perimeter, fire A | step 3 | `lake_2024_rothermel_burn_probability.png`, `_maps.npz` |
| Burn-probability map against the observed perimeter, fire B | step 3 | `silverado_2020_rothermel_burn_probability.png`, `_maps.npz` |
| Arrival-time scatter, A and B | step 3 | `<region>_rothermel_arrival.png`, `_detections.csv` |
| Metrics table with intervals, A and B | step 3 | `<region>_rothermel_metrics.csv`, `_summary.json` |
| Objective surface | step 3 (from step 2's file) | `lake_2024_rothermel_surface.png`, `_surface.csv` |
| Convergence of the ensemble | step 3 | `<region>_rothermel_convergence.png`, `_convergence.csv` |
| Rothermel against Alexandridis | steps 3-4 | `<region>_alexandridis_*`, `<region>_rule_comparison.csv`, `.json` |

Once they exist, this section is to be replaced by the tables and figures, with the suppression
confound discussed against them (section 3), the edge count (M2 requires 0), fire B's cell-hours
under the cap (section 7) and the published comparison, if any (section 9).
`notebooks/validation.ipynb` displays whatever the output directory holds.

## 12. What the tests check (offline)

`tests/test_evaluation.py`, without the real regions:

- the scores and their aggregation against hand calculations on 6 x 6 cell arrays (overlaps,
  interval errors, unreached detections, daily ratios, the consensus mask, bootstrap means);
- the edge rule: a run that reached the edge is counted, listed and left out of every metric
  except the burned area over all runs, labelled a lower bound; with every run at the edge only
  the count is reported and the script exits with 1; and end to end, on a synthetic fire where two
  of three runs reach the edge;
- fire A's path (daily perimeters by each day end and the window end) and fire B's (no daily
  perimeters: only the final perimeter, labelled as including suppression, and the detections) on
  stubbed observations;
- that an ensemble member is bit for bit the member of `run_ensemble` and still counts the
  cell-hours under the cap when the rate is perturbed;
- the whole script in `--synthetic` mode on the committed 5 km crop of fire A with its HRRR
  weather, both rules: every table and figure written, labelled synthetic, the parameters'
  status recorded and stamped, and one and two workers giving byte-identical tables and figures;
- the refusals: an unbuilt region, another rule's parameters, parameters tuned on fire B, seeds
  shared with the calibration; the comparison of two rules; the notebook running with and
  without outputs.

Not tested, because it cannot be offline: reading a real region's observations
(`observed_fire` on a real `FireObservations`), and anything about the real fires.

## References

- Alexandridis, A., Vakalis, D., Siettos, C. I. & Bafas, G. V. (2008). A cellular automata model
  for forest fire spread prediction: the case of the wildfire that swept through Spetses Island
  in 1990. Applied Mathematics and Computation 204, 191-201.
- Efron, B. & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall,
  chapter 13.
- Finney, M. A. (1998). FARSITE: Fire Area Simulator - model development and evaluation. USDA
  For. Serv. Res. Pap. RMRS-RP-4.
- Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I. A. (2014). The New VIIRS 375 m active fire
  detection data product: algorithm description and initial assessment. Remote Sens. Environ.
  143, 85-96.
