# Calibration (WP 3.1)

How the spread rule is tuned to a real fire's early growth, what the objective is, how a trial is
rejected, how to run a study and what the tests show the method can recover. The design is in
`docs/plan.md` section 7; the code is `wildfire_sim.calibration` and `scripts/calibrate.py`;
the settings are `configs/calibration.yaml` and the result goes to `configs/calibrated.yaml`.

**Status.** The committed `configs/calibrated.yaml` is a **placeholder**: neutral values
(`ros_multiplier` 1, `lb_scale` 1), produced by no study. The study on fire A needs its built
region and observations (a LANDFIRE e-mail and a NASA Earthdata token), which were not
available where this work package was written. Run it with the command under *Running a study*
once the region is built.

Units: times in s unless stated (the window in h), rates in m/s, lengths in m; the objective
and both calibration factors are dimensionless.

## The parameters

Only a few parameters are tuned (section 7), each a keyword argument of the spread rule, so a
calibrated value goes into a scenario's `spread.params` and every run, ensemble and CLI command
uses it.

| Rule | Parameter | Range | Meaning |
| --- | --- | --- | --- |
| `rothermel` | `ros_multiplier` | 0.3-3, log scale | rate-of-spread adjustment factor: the head rate times the factor, as FARSITE's adjustment factors (Finney 1998). Byram's intensity `I = H_A R` and the flame length follow from the adjusted rate; heat per unit area is unchanged. |
| `rothermel` | `lb_scale` | 0.5-2 | scales the elongation of the fire ellipse, `LB' = 1 + lb_scale (LB - 1)`, with `LB` Anderson's (1983) length-to-breadth ratio, before the BehavePlus cap (8) and the lattice cap (2.1 on 16 neighbours). `LB' >= 1` always, and a fire without wind or slope stays round. |
| `alexandridis` | `ros_base` | 0.03-0.3 m/s, log scale | the rule's base rate (placeholder range: the plan's multiplier range around the 0.1 m/s placeholder) |
| `alexandridis` | `c2` | 0.0655-0.262 s/m | the wind direction coefficient (placeholder range: 0.5-2 times the paper's 0.131 s/m) |

With both Rothermel factors at 1 the rule's output is bit-identical to an uncalibrated run.
`ros_multiplier` is not the ensemble's random rate factor (`ensemble.ScaledRateRule`, from
`ensemble.ros_multiplier_sigma_frac`): that one scales the directional rates only and leaves the
intensity as it was, standing for uncertainty in the rate; the calibrated factor makes the fire
really spread faster, so it is also more intense.

```yaml
spread:
  rule: rothermel
  params:
    ros_multiplier: 1.0   # from configs/calibrated.yaml
    lb_scale: 1.0
```

In code, `load_calibrated().apply(config)` returns the scenario with the calibrated values.

## The objective

A run starts at the recorded ignition `t_0` and runs, unsuppressed, to the end of the
early-growth window `t_1` (`window_hours`, a placeholder of 48 h, the first two burning periods).
For each of a fixed set of seeds:

```latex
f = w_o\,(1 - J) + w_a\,\frac{\overline{\varepsilon}}{t_1 - t_0}
```

- `J` is the Jaccard overlap between the simulated burned cells and the cells inside a daily
  perimeter posted by `t_1`. Daily perimeters, not the final MTBS perimeter, because a real
  fire is fought and its final perimeter includes the effect of suppression, which would bias the
  rate low (section 7, the suppression confound). A fire without daily perimeters is refused.
- `eps` is the mean arrival-interval error of the VIIRS first detections up to `t_1`
  (`metrics.arrival_interval_error`): zero when the earliest simulated arrival within the 375 m
  footprint falls between the last clear overpass and the detecting one, the distance to the
  nearer end otherwise. Dividing by the window length makes it dimensionless and keeps it in
  [0, 1], like `1 - J`. Without detections in the window the term is 0; the number of
  detections is recorded with the result.
- The weights `w_o = w_a = 0.5` are a placeholder: the plan asks for a weighted sum without
  fixing it.

The objective of a parameter set is `f` averaged over the seeds. The seeds
(`ensemble.derive_seeds(base_seed, n_seeds)`) are the same for every trial, so two parameter
sets are compared on the same link thresholds (common random numbers) and the comparison is not
blurred by the kernel's noise.

**The edge rule.** A run that reaches the grid edge is a lower bound on its fire, so its score
means nothing. A trial in which any run reaches the edge is **rejected, not scored**: Optuna
records it as pruned, it can never become the best trial, and the number of rejected trials is
printed and written to the result. If every trial is rejected the study stops with an error and
writes nothing. The truth run of a synthetic fire must not reach the edge either.

**The search.** An Optuna study with the multivariate TPE sampler, seeded (`sampler_seed`), so a
study reproduces; `--storage sqlite:///runs/calibration.db` keeps a record of it. Afterwards the
objective is evaluated on the grid `best x (1 + step)` with steps of -20, -10, 0, 10 and 20 % for
every parameter (25 points for two), clipped to the bounds. Section 7 asks for this surface: a
narrow minimum means the parameters are identified; a valley along a diagonal means they trade
off. On the committed crop of fire A, for example, a 1-hour fire shows such a valley: a faster
fire with a more elongated ellipse fits nearly as well as a slower, rounder one.

## Running a study

```sh
# fire A, once built: uv run python scripts/download_data.py lake_2024 --observations
uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024

# a synthetic "observed" fire with known parameters, e.g. on the committed crop of fire A
uv run --extra fast --extra opt python scripts/calibrate.py --synthetic crop.yaml \
    --weather tests/fixtures/lake_2024_weather.yaml \
    --truth ros_multiplier=1.2 --truth lb_scale=1.1 --window-hours 1 --trials 3 --seeds 2
```

The region mode ignites the fire at the recorded cell and time, scores it against the daily
perimeters and VIIRS detections and writes `configs/calibrated.yaml`. Listing the VIIRS
overpasses queries NASA's public catalogue, so it needs an internet connection even when every
overpass is cached. `--rule alexandridis` tunes that rule instead. `--window-hours`,
`--trials` and `--seeds` override `configs/calibration.yaml`; `--no-surface` skips the surface.

The synthetic mode takes a scenario file (its grid, weather, ignition cell and time and rule),
runs it once with the `--truth` parameters and the seed `truth_seed` (not a calibration seed),
and uses that run's burned cells at the window end as the observed mask and detections sampled
from its arrival times: a 375 m pixel every 12 cells sees the fire once any cell of its footprint
has burned, at overpasses every 30 min (`overpass_interval_s`; real overpasses are hours apart,
but a short test window needs several), with the interval rule of the real detections. It writes
`runs/calibration_synthetic.yaml` with `status: synthetic`, so it cannot be mistaken for a
calibration.

The result records the parameters with how they were obtained: target, window, seeds, number of
trials and of rejected trials, the objective and its two terms at the optimum, the weights and
bounds, the objective surface, the git commit, the package version, the time and the command.

The cost of a study and the trial and seed counts are in [performance.md](performance.md),
section *Sizing the calibration study*.

## What the method recovers (tests)

`tests/test_calibration.py`, on a synthetic fire: flat uniform GR2, a steady 6 m/s wind, 101 x 101
cells of 30 m, a 1.5-hour window; true `ros_multiplier` 1.5 and `lb_scale` 1.25, so the true ratio
is 1.58, well below the lattice cap of 2.1. Each study: 150 trials x 2 seeds, sampler seed 0,
about 25-40 s on one core with the Numba engine. Two kinds of observed fire:

- **a run of the model** (truth seed 999), as the plan's acceptance asks;
- **the analytic ellipse** of the true parameters (arrival `d / R(theta)` from the rear focus).
  A run-made observed fire shares the lattice's shape error with every calibration run at the
  same bearing, so the error cancels and the bearing test is weak. The ellipse has no lattice, as
  a real fire has none, so this is the case section 2.4 is about: fitted parameters must not
  carry the grid orientation.

| Observed fire | Head bearing (deg) | `ros_multiplier` | `lb_scale` | Objective at the optimum | Rejected trials |
| --- | --- | --- | --- | --- | --- |
| run | 0 | 1.515 (+1.0 %) | 1.281 (+2.5 %) | 0.0173 | 5 |
| run | 13.3 | 1.491 (-0.6 %) | 1.230 (-1.6 %) | 0.0146 | 5 |
| run | 22.5 | 1.486 (-0.9 %) | 1.232 (-1.4 %) | 0.0132 | 4 |
| run | 45 | 1.484 (-1.0 %) | 1.233 (-1.3 %) | 0.0103 | 0 |
| ellipse | 0 | 1.554 (+3.6 %) | 1.295 (+3.6 %) | 0.0168 | 9 |
| ellipse | 13.3 | 1.536 (+2.4 %) | 1.200 (-4.0 %) | 0.0179 | 4 |
| ellipse | 22.5 | 1.534 (+2.3 %) | 1.212 (-3.1 %) | 0.0189 | 4 |
| ellipse | 45 | 1.513 (+0.8 %) | 1.231 (-1.5 %) | 0.0110 | 0 |

Every value is within 10 % of the truth (largest error 4.0 %), and across the four bearings the
recovered multipliers differ by 2.1 % (run) and 2.8 % (ellipse), the recovered `lb_scale` by
4.2 % and 7.9 %. The plan's target, within 10 % and agreeing within 10 %, is met for both kinds;
the tests check it. Against the ellipse the true parameters are not the optimum of the lattice
model (their objective is 0.019-0.032 against 0.011-0.019 at the optimum): the lattice draws the
ellipse slightly wrong, and calibration absorbs that, by up to 4 % at this ratio. With fewer
trials the search had not converged (run-made fire, same sampler seed): 40 trials left errors of
up to 19 % in `lb_scale`, 80 trials up to 6.4 %.

**`lb_scale` is not identified above the lattice cap.** The rule caps the ratio at 2.1 on 16
neighbours (WP 2.1d). Once `1 + lb_scale (LB - 1)` is above the cap, every larger `lb_scale`
draws the same fire, so the objective is flat in it: in a 12 m/s wind over GR2 (`LB` 2.33,
capped for every `lb_scale` above 0.83) the objectives at `lb_scale` 1, 1.5 and 2 are identical
(`test_lb_scale_is_not_identified_above_the_lattice_cap`). A fire that grew mostly at the cap
(counted in each run's `lb_capped_cell_hours`) determines its multiplier but not its shape;
check that count before reading a calibrated `lb_scale`. Fire B, a Santa Ana fire, will very
likely spend its main run above the cap (`docs/plan.md` section 13).

## References

- Anderson, H. E. (1983). Predicting wind-driven wild land fire size and shape. USDA For. Serv.
  Res. Pap. INT-305.
- Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: a next-generation
  hyperparameter optimization framework. Proc. KDD 2019, 2623-2631.
- Finney, M. A. (1998). FARSITE: Fire Area Simulator - model development and evaluation. USDA
  For. Serv. Res. Pap. RMRS-RP-4.
- Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I. A. (2014). The New VIIRS 375 m active fire
  detection data product. Remote Sens. Environ. 143, 85-96.
