# Calibration (WP 3.1)

How the spread rule is tuned to a real fire's early growth, what the objective is, how a trial is
rejected, how to run a study and what the tests show the method can recover. The design is in
`docs/plan.md` section 7; the code is `wildfire_sim.calibration` and `scripts/calibrate.py`;
the settings are `configs/calibration.yaml` and the result goes to `configs/calibrated.yaml`.

**Status.** Fire A is calibrated (section *Fire A*): `configs/calibrated.yaml` (Rothermel,
`ros_multiplier` 0.383, `lb_scale` 0.641) and `configs/calibrated_alexandridis.yaml`
(`ros_base` 0.145 m/s, `c2` 0.174 s/m). The Rothermel factor is about 2.6 times below 1, which
the data do not attribute to the rule: it absorbs the suppression of the real fire and whatever
inputs are still wrong. Read that section before using the values.

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
early-growth window `t_1` (`window_hours`, 48.5 h: the first two burning periods, ending just
after one of fire A's daily perimeters was posted).
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
perimeters and VIIRS detections and writes `configs/calibrated.yaml`. With the
observations cached it needs no network. `--rule alexandridis` tunes that rule instead. `--window-hours`,
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

## Fire A

The 2024 Lake Fire, calibrated on 2026-11-27 on an Apple M4 Pro (macOS 26.6, Python 3.12.15,
NumPy 2.5.4, Numba 0.68.1), from commit `7f359fa`:

```sh
uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024 --window-hours 48.5
uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024 --window-hours 48.5 \
    --rule alexandridis --out configs/calibrated_alexandridis.yaml
```

(`window_hours` in `configs/calibration.yaml` is now 48.5, so the flag can be left out.)

**Target.** The region's 815 x 978 cells of 30 m; ignition 5 July 2024 22:48 UTC at the IRWIN
point; window 48.5 h. The observed mask is the daily perimeter posted 48.1 h after the ignition,
6,699 ha (74,432 cells): a 48 h window would have ended 4 minutes before that posting and
scored the fire against the perimeter of 27.1 h (4,959 ha). 308 VIIRS first detections fall in
the window. Live woody moisture is the measured 88 % of the Santa Barbara County Fire
Department's chamise samples (`configs/live_fuel_moisture.yaml`), not the 60 % placeholder.
Weights 0.5 / 0.5, 100 trials x 2 seeds, sampler seed 0.

| | Rothermel | Alexandridis |
| --- | --- | --- |
| Parameters | `ros_multiplier` 0.383, `lb_scale` 0.641 | `ros_base` 0.145 m/s, `c2` 0.174 s/m |
| Objective | 0.403 | 0.460 |
| 1 - Jaccard | 0.579 | 0.670 |
| Normalized arrival error | 0.227 | 0.249 |
| Trials rejected by the edge rule | 25 of 100 | 34 of 100 |
| Surface points rejected | 0 of 25 | 11 of 25 |
| Wall time, one core | 1 h 40 min | 42 min |

**Rothermel is identified.** All 24 neighbours of the optimum on the surface (plus and minus
10 and 20 % in each parameter) have a larger objective (0.407-0.430), no surface point reaches
the grid edge and both parameters are inside their bounds. The minimum is shallow: 20 % in
either parameter raises the objective by at most 7 %, so the values are known to about that
precision. The lattice cap does not hide `lb_scale`: run at the optimum, at most 5.7 % of the
burned cells (0.7 % on average over the 49 hours) have a capped ellipse in any hour, and
`lb_capped_cell_hours` is about 400. The two seeds give 1 - Jaccard 0.579 and 0.578.

**Alexandridis is limited by the grid edge.** Every surface point faster than the optimum
(larger `ros_base`, or smaller `c2` at the same `ros_base`) reached the edge, and the objective
falls towards them. Its optimum is the fastest fire that stays on the grid within the window,
not a minimum of the objective, so its `ros_base` is a lower bound.

**What the Rothermel factor means.** A multiplier of 0.38 says that, with the measured inputs,
the uncalibrated rule spreads the fire about 2.6 times too fast over the first two burning
periods. Before calibration a probe with the placeholder 60 % live woody moisture needed about
0.30-0.35; the measured moisture explains part of the gap. The rest is not attributable to
Rothermel's equations:

- The Lake Fire was suppressed from the start, and the model has no suppression in Phase 3.
  Calibrating to perimeters of a suppressed fire folds the suppression into the factor
  (Stratton 2006 names "a fire with minimal suppression or knowledge of suppression tactics" as
  a criterion for a calibration fire; Carrasco et al. 2019 make the same point for adjustment
  factors fitted to real fire scars).
- Stratton (2006) takes an adjustment of about 0.3-0.4 away from 1 to mean that a different
  fuel model may be needed. This one is 0.62. About 40 % of the burnable cells are grass or
  grass-shrub models (GR2, GS2, GS3), whose spread does not depend on live woody moisture.
- The wind is one HRRR point at the grid centroid for the whole region; terrain-driven local
  winds are not represented.

**The two terms disagree.** The perimeter favours a slower fire and the VIIRS arrivals a faster
one: at the optimum the simulated fire burns about 11,800 ha against 6,699 ha observed. Along
the surface at `lb_scale` 0.641, 1 - Jaccard falls from 0.688 to 0.523 between multipliers 0.459
and 0.306 while the arrival error rises from 0.166 to 0.309. So the optimum depends on the
weights, which are placeholders. Even at the optimum the burned areas
overlap by 42 % (Jaccard 0.42), and `lb_scale` 0.64 asks for a rounder fire than Anderson's
(1983) ratios: where the fire burns, not only how fast, is off.

**Use.** Report fire B's validation (WP 3.3) with these values as a test of the whole model
including this compensation, not of Rothermel alone. Cruz & Alexander (2013) found mean
rate-of-spread errors of 20 to 310 % across 49 evaluation data sets, mostly under-predictions,
so an over-prediction of this size is unusual and worth revisiting when suppression is modelled
(Phase 4) or the wind is improved.

## References

- Carrasco, J., Pais, C., Shen, Z.-J. M. & Weintraub, A. (2019). Adjusting rate of spread
  factors through derivative-free optimization: a new methodology to improve the performance of
  forest fire simulators. arXiv:1909.05949.
- Cruz, M. G. & Alexander, M. E. (2013). Uncertainty associated with model predictions of
  surface and crown fire rates of spread. Environ. Model. Softw. 47, 16-28.
- Stratton, R. D. (2006). Guidance on spatial wildland fire analysis: models, tools, and
  techniques. USDA For. Serv. Gen. Tech. Rep. RMRS-GTR-183.

- Anderson, H. E. (1983). Predicting wind-driven wild land fire size and shape. USDA For. Serv.
  Res. Pap. INT-305.
- Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: a next-generation
  hyperparameter optimization framework. Proc. KDD 2019, 2623-2631.
- Finney, M. A. (1998). FARSITE: Fire Area Simulator - model development and evaluation. USDA
  For. Serv. Res. Pap. RMRS-RP-4.
- Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I. A. (2014). The New VIIRS 375 m active fire
  detection data product. Remote Sens. Environ. 143, 85-96.
