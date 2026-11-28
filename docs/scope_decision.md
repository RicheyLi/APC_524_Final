# Scope decision: the wind limitation and the plan for the best fit

**Status: for the team to decide.** Written 2026-11-27 after the Phase 3 runs (WP 3.1, 3.3), the
parameter sweep and WP 3.4; revised the same day with the two goals below. The evidence is in
[root_cause_fire_a.md](root_cause_fire_a.md).

**Goals for the rest of Phase 3:**

1. **Acknowledge the limitation.** The wind speed at the fire, at the scale the terrain shapes
   it, is out of scope: no data or model available to this project provides it (section 3).
2. **Get the highest Jaccard we can on both fires with the current model.** Small additions are
   allowed if they keep the core of the model (the Rothermel rule, the lattice, the inputs)
   unchanged; every correction they fit is reported as a correction, not as physics. What is
   left after that is explained as the deficit we cannot model (section 5).

Section 4 is the plan for goal 2, section 6 the decisions it needs, section 7 what it changes and
section 8 the wording for the report.

## 1. Summary

- The model puts fire A (2024 Lake Fire) in partly the wrong place, and the WP 3.1 calibration
  hides it by slowing the fire down: `ros_multiplier` 0.383, Jaccard 0.42 against the 48.5 h
  perimeter. Fire B (2020 Silverado Fire), run with fire A's parameters, scores Jaccard 0.42
  against its final perimeter and misses its western and southern lobes.
- The cause is the **wind at the fire**, not the spread model, the fuels, the moisture, the
  lattice or the breaks: a sweep of every physical placeholder (55 configurations) moves the
  Jaccard by at most 0.03 once the multiplier is refitted, and the spread rule, its conventions
  and the lattice were each verified (section 2.4).
- **Done (WP 3.4):** wind from every node of the HRRR 3 km analysis instead of one point. It
  turns fire A's head in the observed direction, but the head is then too slow.
- **Out of scope:** the wind *speed* along the terrain below 3 km (section 3).
- **Plan:** three small additions (a fitted wind-direction correction, a calibration target for
  fires without daily perimeters, and optionally a fitted break-strength factor), a calibration
  of each fire on its own with fire A's parameters still tested on fire B, a window of one
  burning period for fire A, and an objective weighted towards the overlap. Measured so far, this
  reaches about **Jaccard 0.60 on fire A** over its first burning period (one run; the WP 3.1
  parameters score 0.56 over the same window); fire B has not been screened yet (section 4.5).

## 2. The error

### 2.1 What the real fire did

Fire A ignited on 5 July 2024 at 22:48 UTC (15:48 PDT) near Zaca Lake, in a heat wave (40 deg C,
10 % humidity, 1-h fuel moisture 1.2-3 %). From the VIIRS first detections and the daily
perimeters (`docs/img/validation/lake_2024_rothermel_detections.csv`, NIFC):

- **Day 1:** by the first overpass, 10 h after the ignition, it had run **at least 11 km
  west-north-west** as a streak under 2 km wide (68 detections in the W sector, up to 11.2 km
  away; 60 in the NW), along a ridge at 750-1,200 m.
- **Day 2:** between 23 and 34 h it grew a **south-east lobe**, 18,238 cells or 25 % of the area
  burned by 48 h, while the wind was light.
- Daily perimeters: 1,891 ha at 19.9 h, 4,959 ha at 27.1 h, 6,699 ha at 48.1 h.

### 2.2 What the model does

| | uncalibrated (`ros_multiplier` 1) | calibrated (0.383) | observed |
| --- | --- | --- | --- |
| Head at the W detections | 9.8 h | 22.7 h (median) | by 10 h |
| Burned by 10 h | 5,000 ha, wide (length/breadth 2.2) | 700 ha | narrow streak |
| Grid edge reached | 14.5 h | not within 48.5 h | - |
| South-east lobe | not reached | not reached | 25 % of the area |
| Jaccard | 0.21 against the 27.1 h perimeter (and off the grid) | 0.42 against the 48.1 h perimeter | - |

Uncalibrated, the head is about the right speed but the fire is far too wide and points
west-south-west, into the valley; the calibration then cuts the rate by 2.6 times to trade
the head for fewer false alarms. The 0.383 is therefore not a correction of Rothermel's rate,
and not mainly suppression as `docs/calibration.md` first suggested: it compensates a
direction error.

### 2.3 Why: the wind input

The region build drives the whole 25 x 29 km grid with **one** HRRR 10 m wind, at the grid
centroid. HRRR's own analysis, sampled at nine points over the region, differs by 20-80 deg in
direction and 2-5 times in speed (`root_cause_fire_a.md` section 4). Rotating the single
point by 20 deg, to the wind at the ignition, was the best single change of the whole study
(Jaccard 0.60 against the 27.1 h perimeter).

WP 3.4 now gives every cell the HRRR wind interpolated from its 3 km nodes. The head turns to
276-284 deg at 10 h (was 250-263 deg), the observed direction. But HRRR's wind in the valley the
head ran along is 1-4 m/s, against 7-8 m/s at the centroid, so the head is slower (6.9 km at 10 h
at `ros_multiplier` 1, against 9.0 km with the single point and about 11 km observed) and the
fit does not improve: Jaccard 0.42 (48.5 h) and 0.46 (27.5 h) at their best. An 11 km run in
under 10 hours needs a stronger wind than the 3 km analysis has there: a wind channelled by
the ridges and canyons, on scales the analysis does not resolve.

### 2.4 What is verified and holds

These were checked during the diagnosis and are not part of the error:

- Rothermel's rate, intensity and flame length against BehavePlus (within 0.02 %), Anderson's
  length-to-breadth ratios, the wind adjustment factor.
- Direction conventions: HRRR grid rotation, "from" and "to" bearings, aspect, stencil
  bearings (reversing the wind makes the fit far worse, J 0.19).
- The lattice: head and flank within 10 % of the analytic ellipse up to a length-to-breadth
  ratio of 2.1 on 16 neighbours (`docs/lattice_study.md` section 7); raising the cap to 3-8 or
  changing the stencil, rate sampling or kernel shape does not improve the fit.
- Fuels, live and dead moisture, canopy, breaks and breaching, holdover, urban surrogate: none
  moves the Jaccard by more than 0.03 over its defensible range (`root_cause_fire_a.md`
  appendix).
- WP 3.4's wind field matches the HRRR point series to 0.02 m/s and 0.6 deg at the centre and
  independent HRRR samples to 0.1 m/s and 1 deg elsewhere.

### 2.5 Fire B

Fire B was run in WP 3.3 with fire A's parameters and the single-point wind (before WP 3.4):
Jaccard 0.42 (mean over runs; 0.45 for the consensus mask) against the final perimeter
(5,183 ha, includes suppression), a burned area within 8 % of it (4,771 ha), 24 of 200 runs at
the grid edge, 14 % of the VIIRS detections inside their interval. The simulated fire covers the
centre of the real burn but spreads north and east of the ignition and misses the western and
southern lobes behind a straight edge where a mapped break stopped it; the real fire got past
it, plausibly by spotting (`docs/validation.md` section 11.1). Its HRRR wind reached 20.3 m/s,
above what the lattice draws without rounding the fire (section 5.2). Fire B is not yet built
on the machine where this page was written, so none of the sweep covered it.

## 3. The limitation (goal 1): wind speed below 3 km is out of scope

**The part that could be fixed with our data was fixed (WP 3.4). What remains is the wind speed
along the terrain at scales the HRRR's 3 km grid does not resolve, and nothing available to this
project can supply it.**

Fire A's head ran about 11 km in under 10 hours along a valley and ridge where the HRRR analysis
has 1-4 m/s; with the HRRR wind on every cell the uncalibrated head reaches 6.9 km in that time.
The difference needs a stronger, terrain-channelled wind. It cannot be supplied by:

| Option | Why it is not available |
| --- | --- |
| A finer weather analysis | The finest hourly analyses covering both fires' dates have the same resolution: the HRRR (3 km) and the RTMA (2.5 km). gridMET is 4 km and daily. |
| Weather stations | RAWS stations are single points, a few per county in this terrain; turning them into a wind field near the fire needs the same kind of wind model as the next row. |
| Terrain wind downscaling (WindNinja, Forthofer et al. 2014) | A mass-conserving or momentum solver over the 30 m DEM, hour by hour: a new external C++ dependency, per-region runs of hours, a new data layer and its own validation against station winds. A project in itself, and out of scope. |
| Coupled fire-atmosphere model (WRF-SFIRE, Mandel et al. 2011) | A different kind of model; outside a cellular automaton. |
| Scaling the wind speed up | Tested: a uniform factor of 1.6 or 2 on the wind lowers the best Jaccard (0.434 and 0.419, against 0.448 at the same shape) because it speeds the whole fire, its flanks and the slopes it climbs, not the one channel the head ran in. A speed correction would have to vary in space, which is exactly what a terrain wind model computes. |

What **can** be corrected is a uniform error of the wind's *direction* at the fire: the sweep's
best single change was a 20 deg rotation (section 2.3). That is addition S1 below. It does not
make the head faster, so the speed deficit stays (section 5).

Spotting (WP 6.1) and suppression (Phase 4) are later work packages of the plan. They are not
the cause of fire A's day-1 error, but both contribute to the deficit (section 5).

## 4. Plan to maximise the Jaccard (goal 2)

### 4.1 What is allowed

- **Kept unchanged:** the Rothermel rule and its fuel models, the lattice and kernel, the input
  data (LANDFIRE, 3DEP, OpenStreetMap, HRRR, moisture), the breach model of Wilson (1988).
- **Allowed:** small additions that fit a correction of an *input* per fire (wind direction,
  and optionally the strength of the breaks), choices of the objective and window, and one
  calibration per fire. Each fitted correction is reported with its value and what it stands
  for, and is never carried from one fire to the other.
- **Kept as a test:** fire A's spread parameters are still run on fire B without retuning, and
  that score is reported next to fire B's own fit. A correction that only helps because it
  overfits one fire shows up as a gap between the two.

### 4.2 The levers, measured on fire A

All with one seed, Jaccard of a single run against the perimeter of the window
(`root_cause_fire_a.md` and its appendix). The window of 27.5 h ends just after the 27.1 h
perimeter (4,959 ha).

| Lever | Window | Best Jaccard | Against (same window) | Gain |
| --- | --- | --- | --- | --- |
| Window of one burning period instead of two | 27.5 h | 0.563 (WP 3.1 parameters) | 0.421 at 48.5 h | +0.14 (the day-2 lobe leaves the score) |
| Wind direction +20 deg, single point, `lb_scale` 1 | 27.5 h | **0.598** | 0.537 (same, no rotation) | +0.06 |
| Wind direction +20 deg, single point | 48.5 h | 0.464 | 0.455 | +0.01 (day 2 still missed) |
| `lb_scale` 1 instead of 0.641 | 48.5 h | 0.455 | 0.421 | +0.03 |
| Kernel shape 20 instead of 200 | 48.5 h | 0.479 | 0.455 | +0.02 |
| No breaks at all | 48.5 h | 0.467 | 0.448 | +0.02 |
| Wind field (WP 3.4) instead of the point, `lb_scale` 1 | 27.5 h | 0.460 | 0.537 | -0.08 (head too slow) |
| Wind speed x1.6, x2 | 48.5 h | 0.434, 0.419 | 0.448 | worse |
| Live or dead moisture, canopy, wind limit, urban, holdover, lattice cap, stencil | 48.5 h | 0.43-0.48 | 0.448-0.480 | none beyond noise |

Two things follow. The window and the wind direction are the only large levers; everything else
is within a few hundredths. And at 27.5 h the single point, rotated, fits better than the
physically finer field: the field puts the head in the right direction but with the valley's
weak wind, while the point carries the stronger ridge-level wind. Which wind input to use is
therefore a choice to make by fit (decision D6), recorded as such.

A shorter window fits better still (Jaccard 0.65 at 20.5 h, against the 19.9 h perimeter of
1,891 ha), but scores only the first night's run; 27.5 h is the shortest window that covers a
full burning period.

### 4.3 Small additions

| # | Addition | Size | Why |
| --- | --- | --- | --- |
| S1 | **A fitted wind-direction offset** (degrees, added to every hour and every cell of the wind), calibrated per fire with the spread parameters and stored in the calibrated record | about 100 lines and tests: the calibration searches rule parameters only today; the ensemble already applies a direction offset to the wind and its field (WP 3.4), so the transform exists | The largest lever after the window (+0.06 on fire A). Stands for the error of the HRRR direction at the fire. |
| S2 | **A calibration target for fires without daily perimeters**: the final perimeter, labelled "includes suppression", with the VIIRS detections of the window | about 30 lines and tests (`calibration.observed_target` refuses such a fire today; `scripts/sweep_eval.py` uses it too) | Needed to fit fire B at all, and to screen it with the sweep harness. |
| S3 | **The wind input per fire** (HRRR point or HRRR field) chosen by fit | none: `--no-wind-field` exists on the calibration, validation and sweep scripts | Section 4.2: at 27.5 h the point fits fire A better. |
| S4 | *Optional:* **a fitted break-strength factor** (a scale on the probability that a fire breaches a road or stream, Wilson 1988) for fire B | about 50 lines and tests, the same mechanism as S1 for a parameter file | Fire B's missing western lobe lies behind a break the real fire crossed, probably by spotting, which is not modelled. Decide after the screening (step 2): on fire A, break settings moved the Jaccard by at most 0.02. |

The objective changes too, without code: the weights of `configs/calibration.yaml` move towards
the overlap (decision D3), since the goal is the Jaccard; the arrival error is still reported.

### 4.4 Steps

| Step | What | Who / where | Time |
| --- | --- | --- | --- |
| 1 | Build fire B with its observations and wind field (`scripts/download_data.py silverado_2020 --observations`) | needs the LANDFIRE e-mail and the Earthdata token (`docs/real_data.md`) | under an hour |
| 2 | Implement S2, then screen with `scripts/sweep_eval.py`: fire A at 27.5 h and fire B at 48 h, each with the point and the field, wind offsets from -30 to +30 deg, 3-4 multipliers; fire B also with the breaks weakened (S4's screening, via `--edit breach.yaml:...`) | one process, one seed | about 1.5 h |
| 3 | Implement S1 (and S4 if step 2 shows at least +0.03 on fire B); tests; the configuration changes of section 7 | code | about a day |
| 4 | Calibrate fire A and fire B, each on its own: `ros_multiplier`, `lb_scale`, the wind offset (and the break factor for B if adopted), with the wind input chosen in step 2 | `scripts/calibrate.py`, one core each, in parallel | 2-3 h |
| 5 | Validate (200-run ensembles): fire A with its own fit; fire B with its own fit; fire B with fire A's `ros_multiplier` and `lb_scale` and no offset (the transfer test) | `scripts/validate.py --workers 6` | about 1.5 h |
| 6 | Update `docs/calibration.md` (*Fire A*, a new *Fire B*) and `docs/validation.md` (section 11), stop tuning | docs | |

### 4.5 What to expect

- **Fire A, 27.5 h:** about Jaccard 0.60 for a single run (measured, one seed); a full
  calibration with three parameters may add a little. Against the 48.1 h perimeter the ceiling
  is lower (section 5.1).
- **Fire B:** not screened yet. Its WP 3.3 score (0.42-0.45 against the final perimeter, with
  fire A's parameters and no wind correction) is the floor; the direction offset, its own
  multiplier and, if adopted, the break factor are what may raise it. We do not predict a number
  before step 2.

## 5. The deficit we cannot model

What will remain after section 4, by fire, and what each part would need.

### 5.1 Fire A (2024 Lake Fire)

| Deficit | Size | Why we cannot model it |
| --- | --- | --- |
| **Head speed on day 1** | The fire ran about 11 km in under 10 h; with the HRRR wind at the cells the uncalibrated head reaches 6.9 km. A calibration then either leaves the head late or widens the whole fire to reach it. | Terrain-channelled wind below 3 km (section 3). A direction offset does not change speed; a uniform speed factor makes the fit worse. |
| **Day-2 south-east lobe** | 18,238 cells, 25 % of the area burned by 48.1 h, grown between 23 and 34 h in light wind. Even a perfect day-1 fit scores at most Jaccard 0.75 against the 48.1 h perimeter without it. | Driven by local circulations over the ridge (an afternoon sea breeze from the W-NW in the HRRR at +26 h, light and variable elsewhere): the same out-of-scope wind. Outside the 27.5 h window (D1); stated as a limitation. |
| **Suppression** | Unknown: firing operations and lines from the first day (1,162 personnel by 8 July). Shows as simulated fire beyond held lines. | Phase 4 of the plan; the calibrated multiplier absorbs part of it. |

### 5.2 Fire B (2020 Silverado Fire)

| Deficit | Size | Why we cannot model it |
| --- | --- | --- |
| **Fire shape in the Santa Ana wind** | 4.7 times fire A's capped cell-hours: the fire is drawn rounder than its wind implies, so its flanks run too wide. | The lattice draws the fire ellipse within 10 % only up to a length-to-breadth ratio of 2.1 (`docs/lattice_study.md` section 7); fire B's winds of up to 20.3 m/s ask for more. Raising the cap was tested on fire A and did not help. A property of the cellular automaton. |
| **Lobes beyond breaks** | The western and southern lobes behind a straight break edge. | Spotting (embers carried across the break) is not modelled (WP 6.1). S4 can only stand in for it in part. |
| **Suppression in the target** | Fire B has no daily perimeters, so its only perimeter is the final, fought one; 34 % of its grid is urban (5.5 % inside the perimeter), with placeholder urban parameters. | Phase 4; and no earlier perimeters exist (NIFC's archive starts in 2022). |
| **Terrain wind** | As fire A, through the Santa Ana Mountains' canyons. | Section 3. |
| **Live fuel moisture** | The seasonal placeholder (60 % woody); the measured samples are in Santa Barbara County only. | Measured samples near fire B would be a data task, small, but not found so far. |

## 6. Decisions for the team

Each with the recommended default.

| # | Decision | Recommended default | Alternative |
| --- | --- | --- | --- |
| D1 | Fire A's window | **27.5 h** (one burning period, the 27.1 h perimeter) for calibration and validation; the 48.1 h comparison reported as well, with the day-2 lobe as a stated limitation | Keep 48.5 h as the main score |
| D2 | Milestone M2 | **Report the metrics with their intervals** (Jaccard, VIIRS arrival-interval error, burned-area ratio, edge count); no pass/fail | Keep M2 as written and record it as not met |
| D3 | Objective weights | **0.75 overlap, 0.25 arrival** (the goal is the Jaccard; the arrival term still keeps the timing honest). On fire A it selected the same multiplier as the overlap alone in four of the seven window and shape combinations measured, and a neighbouring one in the other three | 1 / 0 (overlap only); or keep 0.5 / 0.5 |
| D4 | One calibration or two | **Calibrate each fire on its own**, and also run fire B with fire A's spread parameters (the transfer test); report both | Keep fire B as a test only (lower Jaccard on B, stronger claim of transfer) |
| D5 | Wind-direction offset (S1) | **Adopt**, fitted per fire, reported as the HRRR's direction error at the fire | Do not adopt (fire A stays about 0.06 lower) |
| D6 | Wind input | **Choose per fire by fit** (S3), recorded with the reason | Always the field (more physical, lower fit on fire A) |
| D7 | Break-strength factor (S4) | **Decide after the screening**: adopt only if it raises fire B's Jaccard by at least 0.03 | Adopt without screening; or never |
| D8 | Rothermel / Alexandridis comparison | **Drop**: its fires cannot cross breaks (no intensity), so its results are confounded; keep the WP 3.3 results labelled as such | Make it fair (new design work) |
| D9 | Terrain winds | **A limitation in the report** (section 8); no work package | Plan a WindNinja work package (out of scope) |

## 7. What it changes

| Item | Change |
| --- | --- |
| `wildfire_sim.calibration`, `scripts/calibrate.py`, `scripts/validate.py` | S1: the wind-offset parameter in the search, the calibrated record and its `apply`; S2: the final-perimeter target; S4 if adopted. Tests for each. No change to `wildfire_sim.types`, `grid`, `spread.base` or `config`. |
| `scripts/sweep_eval.py` | S2's target, so fire B can be screened |
| `configs/calibration.yaml` | window 27.5 h; weights 0.75 / 0.25; the offset's bounds (for example -45 to +45 deg) with their source |
| `configs/validation.yaml` | fire A's window 27.5 h |
| `configs/calibrated.yaml`, new `configs/calibrated_silverado.yaml` | written by the calibrations |
| `docs/plan.md` | Phase 3 status; M2 as D2; open items: single-point wind resolved (WP 3.4), terrain winds a limitation, the Alexandridis decision closed |
| `docs/calibration.md`, `docs/validation.md` | the new results, the corrections and their meaning, the transfer test, section 5 of this page |
| Runs | about 6-8 h of computing in all, mostly unattended |

## 8. Wording for the report

> The model is calibrated on each fire's own observations, over the first burning period of the
> 2024 Lake Fire and the first two of the 2020 Silverado Fire, with its wind from the hourly HRRR
> analysis at 3 km. Besides the rate-of-spread factor and the ellipse shape, each calibration
> fits one correction of an input, a uniform offset of the wind direction, which stands for the
> error of the 3 km analysis at the fire; it is reported per fire and not carried between them.
> The Lake Fire's parameters are also run on the Silverado Fire without retuning.
>
> What the model cannot reproduce is attributed to inputs and processes outside it: the
> terrain-channelled wind that drove the Lake Fire's head about 11 km in under 10 hours, below
> the resolution of the analysis, which a diagnostic wind model such as WindNinja (Forthofer et
> al. 2014) would be needed to represent; the light-wind growth of its second day; the narrow
> shape of a fire in the Silverado Fire's 20 m/s Santa Ana wind, beyond what the lattice draws;
> spotting across roads; and suppression on both fires. The spread rule, its fuel and moisture
> inputs and the lattice were each verified separately (agreement with BehavePlus, the ellipse
> test, a sweep of every placeholder), and none explains the difference.

## References

- Anderson, H. E. (1983). Predicting wind-driven wild land fire size and shape. USDA For.
  Serv. Res. Pap. INT-305.
- Forthofer, J. M., Butler, B. W. & Wagenbrenner, N. S. (2014). A comparison of three approaches
  for simulating fine-scale surface winds in support of wildland fire management. Part I. Model
  formulation and comparison against measurements. Int. J. Wildland Fire 23, 969-981.
- Mandel, J., Beezley, J. D. & Kochanski, A. K. (2011). Coupled atmosphere-wildland fire
  modeling with WRF 3.3 and SFIRE 2011. Geosci. Model Dev. 4, 591-610.
- Wilson, A. A. G. (1988). Width of firebreak that is necessary to stop grass fires: some field
  experiments. Can. J. For. Res. 18, 682-687.
