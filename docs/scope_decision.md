# Scope decision: the wind error and what to scale back

**Status: for the team to decide.** Written 2026-11-27 after the Phase 3 runs (WP 3.1, 3.3), the
parameter sweep and WP 3.4. The evidence is in [root_cause_fire_a.md](root_cause_fire_a.md);
this page states the conclusion, what is out of scope, and the decisions needed to finish
Phase 3 with results that are correct about what they show. Each decision has a recommended
default (section 4); section 5 lists what the defaults change, and section 6 gives the wording
for the report.

## 1. Summary

- The model puts fire A (2024 Lake Fire) in partly the wrong place and the calibration hides it
  by slowing the fire down: `ros_multiplier` 0.383, Jaccard 0.42 against the 48.5 h perimeter.
- The cause is the **wind at the fire**, not the spread model, the fuels, the moisture, the
  lattice or the breaks. A sweep of every physical placeholder (55 configurations) moves the
  Jaccard by at most 0.03 once the multiplier is refitted; the spread rule, its conventions and
  the lattice were each verified.
- **In scope, done (WP 3.4):** wind from every node of the HRRR 3 km analysis instead of one
  point. It turns the head of fire A in the observed direction (west-north-west), but the head
  is then too slow, so the overlap does not improve.
- **Out of scope:** the remaining error is the wind *speed* along the terrain at scales below
  3 km. Fixing it needs a terrain wind model (for example WindNinja) or a coupled
  fire-atmosphere model. We will not do this; we scale the claims back to what the inputs
  support (section 4).

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

## 3. Is it in scope?

**The part that could be fixed with our data was fixed (WP 3.4). The rest is out of scope.**

| Remedy | What it is | Why not |
| --- | --- | --- |
| Terrain wind downscaling (WindNinja, Forthofer et al. 2014) | A mass-conserving or momentum solver that adjusts the HRRR wind to the 30 m DEM, hour by hour | A new external dependency (C++ solver, its own meshing and stability settings), per-region runs of hours, a new data layer, its own validation against station winds. A project in itself. |
| Coupled fire-atmosphere model (WRF-SFIRE, Mandel et al. 2011) | The fire's heat changes the wind that drives it | A different kind of model, outside a cellular automaton |
| Tuning the wind to the fire | Scale or rotate the wind until the fire fits | Not physical; it would fit fire A and say nothing about fire B. The sweep's +20 deg rotation is evidence of the cause, not a fix |
| Spotting (WP 6.1), suppression (Phase 4) | Long-range ignition by embers; firefighting | Already planned for later phases; neither explains a fast WNW head on day 1 |

The plan assumed that one weather value per time is adequate because the weather sources are
coarser than the study area (`docs/plan.md` section 2, `grid.Weather` docstring before WP 3.4).
For a mountain fire that assumption fails; WP 3.4 removes it for the resolution the HRRR has,
and that is where this project stops.

## 4. Decisions needed

Each decision lists the recommended default. With all the defaults, Phase 3 ends with one
calibration and two validation runs (about 2 h, section 5) and no new modelling code.

| # | Decision | Recommended default | Alternative |
| --- | --- | --- | --- |
| D1 | Comparison window | **The first burning period**: calibrate and validate fire A over 27.5 h (just after the 27.1 h perimeter, 4,959 ha) | Keep 48.5 h and report that day 2 is outside what the wind supports |
| D2 | Milestone M2 | **Report, do not pass/fail**: Jaccard, VIIRS arrival-interval error, burned-area ratio and the edge count, each with its interval; head direction described from the burn-probability map | Keep M2 as written and record it as not met |
| D3 | What is out of the acceptance | **Fire A's south-east lobe and fire B's Santa Ana run** are stated limitations, not targets | Keep them, knowing they cannot be met |
| D4 | Calibrated parameters | **Calibrate `ros_multiplier` only**, with `lb_scale` fixed at 1 (Anderson's published ratios); report the multiplier as a compensation for unresolved wind, with its dependence on the weights | Keep two parameters (`lb_scale` 0.64 fits no better than 1: Jaccard 0.42 against 0.46) |
| D5 | Rothermel against Alexandridis | **Drop the comparison**: keep the WP 3.3 results labelled "confounded, not pursued" | Give the Alexandridis rule an intensity or its own break rule, then rerun (new design work) |
| D6 | Rerun with the wind field | **Rerun** the Rothermel calibration of fire A and the validations of A and B with the field, once | Keep the committed numbers and state they were made with the single point (`--no-wind-field`) |
| D7 | WP 3.4 itself | **Keep** (merged, tested, gets the direction right) | Revert to the single point (more work than keeping it, and loses the evidence) |
| D8 | Terrain winds | **One sentence in the report's limitations** (section 6); no work package, no plan item | Plan a WindNinja work package (scope creep) |

### D1. Comparison window: the first burning period

The calibration and validation windows of fire A are 48.5 h, two burning periods. The second
period's growth (the south-east lobe) depends on a light, terrain-driven wind that neither the
single point nor the 3 km field reproduces. Over the first period the model can be tested on
what it represents: a wind-driven head run.

- Measured over 27.5 h: the best Jaccard is 0.56 with the single point (`lb_scale` 0.641) and
  0.46 with the field; with the field the objective is lowest at `ros_multiplier` 0.6 (0.372).
- Change: `window_hours` 48.5 -> 27.5 in `configs/calibration.yaml` and fire A's entry of
  `configs/validation.yaml`, with the source line updated.
- Fire B has no daily perimeters; its window stays 48 h (its first two burning periods, the
  Santa Ana run) and it is reported, not passed (D2, D3).
- Cost: fewer VIIRS detections score the arrival term (196 instead of 308).

### D2. Milestone M2: report, do not pass/fail

M2 asks that no run reach the grid edge and that the overlap be good. Neither can be promised
with the wind the model has (19 and 24 of 200 runs reached the edge in WP 3.3). Redefine M2 as
a set of **measurements with their uncertainty**, all of which `scripts/validate.py` already
writes:

- Jaccard against the window-end perimeter (fire A) and the final perimeter (fire B, labelled
  as including suppression);
- VIIRS detections inside their interval and the median arrival-interval error;
- burned-area ratio;
- the edge count (runs at the edge are left out of every metric, as now).

The head direction is described from the burn-probability figure. No new metric is written.

### D3. Out of the acceptance: fire A's day 2 and fire B's Santa Ana run

- **Fire A's south-east lobe** needs the light day-2 wind over the ridge (section 2.1). It is
  outside the D1 window anyway; state it as a limitation.
- **Fire B (2020 Silverado)** was driven by Santa Ana winds up to 20.3 m/s at 10 m: above the
  lattice cap (length/breadth 2.1, from about 10.7 m/s over open grass; `docs/validation.md`
  section 7) and through terrain the 3 km field does not resolve either. Keep fire B as the test
  of transfer without retuning (same parameters, burned area, arrival errors), reported with
  these two limitations, not as a pass/fail test.

### D4. Calibrate one parameter, report what it means

- **Fix `lb_scale` at 1.** It is weakly identified (the fit is as good from 0.64 to 2.0;
  `root_cause_fire_a.md` appendix A.2 and A.4) and 1 means Anderson's (1983) ratios with no
  free parameter. Change: delete the `lb_scale` entry under `parameters.rothermel` in
  `configs/calibration.yaml` (the rule's default is 1); no code change. The study then searches
  one parameter, which also halves its cost.
- **Keep the weights at 0.5 / 0.5**, still placeholders, and report how much they move the
  result: over the weights the selected multiplier ranges from 0.3 to 0.5 at 48.5 h (0.3 to
  0.7 over windows and shapes). Above an arrival weight of about 0.6 the grid edge, not the
  data, picks the multiplier, so the arrival weight should not be raised
  (`root_cause_fire_a.md` appendix A.4).
- **Wording:** the calibrated `ros_multiplier` is a compensation for wind the model does not
  resolve (and for suppression), specific to fire A, not a correction of Rothermel's rate.

### D5. Drop the Rothermel / Alexandridis comparison

The Alexandridis rule reports no fireline intensity, so under the breach model of WP 2.9 its
fires never cross a road, stream or built line: on fire B all 200 runs burn the same 221 ha
block and stop (`docs/validation.md` section 11.2). Making the comparison fair is new design
work. Keep the committed results, labelled confounded and not pursued, and do not rerun them.
This removes one calibration and two validations from D6.

### D6. Rerun once with the wind field

Since WP 3.4, `calibration.real_fire` uses a region's wind field whenever the region has one, so
the committed WP 3.1 and 3.3 numbers were made with a different wind than the model now uses.
Rerun once, with D1, D4 and D5 applied:

```sh
uv run --extra geo python scripts/download_data.py lake_2024 --wind-field        # if not yet built
uv run --extra geo python scripts/download_data.py silverado_2020 --observations # fire B, with its field
uv run --extra fast --extra opt --extra geo python scripts/calibrate.py lake_2024
uv run --extra fast --extra geo python scripts/validate.py lake_2024 --workers 6
uv run --extra fast --extra geo python scripts/validate.py silverado_2020 --workers 6
```

Fire A is built (with its field) on the machine where this page was written; fire B is not
(its build needs a LANDFIRE e-mail and an Earthdata token, `docs/real_data.md`). Then
update `docs/calibration.md` (*Fire A*) and `docs/validation.md` (section 11) with the new
numbers, and stop tuning.

### D7. Keep WP 3.4

It is merged and tested, leaves every run without a field bit-identical, and gets the head
direction right. It is also the evidence that the remaining error is wind speed below the HRRR
resolution (section 2.3).

### D8. Terrain winds: a limitation, not a plan item

No WindNinja work package, no roadmap entry. One sentence in the report (section 6).

## 5. What the defaults change

| Item | Change | Code? |
| --- | --- | --- |
| `configs/calibration.yaml` | `window_hours` 27.5; remove `lb_scale` from `parameters.rothermel` | no |
| `configs/validation.yaml` | fire A `window_hours` 27.5 | no |
| `configs/calibrated.yaml` | rewritten by the rerun (one parameter) | no |
| `docs/plan.md` | M2 as in D2; Phase 3 status; open items: single-point wind resolved by WP 3.4, terrain winds a stated limitation; the Alexandridis decision closed as D5 | no |
| `docs/calibration.md`, `docs/validation.md` | rerun results; the wording of D3, D4, D5 | no |
| Tests | `tests/test_calibration.py` checks that the committed record is the fire A study, with its parameters inside the bounds and the default window reproducing it; it follows the new record and window. The synthetic-recovery tests keep both parameters (they test the method, not fire A) | test update only |
| Runs | one calibration, two validations, about 2 h on 6 cores | - |

## 6. Wording for the report

> The model is calibrated and validated over each fire's first burning period. Its wind is the
> hourly HRRR analysis at 3 km, interpolated to every cell. With it, the simulated head of the
> 2024 Lake Fire runs in the observed direction but more slowly than the fire did: the fire ran
> about 11 km in under 10 hours along terrain where the 3 km analysis has 1-4 m/s winds. We
> attribute the difference to terrain-channelled winds below the resolution of the analysis,
> which a diagnostic wind model such as WindNinja (Forthofer et al. 2014) would be needed to
> represent and which is outside this project. The calibrated rate-of-spread factor therefore
> compensates for unresolved wind (and for suppression) on this fire and is not a correction
> of Rothermel's rate. The spread rule, its fuel and moisture inputs and the lattice were each
> verified separately (BehavePlus agreement, the ellipse test, a sweep of every placeholder);
> none explains the difference.

## References

- Anderson, H. E. (1983). Predicting wind-driven wild land fire size and shape. USDA For.
  Serv. Res. Pap. INT-305.
- Forthofer, J. M., Butler, B. W. & Wagenbrenner, N. S. (2014). A comparison of three approaches
  for simulating fine-scale surface winds in support of wildland fire management. Part I. Model
  formulation and comparison against measurements. Int. J. Wildland Fire 23, 969-981.
- Mandel, J., Beezley, J. D. & Kochanski, A. K. (2011). Coupled atmosphere-wildland fire
  modeling with WRF 3.3 and SFIRE 2011. Geosci. Model Dev. 4, 591-610.
