# Why the calibrated model misses fire A: root-cause analysis (2026-11-27)

What is out of scope and the decisions it leaves for the team: [scope_decision.md](scope_decision.md).

Fire A (2024 Lake Fire) built on this machine from the committed configs; baseline reproduces
WP 3.1 exactly (m 0.383, lb_scale 0.641: J 0.421, objective 0.403, 11,843 ha). All numbers below
are from `scripts/sweep_eval.py` records in this directory (`*.json`, `*/SUMMARY.md`).

## 1. What the observations say

Grouping the 308 VIIRS first detections by compass sector from the ignition:

| sector | n | first seen (h) | distance km (median / max) | model arrival (h) |
|---|---|---|---|---|
| W | 68 | 10.0 (first overpass) | 6.7 / 11.2 | 22.7 |
| NW | 60 | 10.4 | 5.2 / 9.4 | 20.0 |
| SE | 114 | 34.1 (clear at 23) | 7.6 / 11.4 | never |

Daily perimeters: 1,891 ha at 19.9 h, 4,959 ha at 27.1 h, 6,699 ha at 48.1 h. So the fire ran
**>= 11 km WNW in < 10 h as a streak < 2 km wide (L/B about 6)**, then grew an **SE lobe of
18,000 cells (25 % of the observed area) between 23 and 34 h**. Observed burn mean elevation
761 m vs 586 m for the grid: it burned along and up the WNW-ESE ridge, not down into the valley.

## 2. What the model does

- Direction conventions are correct: HRRR u/v -> "from" bearing, grid rotation, stencil
  bearings, aspect, wind_to_bearing all check out, and wind +180 deg scores far worse (J 0.19).
- At m = 1 the uncalibrated head is nearly right (W detections reached at 9.8 h; 8.5 km at
  10 h vs 11.2 observed) but the fire is L/B 2.2 and 5,000 ha at 10 h, so it leaves the grid at
  14.5 h. The Jaccard calibration then shrinks everything: m 0.383 makes the head 2.6x too slow.
- The model's head bearing (260 deg, from the centroid wind from 80 deg) sends it WSW into the
  valley (grass, GR2 20 % of cells); the fire went WNW along the ridge.

## 3. Parameter sweep (4 groups, 55 configurations, Opus 5.5 agents)

| group | best J (baseline 0.421) | best objective (0.403) | verdict |
|---|---|---|---|
| weather (wind x0.7-2, live/dead moisture, canopy, wind limit) | 0.480 (finer m scan of baseline itself) | 0.386 (dead moisture x3) | nothing > 0.03 once m is refitted |
| fuel/shape (lb_scale 0.64-3, cap 2.1-8, sampled rates, stencil 8, kernel shape, urban) | 0.479 (lb 1.0 + shape 20) | 0.408 | shape/lattice not the limit |
| breaks (breach floor, Wilson coefs, tree threshold, holdover, no breaks) | 0.467 (no breaks) | 0.398 | +-0.02; SE lobe not held by breaks |
| calibration weights/window | - | - | 0.383 depends on weights (0.3-0.7) but no choice gives m ~ 1 with acceptable J; arrival term never bottoms out, the edge rule picks |
| extra: wind x2 with cap lifted to 8 | 0.434 | 0.446 | elongation alone does not help: head goes WSW, not WNW |
| **extra: wind direction +20 deg** | **0.464 (48.5 h), 0.598 (27.5 h)** | **0.391** | best single lever found |

SE recall never exceeds 0.4 in any configuration without a 3-7x oversized fire.

## 4. Root cause

**One HRRR point for a 25 x 29 km mountain region.** Sampling HRRR's own 3 km analysis at nine
points (`hrrr_points.txt`): at +6..+10 h the wind at the ignition is from 96-99 deg (centroid:
74-81 deg), 1-4 m/s in the WNW valley the head ran along, 6.5-7 m/s on the SE ridge; at +26 h
a W/NW sea breeze (265-316 deg) blows everywhere, strongest over the ridge, pushing ESE: the
SE lobe. The +20 deg rotation that scored best is exactly the ignition-point wind. The
information is in the data already; the model throws it away by using the centroid.

Secondary: (a) the Jaccard objective cannot separate a direction error from a rate error, so
the multiplier absorbs it (0.383 is a compensation, as docs/calibration.md suspected, but for
wind direction, not suppression); (b) the arrival term is monotone in the rate, so above
w_a ~ 0.6 the grid edge, not the data, selects the multiplier.

## 5. Fix

Spatially varying wind from HRRR's native grid (no downscaling): sample every HRRR node over
the region's bounding box per hour (about 9 x 11 nodes), interpolate to the cells, and let the
spread rule use per-cell wind speed and direction (its wind factor, WAF, vector combination
and ellipse already broadcast over arrays). Needs two backward-compatible additions to
frozen contracts (sign-off): an optional per-cell wind field on `Weather` and an optional
field path on `WeatherConfig`. Expected: fire A day 1 head WNW (J ~0.6 on 27.5 h from the
rotation proxy), day 2 SE lobe reachable, a multiplier near 1 instead of 0.38, fewer edge runs.
Then recalibrate (WP 3.1) and revalidate (WP 3.3).

## 6. Fix status (WP 3.4, implemented)

Branch `wp-3.4-spatial-wind` implements section 5 with no downscaling:

- **Data.** `data.hrrr.fetch_hrrr_wind_field` reads the same UGRD/VGRD messages as the point,
  keeps the HRRR nodes around the grid plus one node of margin (fire A: see the build report
  below), rotates each node's grid-relative components to east/north at that node's own
  longitude, and projects every cell centre into the HRRR Lambert grid (PROJ) to store its
  fractional node position. The region gets `wind_field.npz` (format in [data.md](data.md),
  *Wind field*); `scripts/download_data.py <region> --wind-field` adds it to a region already
  built.
- **Run time.** `wildfire_sim.wind_field.WindFieldSeries` interpolates `u` and `v` bilinearly to
  the cells (components, not speed and direction), hour by hour with the weather series' rule;
  `sim.run` attaches the field to each step's `Weather.wind_field`. The Rothermel and
  Alexandridis rules use each cell's 10 m speed and "to" bearing in the midflame wind, wind
  factor, wind-slope vector sum and ellipse; without a field, and with a uniform field equal to
  the scalar, they are bit-identical to before (tested, numpy and numba engines).
- **Everything that builds fire A uses it.** `calibration.real_fire` sets
  `weather.wind_field` when the file exists, so calibration, validation and
  `scripts/sweep_eval.py` use it; `--no-wind-field` returns to the single point. The ensemble,
  validation and sensitivity wind perturbations scale and turn every cell.
- **Contracts.** Two backward-compatible additions, signed off: `grid.Weather.wind_field`
  (`grid.WindField`) and `config.WeatherConfig.wind_field`.

WP 3.1 (calibration) and WP 3.3 (validation) must be rerun with the field.

### 6.1 Fire A with the field (2026-11-27)

Build: `scripts/download_data.py lake_2024 --wind-field` wrote 336 hours on a 14 x 16 node
window (5.8 MB; node speeds 0.0-14.4 m/s, median 2.9). Checks: at the grid centre the field
matches the point series in `hourly.csv` to 0.019 m/s and 0.6 deg (every hour above 1 m/s).
At the ignition cell and 8 km W and E of it, it reproduces the independent samples of
`runs/sweep/hrrr_points.txt` to 0.1 m/s and 1 deg (e.g. +6 h: ignition 6.8 m/s from 99 deg,
8 km W 2.5 from 37, 8 km E 4.6 from 114). The harness still reproduces the single-point baseline
(lb 0.641, m 0.383, 48.5 h: J 0.421, objective 0.403, 11,843 ha).

Runs: `scripts/sweep_eval.py lake_2024 --param lb_scale=1.0 --multipliers 0.383,0.6,0.8,1.0
--seeds 1` (`runs/sweep/field/field_48h.json`, `field_27h.json`). The single-point reference at
the same `lb_scale` 1.0 is `runs/sweep/calib/lb1_w48.json` and `lb1_w27.json`, plus m 0.6 at
27.5 h from `runs/sweep/field/masks_point.json`. Arrival error is given normalized and in hours.

| wind | window | m | J | arrival err | objective | area ratio | edge? | sector recall |
|---|---|---|---|---|---|---|---|---|
| field | 48.5 h | 0.383 | 0.419 | 0.295 / 14.3 h | 0.438 | 1.47 | no | N 0.99 NE 0.95 E 0.86 SE 0.26 S 0.90 SW 0.96 W 0.85 NW 0.88 |
| field | 48.5 h | 0.6 | 0.259 | 0.159 / 7.7 h | 0.450 | 3.17 | no | N 0.99 NE 0.95 E 0.97 SE 0.50 S 0.98 SW 0.97 W 0.97 NW 0.99 |
| field | 48.5 h | 0.8 | 0.171 | 0.102 / 5.0 h | 0.466 | 5.34 | 33.6 h | N 0.99 NE 0.95 E 0.97 SE 0.76 S 0.98 SW 0.97 W 0.97 NW 0.99 |
| field | 48.5 h | 1.0 | 0.134 | 0.069 / 3.4 h | 0.468 | 7.25 | 24.5 h | N 0.99 NE 0.95 E 0.97 SE 0.95 S 0.98 SW 0.97 W 0.97 NW 0.99 |
| point | 48.5 h | 0.383 | 0.455 | 0.289 / 14.0 h | 0.417 | 1.34 | no | N 0.99 NE 0.95 E 0.85 SE 0.30 S 0.98 SW 0.96 W 0.87 NW 0.76 |
| point | 48.5 h | 0.5 | 0.350 | 0.199 / 9.6 h | 0.424 | 2.21 | no | N 0.99 NE 0.95 E 0.95 SE 0.40 S 0.98 SW 0.96 W 0.97 NW 0.97 |
| point | 48.5 h | 0.7 | 0.226 | 0.116 / 5.6 h | 0.445 | 3.87 | 32.7 h | N 0.99 NE 0.95 E 0.97 SE 0.65 S 0.98 SW 0.97 W 0.97 NW 0.99 |
| point | 48.5 h | 1.0 | 0.155 | 0.064 / 3.1 h | 0.454 | 6.25 | 14.5 h | N 0.99 NE 0.95 E 0.97 SE 0.95 S 0.98 SW 0.97 W 0.98 NW 0.99 |
| field | 27.5 h | 0.383 | 0.460 | 0.355 / 9.8 h | 0.448 | 0.85 | no | N 0.99 NE 0.95 E 0.56 SE 0.15 S 0.21 SW 0.77 W 0.67 NW 0.68 |
| field | 27.5 h | 0.6 | 0.455 | 0.199 / 5.5 h | 0.372 | 1.95 | no | N 0.99 NE 0.95 E 0.87 SE 0.61 S 0.93 SW 0.97 W 0.97 NW 0.96 |
| field | 27.5 h | 0.8 | 0.334 | 0.129 / 3.6 h | 0.398 | 2.88 | no | N 0.99 NE 0.95 E 0.96 SE 0.92 S 0.97 SW 0.97 W 0.97 NW 0.99 |
| field | 27.5 h | 1.0 | 0.220 | 0.086 / 2.4 h | 0.433 | 4.41 | 24.5 h | N 0.99 NE 0.95 E 0.97 SE 0.94 S 0.98 SW 0.97 W 0.98 NW 0.99 |
| point | 27.5 h | 0.383 | 0.537 | 0.356 / 9.8 h | 0.409 | 0.77 | no | N 0.99 NE 0.95 E 0.55 SE 0.15 S 0.42 SW 0.94 W 0.74 NW 0.52 |
| point | 27.5 h | 0.5 | 0.536 | 0.262 / 7.2 h | 0.363 | 1.34 | no | N 0.99 NE 0.95 E 0.72 SE 0.36 S 0.80 SW 0.96 W 0.92 NW 0.78 |
| point | 27.5 h | 0.6 | 0.467 | 0.185 / 5.1 h | 0.359 | 1.94 | no | N 0.99 NE 0.95 E 0.88 SE 0.71 S 0.96 SW 0.97 W 0.97 NW 0.95 |
| point | 27.5 h | 0.7 | 0.367 | 0.125 / 3.4 h | 0.379 | 2.60 | no | N 0.99 NE 0.95 E 0.94 SE 0.88 S 0.97 SW 0.97 W 0.97 NW 0.99 |
| point | 27.5 h | 1.0 | 0.205 | 0.058 / 1.6 h | 0.426 | 4.73 | 14.5 h | N 0.99 NE 0.95 E 0.97 SE 0.94 S 0.98 SW 0.97 W 0.98 NW 0.99 |

Head of the fire (farthest burned cell from the ignition; `--save-masks`, 27.5 h runs):

| wind, m | +6 h | +10 h | +20 h |
|---|---|---|---|
| field, 0.6 | 1.6 km at 304 deg | 4.7 km at 284 deg | 8.1 km at 277 deg |
| field, 1.0 | 3.5 km at 247 deg | 6.9 km at 276 deg | 9.6 km at 293 deg |
| point, 0.6 | 1.6 km at 295 deg | 4.9 km at 250 deg | 10.5 km at 274 deg |
| point, 1.0 | 4.7 km at 240 deg | 9.0 km at 263 deg | 14.8 km at 242 deg |
| observed | | about 11 km WNW (W detections by 10 h) | |

Interpretation:

1. **The head turns WNW but slows down.** At 10 h it points 276-284 deg instead of 250-263 deg,
   the +20 deg the rotation proxy suggested. But it reaches only 6.9 km at m 1, against 9.0 km
   with the single point and about 11 km observed. HRRR's own 3 km wind in the WNW valley is
   weak (1-4 m/s, from N to NE at 8 km W), while the centroid point had 7-8 m/s. The field
   faithfully contains the information section 4 found, and that information says the head
   should be slow there.
2. **The SE lobe is not reached any better.** At matched area, SE recall is about the same or
   lower: 0.26 against 0.30 at m 0.383 (48.5 h), and 0.61 against 0.71 at m 0.6 (27.5 h).
   Reaching it still takes area ratios of 3-5. The +26 h sea breeze is in the field, but with
   2-5 m/s it does not carry the fire 8 km ESE in the 23-34 h period.
3. **No better fit, and no multiplier near 1.** The best objective is m 0.383 at 48.5 h
   (0.438, against 0.417 for the single point) and m 0.6 at 27.5 h (0.372, against 0.359 to 0.363).
   J falls by 0.04 to 0.08 at the low multipliers. N and NE false alarms grow (centroid 1.9 km N of
   the observed one at 48.5 h, m 0.383): with the weak local wind, the slope drives the fire up
   the ridge. Section 5's expected J of about 0.6 at 27.5 h is not reached (best 0.460).
4. **Fewer edge runs at high m.** At m 1 the fire reaches the edge at 24.5 h instead of 14.5 h,
   and m 0.8 stays inside at 27.5 h. The slower head leaves more room before the edge rule
   applies.
5. **What this means.** The direction error from the single point is gone, but a rate error in
   the WNW run is now exposed: an 11 km run in under 10 h needs a stronger local wind than 3 km
   HRRR resolves along the valley and ridge (terrain channelling, downslope evening winds), or
   a fuel/rate effect not in the model. The next levers are terrain-aware downscaling (for
   example WindNinja; out of scope here) or weighting the field toward the ridge winds. Rerunning
   WP 3.1/3.3 with the field remains required. The field is the more faithful input even
   though it does not score better on this one fire.

## Appendix: the single-point sweep tables (condensed from `runs/sweep/*/SUMMARY.md`)

All on fire A with the single centroid wind, `scripts/sweep_eval.py lake_2024 --seeds 1`
(seed 4334430513956379144), 48.5 h window unless noted, objective `0.5 (1 - J) + 0.5 E`.
"Best obj" columns are at the minimum-objective multiplier m; "best J" is the scan maximum.
SE recall and area ratio at the best-objective m (best-J m in brackets where it differs).

### A.1 Weather group (`lb_scale` 0.641)

| configuration | best-obj m | J | obj | SE recall | area ratio | best J (m, area) |
|---|---|---|---|---|---|---|
| baseline (WP 3.1, fine scan) | 0.383 | 0.421 | 0.403 | 0.38 | 1.77 | 0.480 (0.29, x1.05) |
| wind x0.7 | 0.6 | 0.293 | 0.427 | 0.59 | 2.89 | 0.444 (0.383, x1.29) |
| wind x1.6 | 0.25 | 0.434 | 0.418 | 0.29 | 1.49 | 0.434 (0.25, x1.49) |
| wind x2.0 | 0.25 | 0.355 | 0.434 | 0.34 | 2.06 | 0.419 (0.15, x0.78) |
| live woody 60 % | 0.25 | 0.468 | 0.409 | 0.30 | 1.37 | 0.468 (0.25, x1.37) |
| live woody 120 % | 0.6 | 0.305 | 0.430 | 0.52 | 2.70 | 0.470 (0.383, x1.10) |
| live herb 60 % | 0.383 | 0.436 | 0.406 | 0.33 | 1.62 | 0.437 (0.25, x0.72) |
| live herb 90 % | 0.383 | 0.459 | 0.408 | 0.31 | 1.42 | 0.459 (0.383, x1.42) |
| dead moisture x2 (fine) | 0.48 | 0.428 | 0.398 | 0.35 | 1.71 | 0.480 (0.383, x1.16) |
| dead moisture x3 | 0.6 | 0.430 | 0.386 | 0.30 | 1.67 | 0.430 (0.383/0.6) |
| canopy height 4 m | 0.383 | 0.417 | 0.403 | 0.39 | 1.79 | 0.449 (0.25, x0.80) |
| crown ratio 0.2 | 0.383 | 0.408 | 0.404 | 0.40 | 1.86 | 0.452 (0.25, x0.83) |
| wind limit on | 0.383 | 0.421 | 0.403 | 0.38 | 1.77 | 0.448 (0.25, x0.79) |

No best-objective run reached the edge. Nothing moves J by more than about 0.03 once m is
refitted; moisture and wind scaling shift m, not the shape.

### A.2 Fuel / shape group (m 0.25, 0.383, 0.6, 1.0 unless noted)

| # | configuration | best m | obj | J | best J (m) | SE recall | area ratio | edge |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline lb 0.641 | 0.383 | 0.403 | 0.421 | 0.421 (0.383) | 0.38 | 1.77 | no |
| 1 | lb 1.0 | 0.383 | 0.417 | 0.455 | 0.455 (0.383) | 0.30 | 1.34 | m=1: 14.5 h |
| 2 | lb 1.5 | 0.6 | 0.431 | 0.328 | 0.468 (0.383) | 0.40/0.18 | 2.36/1.02 | m=1: 15.2 h |
| 3 | lb 2.0 | 0.6 | 0.424 | 0.366 | 0.439 (0.383) | 0.35/0.11 | 2.04/0.88 | m=1: 15.7 h |
| 4 | lb 3.0 | 0.6 | 0.422 | 0.392 | 0.414 (0.383) | 0.30/0.07 | 1.79/0.73 | m=1: 15.9 h |
| 5 | lb 1.0, cap 3.0 | 0.383 | 0.417 | 0.460 | 0.460 (0.383) | 0.29 | 1.31 | m=1: 14.6 h |
| 6 | lb 1.0, cap 4.0 | 0.383 | 0.417 | 0.460 | 0.460 (0.383) | 0.29 | 1.31 | m=1: 14.6 h |
| 7 | lb 2.0, cap 4.0 | 0.6 | 0.427 | 0.421 | 0.422 (0.383) | 0.29/0.07 | 1.55/0.70 | m=1: 18.0 h |
| 8 | lb 1.0, sampled rates, cap 1.5 | 0.383 | 0.408 | 0.417 | 0.433 (0.25) | 0.34/0.13 | 1.70/0.72 | m=0.6: 37.6 h |
| 9 | lb 2.0, sampled rates, cap 3.0 | 0.6 | 0.408 | 0.457 | 0.457 (0.6) | 0.30 | 1.41 | m=1: 22.3 h |
| 10 | lb 1.0, stencil 8 | 0.383 | 0.414 | 0.375 | 0.438 (0.25) | 0.38/0.17 | 2.03/0.88 | m=0.6: 32.2 h |
| 11 | lb 1.0, arrival shape 20 | 0.383 | 0.418 | 0.479 | 0.479 (0.383) | 0.28 | 1.16 | m=1: 16.1 h |
| 12 | lb 1.0, arrival shape 1 (m / 10.8) | 0.056 | 0.417 | 0.429 | 0.429 (0.056) | 0.37 | 1.63 | m=0.093: 36.9 h |
| 13 | lb 1.5 + shape 20 (m 0.3-0.6) | 0.45 | 0.422 | 0.463 | 0.463 (0.45) | 0.27 | 1.20 | no |
| 14 | lb 1.0, urban ros_factor 1.0 | 0.383 | 0.418 | 0.452 | 0.452 (0.383) | 0.30 | 1.36 | m=1: 13.8 h |

Best J 0.479 (+0.058); no configuration beats the baseline objective; SE recall 0.27-0.40.

### A.3 Breaks group (`lb_scale` 0.641)

| configuration | best-obj m | J | obj | best J (m) | SE recall | area ratio |
|---|---|---|---|---|---|---|
| baseline | 0.383 | 0.421 | 0.403 | 0.448 (0.25) | 0.38 (0.14) | 1.77 (0.79) |
| no breaks | 0.383 | 0.357 | 0.431 | 0.467 (0.25) | 0.41 (0.19) | 2.21 (0.92) |
| breach floor 0.05 MW/m | 0.383 | 0.404 | 0.411 | 0.456 (0.25) | 0.38 (0.15) | 1.85 (0.80) |
| breach floor 0.09 MW/m | 0.383 | 0.404 | 0.411 | 0.456 (0.25) | 0.38 (0.15) | 1.85 (0.80) |
| breach floor 0.36 MW/m | 0.383 | 0.423 | 0.402 | 0.459 (0.25) | 0.37 (0.14) | 1.75 (0.74) |
| tree cover min 0.0 / 0.30 | 0.383 | 0.421 | 0.403 | 0.448 (0.25) | 0.38 (0.14) | 1.77 (0.79) |
| Wilson intercept 2.04 | 0.383 | 0.409 | 0.406 | 0.463 (0.25) | 0.40 (0.18) | 1.85 (0.85) |
| Wilson intercept 0.68 | 0.383 | 0.432 | 0.401 | 0.454 (0.25) | 0.36 (0.14) | 1.69 (0.72) |
| Wilson width_coef -1.485 | 0.383 | 0.444 | 0.398 | 0.444 (0.383) | 0.34 (0.08) | 1.59 (0.68) |
| Wilson width_coef -0.66 | 0.383 | 0.376 | 0.423 | 0.463 (0.25) | 0.40 (0.18) | 2.04 (0.85) |
| holdover 2 h / 24 h | 0.383 | 0.421 | 0.403 | 0.448 (0.25) | 0.38 (0.14) | 1.77 (0.79) |

Every m 0.6 and 1.0 run reached the edge (28-36 h and 12-14 h). Breaks move J by about 0.02;
they do not hold the SE lobe: the fire does not reach it in time.

### A.4 Calibration weights and window (edge rule applied; arrival error normalized / in h)

Raw scans, `lb_scale` 0.641:

| window | m | J | arrival err | area ratio | SE recall | edge (h) |
|---|---|---|---|---|---|---|
| 48.5 h | 0.2 / 0.25 / 0.3 / 0.34 | 0.368 / 0.448 / 0.470 / 0.440 | 0.430 / 0.377 / 0.313 / 0.272 | 0.47 / 0.79 / 1.18 / 1.49 | 0.07 / 0.14 / 0.29 / 0.32 | - |
| 48.5 h | 0.383 / 0.43 / 0.5 | 0.421 / 0.343 / 0.281 | 0.226 / 0.186 / 0.144 | 1.77 / 2.31 / 2.98 | 0.38 / 0.44 / 0.55 | - |
| 48.5 h | 0.6 / 0.7 / 1.0 | 0.225 / 0.186 / 0.129 | 0.106 / 0.074 / 0.077 | 3.94 / 5.09 / 7.54 | 0.69 / 0.87 / 0.95 | 35.9 / 22.1 / 13.6 |
| 27.5 h | 0.2 / 0.3 / 0.383 | 0.230 / 0.508 / 0.563 | 0.477 / 0.383 / 0.298 | 0.24 / 0.64 / 1.13 | 0.09 / 0.15 / 0.32 | - |
| 27.5 h | 0.5 / 0.7 / 1.0 | 0.489 / 0.282 / 0.166 | 0.183 / 0.082 / 0.048 | 1.88 / 3.43 / 5.86 | 0.78 / 0.94 / 0.94 | - / 22.1 / 13.6 |
| 20.5 h | 0.2 / 0.3 / 0.383 | 0.403 / 0.652 / 0.434 | 0.446 / 0.375 / 0.306 | 0.44 / 1.22 / 2.10 | 0.25 / 0.70 / 0.78 | - |
| 20.5 h | 0.5 / 0.7 / 1.0 | 0.273 / 0.143 / 0.083 | 0.213 / 0.099 / 0.039 | 3.52 / 6.73 / 11.72 | 0.86 / 0.86 / 0.87 | - / - / 13.6 |

Selected m by weights `(w_o, w_a)` = (1, 0) / (0.75, 0.25) / (0.5, 0.5) / (0.25, 0.75) / (0, 1):

| window | lb_scale | selected m | J at (0.5, 0.5) |
|---|---|---|---|
| 48.5 h | 0.641 | 0.3 / 0.3 / 0.383 / 0.5 / 0.5 | 0.421 |
| 27.5 h | 0.641 | 0.383 / 0.383 / 0.5 / 0.5 / 0.5 | 0.489 |
| 20.5 h | 0.641 | 0.3 / 0.3 / 0.3 / 0.7 / 0.7 | 0.652 |
| 48.5 h | 1.0 | 0.3 / 0.383 / 0.383 / 0.5 / 0.5 | 0.455 |
| 27.5 h | 1.0 | 0.383 / 0.5 / 0.5 / 0.7 / 0.7 | 0.536 |
| 48.5 h | 2.0 | 0.383 / 0.5 / 0.5 / 0.7 / 0.7 | 0.438 |
| 27.5 h | 2.0 | 0.5 / 0.5 / 0.7 / 0.7 / 0.7 | 0.457 |

At 48.5 h and `lb_scale` 0.641, 0.383 is selected for `w_a` 0.37-0.62. The arrival term falls
with m up to the fastest on-grid run, so above `w_a` of about 0.6 the grid edge picks m. No
weight or window selects m > 0.7; every run at m 0.9-1.0 reaches the edge at 13.6-17.6 h.
