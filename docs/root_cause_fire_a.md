# Why the calibrated model misses fire A: root-cause analysis (2026-11-27)

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
