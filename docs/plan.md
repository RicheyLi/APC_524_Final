# Project plan

The single plan for wildfire-sim: the model design and its rationale (sections 1-8), then the
milestones, module map and work packages (WPs) that implement it (sections 9-13). Each WP is
one branch and one pull request (see `CONTRIBUTING.md`).

The plan has no schedule. Work is ordered by dependencies (section 13) and judged by the
acceptance criteria of each WP and the three milestones (section 9). Section 14 records what
the design reviews changed and why.

## 1. Summary

A cellular automaton (CA) in which fire crosses from cell to cell at a physical rate of spread,
inputs come from fire-science GIS datasets, and "optimization" is a search over reactive
suppression policies scored on Monte Carlo ensembles.

- **Spread comes from physics, not hand-picked weights.** The time fire takes to cross to each
  neighbour is drawn from a distribution whose mean is distance over rate of spread, so cells
  and time steps have real meaning. With the default shape of 200 that distribution is narrow
  (7 % per link), so one run is close to a deterministic minimum-travel-time solution on the
  lattice; the spread between runs of an ensemble comes mostly from the sampled inputs.
- **Results do not depend on the time step.** This is tested, not assumed.
- **Burning lasts a physical duration.** A cell is a fire source for as long as it takes the
  front to cross it, then flames for its fuel's residence time.
- **Data comes from LANDFIRE, USGS, gridMET and OpenStreetMap.** A bounding box is the only
  thing the user supplies.
- **Calibration on one real fire and a test on a second, untouched fire are required
  milestones.** Without them the model is not predictive.
- **Firefighting is modelled as resources** with arrival delays, production rates, limits and
  costs, acting on the grid.
- **The optimizer tunes parameterized reactive policies,** for example "build a line 200 m
  ahead of the head when it nears a town".
- **Every result is an ensemble** that samples both the fire's own randomness and the
  uncertainty of its inputs, reported as distributions with confidence intervals and
  burn-probability maps.
- **The real-world pipeline is limited to the US,** where 30 m fuel maps exist.

### Why the design looks like this

Four problems with the original idea (a Drossel-Schwabl forest-fire model fed from Google Maps)
shaped the design:

1. **Google Maps is the wrong data source.** It shows what an area looks like, not fuel type,
   fuel load, canopy, moisture or usable elevation, and its terms of service forbid extracting
   map content.
2. **Parameters cannot be calculated from data alone.** Data gives physical inputs (fuel model,
   slope, wind, moisture). Turning those into spread needs either a physical model (Rothermel)
   or coefficients fitted to a real fire, as Alexandridis et al. did for the 1990 Spetses
   Island fire. Either way a calibration step is needed.
3. **The Drossel-Schwabl baseline has no clock.** A step is not a unit of time, so "the fire
   reached town in 3 hours" has no meaning until the step is linked to a spread rate.
4. **Square grids distort fire shape.** With no wind a fire spreads as a diamond or octagon
   instead of a circle. The model uses true neighbour distances, keeps crossing times random,
   and tests the simulated shape against the analytic circle and ellipse.

## 2. Core model: a CA with physical scale

Cells are 30 m, the native resolution of the US fuel maps.

**Grid.** A 10 km x 10 km study area at 30 m is about 333 x 333 = 111,000 cells. Each cell
stores (see `wildfire_sim.grid.Grid`):

- Static layers: fuel model, canopy cover, elevation, slope, aspect, land use (wildland, urban,
  water, road) and asset value.
- State: unburnable, unburned fuel, burning (with a timer) or burned out; arrival time;
  intensity.
- Suppression layers: fireline present, and retardant strength.

Real grids also carry their coordinate reference system and affine transform in
`Grid.metadata` (keys `crs`, `transform`) so results can be compared with observed perimeters
and exported as GeoTIFF.

**Domain size.** The 10 km area is the reference size for benchmarks and synthetic studies, not
a limit. A real fire's grid is sized from the fire: the bounding box of its observed final
perimeter plus a buffer (WP 2.6). Fire A is far larger than the reference area (section 6), so
its grid is on the order of a million cells. A fire that reaches the edge of its grid can no
longer be trusted, because the model cannot say what would have burned beyond it: every run
records whether and when it reached the edge (WP 1.11), and such runs are reported as lower
bounds, never silently averaged in.

### 2.1 Ignition kernel

A *link* is a burning cell i and one of its neighbours j. Fire crosses a link in a random time
with mean d\_ij / R\_ij, where d\_ij is the centre-to-centre distance (dx, or sqrt(2) dx on a
diagonal) and R\_ij is the link's rate of spread. The crossing time is **gamma-distributed with
shape k** (coefficient of variation 1 / sqrt(k)).

**Link rate.** Fire going from the centre of i to the centre of j crosses half of each cell,
so the link rate is the harmonic mean of the two cells' rates in that direction
(`link_rate_of_spread`):

```latex
R_{ij} = \frac{2\,R_i\,R_j}{R_i + R_j}
```

It is zero if either cell cannot spread. Using the source cell's rate alone would let fire
enter slow fuel, or a retardant drop, at the fast cell's speed.

**Thresholds.** The simulator samples the crossing-time law once per link for the whole run
(`draw_crossing_thresholds`): a dimensionless threshold tau\_ij from a gamma distribution with
mean 1 and shape k. Each burning cell accumulates progress along each of its links, and the
neighbour ignites when the progress reaches the threshold:

```latex
x_{ij}(t) = \int_{t_i}^{t} \frac{R_{ij}(s)}{d_{ij}}\,ds, \qquad
\text{j ignites when } x_{ij} \ge \tau_{ij}
```

A cell with several burning neighbours ignites when the first of those links completes, which
keeps the "more burning neighbours means sooner" idea. When a link completes part-way through
a step, the arrival time is interpolated within the step rather than rounded to its end.

This form was chosen over drawing a random number every step because:

- it is exact when the rate changes during a crossing (a wind shift, a retardant drop);
- the result does not depend on dt beyond the interpolation error;
- there are no special functions in the time loop;
- it gives **common random numbers**: two runs with the same seed and different suppression
  policies see the same fire "luck" on every link, so their difference is the policy.

The equivalent per-step form (`ignition_probability` with `age` and `shape`,
`neighbour_ignition_probability`) is kept for analysis and tests:

```latex
p_{ij} = 1 - \frac{S(a_i + \Delta t)}{S(a_i)}, \qquad
S(t) = Q\!\left(k,\; k\,\frac{R_{ij}\,t}{d_{ij}}\right), \qquad
P_j = 1 - \prod_{i} \left(1 - p_{ij}\right)
```

where a\_i is the time since cell i ignited and Q is the regularized upper incomplete gamma
function.

**Why the shape is not 1.** The first version of this plan used p = 1 - exp(-R dt / d), which
is the k = 1 (memoryless) case. Its mean crossing time for one link is d / R, but on a lattice
the front follows the fastest of many random paths, so the fire runs several times faster than
R and gets faster as the step shrinks. Measured with `scripts/kernel_front_speed.py` (per-step
form, flat uniform fuel, no wind, R = 0.2 m/s, dx = 30 m, 40 simulated minutes, mean of 6
seeds; values are front speed / R along the grid axes):

| Shape k | 8 neighbours, dt 30 s | dt 15 s | dt 7.5 s | 16 neighbours, dt 30 s | dt 7.5 s |
| --- | --- | --- | --- | --- | --- |
| 1 (memoryless) | 3.07 | 3.85 | 4.47 | 5.80 | off grid |
| 5 | 1.27 | 1.38 | 1.44 | 1.72 | 1.90 |
| 20 | 1.01 | 1.03 | 1.07 | 1.15 | 1.21 |
| 50 | 0.93 | 0.96 | 0.99 | 1.02 | 1.07 |

The diagonal speed is within 6 % of the axis speed in every row, so the burn stays round.
Shape 20 was the first default. Phase 1 showed two consequences, and the default is now
**`spread.arrival_shape = 200`**:

- A residual speed bias remains and differs by stencil. WP 1.4 measured it for the real engine
  (the threshold form runs a little faster than the per-step form in the table: 1.126 on 8
  neighbours and 1.277 on 16 at shape 20) and stored a **front-speed correction** per
  (stencil, shape) in `configs/kernel_correction.yaml`; the simulator divides link rates by
  it. **Known weakness:** the correction is measured on round fires. Kernel noise speeds up a
  round front but hardly speeds up the head of a narrow fire, so dividing by the correction
  leaves such a head 11 % (8) to 22 % (16) slow. At shape 200 the correction is within 4 % of
  1 (0.992 on 8 neighbours, 1.042 on 16) and the problem nearly disappears, which is why 200
  is the default.
- The kernel's own noise is small (coefficient of variation 7 % per link at shape 200), so an
  ensemble that varies only the seed would produce nearly identical fires. Ensembles
  therefore also sample input uncertainty (section 6).

### 2.2 Rate of spread

The head-fire rate comes from a spread rule behind one interface (`SpreadRule`). Rules return a
rate per stencil direction for every cell, plus head intensity and flame length.

- *Constant rule*: one rate in every direction; the baseline used by the kernel tests.
- *Alexandridis rule*: a base rate times vegetation, density, wind and slope factors
  (Alexandridis et al. 2008). Quick to build, calibrated on real fires, and the comparison
  point for Rothermel.
- *Rothermel rule*: the surface fire spread model (Rothermel 1972 with the Albini 1976
  revisions) for the Scott & Burgan 40 fuel models, implemented in vectorized NumPy and
  checked against the BehavePlus fire behaviour engine (the `behave` library, run through
  `pyrothermel`). This is the model the validation and the strategy search run on. As built
  (WP 2.1a) it agrees with that engine to 0.02 % for all 53 fuel models. The reference values
  in the test were produced by running the engine, not copied from a printed BehavePlus table;
  `pyrothermel`'s metric preset overstates intensity 2.33-fold (heat content in kJ/kg read as
  Btu/lb), so the comparison uses its US preset.

For Rothermel the directional rates follow an ellipse:

- Wind and slope are combined as vectors, so the direction of maximum spread is not the wind
  direction when the slope faces elsewhere.
- The wind adjustment factor models are built on the 20-ft wind, while `Weather.wind_speed`
  and the weather sources give the 10 m wind. The 10 m wind is first divided by 1.15 to get
  the 20-ft wind (Turner and Lawson 1978, as BehavePlus does; Andrews 2012), then reduced to
  midflame height with the wind adjustment factor, which depends on canopy cover and fuel bed
  depth (Andrews 2012).
- The length-to-breadth ratio comes from the effective midflame wind U (m/s), and the rate at
  angle theta from the direction of maximum spread follows from the eccentricity e:

```latex
LB = 0.936\,e^{0.2566 U} + 0.461\,e^{-0.1548 U} - 0.397, \qquad
e = \frac{\sqrt{LB^2 - 1}}{LB}, \qquad
R(\theta) = R_{\text{head}}\,\frac{1 - e}{1 - e\cos\theta}
```

**Intensity follows the front, not the head.** Byram's fireline intensity is heat per unit
area times the rate at which the front advances normal to itself, so the flanks and the back
of a fire are much cooler than its head: on the analytic ellipse the flank burns at
sqrt((1 - e) / (1 + e)) of the head intensity and the back at (1 - e) / (1 + e), which is 17 %
and 3 % at a length-to-breadth ratio of 3. A spread rule reports head values only, so the
simulator computes each cell's intensity from the local front-normal rate, estimated from the
arrival times of the cell and its already-burned neighbours as 1 / |grad T| (WP 1.12). Flame
length follows from that intensity (Byram 1959). Giving every cell the head intensity, as
Phase 1 did before WP 1.12, would make the crew safety limit and line breaching (section 4)
treat a flank like a head and rule out flank attack, the standard tactic. Scaling by the rate along the link
that ignited the cell is not enough either: fire reaches the flank along a ray from the
origin, whose speed is about three times the front-normal speed there at a ratio of 3.

As built (WP 1.12), the estimate is exact on the polygon the lattice burns, so it inherits the
lattice's shape error: within 3 % on a round fire and 20 % at a ratio of 1.5, but at a ratio
of 3 the flank reads 27-45 % too hot and a head that lies between two stencil directions
reads 0.4 of the head value ([lattice_study.md](lattice_study.md), section 6). Suppression
results on elongated fires therefore also wait for WP 2.1d.

After WP 2.1d the `"rothermel"` rule caps the ratio at 2.1, so a ratio of 3 does not occur.
With fitted rates, at a ratio of 1.5 every part is within 13 % and at the cap within 17 %,
except one case: with the head between two stencil directions the cells on the head bearing
read 0.74-0.84 of the head intensity ([lattice_study.md](lattice_study.md), section 7).

**Why Rothermel is implemented here and not only wrapped.** `pyrothermel` publishes only a
Windows wheel; on Linux it needs a C++ toolchain, so in CI (Ubuntu) every Rothermel test would
be skipped. It is also called one cell at a time. A NumPy implementation runs everywhere, is
fast over a grid, and is ordinary testable code. `pyrothermel` stays as an optional
cross-check.

**Where temperature and humidity enter.** They set dead fuel moisture, which feeds Rothermel.
One-hour fuel moisture is computed each hour from temperature and relative humidity with an
equilibrium moisture content model, so the daily cycle of burning conditions is represented.
As built (WP 2.1c) the model is Simard's (1968) regression, in the form of the 1978 National
Fire-Danger Rating System (Cohen & Deeming 1985): it reproduces the Wood Handbook's
equilibrium moisture table within 1.0 percentage point from 30 to 110 deg F. The air at the
fuel is corrected for sunshine by a cloudiness code, and the 1-h and 10-h moisture are 1.03
and 1.28 times its equilibrium moisture, so the 10-h value is 1.243 times the 1-h value.
gridMET provides 100-hour and 1000-hour moisture only (`fm100`, `fm1000`), so hundred-hour
moisture comes from gridMET. Thousand-hour moisture is not used: `Weather` has no field for
it. Both computed moistures are equilibrium values without a time lag: right for 1-h fuels,
but applied every hour it swings the 10-h moisture through the whole daily cycle, which a
10-hour time lag would damp (section 13).

**Limitations stated up front.** Surface fire only until the crown-fire extension (WP 6.2).
Wind is uniform over the grid; terrain-channelled wind is out of scope.

### 2.3 Burn duration

A cell has two clocks. It is an *ignition source* (state BURNING) until none of its links can
still complete. It is *flaming* (contributes `intensity`, blocks crews) for the time the front
takes to cross it plus its fuel's residence time (seconds for grass, longer for timber).
Burning out a cell after the residence time alone would stop the fire: grass flames for under
10 s while a 30 m cell takes 150 s to cross at 0.2 m/s (measured with the memoryless kernel:
with a 30 s burn the fire dies after about 10 cells).

**Dormant sources.** A spread rate can fall to zero and come back: fine dead fuel passes its
moisture of extinction on a humid night, and a fresh retardant drop can stop spread until it
fades. Real fires hold over through such periods and resume. A cell whose remaining links are
stalled this way therefore stays a *dormant* source, keeping its link progress, for up to a
holdover time per fuel model, and resumes when the rate returns (WP 1.10). It is not flaming
while dormant. Before WP 1.10 the engine ended the cell, and with it the fire: a 10-minute
stall on `flat_no_wind` ended the run with 195 cells burned instead of 4,087 (fourth review,
section 14). A link that is blocked by barrier geometry does not make a cell dormant; that
cannot end.

### 2.4 Time step, barriers and stencils

**Time step.** Arrival times do not depend on dt: a link may complete, and the newly lit cell
make progress, within the same step, and the in-step cascade is exact (third review, section
14), so a fast fire is never capped at one cell per step. dt only sets how often the weather
is looked up and hooks (suppression) act.

**Barriers.** With the 8-neighbour stencil, fire passes diagonally between two barrier cells
that touch only at a corner, so rasterized lines must be 4-connected. With the 16-neighbour
stencil a knight move would jump a barrier one cell wide, so the simulator blocks knight moves
whose intermediate cells are unburnable.

**Which stencil.** A lattice cannot represent a narrow ellipse well: after time T the burn is
the polygon spanned by the stencil directions, each scaled by its rate, and that polygon sits
inside the ellipse. WP 1.4 measured this ([lattice_study.md](lattice_study.md)): with the head
in the worst direction, head and flank distances stay within 10 % of the analytic ellipse only
up to a length-to-breadth ratio of about 1.0 on 8 neighbours and 1.5 on 16; at a ratio of 3 the
head is 59 % (8) and 32 % (16) short. Larger stencils help slowly (2.1 at 32 neighbours, 2.75 at
48).

The error depends on the bearing of the head relative to the grid axes, so it cannot be left
for calibration to absorb: a spread multiplier and ellipse shape fitted on one fire would
carry that fire's grid orientation to the next one. This is the same mistake the second review
removed for the front speed. It has to be dealt with before Phase 3, in WP 2.1d: directional
rates fitted so the lattice polygon best matches the ellipse, a cap on the length-to-breadth
ratio at the largest value the lattice represents within 10 %, a count of how often the cap
bites, and a calibration test that recovers the same parameters at every head bearing (WP
3.1). The choice among the remaining options is recorded in section 13.

As built (WP 2.1d, [lattice_study.md](lattice_study.md) section 7): fitted rates halve the
polygon's shape error and raise the largest ratio drawn within 10 % at every tested bearing
from 1.5 to 2.1 on 16 neighbours (1.2 to 1.4 on 8). The rule caps the ratio there, keeping the
head rate, so a fire in a stronger wind (above about 3.4 m/s effective midflame wind, a 10 m
wind of about 10.7 m/s over open grass) is drawn rounder than it is: its flanks run too wide.
Each run records the burning cell-hours under the cap (`lb_capped_cell_hours`) and the CLI
warns. The fitted rates do not reach a ratio of 3, which reopens the choice of method
(section 13).

### 2.5 Extensions

Spotting (embers that carry fire over lines) and crown fire are designed in Phase 6. They are
real physics the base model lacks, not decoration: without spotting any wide line is perfect
against a surface fire, and without crown fire timber stands burn too slowly in strong wind.

Two results depend on them, so they are ordered as prerequisites, not left as optional extras
(section 13). The strategy experiments (WP 4.7) need spotting (WP 6.1): without it they would
overstate the value of indirect line. Validation on a fire in timber needs crown fire (WP
6.2); until that is merged, both validation fires must be ones where surface fire dominated.

### 2.6 Urban cells

Urban cells burn. They follow the same logic as wildland cells, scaled to be harder to burn
(WP 1.13): an urban cell takes its rate of spread from a surrogate timber fuel model,
multiplied by a factor below 1 so fire takes longer to cross into and through it, and it
flames for longer than the surrogate fuel's residence time. The surrogate model and both
factors live in `configs/urban.yaml`. They are team assumptions, not measurements: this is a
stand-in for structure-to-structure spread, not a model of it, and the factors are part of the
sensitivity analysis (WP 3.2). Since WP 2.1a the surrogate's residence time is cited (TL3,
15 s, Anderson 1969) instead of a 120 s placeholder, so with `residence_factor` 30 an urban
cell flames for 7.5 min, not the hour the factor was chosen to give (section 13). An urban cell is lost when it burns, like any other cell with
an asset value. Before this change urban cells could not burn and counted as lost only when a
neighbour burned, so fire could never run through a town and only its rim could be lost.
Water and wide roads stay unburnable.

## 3. Real-world data pipeline

Every model input has a free, public US source. The pipeline is: user gives a bounding box and
a date range, download each layer, reproject everything onto one 30 m grid in a projected CRS,
cache to disk with a manifest, run.

| Input | Source | Resolution | Notes |
| --- | --- | --- | --- |
| Fuel model, canopy cover, height, base height, bulk density | LANDFIRE (Scott & Burgan 40 fuel models) | 30 m | US only. The LANDFIRE **version must predate the fire** being simulated, or the fuels already show its scar |
| Elevation, slope, aspect | USGS 3DEP | 10-30 m | Slope and aspect computed here |
| Wind, temperature, humidity | Hourly analysis (RTMA or HRRR) where available; gridMET daily otherwise | 2.5-4 km | One value per hour for the grid. Daily data is disaggregated with a documented diurnal profile |
| Dead fuel moisture | 1-h computed from T and RH; 10-h from the 1-h value by a cited offset; 100-h from gridMET (`fm100`) | about 4 km | gridMET has no 1-h or 10-h moisture |
| Live fuel moisture | National Fuel Moisture Database samples near the site; seasonal default otherwise | Point | Matters most in shrub fuels such as chaparral |
| Roads, rivers, lakes, buildings | OpenStreetMap (osmnx) | Vector | See the barrier rule below |
| Asset value | Building footprints x an assumed USD per building | Vector | A team assumption, stated in the README |
| Historical fires (validation) | MTBS burn perimeters; NASA FIRMS VIIRS hotspots; daily perimeters where published | Perimeter; 375 m detections | Final burned area and timing of spread |

**Barrier rule.** A 30 m cell is wider than most roads and streams, so marking every road cell
unburnable would create barriers that do not exist. Only features at least as wide as a
documented threshold (motorways and trunk roads, rivers and lakes mapped as polygons) become
unburnable cells.

**Breaks.** A narrower road or stream is a *break*: the cell keeps its fuel and records the
feature's width in metres. A break either holds or is crossed, depending on the fire's
intensity relative to its width; it does not slow the fire by the fraction of the cell it
covers, which is what the plan said before and has no physical basis. A hand line or dozer
line built by a crew is the same thing, a break of the line's width, so roads and constructed
lines go through one breach model (WP 2.9) and a crew can anchor a line on a road. The model
is the one of section 4: one fixed random number per cell, crossed the first time the breach
probability of the highest intensity the break has faced exceeds it.

The breach probability is Wilson's (1988, eqs. 2 and 4), fitted to 113 experimental head fires
in cured grass against breaks 1.5 to 15 m wide, with intensities up to 17 MW/m:

```latex
p_b = \frac{e^{b}}{1 + e^{b}}, \qquad b = 1.36 + 0.36\,I + (0.61\,T - 0.99)\,W
```

with I the fireline intensity in MW/m, W the break width in m, and T = 1 when trees stand
within 20 m of the break and 0 otherwise. It gives 29 %, 55 % and 88 % for a 3 m break without
trees at 2, 5 and 10 MW/m, as the paper states. Two limits follow from the paper and are part
of the model, not footnotes. The fit must not be used below the intensities it was made on:
its intercept gives a 47 % chance of crossing a 1.5 m break at zero intensity. And its use in
other fuels, at other widths and for flank or backing fire is an extrapolation.

**Reproducibility of data.** Downloaded data is never committed. Every cached region has a
manifest (source, version, access date, bounding box, CRS, SHA-256 of each file), and
`docs/data.md` records each source's access method, licence and attribution (OpenStreetMap
data is ODbL). Access methods are checked when each WP starts, because download services
change; a scheduled CI job runs the `network` tests weekly to catch it when they do.

## 4. Firefighting tactics

Every tactic is a resource that arrives after a dispatch delay, works at a limited rate, costs
money, and changes cells on the grid.

| Tactic | Effect on the grid | Key parameters | Cost drivers |
| --- | --- | --- | --- |
| Indirect attack (control line far from the fire) | Cells along a planned path become unburnable, one segment at a time as it is built. An unfinished segment does not stop fire. | Path, line width, build rate, start time | Crew and dozer hours |
| Parallel attack (line close to the fire) | Same as indirect, but the path is the current fire edge pushed outward by an offset d, recomputed as the fire moves | Offset d, which flank, build rate | Crew and dozer hours |
| Direct attack: ground crews and engines | Crews move along the burning edge and extinguish it at a production rate (metres of fire edge per hour); a burning cell is out when the work done on it covers its edge, and it then stops being a source | Speed along the edge, production rate, maximum flame length | Crew and engine hours |
| Direct attack: aircraft | A drop covers a strip of cells and multiplies their spread rate by a factor below 1, which fades over hours (retardant) or minutes (water) | Drop footprint, reduction factor, decay time, turnaround time | Flight hours, litres dropped |

**Realism the optimizer must not be able to exploit:**

- **Line breaching.** A fire crosses a line with a probability that rises with fire intensity
  relative to line width. A breach turns the line cell back into fuel.
- **No chance per time step.** A probability applied every step makes the outcome depend on
  dt and on how long a cell flames, which the fire model was built to avoid. Suppression
  randomness is therefore drawn once per cell at the start of the run, like the link
  thresholds: a line cell breaches the first time the breach probability of the highest
  intensity it has faced exceeds its own fixed random number. Every policy then meets the same
  luck in the same cell, so paired comparisons stay valid, and work that takes time
  (extinguishing, line building) is accumulated at a rate, never rolled for.
- **Crew safety limit.** Ground crews cannot work an edge whose flame length exceeds the NWCG
  limit for their resource type. The flame length is the local one (section 2.2), so a flank
  can be workable while the head is not.
- **Production rates depend on fuel and slope.** Line is built faster in grass than in timber
  and slower on steep ground (NWCG fireline production rate tables).
- **Travel.** A resource has a position and moves at a finite speed; work starts when it
  arrives at the work site, not when it is ordered.
- **Availability.** Each resource type has a maximum count, a mobilization cost paid once, and
  a shift limit (hours worked per day). "Unlimited but costed" would let the optimizer order
  a thousand crews for the first hour.
- **Daylight.** Aircraft do not fly at night.

Parameter values (build rates, drop sizes, decay times, limits, costs) come from
fire-management references and live in `configs/` with citations.

## 5. Strategy search

The code tunes a family of reactive rules rather than searching over raw line placements. A
rule family has 5-15 numbers, which a black-box optimizer can handle in a few thousand
ensemble runs.

**What a reactive policy is.** At each decision point (every 30-60 simulated minutes) the
policy looks at the fire's current state and decides what to deploy. The four rules of the
parameterized policy (`ReactivePolicy`, WP 4.4), each controlled by numbers the optimizer
tunes:

1. Dispatch k crews, m dozers and n aircraft at first report.
2. Build a parallel line at offset d on the flank nearest the assets.
3. When the head's spread rate passes a threshold, switch from direct attack to an indirect
   line L metres ahead, downwind.
4. Send aircraft to the head of the fire unless an asset is within a set distance, then protect
   the asset.

The policy reads only things a real incident commander would know: perimeter, head direction,
spread rate, intensity and distance to assets, as they were at the last decision point.

**Cost function.** The optimizer minimizes the expected total cost over an ensemble of runs:

```latex
J(\theta) = \mathbb{E}\left[\sum_{r} \left(m_r N_r(\theta) + c_r\,T_r(\theta)\right) + c_w V_w(\theta) + c_{rt} V_{rt}(\theta) + \sum_{k\,\in\,\text{burned}} v_k\right]
```

theta is the policy's parameter vector; N\_r is the number of resources of type r mobilized at
one-off cost m\_r; T\_r is hours worked at hourly cost c\_r; V\_w and V\_rt are litres of water
and retardant at unit costs c\_w and c\_rt; v\_k is the value of burned cell k.

**Tail risk.** An average can hide the rare runs where a town burns. The search is therefore
run as a two-objective study (mean cost and 90th-percentile cost) and the Pareto front is
reported, with the mean-cost optimum as the headline policy.

**Optimizer.** Optuna: it handles a mix of whole-number parameters (crew counts) and continuous
ones (offsets), tolerates noisy scores, supports multiple objectives, and can stop unpromising
candidates early.

**Baselines.** No suppression, direct attack only and indirect attack only. The optimized
policy has to beat all three for the result to mean anything.

**Guarding against a policy that only fits its training runs.** The optimizer sees a fixed set
of seeds and one scenario, so its winner can be tuned to that luck. The reported comparison
therefore uses:

- fresh seeds the optimizer never saw;
- held-out scenarios (different ignition point, wind direction and moisture);
- paired comparisons on common random numbers, with bootstrap confidence intervals on the
  cost difference against each baseline.

**Extension: reinforcement learning** (WP 6.3), using the same observation and action
interfaces.

## 6. Ensembles and their cost

**What an ensemble samples.** Each run draws (a) the kernel's link thresholds and (b) a
perturbation of the inputs: wind speed and direction, fuel moisture, a rate-of-spread
multiplier and, optionally, the ignition location, each from a distribution given in the
scenario's `ensemble` section. Without (b) the burn-probability map would be almost binary and
would understate how uncertain a forecast is.

**Random streams.** One `SeedSequence` per run spawns independent generators for spread,
input perturbation and suppression. Suppression can then never change the fire's draws, which
is what makes paired policy comparisons valid.

**How many runs.** Reported means carry a confidence interval; the run count is the smallest
that brings the interval's half-width under a stated fraction of the mean (50-200 runs is
typical), shown with a running-mean plot.

**Compute.** One run is cheap; the cost comes from multiplying runs per ensemble by candidate
strategies. Measured in WP 1.7 for the 10 km area (333 x 333 cells) and a 12-hour fire at
dt = 30 s (1,440 steps), 8-neighbour / 16-neighbour stencil, on an 8-thread laptop (details and
the profile in [performance.md](performance.md)):

| Workload | Runs | NumPy engine, 1 core | NumPy, 8 workers | Numba engine, 1 core | Numba, 8 workers |
| --- | --- | --- | --- | --- | --- |
| One scenario | 1 | 1.1 / 2.5 s | | 0.14 / 0.23 s | |
| One ensemble | 200 | 3.8 / 8.3 min | 1.3 / 2.5 min | 29 / 47 s | 20 / 25 s |
| One optimization | 300 candidates x 50 runs = 15,000 | 4.8 / 10.4 h | 1.4 / 3.0 h | 36 / 59 min | 7 / 12 min |

Starting a worker pool costs 9-17 s, so the optimizer (WP 4.5) must keep one pool for the whole
search instead of starting one per candidate.

**These numbers do not cover fire A.** The Chimney Fire burned 46,344 acres (about 18,750 ha)
in 24 days (13 August to 6 September 2016), which is almost twice the whole reference area. Its grid has about ten times the cells, and
even the early-growth window used for calibration is days, not 12 hours. Calibration cost is
therefore one to two orders of magnitude above the table. It is measured on the fire A grid
before the calibration study is sized (WP 3.1), and the same is done for runs with
suppression before the strategy search is sized (WP 4.5): the table has no suppression in it.

Runs are independent and split across CPU cores. The engine updates only cells on the burning
front; spread-rule output is cached and recomputed only when weather or suppression layers
change.

## 7. Validation

**Calibrate on fire A, test on fire B without retuning.** Fire A is the 2016 Chimney Fire, San
Luis Obispo County, CA: it has an MTBS perimeter, VIIRS hotspots showing daily growth,
pre-fire LANDFIRE fuels, and a published CA simulation to compare with (arXiv 2507.00761).
Fire B has the same data available and, like A, is a fire where surface fire dominated (see
*Open decisions*).

**What fire B can and cannot show.** The calibrated spread multiplier mostly corrects the bias
of the fuel models that dominate fire A; FARSITE's rate-of-spread adjustment factors are
specific to a fuel model for this reason (Finney 1998). A fire B in a different fuel type would therefore be expected to fail
"no retuning" for a reason known in advance, and a pass would be luck. The untouched test is
a fire in the same broad fuel group as A (shrub and grass) but a different place, wind
direction and season. A fire in a different fuel group is a useful second test only if it is
reported as a stress test of transfer across fuels, and a timber fire only once crown fire (WP
6.2) is in the model.

**The suppression confound.** Real fires are fought; the calibration runs are not. Calibrating
unsuppressed runs to a suppressed fire's final perimeter biases the spread rate downward. To
limit this:

- calibrate on the **growth over the first burning periods**, before containment lines held,
  using VIIRS detections and daily perimeters, not only on the final perimeter;
- report the final-perimeter overlap separately and say plainly that it includes suppression.

**Metrics:**

- Overlap between simulated and real burned area (Jaccard and Sorensen), using cells burned in
  at least half the runs.
- Arrival-time error, scored against an interval. A VIIRS detection does not say when the
  fire arrived, only that it had arrived by that overpass; Suomi NPP passes at about 01:30 and
  13:30 local time, so the arrival lies between the detecting overpass and the last earlier
  one that saw the place clear and without fire (or the ignition time). The error of a
  detection is zero when the simulated arrival falls inside that interval and the distance to
  the nearer end otherwise. A detection the simulated fire never reaches is scored against the
  end of the comparison window. A 375 m detection covers about 12 x 12 cells; the simulated
  arrival is the earliest among them. Treating the detection time as the arrival time would
  bias the calibrated spread rate low by several hours per detection.
- Ratio of simulated to real burned area, per day.

**Calibration.** Tune only a few parameters (an overall spread-rate multiplier, the ellipse
shape) against a weighted sum of overlap and arrival-time error. Report the objective surface
around the optimum, so it is visible whether the parameters are actually identified by the
data or trade off against each other.

**Sensitivity analysis.** A Morris screening over the main inputs (wind speed and direction,
fuel moistures, rate multiplier, kernel shape, stencil) shows which ones drive burned area and
therefore where data quality matters.

**Physics tests in the test suite** (WP 1.4):

- No wind, flat, uniform fuel: a circular burn whose front speed matches R.
- Steady wind: the analytic ellipse (head, flank and back distances).
- Halving dt: the same distribution of arrival times.
- A complete line: fire stops. Water cells never burn; urban cells burn more slowly than the
  surrogate fuel (WP 1.13).
- A slow-fuel strip delays the front by the predicted time.

## 8. Software quality

Beyond the rules in `CONTRIBUTING.md`:

- **Property-based tests** (Hypothesis) for the invariants: probabilities stay in [0, 1], a
  burned cell never re-ignites, burned area never decreases, a saved grid loads back equal.
- **Golden-run regression tests.** A few small scenarios with fixed seeds have their arrival
  time arrays committed; any change to them must be deliberate and explained in the PR.
- **Run provenance.** Every run directory records the full resolved config, the seed, the git
  commit, package versions and the data manifest hash, so any figure can be traced to what
  produced it.
- **Decision records.** Each design decision that was argued over gets a short record in
  `docs/decisions/` (context, options, choice, consequences), starting with the ignition
  kernel and the Rothermel implementation.
- **Coverage and performance in CI.** The coverage floor rises from 50 % to 85 % once Phase 1
  is merged; a benchmark job fails if a run on the reference grid slows by more than 20 %.
- **Figures regenerate.** Every figure in the report is produced by a committed script from
  committed configs and seeds.

## 9. Milestones

Undated; each is a set of conditions that are either met or not.

**M1 - simulator works.** Constant, Alexandridis and Rothermel rules pass every physics test in
`tests/test_physics.py` on synthetic landscapes; an ensemble and a GIF are reproducible from a
YAML file and a seed; the engine meets the performance target recorded in
`docs/performance.md`; a fire survives a period of zero spread rate, reports reaching the
edge of its grid, has front-normal intensity and can burn through a town (WPs 1.10-1.13).
Status: met. The Rothermel rule (WPs 2.1a-b) passes the physics tests: on flat uniform fuel
it gives exactly the rates of the elliptical rule the ellipse tests use, and simulated with
it the burn is round without wind and within 10 % of the analytic ellipse at a ratio of 1.5
(`tests/test_physics.py`). One cost is not in the benchmark: the rule takes 0.11 s per
evaluation on the 333 x 333 reference grid, and it is re-evaluated whenever the weather
changes, so a 12-hour run with hourly weather spends about 1.3 s in it, more than the Numba
engine's whole 0.5 s budget. WP 3.1 benchmarks with this rule on the fire A grid.

**M2 - model is validated.** Calibrated on fire A; run on fire B without retuning; overlap and
arrival-time errors reported for both, with the sensitivity analysis. No reported run reaches
the edge of its grid, and calibration recovers the same parameters at every head bearing.

**M3 - strategies are evaluated.** The optimized policy is compared with the three baselines on
unseen seeds and held-out scenarios, with confidence intervals and the mean / 90th-percentile
Pareto front, with spotting switched on.

## 10. Scope decisions

| Decision | Choice |
| --- | --- |
| Ignition kernel | Gamma-distributed link crossing time sampled once per link as a threshold; `spread.arrival_shape = 200`; link rates as the harmonic mean over the cells a link crosses; per-(stencil, shape) front-speed correction. |
| Rothermel implementation | Native vectorized NumPy (WP 2.1), cross-checked against `pyrothermel` (optional extra `rothermel`) and published BehavePlus values. |
| Default stencil | 16 neighbours, from the lattice study in WP 1.4: better than 8 at every length-to-breadth ratio, at about twice the run time. |
| Package name | `wildfire_sim` (CLI `wildfire-sim`). |
| Validation fires | A: 2016 Chimney Fire (calibration). B: to be chosen (test, no retuning). |
| Weather | One hourly scalar series per run. Hourly analysis where available, disaggregated gridMET otherwise. Spatially varying wind is out of scope. |
| Primary Python | 3.12. 3.13 tested in CI; 3.14 allowed to fail. |
| Suppression set | Hand crews, dozers, engines and aircraft. |
| Extensions (Phase 6) | Spotting, crown fire, reinforcement learning. |
| Out of scope | Non-US data; terrain-modified wind; burnout and backfire operations; a physical model of structure-to-structure fire spread (urban cells burn by a scaled surrogate, section 2.6). |
| File paths in scenario files | Relative paths resolve against the working directory; commands are run from the repository root. Parameter tables live in `configs/` and are read from there, so the package is used from a checkout, not from an installed wheel. |

## 11. Module map

```text
src/wildfire_sim/
  types.py            CellState, LandUse, Stencil (8/16)           [frozen, Phase 0]
  grid.py             Grid layers, Weather, WeatherSeries          [frozen, Phase 0]
  config.py           Pydantic schemas, YAML loaders               [frozen, Phase 0]
  spread/base.py      SpreadRule protocol, ignition kernel         [frozen, Phase 0]
  spread/__init__.py  rule registry (register, make_spread_rule)   [Phase 0]
  spread/constant.py                                               [WP 1.1]
  spread/alexandridis.py                                           [WP 1.2]
  spread/rothermel.py, spread/ellipse.py, spread/moisture.py       [WP 2.1]
  sim.py              time loop, event log, RunResult, provenance,  [WP 1.3, 1.10-1.14]
                      dormant sources, edge detection, front-normal
                      intensity, flame length
  urban.py            urban-cell parameters (surrogate fuel, factors) [WP 1.13]
  viz.py              maps, burn-probability plots, GIF            [WP 1.5]
  ensemble.py         parallel seeded runs, input perturbation     [WP 1.6]
  sim_fast.py         Numba front-only engine                      [WP 1.7]
  cli.py              validate (Phase 0) / run / ensemble / plot   [WP 1.8]
  data/synthetic.py   test landscapes                              [Phase 0]
  data/landfire.py, terrain.py, weather.py, osm.py, cache.py, validation.py  [WP 2.2-2.7]
  metrics.py          burned area, overlap, cost, intervals        [WP 2.8]
  breaks.py           breach of roads, streams and built lines     [WP 2.9]
  suppression/geometry.py, resources.py, tactics.py, policies.py  [WP 4.1-4.4]
  optimize.py         Optuna study                                 [WP 4.5]
  spotting.py, spread/crown.py                                     [WP 6.1, 6.2]
configs/              scenarios/*.yaml, costs.yaml, resources.yaml, alexandridis.yaml,
                      residence_time.yaml, holdover.yaml, urban.yaml, flame_length.yaml,
                      breach.yaml, kernel_correction.yaml, fuel_models.csv, calibrated.yaml
scripts/              kernel_front_speed.py, bench.py, download_data.py, calibrate.py,
                      sensitivity.py, run_experiments.py
docs/decisions/       one short record per design decision
tests/                offline, deterministic; fixtures/ holds committed .npz grids;
                      golden/ holds committed arrival-time arrays
```

Frozen modules are the contracts every other WP builds on. Changing them needs a PR that
updates all callers and explicit sign-off. Adding an optional config field with a default is
allowed.

A spread rule registered as `x` must live in `spread/x.py`; `make_spread_rule("x")` imports that
module on first use, so a scenario file can select a rule by name.

## 12. Work packages

Each row is self-contained: a model or teammate handed one row plus the frozen modules should
be able to complete it without further context. "Consumes" lists the interfaces the WP relies
on; "Acceptance" is what the PR's tests must show.

### Phase 0 - Foundations (complete)

| WP | Deliverable | Status |
| --- | --- | --- |
| 0.1 | Repo hygiene: LF line endings, rename to `wildfire_sim`, correct URLs, 3.12 primary, `AGENTS.md`, data dirs ignored | done |
| 0.2 | Dependency groups (`fast`, `geo`, `rothermel`, `opt`, `all`) | done |
| 0.3 | `types.py` (`CellState`, `LandUse`, `Stencil`, `STENCIL_8`, `STENCIL_16`), `grid.py` (`Grid`, `Weather`, `WeatherSeries`, save/load) | done |
| 0.4 | `spread/base.py`: `SpreadOutput`, `SpreadRule`, `ignition_probability`, `neighbour_ignition_probability`, `max_stable_dt`; registry in `spread/__init__.py` | done |
| 0.5 | `config.py`: `ScenarioConfig`, `CostConfig` with `CitedValue`, loaders; three scenario YAMLs; `configs/costs.yaml` placeholders | done |
| 0.6 | `data/synthetic.py` generators, `build_grid`, `tests/conftest.py`, committed fixture `tests/fixtures/flat_21x21.npz` | done |
| 0.7 | This page | done |
| 0.8 | First review fixes: gamma kernel and `spread.arrival_shape`; lazy rule import; `n_steps` rounding; ignition-cell bounds; `engine_usd_per_h`; finite-value grid validation; CLI error handling; `scripts/kernel_front_speed.py` | done |
| 0.9 | Second review: `link_rate_of_spread` (harmonic-mean link rates), `draw_crossing_thresholds` (per-link thresholds, common random numbers) | done |

### Phase 1 - Simulator core (complete: 1.1-1.9, see the third review in section 14; 1.10-1.14, added by the fourth review, see the fifth)

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 1.1 | `spread/constant.py` | 0.4 | `ConstantRule(ros: float)` registered as `"constant"`: same `ros` (m/s) in every direction for burnable cells, zero elsewhere, times `(1 - retardant)`; `intensity`/`flame_length` zero | Output validates; non-burnable cells zero; retardant 0.5 halves ros; `make_spread_rule("constant", ros=0.2)` works without importing the module first |
| 1.2 | `spread/alexandridis.py`, `configs/alexandridis.yaml`, `configs/scenarios/steady_wind.yaml` | 0.4 | Alexandridis et al. 2008 rule expressed as a rate: `ros_k = ros_base (1 + p_veg)(1 + p_den) p_w(theta_k) p_s(k)` where `ros_base` (m/s) replaces the paper's base probability `p_h` and is the calibration parameter; wind factor `p_w = exp(c1 V) exp(c2 V (cos(theta) - 1))` with `theta` the angle between spread direction k and the wind-to bearing; slope factor `p_s = exp(a * theta_s)` with `theta_s = atan((elev_neighbour - elev_cell) / d_k)` in degrees; fuel-model-to-(veg, density) mapping table and constants (paper values `a = 0.078`, `c1 = 0.045`, `c2 = 0.131`, verified against Table 4 of the paper) cited in the YAML; switch `steady_wind.yaml` to this rule | Downwind ros > upwind; upslope > downslope; constants loaded from YAML, not hard-coded; edge cells use zero slope toward missing neighbours |
| 1.3 | `sim.py`, `configs/residence_time.yaml` | 0.3-0.9, any rule | `run(config, seed, rule=None) -> RunResult` with `RunResult(grid, arrival_time, burned_mask, events, elapsed_s, provenance)`. Setup: spawn spread / perturbation / suppression generators from `SeedSequence(seed)`; `draw_crossing_thresholds` once. Per step: weather lookup; `rate_of_spread` (cached until weather or suppression layers change); `link_rate_of_spread`; divide by the front-speed correction; add `R_link dt / d` to the progress of each burning cell's links; ignite FUEL neighbours whose link progress reaches its threshold, with `arrival_time` interpolated inside the step and the leftover time applied to the new cell's links in the same step. A BURNING cell becomes BURNED when no link of it can still complete (no FUEL neighbour with link rate > 0), or its age exceeds an optional `max_source_age_s`; `burn_timer` holds the remaining flaming time `d / ros_head + residence_time(fuel_model)`. Knight moves (16-stencil) are blocked when both intermediate cells are unburnable. Only cells on the front are updated. `events` is a list of `(t, kind, row, col)`; ignition from `IgnitionConfig`, erroring if a cell is outside the grid or unburnable. Residence times are `CitedValue`s keyed by fuel model with a default. `provenance` holds resolved config, seed, git commit and package versions | Water/urban never burn; identical output for identical seed; fire ends when no cell is burning; `arrival_time` NaN exactly where never burned; burning set stays a thin front on `flat_no_wind`; a fast fire (`ros dt > dx`) advances more than one cell per step; a run with a no-op suppression hook equals a run without it bit-for-bit |
| 1.4 | `tests/test_physics.py`, `scripts/kernel_front_speed.py`, `configs/kernel_correction.yaml`, `docs/lattice_study.md` | 1.1, 1.2, 1.3 | Physics tests: (a) no wind, constant rule, 20 runs: diagonal extent / axis extent in [0.95, 1.05] and boundary-radius standard deviation / mean < 0.1 (an edge-counting isoperimetric quotient cannot be used: it is 0.62 for a perfect raster disc); (b) steady wind with an elliptical rule: head, flank and back distances each within 10 % of the analytic ellipse, for length-to-breadth 1.5, 3 and 6; (c) dt halving (15 s vs 7.5 s), 50 runs each: mean arrival time along a transect within 2 % and KS `p > 0.01`; (d) complete 3-cell-wide fireline ring: no cell outside ignites; (e) river scenario: no cell east of the river ignites, for every stencil; (f) front speed after correction: axis extent / (R T) in [0.97, 1.03], and > 2 uncorrected with `arrival_shape = 1`; (g) a strip of fuel four times slower delays the front by the predicted time within 10 %; (h) a wind shift half-way through a run gives the same mean arrival times for dt and dt/2. Lattice study: repeat (b) for each stencil (adding a 32-neighbour stencil to `types.py` if 16 fails), measure the front-speed correction per (stencil, shape), write both to the YAML and the doc, and set the default stencil | All tests pass with thresholds stated in docstrings; the study's figures regenerate from the script |
| 1.5 | `viz.py` | `RunResult`, `Grid` | `plot_state(grid, ax)`, `plot_arrival(result)`, `plot_burn_probability(stack)`, `animate(results_frames, path.gif)` using matplotlib + imageio; colour map keyed on `CellState` and safe for colour-blind readers | Smoke tests write PNG/GIF to `tmp_path`; no `plt.show()`; Agg backend |
| 1.6 | `ensemble.py`, `config.py` (optional `ensemble` section) | 1.3 | `run_ensemble(config, n_runs, base_seed, n_workers) -> EnsembleResult(burn_probability, arrival_times (n,H,W), burned_area_m2 (n,), seeds, perturbations)`; seeds from `np.random.SeedSequence(base_seed).spawn(n)`; per-run input perturbations (wind speed and direction, moistures, rate multiplier, optional ignition jitter) drawn from distributions in `EnsembleConfig`, default none; `multiprocessing` pool with the `spawn` start method (works on Windows); `running_mean` and `mean_ci(values, level)` helpers | n=4 parallel equals n=4 serial bit-for-bit; seeds and drawn perturbations recorded; with perturbations off, results equal plain seeded runs |
| 1.7 | `sim_fast.py`, `scripts/bench.py`, `docs/performance.md`, CI benchmark job | 1.3 | Profile the NumPy engine first and record where time goes. Numba engine updating only front cells; selected by `config.engine = "numba"` (optional field, default `"numpy"`); benchmark prints s/run on 333 x 333, 1440 steps | Same result as the NumPy engine for the same seed; measured numbers and the target in `docs/performance.md`; CI fails on a 20 % slowdown |
| 1.8 | `cli.py` | 0.5, 1.3, 1.5, 1.6 | `run SCENARIO --seed --out DIR` (writes `grid.npz`, `events.csv`, `summary.json`, `provenance.json`, `final.png`); `ensemble SCENARIO --n --out`; `plot DIR` | `CliRunner` tests for each command on `flat_no_wind` |
| 1.9 | `tests/test_properties.py`, `tests/golden/`, `pyproject.toml` (add `hypothesis` to the test group, raise `fail_under` to 85) | 1.3 | Hypothesis tests for the invariants in section 8; golden arrival-time arrays for `flat_no_wind`, `steady_wind`, `river_barrier` at a fixed seed with a documented regeneration command | Property tests pass; changing the kernel makes a golden test fail |
| 1.10 | `sim.py`, `sim_fast.py`, `configs/holdover.yaml` | 1.3, 1.7 | Dormant sources (section 2.3). A BURNING cell that has a FUEL neighbour on an unblocked link whose rate is zero now (zero rate in either cell under the current weather or retardant) stays BURNING as a dormant source after its flaming clock runs out: link progress is kept, `intensity` is 0, and it spreads again when the rate returns. It becomes BURNED once it has been dormant for longer than its holdover time, a `CitedValue` keyed by fuel model with a default (same schema as the residence times). Links blocked by barrier geometry do not count. Both engines. The run still ends when no cell is BURNING | On `flat_no_wind`, a hook that sets retardant to 1 everywhere from 1200 s to 1800 s gives the unstalled run's arrival times shifted by 600 s for every cell first reached after 1200 s (as built: the run ends at 1380 s with 195 of 4,087 cells burned); a stall longer than the holdover ends the fire; a weather series with zero rate overnight resumes the next morning; both engines agree; golden runs unchanged |
| 1.11 | `sim.py`, `sim_fast.py`, `ensemble.py`, `cli.py` | 1.3, 1.6, 1.8 | Edge detection (section 2). `RunResult.provenance["reached_edge_s"]` is the earliest arrival time at a cell within `stencil.max_offset` cells of the grid edge, or None, with an `"edge"` event; `EnsembleResult` records it per run; `summary.json` and the CLI output carry it and warn when it is set | `flat_no_wind` run for 7500 s reports the time it reached the edge (7,403 s); the shipped 5400 s run reports None; arrival times bit-identical to before. (Restated in the fifth review: this row asked for 7200 s, by which time the fire has not reached the edge band with the current defaults) |
| 1.12 | `sim.py`, `sim_fast.py`, `spread/base.py` (optional field, signed off) | 1.3, 1.7 | Front-normal intensity (section 2.2). `SpreadOutput` gains an optional `heat_per_area` (`(H, W)`, kJ/m^2, default None). When a cell ignites, the engine estimates the front-normal rate `R_n = 1 / abs(grad T)` from a least-squares plane through the arrival times of the cell and its already-burned stencil neighbours (falling back to the igniting link's rate when fewer than two non-collinear neighbours have burned) and sets `grid.intensity = min(heat_per_area * R_n, head intensity)`; with `heat_per_area` None the intensity is the rule's head value scaled by `R_n / max_k ros`, capped at the head value. Initial ignition cells get the head value. A helper returns flame length from intensity (Byram 1959) for WP 4.3 | Elliptical test rule with a constant `heat_per_area`, length-to-breadth 1.5 and 3, head on an axis and at 22.5 degrees: mean intensity over head, flank (widest point) and back cells against `I_head`, `I_head sqrt((1 - e) / (1 + e))` and `I_head (1 - e) / (1 + e)`, within tolerances set from the measurement and stated in the docstrings; a round fire has `I_head` everywhere within the same tolerance; arrival times bit-identical to before. (As built: 3 % on a round fire, 20 % at a ratio of 1.5; the ratio-3 flank and off-axis head are a strict expected failure until WP 2.1d, see the fifth review) |
| 1.13 | `types.py`, `grid.py` (both frozen, signed off), `sim.py`, `sim_fast.py`, `data/synthetic.py`, `configs/urban.yaml` | 1.3, 1.7, 1.10 | Burnable urban cells (section 2.6). `LandUse.URBAN.burnable` becomes True and `Grid.burnable_mask` follows; an urban cell carries the surrogate fuel model named in `configs/urban.yaml`. After the spread rule runs, the engine multiplies the rates of urban cells by `ros_factor` (below 1) and their residence time by `residence_factor` (above 1), so every rule is scaled the same way; all three are `CitedValue`s marked as team assumptions. `with_town` writes the surrogate fuel model into its town cells. Roads and water are unchanged | Fire crosses a town on `with_town` and every town cell it reaches is BURNED; the front takes `1 / ros_factor` times as long to cross a strip of urban cells as the same strip of the surrogate fuel, within the tolerance of physics test (g); urban cells flame `residence_factor` times as long; with `ros_factor` 0 the town holds as before; golden runs without urban cells unchanged, the others regenerated and explained in the PR |
| 1.14 | `config.py` (optional field, signed off), `sim.py` | 0.5, 1.3 | Wall clock. `ScenarioConfig.start_time_utc` (timezone-aware datetime, default None) is the instant of simulated time 0; `ScenarioConfig.time_at(t_s)` returns the datetime of a simulated time, or None without a start; the start is written to `provenance`. Consumers: WP 2.4 aligns the hourly weather series to it, WP 2.7 converts overpass times to simulated seconds with it, WP 4.2 takes aircraft daylight from it and the grid's location (WP 2.3 writes `centroid_lon` and `centroid_lat` to `Grid.metadata`) | A scenario without the field loads and runs bit-identically; a datetime without a timezone is rejected; `time_at(3600)` is one hour after the start; the start appears in `provenance.json` |

The thresholds in WP 1.4 are targets set before the engine exists. If the lattice study shows
that no stencil can meet one, the limit is written up in `docs/lattice_study.md` and the
threshold changed there and in the test together, never loosened silently.

### Phase 2 - Rothermel and real data

Status: 2.1a complete (branch `wp-2.1a-rothermel`; its four interpretation decisions accepted by the team, section 13): `spread/rothermel.py` computes head rate,
intensity, flame length, heat per unit area and residence time per cell, and exposes the wind
and slope factors separately (`surface_fire`, `head_fire`, `FuelBed`) for WP 2.1b to combine
as vectors. `configs/fuel_models.csv` holds the 40 Scott & Burgan and the 13 Anderson models.
`configs/residence_time.yaml` now holds Anderson's (1969) values for all 53 models. Arrival
times are unchanged (golden runs pass); cells stop flaming sooner. Rothermel's wind limit is an
option, off by default (section 13).

2.1b built (branch `wp-2.1b-rothermel-rule`): `RothermelRule`, registered as `"rothermel"` in
`spread/rothermel.py`, with the wind reduction, wind-slope vector sum and ellipse in
`spread/ellipse.py` and its parameters in `configs/rothermel.yaml`. Against the BehavePlus
engine it agrees to better than 1e-7 in head rate, head direction and the rate in eight
directions (10 m wind, open and sheltered fuel, wind across and against the slope). The
length-to-breadth ratio is capped at 8 as in BehavePlus. Canopy height and crown ratio, needed
only for sheltered fuel, are uniform placeholders until real layers exist (section 13).

2.1c built (branch `wp-2.1c-fuel-moisture`): `spread/moisture.py` computes 1-h and 10-h dead
fuel moisture from temperature, relative humidity and an optional cloudiness code, per hour
(`apply_hourly` on a `WeatherSeries`), with its parameters in `configs/fuel_moisture.yaml`.

2.1d built (branch `wp-2.1d-lattice-fit`): lattice-fitted rates (`spread/ellipse.py`:
`fit_rates`, `LatticeFitTable`; tables `configs/lattice_fit_8.csv`, `configs/lattice_fit_16.csv`
written by `scripts/lattice_study.py --fit-tables`), used by the `"rothermel"` rule
(`lattice_rates: fitted` in `configs/rothermel.yaml`), and a cap on the length-to-breadth ratio
at 2.1 (16 neighbours) and 1.4 (8). Head and flank are within 10 % of the analytic ellipse at
head bearings 0, 13.3, 22.5 and 45 degrees up to the cap (`tests/test_physics.py`); a round fire
keeps its rates exactly; constant and Alexandridis runs are unchanged (golden runs pass).
`provenance` and `summary.json` carry `lb_capped_cell_hours` and `wildfire-sim run` warns. Like
WP 2.1a, it also touches files outside its list, because the plan's deliverable needs them:
`sim.py` (the cell-hour count), `io.py` and `cli.py` (summary and warning), `configs/`, and
the tests of WP 1.12 (`tests/test_intensity.py`, re-measured) and of the CLI. The remaining WPs
have not started.

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 2.1a | `spread/rothermel.py`, `configs/fuel_models.csv` | 0.4 | Native Rothermel (1972) surface spread with Albini (1976) revisions, vectorized over the grid: reaction intensity, propagating flux, wind and slope factors, moisture damping, live fuel extinction moisture, dynamic herbaceous transfer for the Scott & Burgan dynamic models. Fuel parameters for all 40 models (loads, SAV ratios, depth, extinction moisture, heat content) in the CSV with the source cited. Outputs head ros (m/s), Byram intensity (kW/m), flame length (m, Byram 1959), heat per unit area, and residence time `t_r = 384 / sigma` min (Anderson 1969; Andrews 2018), which replaces the placeholder values in `configs/residence_time.yaml`. The fuel-bed part is evaluated once per fuel model present (moisture is uniform over the grid) and mapped back to cells; the wind and slope factors, which vary almost cell by cell, are computed per cell | Head ros, intensity and flame length for at least GR1, GR4, SH2, SH5, TU1, TL3 at three moisture/wind/slope cases within 2 % of BehavePlus values recorded in the test with their source (values produced by running the BehavePlus engine, `behave` via `pyrothermel`, are accepted as that source); zero ros at extinction moisture; agrees with `pyrothermel` within 2 % where installed (`pytest.importorskip`) |
| 2.1b | `spread/ellipse.py`, `spread/rothermel.py` | 2.1a | Midflame wind: the 10 m wind divided by 1.15 to give the 20-ft wind (Turner and Lawson 1978; the factor in `configs/` with its citation), then times the wind adjustment factor (sheltered and unsheltered, Andrews 2012); vector combination of wind and slope into direction of maximum spread and effective wind; length-to-breadth and directional rates from the formulas in section 2.2 (U capped at a documented value); `RothermelRule` registered as `"rothermel"` in `spread/rothermel.py` (the registry finds a rule named `x` only in `spread/x.py`), with the ellipse helpers in `spread/ellipse.py`, returning `(K, H, W)` rates times `(1 - retardant)` | A 10 m wind of 11.5 m/s gives the midflame wind of a 20-ft wind of 10 m/s; upslope with no wind spreads uphill; cross-slope wind gives a direction between wind and slope; back rate equals `R_head (1 - e) / (1 + e)`; physics tests 1.4 pass with this rule |
| 2.1c | `spread/moisture.py` | 0.3 | One-hour dead fuel moisture from temperature and relative humidity by an equilibrium moisture content model (source cited), applied hourly; documented offsets for 10-h | Reproduces the source's tabulated values; moisture rises overnight and falls in the afternoon for a synthetic diurnal series |
| 2.1d | `spread/ellipse.py`, `scripts/lattice_study.py`, `docs/lattice_study.md`, `tests/test_physics.py` | 2.1b, 1.4 | Elongated fires on the lattice (section 2.4). (1) Lattice-fitted rates: for a given length-to-breadth ratio and head bearing, per-direction rates chosen to minimize the largest radial error between the lattice polygon (convex hull of the rate-scaled stencil directions) and the analytic ellipse, replacing rates sampled from the ellipse; selected by a rule parameter whose default the study sets. (2) The lattice study repeated with them, at head bearings 0, 13.3 (midway between adjacent 16-neighbour directions), 22.5 and 45 degrees. (3) A cap `lb_max` at the largest ratio for which head and flank stay within 10 % at every bearing, applied in the ellipse rule; `provenance` counts the cell-hours in which it applied and the CLI warns | Head and flank within 10 % of the analytic ellipse at every tested bearing for every ratio up to `lb_max`, in `tests/test_physics.py`; the strict expected-failure record updated; if the fitted rates gain less than the study's stated margin over sampled rates, the doc says so and only the cap ships; with a constant-rate rule, results bit-identical to before; the intensity bounds of `tests/test_intensity.py` re-measured with the fitted rates and its ratio-3 expected failure removed or restated (WP 1.12) |
| 2.2 | `data/landfire.py` | 0.3 | Fetch LANDFIRE fuel model (FBFM40), canopy cover, height, base height and bulk density for a lat/lon bbox and a **named LANDFIRE version** via the LANDFIRE Product Service; write GeoTIFFs under `data/cache/<bbox-hash>/`; map non-burnable codes (91-99) to `LandUse`; urban cells get the surrogate fuel model of `configs/urban.yaml` (WP 1.13), without which they cannot burn | Request builder unit-tested with mocked HTTP; refuses a version newer than the fire date; live test marked `network` |
| 2.3 | `data/terrain.py` | 0.3 | 3DEP DEM via `py3dep` for the bbox; reproject/resample onto the LANDFIRE grid with rasterio; slope (rise/run) and aspect (downhill bearing) from `np.gradient`; CRS, transform and the grid centroid (`centroid_lon`, `centroid_lat`, degrees) written to `Grid.metadata` | Slope and aspect of `synthetic.slope_ramp` elevation recovered to 1e-3 / 0.5 deg |
| 2.4 | `data/weather.py` | 0.3, 2.1c | Hourly wind, temperature and humidity for the bbox centroid and date range from an hourly analysis where available, else gridMET daily disaggregated with a documented diurnal profile; 100-h moisture from gridMET (`fm100`; gridMET has no 1-h or 10-h moisture, and 1000-h is not used); 10-h from the 1-h value by the offset of WP 2.1c; live fuel moisture from the National Fuel Moisture Database with a seasonal default; output is a `WeatherConfig` with one point per hour | Mocked datasets convert to a valid `WeatherConfig`; disaggregated daily data preserves the daily mean; live test marked `network` |
| 2.5 | `data/osm.py` | 0.3 | `osmnx` roads, water, buildings for the bbox; apply the barrier rule of section 3 (wide features become unburnable cells; narrow ones become breaks, their width in metres stored in a new optional `Grid` layer `break_width_m`, signed off); `asset_value = buildings per cell * usd_per_building` from `costs.yaml` | Known polygon rasterizes to the expected cells; a 6 m road creates no unburnable cell and a line of cells with `break_width_m = 6` |
| 2.6 | `scripts/download_data.py`, `data/cache.py`, `docs/data.md`, weekly `network` CI job | 2.2-2.5 | One command builds `data/cache/<region>/grid.npz`, `weather.yaml` and `manifest.json` (sources, versions, dates, SHA-256) for a named region; regions defined for fires A and B, each as the bounding box of the fire's observed final perimeter plus a buffer of at least 3 km on every side (so the grid size follows the fire, section 2); writes a 5 km x 5 km crop of each to `tests/fixtures/` (committed); `docs/data.md` records access method, licence and attribution per source | Script runs end to end; cached grid validates; no cell of the observed perimeter lies within the buffer of the grid edge; fixtures under 2 MB each; rebuilding from cache reproduces the manifest hashes |
| 2.7 | `data/validation.py` | 0.3 | MTBS perimeter, VIIRS detections and daily perimeters where published, for fires A and B; ignition point and time from the incident record with citation; `observed_burn_mask(grid, until=None) -> bool array`, `detections(grid) -> list[(t_clear, t_detect, row, col)]`, where `t_detect` is the detecting overpass and `t_clear` the last earlier overpass that viewed the place without cloud and without fire, or the ignition time (section 7); the overpass times of the satellite come from the product's own records, not from a nominal schedule | Mocked inputs rasterize correctly; a place detected on the third overpass after a cloudy second one gets `t_clear` from the first; live test marked `network` |
| 2.9 | `breaks.py`, `sim.py`, `sim_fast.py`, `configs/breach.yaml` | 1.12, 2.5 | Break crossing (section 3). A cell with `break_width_m > 0` or a constructed line acts as a barrier until it is breached. One uniform number `u_c` per cell is drawn at setup from a fourth child of the run's `SeedSequence` (`spawn(4)`; the first three children are unchanged, so existing runs keep their draws). A break is breached the first time `p_b(I_peak, width) >= u_c`, with `I_peak` the highest `grid.intensity` any adjacent flaming cell has had; it then burns as ordinary fuel. `p_b` is Wilson's (1988) logistic model with the coefficients of section 3 as `CitedValue`s in `configs/breach.yaml` (intensity converted from kW/m to MW/m; tree presence from the canopy cover of the cell and its neighbours), together with the fitted range of width and intensity and a minimum intensity (placeholder 0.18 MW/m, section 13) below which `p_b` is 0. WP 4.3 calls the same function for the lines crews build | Breach probability rises with intensity and falls with width; reproduces the paper's 29 %, 55 % and 88 % for a 3 m break without trees at 2, 5 and 10 MW/m and 72 % with trees at 2 MW/m; below the minimum intensity no break is breached, so a rule with zero intensity never breaches; a grid without breaks is bit-identical to before; the set of breached cells is the same for dt and dt / 2; two runs with one seed and different hooks see the same `u_c` |
| 2.8 | `metrics.py` | `RunResult`, `EnsembleResult` | `burned_area_ha`, `jaccard(sim_mask, obs_mask)`, `sorensen`, `arrival_interval_error(arrival, detections, footprint_m, window_end_s)` (the interval score of section 7: zero inside `(t_clear, t_detect]`, distance to the nearer end outside, earliest simulated arrival within the detection footprint, never-reached detections scored against the window end), `area_ratio`, `suppression_cost(fleet, costs)`, `asset_loss(grid)` (the asset value of every burned cell, urban cells included), `bootstrap_ci(values, level)`, `paired_difference_ci(a, b, level)` | Hand-computed cases; intervals cover a known mean at the stated rate in a simulation test |

### Phase 3 - Calibration and validation

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 3.1 | `scripts/calibrate.py`, `configs/calibrated.yaml`, `docs/performance.md` | 2.1 (a-d), 1.10, 1.11, 2.6-2.9, optuna | First, `scripts/bench.py` on the fire A grid over the early-growth window, recorded in `docs/performance.md`; trial and seed counts are chosen from it. Then an Optuna study on fire A over the spread-rate multiplier in [0.3, 3] and ellipse `lb_scale` in [0.5, 2] (and `ros_base`, `c2` for the Alexandridis rule), minimizing a weighted sum of (1 - Jaccard at the end of the early-growth window) and normalized arrival-interval error (section 7), averaged over fixed seeds; writes best parameters and the objective evaluated on a grid around them | Runs on the committed crop in CI with 3 trials x 2 seeds (marked `slow`); on a synthetic "observed" fire generated with known parameters, recovers them within 10 %, and the recovered values agree within 10 % when that fire's wind blows along a grid axis, at 13.3, at 22.5 and at 45 degrees to it; a trial in which any run reaches the grid edge is rejected, not scored |
| 3.2 | `scripts/sensitivity.py`, `docs/sensitivity.md` | 1.6, 2.1 | Morris screening of burned area and head arrival time over wind speed, wind direction, 1-h and live moisture, rate multiplier, kernel shape and stencil | Ranks a deliberately inert parameter last; figures regenerate from the script |
| 3.3 | `docs/validation.md`, `notebooks/validation.ipynb` | 3.1, 3.2 | Fire A (calibrated) and fire B (not retuned): burn-probability map vs. observed perimeter, arrival-time scatter, metrics table with intervals, objective surface, discussion of the suppression confound; Rothermel vs. Alexandridis; comparison with the published Chimney Fire simulation | Figures regenerate from a script |

### Phase 4 - Suppression and strategy search

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 4.1 | `suppression/geometry.py` | 0.3 | `perimeter_cells(burned_mask)`, `rasterize_polyline(points, width_cells, shape)` producing **4-connected** lines (no diagonal gaps), `offset_perimeter(burned_mask, offset_m, dx)` via `scipy.ndimage.distance_transform_edt`, `head_direction(arrival_time, window_s)` from recent ignition centroid drift, `distance_to_assets(grid)`, `travel_time(grid, origin, speed)` over passable cells | Hand-drawn shapes; offset of a filled disc is a ring at the right radius; a diagonal line has no corner-only contacts |
| 4.2 | `suppression/resources.py`, `config.py` (optional fields on `ResourceConfig`: `max_count`, `mobilization_cost_usd`, `shift_hours`, `travel_speed_m_s`, `night_ops`) | 0.5, 4.1 | `Resource` dataclass from `ResourceConfig` with position and status; `Fleet` tracks dispatch time, travel, arrival, hours worked against the shift limit, litres dropped; cost accrual per step using the `CostConfig` rate for the type when `hourly_cost_usd` is 0, plus mobilization cost once; orders beyond `max_count` are refused | Hours and cost match hand calculation; a resource does no work before it arrives or beyond its shift; aircraft idle at night |
| 4.3 | `suppression/tactics.py`, `configs/production_rates.yaml` | 4.1, 4.2, 1.3 hooks, 1.12, 2.9 | `IndirectLine(path)`, `ParallelLine(offset, flank)`, `DirectAttack(crews)`, `AirDrop(footprint, factor, decay_s)`; line segments become `fireline = True` at a build rate looked up by resource type, fuel model and slope class (then `grid.apply_unburnable()`); breach (section 4, no chance per step) is the break model of WP 2.9 applied to the line's width, with the same per-cell `u_c`, so a built line and a road of equal width behave alike and a line can end on a road; direct attack is accumulated work: a crew extinguishes fire edge at a production rate (m/h by resource type, fuel model and slope class) and a burning cell becomes BURNED when the work done on it reaches `dx`; crews skip cells whose local flame length (from `grid.intensity`, WP 1.12) exceeds `max_flame_length_m`; any other suppression randomness is likewise drawn per cell or per resource at setup; `sim.py` gains a `tactics` hook called each step | Complete wide line stops fire; breach probability rises with intensity; crews refuse an unsafe head but work a flank of the same fire; line is built slower in timber and on steep slopes; retardant decays to zero; the fire's link thresholds are identical with and without suppression; two policies see the same `u_c` field for one seed; with a line finished before the fire arrives, the set of breached cells is the same for dt and dt / 2 |
| 4.4 | `suppression/policies.py` | 4.1-4.3 | `Observation` dataclass (perimeter length, head bearing, head ros, max intensity, distance to nearest asset, burned area, elapsed time, fleet status) built at each decision point; `Policy` protocol `act(obs, fleet, t) -> list[Action]`; baselines `NoSuppression`, `DirectOnly`, `IndirectOnly`; `ReactivePolicy(params)` implementing the four rules in section 5 | Each baseline runs end to end on `with_town`; a policy cannot read the grid directly |
| 4.5 | `optimize.py` | 4.4, 1.6, 2.8, optuna | `optimize_policy(config, n_trials, runs_per_trial, seeds)`: two objectives (mean total cost, 90th-percentile cost); common random numbers; pruning after the first 5 runs for the single-objective mode; study saved to SQLite; `evaluate_policy(params, scenarios, seeds)` for held-out evaluation with paired intervals against baselines | Toy problem with a known optimum is recovered; held-out evaluation uses no seed seen in the study |
| 4.6 | `configs/costs.yaml`, `configs/resources.yaml`, `configs/production_rates.yaml` | 0.5 | Replace every PLACEHOLDER with a cited value (NWCG, USFS cost guides); resource build rates, limits, mobilization costs and safety limits | Validators pass; no `PLACEHOLDER` string remains |
| 4.7 | `scripts/run_experiments.py`, `configs/scenarios/experiments/`, `docs/results.md` | everything, including 6.1 | Optimized policy vs. three baselines on the fire A grid with a synthetic town, with spotting on, on unseen seeds and on held-out scenarios; the headline comparison repeated with spotting off to show what it changes; cost distributions with intervals, Pareto front, burn-probability maps, convergence plot of running mean vs. run count; an ablation showing what each realism constraint (breaching, safety limit, availability) changes | Figures regenerate from committed configs and seeds |

Intensity-dependent pieces (breaching, crew safety) read the front-normal `grid.intensity` of
WP 1.12 and the flame length derived from it, which need the Rothermel rule's head intensity
and heat per unit area. The constant and Alexandridis rules return zeros, so suppression
experiments run on Rothermel. The run-time estimate of section 6 has no suppression in it;
WP 4.5 measures a run with tactics before fixing the number of trials.

### Phase 5 - Deliverables

README example run with a GIF, `docs/` page per module, decision records in `docs/decisions/`,
report assembled from `docs/validation.md`, `docs/sensitivity.md` and `docs/results.md` with a
limitations section, final changelog, tagged release.

### Phase 6 - Extensions

| WP | Files | Consumes | Deliverable | Acceptance |
| --- | --- | --- | --- | --- |
| 6.1 | `spotting.py`, `configs/spotting.yaml` | 1.3, 2.1 | Ember spotting: each flaming cell above an intensity threshold launches embers at a rate; landing distance drawn from a wind-aligned lognormal distribution with cited parameters; an ember ignites a FUEL cell with a moisture-dependent probability; uses its own random stream | Off by default and then bit-identical to the base model; with it on, fire crosses a river or line that stops the base model at a rate that rises with wind |
| 6.2 | `spread/crown.py` | 2.1, 2.2 | Crown fire: Van Wagner (1977) initiation from surface intensity, canopy base height and foliar moisture; active crown rate of spread from Rothermel (1991); crown fraction burned blends surface and crown rates | Published worked examples reproduced; no crowning where canopy cover is zero |
| 6.3 | `suppression/rl.py` | 4.4 | Gymnasium environment wrapping the simulator with the `Observation`/`Action` interfaces; one baseline agent trained and compared with the optimized reactive policy | Environment passes the Gymnasium checker; comparison uses the held-out protocol of WP 4.5 |

## 13. Dependencies and open decisions

**Order of work.** No dates; a WP can start when the WPs it consumes are merged.

- 1.1, 1.2, 1.5, 2.1a, 2.1c, 2.2-2.5, 4.1 depend only on Phase 0 and can start now, in parallel.
- 1.3 follows 1.1. Then 1.4, 1.6, 1.8, 1.9 and 2.1b; 1.7 after 1.4.
- 1.10, 1.11, 1.12 and 1.13 follow 1.7. All four edit both engines, so they are merged one at
  a time, in that order. 1.14 depends only on Phase 0; 2.4, 2.7 and 4.2 consume it.
- 2.1d follows 2.1b.
- 2.6 and 2.7 follow 2.2-2.5; 2.8 follows 1.6; 2.9 follows 1.12 and 2.5.
- Phase 3 follows M1 (which now includes 1.10-1.13), 2.1d and 2.6-2.9. If fire B burned in
  timber, 6.2 comes before Phase 3 as well.
- 4.2 follows 4.1; 4.3 follows 4.2, 1.3, 1.12 and 2.9; then 4.4, 4.5; 4.7 follows Phase 3 and 6.1.
- 6.1 and 6.2 follow M1; 6.3 follows 4.5. 6.1 is required by 4.7, so it is an extension in
  name only.

A handoff prompt for one WP contains: the WP row above, the source of the frozen modules it
consumes (pasted, not linked), the rules from `CONTRIBUTING.md`, and the command
`just check`.

**Open decisions.**

- [ ] **Which fire is fire B.** The criteria are settled (below). Candidates, all wind-driven
      brush fires in southern California inside the size range, with sizes and dates from
      the incident records:

      | Fire | Year | Size | Active | Notes |
      | --- | --- | --- | --- | --- |
      | Springs (Ventura Co.) | 2013 | 24,251 acres (9,814 ha) | 2-6 May | Chaparral; long-range spotting reported; predates the HRRR archive |
      | Canyon 2 (Orange Co.) | 2017 | 9,217 acres (3,730 ha) | 9-17 Oct | About 7,500 acres on the first day; 25 structures destroyed |
      | Lilac (San Diego Co.) | 2017 | 4,100 acres (1,659 ha) | 7-16 Dec | Reached nearly its full size within hours, so at most one or two overpasses saw it grow; 157 structures destroyed |
      | Maria (Ventura Co.) | 2019 | 9,999 acres (4,046 ha) | 31 Oct-6 Nov | Started at night on South Mountain; burned into orchards and farmland |
      | Silverado (Orange Co.) | 2020 | 13,390 acres (5,419 ha) | 26 Oct-7 Nov | Gusts to 80 mph; aircraft grounded at first, so little early air attack; slightly above the size range |

      To check before choosing: the wind during the main run against the length-to-breadth
      cap of WP 2.1d (2.1: an effective midflame wind of about 3.4 m/s, a 10 m wind of about
      10.7 m/s over open grass; a Santa Ana fire may sit above it for most of its growth); how many
      VIIRS overpasses saw the fire growing; that an MTBS perimeter and a LANDFIRE version
      older than the fire exist.
- [ ] Refresh `scripts/bench_baseline.json` from a CI run now that WPs 1.10-1.13 are in the
      engines (their measured cost is in `docs/performance.md`); the CI benchmark check stays
      non-blocking until then.
- [ ] Minimum set of values still to be sourced, each with a flagged placeholder so no work
      package is blocked: holdover times per fuel model (WP 1.10; placeholder 12 h for every
      fuel, enough to carry a fire through one night); the urban surrogate (WP 1.13;
      placeholders: fuel model TL3, `ros_factor` 0.5, `residence_factor` 30; the factor was
      chosen to make a town flame for about an hour on a 120 s TL3 placeholder, and with the
      cited 15 s it gives 7.5 min); the minimum
      intensity of the breach model (WP 2.9; placeholder 0.18 MW/m). All five are in the
      sensitivity analysis (WP 3.2).
- [ ] Whether the HRRR archive has the analysis hours for both fires' dates (WP 2.4). The
      RTMA archive on AWS starts in 2019, so it cannot serve fire A; the HRRR archive starts
      in 2014 and has gaps. If either fire is not covered, both use disaggregated gridMET.
- [ ] Cost figures with sources: hourly rates and mobilization costs per resource type; water
      and retardant per litre. Left as placeholders until WP 4.6.
- [ ] **Rothermel's wind limit** (WP 2.1b). The original limit caps the effective midflame
      wind at `0.9 I_R` (ft/min, `I_R` in Btu/ft^2/min; Rothermel 1972). Andrews et al. (2013)
      found it too restrictive. The BehavePlus engine (`behave`) applies it by default, and it
      binds in ordinary conditions: GR1 at moderate moisture, with a 3 m/s midflame wind up a
      0.6 slope, spreads at 0.025 m/s with the limit and 0.090 m/s without. The rule reads it
      from `configs/rothermel.yaml` (`apply_wind_limit`, false until the team decides; a
      scenario can override it in `spread.params`). The choice enters the sensitivity analysis
      (WP 3.2).
- [ ] **Canopy height and crown ratio for the wind adjustment factor** (WP 2.1b). Sheltered
      fuel (crown fill `canopy_cover * crown_ratio / 3` above 0.05) needs both (Andrews 2012),
      and `Grid` has no layers for them. The rule uses one value each for the whole grid,
      placeholders 15 m and 0.4 in `configs/rothermel.yaml`. WP 2.2 fetches LANDFIRE canopy
      height and base height (crown ratio = (height - base) / height); carrying them per cell
      needs optional `Grid` layers, a change to a frozen module that needs sign-off, or the
      placeholders stay.
- [ ] **Front propagation above a ratio of 2.1** (WP 2.1d). The decision after the fourth review
      said a different front-propagation method is reconsidered only if the fitted rates do
      not reach a ratio of 3 at every bearing. They reach 2.1 on 16 neighbours
      ([lattice_study.md](lattice_study.md) section 7), so the condition is met. Options:
      (a) accept the cap: fires in winds above it are drawn rounder (head right, flanks too
      wide), counted in `lb_capped_cell_hours` and flagged; (b) a method without the lattice
      limit, such as a fast-marching or level-set solver of the arrival time with the elliptical
      rate as a continuous function of direction, at the cost of rewriting the engine and its
      kernel noise; (c) larger stencils, which the decision rules out (a 48-neighbour stencil
      reaches about 2.75 with sampled rates). Until decided, (a) is what ships. Fire A's and
      fire B's winds against the cap should inform it.
- [ ] **Inputs and time lag of the hourly fuel moisture** (WP 2.1c, for WP 2.4). (1) The
      sunshine correction needs a cloudiness code (0 clear to 3 overcast) for each daylight
      hour; WP 2.4 must take it from the hourly source's cloud cover or pass none, which treats
      every hour as shaded and makes the fuel moister by day. (2) The 10-h moisture is the
      NFDRS equilibrium rule (1.28 EMC) applied every hour, so it has no 10-hour lag; a lag
      (`m(t + dt) = m_eq + (m(t) - m_eq) exp(-dt / 10 h)`, Fosberg and Deeming 1971) would be
      more faithful but is not in the plan.

Resolved: study region and fire A (Chimney Fire); ignition kernel; default stencil 16 and
`arrival_shape` 200 (after the lattice study); Rothermel implemented
natively with `pyrothermel` as a cross-check; package name; frozen-module changes signed
off: optional `SpreadOutput.heat_per_area` (WP 1.12), `LandUse.URBAN` burnable (WP 1.13),
optional `Grid` layer `break_width_m` (WP 2.5), optional `ScenarioConfig.start_time_utc`
(WP 1.14); breach model coefficients (Wilson 1988, verified against the paper).

Decided after the fourth review:

- **Elongated fires on the lattice:** lattice-fitted directional rates with a cap on the
  length-to-breadth ratio (WP 2.1d, before Phase 3). A different front-propagation method is
  reconsidered only if the fitted rates do not reach a ratio of 3 at every bearing. Larger
  stencils are not added. Done in WP 2.1d: the fitted rates reach 2.1 (16 neighbours), not
  3, so the method question is open again (open decisions above).
- **Fire B criteria:** a US fire in the same broad fuel group as the Chimney Fire (shrub and
  grass, surface fire dominant) but a different place and wind direction, wind-driven, with
  an MTBS perimeter, VIIRS coverage and little early suppression, 1,000-10,000 ha. A fire in
  another fuel group may be added as a stress test, reported as such; a timber fire needs WP
  6.2 first.
- **Breach floor:** a minimum intensity. Below it a break always holds; at or above it
  Wilson's probability applies unchanged, so his fitted numbers are kept (WP 2.9). The floor
  is 0.18 MW/m, a placeholder: it is the lower bound implied by the ranges the paper states
  (slowest spread 0.1 m/s, lightest fuel 1 t/ha, `I = 1.8 w r`), not an observed fire, and
  the lowest intensity in Wilson's data may be higher. Known limits: the fit starts at
  1.5 m, so a 1 m hand line is an extrapolation and comes out at 61 % crossed just above the
  floor; Wilson himself notes that Kuznetsov (1985) saw 1 of 43 breaks 1.4 m wide crossed
  where this model predicts 84-94 %. Flame contact (a break holding while the flames are
  shorter than it is wide) was considered and rejected because it overrides the fit where
  Wilson's 3 m breaks were crossed 29-55 % of the time.
- **Burned wildland has no dollar value** (`usd_per_ha_wildland` = 0), so every strategy
  experiment includes a town.
- **A building is worth 400,000 USD,** a team assumption stated in the README.
- **Residence time:** Anderson (1969), `t_r = 384 / sigma` minutes with `sigma` the
  characteristic surface-area-to-volume ratio in 1/ft, as given in Andrews (2018, section
  4.2). Done in WP 2.1a. For a dynamic fuel model `sigma` depends on how much grass has
  cured, so the table is evaluated at the default `Weather` moisture (Scott & Burgan's D2L2:
  6/7/8 % dead, 60 % live herbaceous, 90 % live woody), for all 53 models, 6.6-20.1 s; fuel
  models not in the table get 15 s, near the median. Both choices accepted by the team. The
  Rothermel rule also reports the moisture-dependent value.
- **WP 2.1a acceptance, as interpreted and accepted by the team:**
  - Reference values produced by running the BehavePlus engine (the `behave` library through
    `pyrothermel`) count as "BehavePlus values with their source"; no printed table is needed.
  - "Evaluated once per unique combination" is met by evaluating the fuel bed once per fuel
    model and the wind and slope factors per cell: the results are identical (tested) and the
    cost is lower, since wind and slope differ almost cell by cell.
  - The tests of other modules that pinned the placeholder residence times
    (`tests/test_dormant.py`, `tests/test_sim.py`, `tests/test_urban.py`), the note in
    `configs/urban.yaml`, the `spread/__init__.py` docstring and this plan change in the WP 2.1a
    pull request, an exception to the one-WP-owns-its-files rule of `CONTRIBUTING.md`:
    replacing the residence times is part of the WP and would fail CI without them.
- **Hourly weather:** one source for both fires; HRRR analysis hours if the archive covers
  both, disaggregated gridMET for both otherwise.
- **Module owners:** all three team members own every module.

## 14. Design review log

### First review: plan against the Phase 0 code

The two original plan documents were merged and each claim checked against the code. WPs
0.1-0.7 all existed and their tests passed.

| Problem | Fix |
| --- | --- |
| The memoryless ignition kernel made the front run 3 to 4.5 times faster than R and depend on dt, contradicting the plan's central claim | Gamma crossing-time kernel; `spread.arrival_shape` (default 20); evidence script `scripts/kernel_front_speed.py` |
| `make_spread_rule("constant")` would fail unless the caller had imported the rule module | Registry imports `wildfire_sim.spread.<name>` on first use |
| `ScenarioConfig.n_steps` lost a step to float rounding (3600 s / 0.1 s gave 35999) | Rounding-tolerant floor |
| Ignition cells outside a synthetic grid passed validation | Checked in `ScenarioConfig` |
| `ResourceConfig` has an `engine` type but `CostConfig` had no engine rate | `engine_usd_per_h` added |
| `Grid.validate` accepted NaN and negative asset values | Finite and non-negative checks |
| `wildfire-sim validate` printed tracebacks for bad input | Reported as errors with exit code 1 |
| Burn duration equal to the residence time alone extinguishes the fire on 30 m cells | Two clocks (section 2.3) |
| Circularity test used a measure a raster disc cannot satisfy; wind test used a rule that ignores wind | Tests rewritten (WP 1.4) |
| Alexandridis probability-to-rate conversion, breaching of unburnable cells, and diagonal or knight-move leaks through barriers were unspecified | Specified (WPs 1.2, 4.3, 1.3, 4.1) |

### Second review: the design as a whole

| Weakness | Change |
| --- | --- |
| A link used only the source cell's rate, so fire entered slow fuel or retardant at the fast cell's speed | Harmonic-mean link rates (`link_rate_of_spread`, section 2.1); physics test (g) |
| The per-step kernel assumed a constant rate during a crossing and rounded arrivals to step ends | Per-link thresholds with accumulated progress and in-step interpolation (`draw_crossing_thresholds`); physics tests (c), (h) |
| Suppression could change the fire's random draws, breaking paired comparisons | Separate random streams; thresholds drawn once per run |
| Residual front-speed bias (up to 21 % on 16 neighbours) was left for calibration to absorb, mixing a numerical artefact into a physical parameter | Measured per-stencil correction (WP 1.4) |
| The 8-neighbour stencil was assumed adequate for elongated fires; the wind test checked only that the burn was longer downwind | Lattice study against the analytic ellipse; default stencil chosen from it (WP 1.4) |
| With low kernel noise, seed-only ensembles understate uncertainty | Input perturbations in ensembles (section 6, WP 1.6) |
| Rothermel wrapped a library with no Linux wheel, so CI would have skipped every Rothermel test | Native NumPy implementation, library as cross-check (WPs 2.1a-c) |
| Wind-slope vectoring, midflame wind reduction and the backing rate were unspecified | Specified (section 2.2, WP 2.1b) |
| Temperature and humidity were carried but unused; weather was daily | Hourly weather and computed 1-h moisture (WPs 2.1c, 2.4) |
| LANDFIRE version was not pinned, so fuels could postdate the fire | Version required and checked (WP 2.2) |
| Every road and stream cell became a full barrier at 30 m | Width-based barrier rule (section 3, WP 2.5) |
| Grids carried no CRS or transform; cached data had no manifest | Metadata keys and manifest (WPs 2.3, 2.6) |
| Calibration used only the final perimeter of a suppressed fire, with no second fire and no check that parameters were identified | Early-growth calibration, fire B without retuning, objective surface, parameter-recovery test, sensitivity analysis (section 7, Phase 3) |
| Resources were unlimited, appeared at the fire instantly once dispatched, and built line at one rate in any fuel | Availability, mobilization cost, shifts, travel, fuel- and slope-dependent production (section 4, WPs 4.2, 4.3) |
| The optimized policy was scored on the seeds and scenario it was tuned on; tail risk was reported but not optimized | Held-out seeds and scenarios, paired intervals, two-objective search (section 5, WPs 4.5, 4.7) |
| No property tests, regression baselines, run provenance or decision records | Section 8, WPs 1.3, 1.9, Phase 5 |
| Spotting and crown fire were unowned stretch goals | Specified as WPs 6.1, 6.2 |

### Third review: Phase 1 as built

Each work package was implemented by a separate agent and then checked independently before
merging. The simulator was verified against a from-scratch Dijkstra solution on a random
landscape with barriers, retardant and both stencils: arrival times agree to 1e-12 s for both
engines and any time step.

| Finding | Outcome |
| --- | --- |
| The in-step cascade specified in WP 1.3 (cells lit earlier in a step are never revised) biased arrival times by up to 4 % when fire crossed several cells per step | Replaced by an exact label-correcting cascade; arrival times no longer depend on dt at all |
| `Grid.apply_unburnable` overwrote BURNING and BURNED cells | Fixed |
| Knight links (16 neighbours) used only their end cells' rates and tunnelled through slow fuel (strip delay 12 % short) | `link_rate_of_spread` now averages over the four cells a knight link crosses; exact in the deterministic limit |
| Diagonal moves leaked between barrier cells touching at a corner | The simulator blocks diagonal and knight links that squeeze between unburnable cells, so 8-connected rasterized barriers hold |
| The lattice cannot reproduce elongated ellipses to the 10 % target (section 2.4) | Default stencil changed from 8 to 16; with the new defaults the target is met at a ratio of 1.5 and missed by 3 points at 3 (`test_default_kernel_ellipse`); the shape-20 failures stay recorded as strict expected failures; beyond a ratio of 3 is an open decision |
| The single front-speed correction over-slows elongated fires (section 2.1) | Default `arrival_shape` raised from 20 to 200 |
| Seed-only ensembles vary burned area by 4 %; with input perturbation 35 % | Confirms section 6 |
| `ensemble.mean_ci` returned NaN near the float64 limit (found by a property test) | Fixed |
| The plan's physics-test thresholds for dt halving and the slow strip needed restating | Thresholds and evidence in `docs/lattice_study.md` |
| Starting a worker pool costs 9-17 s | The optimizer must reuse one pool (WP 4.5) |

Left for later: `SpreadConfig` defaults to the constant rule with no `ros`, so a scenario that
omits `spread` fails at run time; `SpreadOutput.validate` does not check intensity or flame
length; the CI benchmark check is non-blocking until its baseline is refreshed from a CI run.

### Fourth review: the plan against what validation and the strategy search need

The plan and the Phase 1 code were read together, looking for places where Phase 3 or Phase 4
would produce a wrong answer and not an error. The first finding was confirmed by running the
simulator; the others come from reading the code and the plan.

| Hole | Change |
| --- | --- |
| A cell is a source only while one of its links has a rate above zero, so a period of zero rate (extinction moisture overnight, a full-strength retardant drop) burns every cell out and ends the fire for good. Measured on `flat_no_wind`: a 10-minute stall ended the run at 1380 s with 195 cells burned, against 4,087 without it | Dormant sources with a holdover time (section 2.3, WP 1.10) |
| The grid and the run-time estimate are sized for a 10 km area and a 12-hour fire, but fire A is roughly twice that area and burned for weeks; nothing detects a fire reaching the edge of its grid | Grid sized from the observed perimeter plus a buffer; edge detection; calibration cost measured before the study is sized (sections 2 and 6, WPs 1.11, 2.6, 3.1) |
| The lattice's ellipse error depends on the head's bearing relative to the grid, and calibration would absorb it into the spread multiplier and ellipse shape, which then would not transfer to a fire with another wind direction | Lattice-fitted rates and a length-to-breadth cap before Phase 3; calibration must recover the same parameters at every bearing (section 2.4, WPs 2.1d, 3.1) |
| Every cell gets the head intensity, so the crew safety limit and line breaching treat flanks and the back of a fire like its head | Front-normal intensity from the arrival-time gradient (section 2.2, WPs 1.12, 4.3) |
| Line breaching and direct attack were specified as a probability per step, which depends on dt and on flaming time, and policies would consume the suppression stream at different rates | One random number per cell drawn at setup; work accumulated at a rate (section 4, WP 4.3) |
| Fire B had to differ in fuel type, where a fuel-specific multiplier is expected to fail; a timber fire B needs crown fire, and the strategy experiments need spotting, yet both were optional extensions after the work that depends on them | Fire B criteria restated; 6.1 required by 4.7 and 6.2 by a timber fire B (sections 2.5, 7 and 13) |

Smaller findings of the same review, each checked against a source where it rested on memory:

| Finding | Outcome |
| --- | --- |
| The wind adjustment factor applies to the 20-ft wind, but the plan fed it the 10 m wind | Confirmed: Andrews (2012) states that the models are based on the 20-ft wind and that BehavePlus divides a 10 m wind by 1.15 (Turner and Lawson 1978). Conversion added (section 2.2, WP 2.1b) |
| Sections 2.2 and 3 took 10-h moisture from gridMET | Confirmed that gridMET has only `fm100` and `fm1000`. 10-h now comes from the 1-h value by a cited offset; 1000-h dropped (sections 2.2 and 3, WP 2.4) |
| A VIIRS detection is an upper bound on the arrival time | Arrival-time error scored against the interval between overpasses (section 7, WPs 2.7, 2.8) |
| A scenario has no start date and time | Optional `ScenarioConfig.start_time_utc` (WP 1.14) |
| A narrow road slowed a cell while a hand line made a whole cell unburnable | One breach model for roads, streams and built lines, with Wilson's (1988) coefficients checked against the paper (section 3, WPs 2.5, 2.9, 4.3) |
| The constants in `configs/alexandridis.yaml` were marked "check against the paper" | Checked against the paper itself: all nine match (density factors Table 2, vegetation factors Table 3, a, c1 and c2 Table 4; wind speed in m/s). The paper does not state the unit of the slope angle; degrees are assumed. The vegetation and density factors were chosen empirically by the authors and the other four fitted to the burned area of the Spetses fire. Notes in the YAML updated. The rule still uses them as rate multipliers, which the paper does not |
| Urban cells could not burn, so only the rim of a town could be lost | Urban cells burn by a scaled surrogate (section 2.6, WP 1.13) |
| Section 2.4 chose dt with `max_stable_dt`; the summary presented a nearly deterministic model as a stochastic one | Both passages rewritten |
| `KernelCorrectionTable.value_for` returns 1.0 for a shape not in the table; retardant decay defeats the spread-rule cache | Accepted as they are |

### Fifth review: WPs 1.10-1.14 as built

The five work packages the fourth review added were implemented one per branch, in the order
1.14, 1.10, 1.11, 1.12, 1.13, each with its acceptance tests on both engines. The golden
arrival-time arrays did not move in any of them. What the work turned up:

| Finding | Outcome |
| --- | --- |
| WP 1.11's acceptance run does not reach the edge. With the defaults set after the lattice study, `flat_no_wind` first burns a cell with a link leaving the grid at 7,403 s (the band starts 49 cells out: 7,350 s at 0.2 m/s); at 7200 s the front is 47 cells out | Acceptance restated at 7500 s; a test pins that the 7200 s run reports None |
| Front-normal intensity (WP 1.12) is exact on the polygon the lattice burns (within 1 % without kernel noise, on every side of the polygon), and so inherits its shape error against the ellipse: at a ratio of 3 the flank is 27-45 % too hot and a head between two stencil directions reads 0.4 of the head value | Bounds set from the measurement; the ratio-3 target is a strict expected failure; WP 2.1d re-measures (`docs/lattice_study.md`, section 6) |
| With `ros_factor` 0, urban cells would have been zero-rate fuel, and after WP 1.10 the cells around a town would have waited as dormant sources for the whole holdover time | With `ros_factor` 0 urban cells are made UNBURNABLE at the start of the run, exactly the behaviour before WP 1.13 |
| A rule whose rate is zero everywhere now keeps its ignition cell BURNING, dormant, for the holdover time (12 h placeholder), where the run used to end at once | Accepted: it follows from section 2.3. The engine cannot tell a rate that is zero for good from one that is zero for now |
| The dormant clock counts whole steps, from the end of the first step that leaves a cell dormant, and restarts when the cell spreads again | Accepted: holdover times are hours, steps are seconds |
| Unguarded, the additions slowed the NumPy engine by 18-28 % on the reference run | A stalled-link scan only for cells that have a zero-rate link at all, and no plane fit where the rule reports no intensity: back to the old speed for such rules; 5-8 % (NumPy) and 2 % (Numba) for a rule with intensity (`docs/performance.md`) |
| The intensity a cell had when the front crossed it was lost once the cell stopped flaming | `RunResult.front_intensity`, for WP 2.9 and for analysis |
| The urban parameters are read by the simulator and by the data generators, which must not import the simulator | New module `wildfire_sim.urban` (added to the module map) |
| Byram's flame-length constants had no home in `configs/` | `configs/flame_length.yaml`, the coefficient derived from Byram's units in the file |
| The CI job for the Numba engine ran only `tests/test_sim_fast.py`, so the Numba halves of the new tests would never run in CI | The job now runs the new test files too |
| An urban cell with `fuel_model` 0 cannot burn | WP 2.2 writes the surrogate fuel model into urban cells (row updated) |

Left for later, unchanged from the third review: `SpreadConfig` defaults to the constant rule
with no `ros`; `SpreadOutput.validate` does not check intensity or flame length. The viz
legend still describes urban land use as non-burnable in two docstrings (`viz.py` was not in
any of these work packages).

## 15. Sources

- [Alexandridis et al. 2008, CA model of the Spetses Island fire](https://dspace.lib.ntua.gr/xmlui/handle/123456789/18520)
- [Alexandridis et al. 2011, CA spread with suppression tactics (Int. J. Wildland Fire)](https://publish.csiro.au/wf/ExportCitation/wf09119)
- [Russo et al. 2013, same CA applied to the 2008 Rhodes fire without retuning](https://cetjournal.it/index.php/cet/article/view/6182)
- [CA wildfire simulator applied to the 2016 Chimney Fire (arXiv 2507.00761)](https://arxiv.org/pdf/2507.00761)
- [pyretechnics: Python fire behavior equations](https://ost.ecosyste.ms/projects/351112)
- [pyrothermel: Python bindings for Rothermel models](https://pypi.org/project/pyrothermel/)
- [behave: the BehavePlus fire behaviour library](https://github.com/firelab/behave)
- [Cohen & Deeming 1985, the National Fire-Danger Rating System: basic equations (PSW-GTR-82)](https://www.fs.usda.gov/psw/publications/documents/psw_gtr082/psw_gtr082.pdf)
- [Glass & Zelinka 2010, moisture relations and physical properties of wood, Wood Handbook chapter 4 (FPL-GTR-190)](https://research.fs.usda.gov/treesearch/37428)
- Simard 1968, the moisture content of forest fuels I (Forest Fire Research Institute, Ottawa, FF-X-14)
- Rothermel 1972, a mathematical model for predicting fire spread in wildland fuels (USDA For. Serv. Res. Pap. INT-115)
- Albini 1976, estimating wildfire behavior and effects (USDA For. Serv. Gen. Tech. Rep. INT-30)
- Anderson 1969, heat transfer and fire spread (USDA For. Serv. Res. Pap. INT-69)
- Scott & Burgan 2005, standard fire behavior fuel models (USDA For. Serv. Gen. Tech. Rep. RMRS-GTR-153)
- Andrews 2018, the Rothermel surface fire spread model and associated developments (USDA For. Serv. Gen. Tech. Rep. RMRS-GTR-371)
- Andrews, Cruz & Rothermel 2013, examination of the wind speed limit function in the Rothermel model (Int. J. Wildland Fire 22: 959-969)
- [Andrews 2012, wind adjustment factor and midflame wind speed (RMRS-GTR-266)](https://www.fs.usda.gov/rm/pubs/rmrs_gtr266.pdf)
- [gridMET variables (Earth Engine catalog)](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET)
- [Russo et al. 2014, CA risk assessment on Spetses, with the 2008 parameter table (Chem. Eng. Trans. 36)](https://www.aidic.it/cet/14/36/043.pdf)
- [Wilson 1988, width of firebreak necessary to stop grass fires (Can. J. For. Res. 18)](https://www.frames.gov/catalog/11823)
- [Andrews 2018, the Rothermel surface fire spread model and associated developments (RMRS-GTR-371)](https://www.fs.usda.gov/research/treesearch/55928)
- [NASA VIIRS 375 m active fire product](https://viirsland.gsfc.nasa.gov/Products/NASA/FireESDR.html)
- [Finney 1998, FARSITE model development and evaluation (RMRS-RP-4)](https://www.fs.usda.gov/rm/pubs/rmrs_rp004.pdf)
- [Forest-fire model (the original baseline)](https://en.wikipedia.org/wiki/Forest-fire_model)

Cited by name and to be linked from the relevant config or doc when its WP is done: Rothermel
(1972, 1991); Albini (1976); Anderson (1969, 1983); Van Wagner (1977); Byram (1959); Scott and
Burgan (2005); Andrews (2012); Fosberg and Deeming (1971); Simard (1968); Finney (1998,
FARSITE); Wilson (1988); NWCG fireline production rates and resource typing guides.
