# Wildfire Spread & Suppression Model — Revised Plan

Oct 8, 2026 · @Richey

## Summary of changes

Keep the cellular automaton (CA), but tie every ignition probability to a physical rate of spread, pull inputs from fire-science GIS datasets instead of Google Maps, and treat "optimization" as a search over reactive suppression policies scored on Monte Carlo ensembles.

- **Ignition probability comes from physics, not hand-picked weights.** Each neighbor's ignition chance is computed from a directional rate of spread (Rothermel), so cells and time steps have real meaning.
- **Time step is tied to cell size.** The step size is chosen from the cell size and the fastest spread, and the model is tested to give the same answer when the step is halved.
- **Burning lasts a realistic duration.** A cell stays on fire for its fuel's residence time, not one step.
- **Data comes from LANDFIRE, USGS, gridMET and OpenStreetMap.** A map is still how the user picks the area, but not where the physics inputs come from (see the next section).
- **Calibration against a real fire is a required milestone,** not an afterthought. Without it the model is not predictive.
- **Firefighting is modeled as resources** with arrival delays, production rates and hourly costs, acting on the grid.
- **The optimizer tunes parameterized reactive policies,** for example "build a line 200 m ahead of the head when it nears a town". Reinforcement learning is a stretch goal.
- **Every result is an ensemble** of runs, reported as distributions and burn-probability maps.
- **The real-world pipeline is a must-have but limited to the US,** where 30 m fuel maps exist, starting with one cached study region.

## Critique: real-world data and tunable parameters

The idea is sound and it is how published CA fire models work, but it needs four corrections.

1. **Google Maps is the wrong data source.** It shows what an area looks like, not fuel type, fuel load, canopy, moisture or usable elevation, and its terms of service forbid extracting map content. Use a map only to let the user draw a bounding box, then pull physics inputs from fire-modeling datasets (next sections).
2. **Parameters cannot be calculated from data alone.** Data gives physical inputs such as fuel model, slope, wind and moisture. Turning those into a probability needs either a physical spread model (Rothermel) or coefficients fitted to a real fire. Alexandridis et al. took the second route for the 1990 Spetses Island fire, tuning parameters by black-box optimization against GIS data of the actual burn. Either way, plan a calibration step.
3. **The Wikipedia baseline has no clock.** In the Drossel–Schwabl model a step is not a unit of time, so "the fire reached town in 3 hours" has no meaning until you link the step to a spread rate. Section 3 does this.
4. **Square grids distort fire shape.** On a square grid, a fire with no wind spreads as a diamond or octagon instead of a circle. Use the true diagonal distance (√2 × cell size), keep ignition random, and add a test that a no-wind fire on flat uniform fuel comes out nearly circular.

## Core model: a CA with physical scale

Use 30 m cells, the native resolution of the US fuel maps. Derive each neighbor's ignition probability from a directional rate of spread so that results do not depend on the time step.

**Grid.** A 10 km × 10 km study area at 30 m is about 333 × 333 ≈ 111,000 cells. Each cell stores:

- Static layers: fuel model, canopy cover, elevation, slope, aspect, land use (urban, water, road) and asset value.
- Weather layers that change over time: wind speed and direction, and dead fuel moisture.
- State: unburnable, unburned fuel, burning (with a countdown timer) or burned out.
- Suppression layers: fireline present, and retardant strength.

**Ignition rule.** Treat spread from a burning cell i to an unburned neighbor j as a random process whose average arrival time is the travel distance divided by the spread rate:

```latex
p_{ij} = 1 - \exp\!\left(-\frac{R_{ij}\,\Delta t}{d_{ij}}\right)
```

Here R\_ij is the rate of spread from i toward j (m/s), d\_ij is the center-to-center distance (Δx, or √2 Δx on a diagonal) and Δt is the time step. With several burning neighbors, the cell ignites with probability:

```latex
P_j = 1 - \prod_{i \in \text{burning neighbors}} \left(1 - p_{ij}\right)
```

This keeps your original "more burning neighbors means more likely" idea and makes the mean arrival time equal d / R regardless of Δt.

**Rate of spread.** Compute the head-fire rate from the Rothermel surface fire model using fuel model, fuel moisture, wind and slope. Spread in other directions follows an ellipse whose length-to-breadth ratio grows with wind speed. Python implementations exist (pyretechnics, pyrothermel); writing the equations yourselves is good report material but costs about a week.

**Where temperature and humidity enter.** They act mainly through dead fuel moisture, which feeds Rothermel. Model them as a moisture field, not as separate multipliers on probability.

**Burn duration.** A burning cell stays on fire for its fuel's residence time (seconds for grass, much longer for timber) before becoming burned out.

**Time step.** Pick Δt so that p\_ij stays well below 1 for the fastest spread in the domain, roughly Δt ≤ Δx / R\_max. Fast grass fire at 1 m/s on 30 m cells gives Δt ≈ 30 s.

**Intermediate model.** Before Rothermel, implement the Alexandridis rule, which multiplies a base probability by vegetation, density, wind and slope factors. It is quick to build, already calibrated on real fires, and gives a comparison point in your write-up.

**Stretch: spotting.** Embers thrown ahead of the front, which let fire jump firelines. Alexandridis et al. include a simple version.

## Real-world data pipeline

Every model input has a free, public US source. The pipeline is: user picks a bounding box → download each layer → reproject everything onto one 30 m grid → cache to disk → run.

| Input | Source | Resolution | Notes |
| --- | --- | --- | --- |
| Fuel model, canopy cover, canopy height | LANDFIRE (Scott & Burgan 40 fuel models) | 30 m | US only; this is why the project is limited to the US |
| Elevation → slope, aspect | USGS 3DEP | 10–30 m | Compute slope and aspect yourselves |
| Wind, temperature, humidity | gridMET (daily, history) or HRRR (hourly forecast) | \~4 km / 3 km | Interpolate to the grid; HRRR if you want hour-by-hour wind shifts |
| Dead fuel moisture | gridMET 100-hr and 1000-hr fuel moisture | \~4 km | Live fuel moisture: assume a seasonal value |
| Roads, rivers, lakes, buildings | OpenStreetMap (osmnx) | Vector | Natural firebreaks, crew access, assets |
| Asset value | Building footprints × an assumed $ per building | Vector | A team assumption; state it in the README |
| Historical fires (validation) | MTBS burn perimeters; NASA FIRMS VIIRS hotspots | Perimeter; 375 m detections | Final burned area and timing of spread |

Useful Python libraries: rasterio and xarray for rasters, geopandas and osmnx for vectors. Do not commit downloaded data to the repo; commit a download script and cache files locally.

Verify each source's current access method in week 1. Download services change, and a dead link found in week 6 is costly.

## Firefighting tactics

Model every tactic as a resource that arrives after a dispatch delay, works at a limited rate, costs money per hour, and changes cells on the grid. That answers your "how to model?" questions and makes "unlimited but costed" realistic: you can call more crews, but no line is built instantly.

| Tactic | Effect on the grid | Key parameters | Cost drivers |
| --- | --- | --- | --- |
| Indirect attack (control line far from the fire) | Cells along a planned path become unburnable, one segment at a time as it is built. An unfinished segment does not stop fire. | Path, line width, build rate (m/h per crew or dozer), start time | Crew and dozer hours |
| Parallel attack (line close to the fire) | Same as indirect, but the path is the current fire edge pushed outward by an offset d, recomputed as the fire moves | Offset d, which flank, build rate | Crew and dozer hours |
| Direct attack: ground crews | Crews move along the burning edge and put out burning cells next to them with some probability per step | Crew speed along the edge, extinguish rate, maximum fire intensity crews can face | Crew hours |
| Direct attack: aircraft | A drop covers a strip of cells and multiplies their spread rate by a factor below 1, which fades over hours (retardant) or minutes (water) | Drop footprint, reduction factor, decay time, turnaround time | Flight hours, liters dropped |

**Line breaching.** Allow a fire to cross a line with a probability that rises with fire intensity relative to line width. Without this, any line is perfect and the optimizer will exploit it.

**Crew safety limit.** Ground crews cannot work an edge whose flame lengths are too high; NWCG guidance sets such limits by resource type. Enforce a limit and the optimizer will learn when direct attack is not an option.

The parameter values (build rates, drop sizes, decay times, costs) should come from fire-management references and be listed in one config file with citations.

## Strategy search

Have the code tune a family of reactive rules rather than search over raw line placements. A rule family has perhaps 5–15 numbers, which a black-box optimizer can handle in a few thousand ensemble runs.

**What a reactive policy is.** At each decision point (every 30–60 simulated minutes) the policy looks at the fire's current state and decides what to deploy. Example rules, each controlled by a number the optimizer tunes:

- Dispatch k crews, m dozers and n aircraft at first report.
- Build a parallel line at offset d on the flank nearest the assets.
- When the head's spread rate passes a threshold, switch from direct attack to an indirect line L meters ahead, downwind.
- Send aircraft to the head of the fire unless an asset is within a set distance, then protect the asset.

The policy reads only things a real incident commander would know: perimeter, head direction, spread rate, intensity and distance to assets. User-defined strategies from your original plan still work: a user writes a policy and the optimizer tunes its numbers.

**Cost function.** The optimizer minimizes the expected total cost over an ensemble of runs:

```latex
J(\theta) = \mathbb{E}\left[\sum_{r} c_r\,T_r(\theta) + c_w V_w(\theta) + c_{rt} V_{rt}(\theta) + \sum_{k\,\in\,\text{burned}} v_k\right]
```

θ is the policy's parameter vector; T\_r is hours worked by resource type r at hourly cost c\_r; V\_w and V\_rt are liters of water and retardant at unit costs c\_w and c\_rt; v\_k is the value of burned cell k. Also report a tail measure such as the 90th-percentile cost, because an average can hide the rare runs where a town burns.

**Optimizer.** Use Optuna: it handles a mix of whole-number parameters (crew counts) and continuous ones (offsets), tolerates noisy scores, and can stop unpromising candidates early. CMA-ES (the `cma` package) is a good alternative if all parameters are continuous.

**Baselines.** Compare against no suppression, direct attack only and indirect attack only. The optimized policy has to beat all three for the result to mean anything.

**Stretch: reinforcement learning.** An RL agent picking actions each decision point is the natural next step, but it needs far more simulated runs. Attempt it only if the rest is done.

## Stochastic runs: how expensive they are

One run is cheap; the cost comes from multiplying runs per ensemble by candidate strategies. With reasonably written code the whole project fits on laptops.

Rough estimates for the 10 km area (111,000 cells), a 12-hour fire at Δt = 30 s, so about 1,440 steps. These are order-of-magnitude guesses; benchmark your own code in week 2.

| Workload | Runs | Plain NumPy, whole grid each step (\~1–2 s/run) | Numba, burning front only (\~0.1 s/run) |
| --- | --- | --- | --- |
| One scenario | 1 | 1–2 s | 0.1 s |
| One ensemble | 200 | 3–7 min | 20 s |
| One optimization | 300 candidates × 50 runs = 15,000 | 4–8 h | 25 min |

Runs are independent, so they split across CPU cores with no extra work: an 8-core laptop is roughly 8× faster than the table.

**How many runs per ensemble.** The uncertainty of an average shrinks with the square root of the run count: quadrupling runs halves the error bar. Plot the running mean of burned area against run count and stop when it settles; 50–200 runs is typical.

**Keeping it cheap:**

- Update only cells near the burning front, compiled with Numba.
- Score all candidate strategies on the same set of random seeds, so differences reflect the strategy, not luck.
- Give every candidate a few runs first, and spend more runs only on the promising ones (Optuna's pruning does this).
- Record every seed so any run can be reproduced exactly.

## Validation

Calibrate on one well-documented historical US fire and test on a second. This is the step that makes the model "predictive".

**Choosing a fire.** Look for one with an MTBS perimeter, VIIRS hotspots showing its daily growth, and pre-fire LANDFIRE fuels. A medium fire (about 1,000–10,000 ha) driven by wind is easiest; little documented suppression keeps the comparison clean. The 2016 Chimney Fire in California has already been simulated with an Alexandridis-style CA, which gives you a published comparison.

**Metrics:**

- Overlap between simulated and real burned area (Jaccard or Sørensen index), using cells burned in at least half the runs.
- Arrival-time error: when the simulated fire reaches each VIIRS detection versus when it was detected.
- Ratio of simulated to real burned area.

**Calibration.** Tune only a few parameters (an overall spread-rate multiplier, the ellipse shape) with the same optimizer, then run the second fire without retuning.

**Physics tests in the test suite:**

- No wind, flat, uniform fuel → a near-circular burn.
- Steady wind → an ellipse stretched downwind.
- Halving Δt → the same distribution of arrival times.
- A complete line wider than any breach distance → fire stops.
- Water and urban cells never burn.

## Repo structure

Since the repo is the deliverable, organize it as an installable Python package with swappable spread rules, config-driven runs and automated tests.

```text
wildfire-sim/
  pyproject.toml
  README.md                 # what it does, install, one example run, a GIF
  src/wildfire/
    grid.py                 # Grid: layers, cell states, geometry
    data/                   # landfire.py, terrain.py, weather.py, osm.py, cache.py
    spread/                 # drossel.py, alexandridis.py, rothermel.py (same interface)
    suppression/            # resources.py, tactics.py, policies.py
    sim.py                  # time loop, event log
    ensemble.py             # parallel Monte Carlo runs
    optimize.py             # Optuna study over policy parameters
    metrics.py              # burned area, cost, overlap scores
    viz.py                  # maps, burn-probability plots, animations
  configs/                  # YAML scenarios and cited parameter values
  scripts/                  # run_scenario.py, calibrate.py, download_data.py
  notebooks/                # exploration and figures
  tests/
```

**Practices that pay off:**

- Every spread rule takes the same inputs and returns ignition probabilities, so models swap with one config line.
- Every run is defined by a YAML file plus a seed, so results are reproducible.
- pytest with GitHub Actions runs the physics tests on every push.
- Each person owns one or two modules and changes others' through pull requests.

## Timeline

The plan front-loads the physics and data so the model is validated by mid-November, leaving the last three weeks for strategy search, experiments and a buffer.

&#91;embedded content: 8-week plan · 2 gates\]

If the first gate slips, cut the Rothermel step and finish on the Alexandridis rule; the rest of the pipeline works with either. Stretch goals (spotting, reinforcement learning, non-US data) start only after the second gate.

## Open decisions

- [ ] Pick the study region and the two validation fires (US only).
- [ ] Use an existing Rothermel library or write the equations yourselves.
- [ ] Set asset values: dollars per building, and whether forest and other land have a value.
- [ ] Collect cost figures with sources: crew, dozer and aircraft hourly rates; water and retardant per liter.
- [ ] Assign module owners.

## Sources

- [Alexandridis et al. 2008, CA model of the Spetses Island fire](https://dspace.lib.ntua.gr/xmlui/handle/123456789/18520)
- [Alexandridis et al. 2011, CA spread with suppression tactics (Int. J. Wildland Fire)](https://publish.csiro.au/wf/ExportCitation/wf09119)
- [Russo et al. 2013, same CA applied to the 2008 Rhodes fire without retuning](https://cetjournal.it/index.php/cet/article/view/6182)
- [CA wildfire simulator applied to the 2016 Chimney Fire (arXiv 2507.00761)](https://arxiv.org/pdf/2507.00761)
- [pyretechnics: Python fire behavior equations](https://ost.ecosyste.ms/projects/351112)
- [pyrothermel: Python bindings for Rothermel models](https://pypi.org/project/pyrothermel/)
- [Forest-fire model (your baseline)](https://en.wikipedia.org/wiki/Forest-fire_model)
