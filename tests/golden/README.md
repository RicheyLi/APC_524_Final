# Golden runs

Committed arrival-time arrays for three small scenarios at a fixed seed (WP 1.9,
`docs/plan.md` section 8). `tests/test_golden.py` reruns each one and fails if the result moved.

| Scenario | Rule | Stencil | What it pins down |
| --- | --- | --- | --- |
| `flat_no_wind.yaml` | constant, no wind | 8 | isotropic spread; the whole 31 x 41 grid burns |
| `steady_wind_slope.yaml` | Alexandridis, 5 m/s west wind, west-facing 0.2 slope | 8 | wind and slope factors; the run stops with the head inside the grid |
| `river_barrier.yaml` | constant, one-cell river at column 25 | 16 | knight moves and barrier blocking; only the west side burns |

For each scenario `<name>.npz` holds `arrival_time`, the `(31, 41)` float64 arrival times in
seconds (NaN where a cell never burned), and `<name>.json` records the scenario file, seed,
spread rule, stencil, arrival shape, kernel correction and its source, package versions and
git commit of the run that wrote it.

## Tolerance

The burned set must match exactly and arrival times within `rtol = 1e-7` (plus `atol = 1e-6` s
for the ignition at t = 0). Runs are bit-stable on one platform; the slack covers last-digit
differences across platforms and NumPy builds. Link thresholds are stored as float32, so such a
difference can at worst flip one float32 rounding (a relative change of 6e-8). A 0.1 % change of
the kernel correction moves arrival times by 1e-3, far beyond the tolerance.

## Regenerating

If a golden test fails because you changed the model on purpose:

```sh
uv run python tests/golden/regenerate.py           # rewrites only files whose array changed
uv run python tests/golden/regenerate.py --force   # rewrites all, refreshing the metadata
```

It prints, per scenario, how many cells changed burned status and the largest arrival-time
change. Commit the new `.npz` and `.json` files and explain in the pull request why the
arrival times changed. Never regenerate just to make a failing test pass without knowing why.
