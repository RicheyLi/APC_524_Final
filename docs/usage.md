# Usage

Every run is defined by a YAML scenario file plus a seed.

```sh
uv run wildfire-sim run configs/scenarios/flat_no_wind.yaml --seed 0 --out runs/demo
```

From Python:

```python
from wildfire_sim.config import load_scenario

cfg = load_scenario("configs/scenarios/flat_no_wind.yaml")
```

The `run`, `ensemble` and `plot` subcommands are implemented in WP 1.8; until then the CLI
only validates configuration files.
