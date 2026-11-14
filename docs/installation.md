# Installation

wildfire-sim is not published on PyPI. Install from source.

```sh
git clone https://github.com/RicheyLi/APC_524_Final.git
cd APC_524_Final
uv sync
```

Optional extras:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `fast` | numba | compiled front-only update kernel |
| `geo` | rasterio, xarray, geopandas, osmnx, py3dep | real-world data pipeline (LANDFIRE, 3DEP, gridMET, OSM) |
| `rothermel` | pyrothermel | optional cross-check of the built-in Rothermel spread rule |
| `opt` | optuna | policy optimization and calibration |
| `all` | everything above | |

```sh
uv sync --extra all
```

Requires Python 3.12 or newer.
