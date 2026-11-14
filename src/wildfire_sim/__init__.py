"""wildfire-sim: a cellular-automaton wildfire spread and suppression model.

See ``docs/plan.md`` for the model design, the module map and the work packages.
"""

from wildfire_sim.grid import Grid, Weather, WeatherSeries
from wildfire_sim.types import CellState, LandUse, Stencil, get_stencil

__version__ = "0.1.0"

__all__ = [
    "CellState",
    "Grid",
    "LandUse",
    "Stencil",
    "Weather",
    "WeatherSeries",
    "__version__",
    "get_stencil",
]
