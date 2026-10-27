"""Grid construction: synthetic generators now, the real-world pipeline in Phase 2.

Phase 2 modules (``landfire``, ``terrain``, ``weather``, ``osm``, ``cache``, ``validation``)
produce grids saved with :meth:`wildfire_sim.grid.Grid.save`; runs then load them through
:class:`~wildfire_sim.config.CachedGridConfig`.
"""

from __future__ import annotations

from wildfire_sim.config import CachedGridConfig, SyntheticGridConfig
from wildfire_sim.data.synthetic import GENERATORS
from wildfire_sim.grid import Grid


def build_grid(config: SyntheticGridConfig | CachedGridConfig) -> Grid:
    """Build or load the grid described by a scenario's ``grid`` section."""
    if isinstance(config, SyntheticGridConfig):
        generator = GENERATORS[config.generator]
        return generator(shape=config.shape, dx=config.dx, fuel_model=config.fuel_model, **config.params)
    return Grid.load(config.path)


__all__ = ["GENERATORS", "build_grid"]
