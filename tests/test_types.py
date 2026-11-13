"""Tests for wildfire_sim.types: stencils and enums."""

from typing import Any

import numpy as np
import pytest

from wildfire_sim.types import STENCIL_8, STENCIL_16, CellState, LandUse, Stencil, get_stencil


def test_stencil_8_geometry() -> None:
    assert STENCIL_8.size == 8
    assert STENCIL_8.max_offset == 1
    # orthogonal neighbours at distance 1, diagonals at sqrt(2)
    ortho = np.abs(STENCIL_8.offsets).sum(axis=1) == 1
    assert np.allclose(STENCIL_8.distances[ortho], 1.0)
    assert np.allclose(STENCIL_8.distances[~ortho], np.sqrt(2))


def test_stencil_8_bearings_clockwise_from_north() -> None:
    assert np.allclose(STENCIL_8.bearings_deg, [0, 45, 90, 135, 180, 225, 270, 315])
    # first offset is north: row decreases
    assert tuple(STENCIL_8.offsets[0]) == (-1, 0)
    # east: col increases
    assert tuple(STENCIL_8.offsets[2]) == (0, 1)


def test_stencil_16_geometry() -> None:
    assert STENCIL_16.size == 16
    assert STENCIL_16.max_offset == 2
    knight = np.abs(STENCIL_16.offsets).max(axis=1) == 2
    assert knight.sum() == 8
    assert np.allclose(STENCIL_16.distances[knight], np.sqrt(5))
    # bearings strictly increasing clockwise
    assert np.all(np.diff(STENCIL_16.bearings_deg) > 0)
    # every 8-stencil direction is present in the 16-stencil
    for off in STENCIL_8.offsets:
        assert any(np.array_equal(off, o) for o in STENCIL_16.offsets)


def test_stencils_have_no_duplicate_offsets() -> None:
    for st in (STENCIL_8, STENCIL_16):
        assert len({tuple(o) for o in st.offsets}) == st.size


def test_stencil_arrays_are_read_only() -> None:
    with pytest.raises(ValueError):
        STENCIL_8.offsets[0, 0] = 5


def test_stencil_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError):
        Stencil(offsets=np.zeros((3, 2), int), distances=np.ones(2), bearings_deg=np.zeros(3))


def test_get_stencil() -> None:
    assert get_stencil(8) is STENCIL_8
    assert get_stencil(16) is STENCIL_16
    bad: Any = 4
    with pytest.raises(ValueError):
        get_stencil(bad)


def test_enums() -> None:
    assert CellState.UNBURNABLE == 0
    assert len(CellState) == 4
    assert LandUse.WILDLAND.burnable
    assert not LandUse.WATER.burnable
    assert LandUse.URBAN.burnable  # urban cells burn by a scaled surrogate fuel (WP 1.13)
    assert not LandUse.ROAD.burnable
