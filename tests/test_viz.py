"""Tests for :mod:`wildfire_sim.viz` (WP 1.5).

The simulator (WP 1.3) is not imported: fake results are built with :class:`FakeResult`, which
satisfies the ``HasArrival`` protocol that ``viz`` is written against. Every figure is drawn on
an off-screen Agg canvas; nothing touches pyplot's global state or opens a window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from wildfire_sim import viz
from wildfire_sim.data.synthetic import flat_uniform, with_river, with_town
from wildfire_sim.grid import Grid
from wildfire_sim.types import CellState, FloatArray, LandUse

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _agg_backend():
    """Force a non-interactive backend and close any pyplot figures a test might leak."""
    matplotlib.use("Agg", force=True)
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@dataclass
class FakeResult:
    """Minimal stand-in for ``wildfire_sim.sim.RunResult``."""

    grid: Grid
    arrival_time: FloatArray


def _radial_result(grid: Grid, speed_cells_per_s: float = 0.01) -> FakeResult:
    """Fire spreading radially from the grid centre over the burnable cells.

    Arrival time is distance / speed, NaN on unburnable cells and beyond 40 % of the grid size.
    """
    h, w = grid.shape
    rows, cols = np.mgrid[0:h, 0:w]
    r0, c0 = h // 2, w // 2
    if not grid.burnable_mask()[r0, c0]:
        c0 = 2  # e.g. a river or town in the middle: ignite near the west edge instead
    dist = np.hypot(rows - r0, cols - c0)
    arrival = dist / speed_cells_per_s
    arrival[dist > 0.4 * max(h, w)] = np.nan
    arrival[~grid.burnable_mask()] = np.nan
    burned = np.isfinite(arrival)
    final = grid.copy()
    final.state[burned] = int(CellState.BURNED)
    final.state[burned & (arrival > np.nanmax(arrival) * 0.9)] = int(CellState.BURNING)
    final.arrival_time[:] = arrival.astype(np.float32)
    return FakeResult(grid=final, arrival_time=arrival.astype(np.float64))


@pytest.fixture
def result(small_grid: Grid) -> FakeResult:
    return _radial_result(small_grid)


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ------------------------------------------------------------------ colours


def test_colour_mapping_covers_every_category() -> None:
    assert set(viz.CATEGORY_COLOURS) == set(viz.DisplayClass)
    # distinct colours so categories can be told apart
    assert len(set(viz.CATEGORY_COLOURS.values())) == len(viz.CATEGORY_COLOURS)
    for state in (CellState.FUEL, CellState.BURNING, CellState.BURNED, CellState.UNBURNABLE):
        assert state in viz.STATE_COLOURS
    for lu in LandUse:
        assert lu in viz.LAND_USE_COLOURS


def test_classify_grid_categories() -> None:
    grid = with_river((10, 10), 30.0, col=4, width=2)
    grid.land_use[0, 0] = int(LandUse.URBAN)
    grid.land_use[0, 1] = int(LandUse.ROAD)
    grid.fireline[9, 9] = True
    grid.fuel_model[5, 0] = 0  # bare ground: other unburnable
    grid.apply_unburnable()
    grid.state[2, 2] = int(CellState.BURNING)
    grid.state[3, 3] = int(CellState.BURNED)
    cats = viz.classify_grid(grid)
    assert cats.shape == grid.shape
    assert cats[0, 4] == viz.DisplayClass.WATER
    assert cats[0, 0] == viz.DisplayClass.URBAN
    assert cats[0, 1] == viz.DisplayClass.ROAD
    assert cats[9, 9] == viz.DisplayClass.FIRELINE
    assert cats[5, 0] == viz.DisplayClass.OTHER_UNBURNABLE
    assert cats[2, 2] == viz.DisplayClass.BURNING
    assert cats[3, 3] == viz.DisplayClass.BURNED
    assert cats[8, 8] == viz.DisplayClass.FUEL


# ------------------------------------------------------------------ plot_state


def test_plot_state_returns_axes_and_writes_png(result: FakeResult, tmp_path: Path) -> None:
    ax = viz.plot_state(result.grid)
    assert isinstance(ax, Axes)
    out = viz.save_figure(ax, tmp_path / "state.png")
    assert out == tmp_path / "state.png"
    assert _nonempty(out)


def test_plot_state_uses_given_axes_and_has_legend(small_grid: Grid) -> None:
    fig = Figure()
    ax = fig.add_subplot()
    small_grid.retardant[3:6, 3:6] = 0.8
    out = viz.plot_state(small_grid, ax=ax)
    assert out is ax
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert "Unburned fuel" in labels
    assert any("etardant" in label for label in labels)
    assert ax.get_aspect() == 1.0


def test_plot_state_north_up_extent(small_grid: Grid) -> None:
    ax = viz.plot_state(small_grid)
    image = ax.get_images()[0]
    x0, x1, y0, y1 = image.get_extent()
    # row 0 (north) is drawn at the top: y increases upward, image origin "upper"
    assert image.origin == "upper"
    assert x0 == 0 and y0 == 0
    assert x1 == pytest.approx(21 * 30.0) and y1 == pytest.approx(21 * 30.0)


@pytest.mark.parametrize("make", [with_river, with_town])
def test_plot_state_with_river_and_town(make, tmp_path: Path) -> None:
    grid = make((41, 41), 30.0)
    res = _radial_result(grid)
    out = viz.save_figure(viz.plot_state(res.grid), tmp_path / "s.png")
    assert _nonempty(out)


# ------------------------------------------------------------------ plot_arrival


@pytest.mark.parametrize("contours", [True, False])
def test_plot_arrival_writes_png(result: FakeResult, tmp_path: Path, contours: bool) -> None:
    ax = viz.plot_arrival(result, contours=contours)
    assert isinstance(ax, Axes)
    assert _nonempty(viz.save_figure(ax, tmp_path / "arrival.png"))


def test_plot_arrival_colourbar_states_units(result: FakeResult) -> None:
    fig = Figure()
    ax = fig.add_subplot()
    assert viz.plot_arrival(result, ax=ax) is ax
    colourbar_axes = [a for a in fig.axes if a is not ax]
    assert colourbar_axes, "expected a colour bar"
    label = colourbar_axes[0].get_ylabel()
    assert "Arrival time" in label
    assert "(min)" in label or "(h)" in label


def test_plot_arrival_all_unburned(small_grid: Grid, tmp_path: Path) -> None:
    res = FakeResult(grid=small_grid, arrival_time=np.full(small_grid.shape, np.nan))
    assert _nonempty(viz.save_figure(viz.plot_arrival(res), tmp_path / "a.png"))


@pytest.mark.parametrize("make", [with_river, with_town])
def test_plot_arrival_with_river_and_town(make, tmp_path: Path) -> None:
    res = _radial_result(make((41, 41), 30.0))
    assert _nonempty(viz.save_figure(viz.plot_arrival(res), tmp_path / "a.png"))


# ------------------------------------------------------------------ plot_burn_probability


def test_plot_burn_probability_from_stack(rng: np.random.Generator, tmp_path: Path) -> None:
    stack = rng.random((5, 12, 15)) < 0.4
    ax = viz.plot_burn_probability(stack, dx=30.0)
    assert isinstance(ax, Axes)
    image = ax.get_images()[0]
    np.testing.assert_allclose(np.asarray(image.get_array()), stack.mean(axis=0))
    assert image.get_clim() == (0.0, 1.0)
    assert _nonempty(viz.save_figure(ax, tmp_path / "p.png"))


def test_plot_burn_probability_from_probability(rng: np.random.Generator, tmp_path: Path) -> None:
    prob = rng.random((12, 15))
    fig = Figure()
    ax = fig.add_subplot()
    assert viz.plot_burn_probability(prob, dx=30.0, ax=ax) is ax
    labels = [a.get_ylabel() for a in fig.axes if a is not ax]
    assert any("probability" in label.lower() for label in labels)
    assert _nonempty(viz.save_figure(fig, tmp_path / "p.png"))


@pytest.mark.parametrize(
    "bad",
    [
        np.full((4, 4), 1.5),
        np.full((4, 4), -0.1),
        np.full((2, 4, 4), 2.0),
        np.full((4, 4), np.nan),
        np.zeros(4),
        np.zeros((2, 2, 2, 2)),
    ],
)
def test_plot_burn_probability_rejects_bad_input(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        viz.plot_burn_probability(bad, dx=30.0)


def test_plot_burn_probability_rejects_bad_dx() -> None:
    with pytest.raises(ValueError):
        viz.plot_burn_probability(np.zeros((3, 3)), dx=0.0)


# ------------------------------------------------------------------ frames_from_arrival


def test_frames_from_arrival_times_and_shapes(result: FakeResult) -> None:
    frames = viz.frames_from_arrival(result, n_frames=7)
    assert len(frames) == 7
    times = [t for t, _ in frames]
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(np.nanmax(result.arrival_time))
    assert all(b > a for a, b in zip(times, times[1:], strict=False))
    for _, cats in frames:
        assert cats.shape == result.grid.shape
        assert np.issubdtype(cats.dtype, np.integer)


def test_frames_from_arrival_initially_unburned_except_ignition(result: FakeResult) -> None:
    _, first = viz.frames_from_arrival(result, n_frames=10)[0]
    fire = (first == viz.DisplayClass.BURNING) | (first == viz.DisplayClass.BURNED)
    np.testing.assert_array_equal(fire, result.arrival_time == 0.0)
    assert fire.sum() == 1


def test_frames_from_arrival_burned_area_non_decreasing(result: FakeResult) -> None:
    frames = viz.frames_from_arrival(result, n_frames=12)
    areas = [int(((c == viz.DisplayClass.BURNING) | (c == viz.DisplayClass.BURNED)).sum()) for _, c in frames]
    assert all(b >= a for a, b in zip(areas, areas[1:], strict=False))
    # by the last frame every cell that ever burned shows as burning or burned
    assert areas[-1] == int(np.isfinite(result.arrival_time).sum())
    # burned-out cells appear once the front has moved on
    assert (frames[-1][1] == viz.DisplayClass.BURNED).any()


def test_frames_from_arrival_keeps_land_use(with_river_result: FakeResult) -> None:
    for _, cats in viz.frames_from_arrival(with_river_result, n_frames=5):
        water = with_river_result.grid.land_use == int(LandUse.WATER)
        assert (cats[water] == viz.DisplayClass.WATER).all()


@pytest.fixture
def with_river_result() -> FakeResult:
    return _radial_result(with_river((31, 31), 30.0))


def test_frames_from_arrival_validates_n_frames(result: FakeResult) -> None:
    with pytest.raises(ValueError):
        viz.frames_from_arrival(result, n_frames=0)


def test_frames_from_arrival_never_burned(small_grid: Grid) -> None:
    res = FakeResult(grid=small_grid, arrival_time=np.full(small_grid.shape, np.nan))
    frames = viz.frames_from_arrival(res, n_frames=3)
    assert [t for t, _ in frames] == [0.0, 0.0, 0.0]
    for _, cats in frames:
        assert (cats == viz.DisplayClass.FUEL).all()


# ------------------------------------------------------------------ animate


def test_animate_writes_gif_with_requested_frames(result: FakeResult, tmp_path: Path) -> None:
    out = viz.animate(result, tmp_path / "fire.gif", n_frames=6, fps=5)
    assert out == tmp_path / "fire.gif"
    assert _nonempty(out)
    frames = iio.imread(out, index=None)
    assert frames.shape[0] == 6


@pytest.mark.parametrize("make", [with_river, with_town])
def test_animate_with_river_and_town(make, tmp_path: Path) -> None:
    res = _radial_result(make((31, 31), 30.0))
    out = viz.animate(res, tmp_path / "fire.gif", n_frames=4)
    assert iio.imread(out, index=None).shape[0] == 4


def test_animate_static_fire_still_has_all_frames(small_grid: Grid, tmp_path: Path) -> None:
    arrival = np.full(small_grid.shape, np.nan)
    arrival[10, 10] = 0.0
    out = viz.animate(FakeResult(grid=small_grid, arrival_time=arrival), tmp_path / "f.gif", n_frames=3)
    assert iio.imread(out, index=None).shape[0] == 3


def test_animate_does_not_change_backend(result: FakeResult, tmp_path: Path) -> None:
    before = matplotlib.get_backend()
    viz.animate(result, tmp_path / "f.gif", n_frames=2)
    assert matplotlib.get_backend() == before


# ------------------------------------------------------------------ save_figure / hygiene


def test_save_figure_accepts_figure_and_creates_parent(tmp_path: Path) -> None:
    ax = viz.plot_state(flat_uniform((5, 5), 30.0))
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    out = viz.save_figure(fig, tmp_path / "sub" / "x.png", dpi=50)
    assert _nonempty(out)


def test_no_plt_show_in_viz() -> None:
    source = (REPO_ROOT / "src" / "wildfire_sim" / "viz.py").read_text(encoding="utf-8")
    assert "plt.show" not in source
    assert "pyplot" not in source
