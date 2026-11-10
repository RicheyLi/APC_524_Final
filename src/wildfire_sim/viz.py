"""Maps, burn-probability plots and GIF animations of simulation results.

Everything here uses matplotlib's object-oriented API on an off-screen Agg canvas
(:class:`matplotlib.figure.Figure` + :class:`~matplotlib.backends.backend_agg.FigureCanvasAgg`).
No function opens a window, touches matplotlib's global figure manager or changes the
user's backend, so the module works on headless machines and inside tests. Every plotting
function accepts an optional ``ax`` to draw on and returns the :class:`~matplotlib.axes.Axes`
it used; :func:`save_figure` writes it to disk.

Orientation
-----------
North is up: row 0 of every ``(H, W)`` layer is drawn at the top, column 0 at the left
(west). Axes show distance from the south-west corner of the grid, in metres for grids
smaller than 1 km and in kilometres otherwise, with equal aspect so cells are square.

Colour choices
--------------
Categorical maps use the Okabe-Ito palette (Okabe & Ito 2008, "Color Universal Design"),
whose eight colours stay distinguishable under the common forms of colour-vision deficiency
(protanopia, deuteranopia, tritanopia). The one mapping, :data:`CATEGORY_COLOURS`, is keyed
on :class:`DisplayClass`; :data:`STATE_COLOURS` and :data:`LAND_USE_COLOURS` are views of it
keyed on :class:`~wildfire_sim.types.CellState` and :class:`~wildfire_sim.types.LandUse`.
Burned cells are black and burning cells vermillion, so burned area also reads correctly in
greyscale. Continuous quantities use perceptually uniform sequential maps from matplotlib:
``viridis`` for arrival time and ``cividis`` for burn probability, each with a labelled
colour bar stating units. On the arrival map firelines are drawn black rather than yellow,
since yellow is the late end of ``viridis``.

Results
-------
The simulator's ``RunResult`` is not imported here; :func:`plot_arrival`,
:func:`frames_from_arrival` and :func:`animate` accept anything matching the
:class:`HasArrival` protocol (a ``grid`` and an ``arrival_time`` array).
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Protocol

import imageio.v3 as iio
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Patch

from wildfire_sim.grid import Grid
from wildfire_sim.types import CellState, FloatArray, IntArray, LandUse

__all__ = [
    "ARRIVAL_CMAP",
    "ARRIVAL_FIRELINE_COLOUR",
    "CATEGORY_COLOURS",
    "CATEGORY_LABELS",
    "LAND_USE_COLOURS",
    "PROBABILITY_CMAP",
    "RETARDANT_COLOUR",
    "STATE_COLOURS",
    "UNBURNED_COLOUR",
    "DisplayClass",
    "HasArrival",
    "animate",
    "classify_grid",
    "frames_from_arrival",
    "plot_arrival",
    "plot_burn_probability",
    "plot_state",
    "save_figure",
]


class HasArrival(Protocol):
    """What the viz functions need from a simulation result (e.g. ``wildfire_sim.sim.RunResult``).

    Attributes:
        grid: The final :class:`~wildfire_sim.grid.Grid` of the run (supplies ``dx`` and land use).
        arrival_time: ``(H, W)`` float array, seconds since the start of the run at which each
            cell ignited; NaN where it never burned.
    """

    grid: Grid
    arrival_time: FloatArray


class DisplayClass(IntEnum):
    """What a cell is drawn as on a categorical map. Combines fire state, land use and firelines."""

    FUEL = 0
    """Unburned, burnable wildland fuel."""
    BURNING = 1
    BURNED = 2
    WATER = 3
    URBAN = 4
    ROAD = 5
    FIRELINE = 6
    """A completed control line (``Grid.fireline``) on wildland."""
    OTHER_UNBURNABLE = 7
    """Unburnable for any other reason, e.g. fuel model 0 (bare rock, barren ground)."""


# Okabe-Ito palette; the light grey for "other unburnable" is a neutral outside the palette.
CATEGORY_COLOURS: dict[DisplayClass, str] = {
    DisplayClass.FUEL: "#009E73",  # bluish green
    DisplayClass.BURNING: "#D55E00",  # vermillion
    DisplayClass.BURNED: "#000000",  # black
    DisplayClass.WATER: "#0072B2",  # blue
    DisplayClass.URBAN: "#CC79A7",  # reddish purple
    DisplayClass.ROAD: "#999999",  # mid grey
    DisplayClass.FIRELINE: "#F0E442",  # yellow
    DisplayClass.OTHER_UNBURNABLE: "#DDDDDD",  # light grey
}
"""Colour of each :class:`DisplayClass` (hex RGB). The single source of categorical colours."""

CATEGORY_LABELS: dict[DisplayClass, str] = {
    DisplayClass.FUEL: "Unburned fuel",
    DisplayClass.BURNING: "Burning",
    DisplayClass.BURNED: "Burned",
    DisplayClass.WATER: "Water",
    DisplayClass.URBAN: "Urban",
    DisplayClass.ROAD: "Road",
    DisplayClass.FIRELINE: "Fireline",
    DisplayClass.OTHER_UNBURNABLE: "Other unburnable",
}
"""Legend label of each :class:`DisplayClass`."""

STATE_COLOURS: dict[CellState, str] = {
    CellState.FUEL: CATEGORY_COLOURS[DisplayClass.FUEL],
    CellState.BURNING: CATEGORY_COLOURS[DisplayClass.BURNING],
    CellState.BURNED: CATEGORY_COLOURS[DisplayClass.BURNED],
    CellState.UNBURNABLE: CATEGORY_COLOURS[DisplayClass.OTHER_UNBURNABLE],
}
"""Colour of each :class:`~wildfire_sim.types.CellState`, taken from :data:`CATEGORY_COLOURS`."""

LAND_USE_COLOURS: dict[LandUse, str] = {
    LandUse.WILDLAND: CATEGORY_COLOURS[DisplayClass.FUEL],
    LandUse.URBAN: CATEGORY_COLOURS[DisplayClass.URBAN],
    LandUse.WATER: CATEGORY_COLOURS[DisplayClass.WATER],
    LandUse.ROAD: CATEGORY_COLOURS[DisplayClass.ROAD],
}
"""Colour of each :class:`~wildfire_sim.types.LandUse`, taken from :data:`CATEGORY_COLOURS`."""

RETARDANT_COLOUR = "#E69F00"
"""Okabe-Ito orange; retardant is overlaid in this colour with opacity proportional to strength."""

ARRIVAL_CMAP = "viridis"
"""Perceptually uniform sequential colour map for arrival time."""

PROBABILITY_CMAP = "cividis"
"""Perceptually uniform sequential colour map for burn probability (also colour-blind optimised)."""

UNBURNED_COLOUR = "#DDDDDD"
"""Neutral light grey for wildland that never burned, under the arrival-time map."""

ARRIVAL_FIRELINE_COLOUR = "#000000"
"""Firelines on the arrival-time map: black, because the categorical yellow clashes with viridis."""

_N_CLASSES = len(DisplayClass)
_CATEGORY_CMAP = ListedColormap([CATEGORY_COLOURS[c] for c in DisplayClass], name="wildfire_categories")
_CATEGORY_NORM = BoundaryNorm(np.arange(_N_CLASSES + 1) - 0.5, _N_CLASSES)


# ------------------------------------------------------------------ helpers


def _new_axes(figsize: tuple[float, float] = (7.5, 5.5), dpi: float = 100.0) -> Axes:
    """A fresh Axes on a new off-screen Agg figure (not registered with any global figure manager)."""
    fig = Figure(figsize=figsize, dpi=dpi, layout="constrained")
    FigureCanvasAgg(fig)
    return fig.add_subplot()


def _parent(ax: Axes) -> Figure | SubFigure:
    fig = ax.get_figure()
    if fig is None:  # an Axes always belongs to a figure once created via add_subplot
        raise ValueError("Axes is not attached to a figure")
    return fig


def _distance_scale(shape: tuple[int, int], dx: float) -> tuple[float, str]:
    """Divisor and unit name for the axis coordinates: km for grids >= 1 km across, else m."""
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx!r}")
    return (1000.0, "km") if max(shape) * dx >= 1000.0 else (1.0, "m")


def _extent(shape: tuple[int, int], dx: float) -> tuple[tuple[float, float, float, float], float, str]:
    """``imshow`` extent (left, right, bottom, top) with the SW corner at the origin, plus the scale."""
    scale, unit = _distance_scale(shape, dx)
    h, w = shape
    return (0.0, w * dx / scale, 0.0, h * dx / scale), scale, unit


def _decorate(ax: Axes, unit: str, title: str | None) -> None:
    ax.set_xlabel(f"Easting from west edge ({unit})")
    ax.set_ylabel(f"Northing from south edge ({unit})")
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)


def _show_categories(ax: Axes, cats: IntArray, dx: float) -> str:
    """Draw a ``(H, W)`` :class:`DisplayClass` array north-up; return the axis unit."""
    extent, _, unit = _extent((int(cats.shape[0]), int(cats.shape[1])), dx)
    ax.imshow(
        cats,
        cmap=_CATEGORY_CMAP,
        norm=_CATEGORY_NORM,
        origin="upper",
        extent=extent,
        interpolation="nearest",
    )
    return unit


def _legend_handles(classes: list[DisplayClass]) -> list[Patch]:
    return [
        Patch(facecolor=CATEGORY_COLOURS[c], edgecolor="#555555", linewidth=0.5, label=CATEGORY_LABELS[c])
        for c in classes
    ]


def _place_legend(ax: Axes, handles: list[Patch], below: bool = False) -> None:
    """Legend outside the map (right of it, or below it in one row) so it never hides cells."""
    if below:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncols=min(len(handles), 4),
            borderaxespad=0.0,
            fontsize="small",
        )
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize="small")


def _base_categories(grid: Grid) -> IntArray:
    """Classes from static layers only (land use, fireline, unburnable fuel), ignoring fire state."""
    cats = np.full(grid.shape, int(DisplayClass.FUEL), dtype=np.int8)
    other = ~((grid.land_use == int(LandUse.WILDLAND)) & (grid.fuel_model > 0))
    cats[other | (grid.state == int(CellState.UNBURNABLE))] = int(DisplayClass.OTHER_UNBURNABLE)
    cats[grid.fireline] = int(DisplayClass.FIRELINE)
    cats[grid.land_use == int(LandUse.ROAD)] = int(DisplayClass.ROAD)
    cats[grid.land_use == int(LandUse.URBAN)] = int(DisplayClass.URBAN)
    cats[grid.land_use == int(LandUse.WATER)] = int(DisplayClass.WATER)
    return cats


def _format_time(t_s: float, unit: str) -> str:
    return f"{t_s / 3600.0:.2f} h" if unit == "h" else f"{t_s / 60.0:.1f} min"


def _time_unit(max_s: float) -> tuple[float, str]:
    """Divisor and unit for displaying times up to ``max_s`` seconds: minutes below 3 h, else hours."""
    return (3600.0, "h") if max_s >= 3 * 3600.0 else (60.0, "min")


# ------------------------------------------------------------------ public API


def classify_grid(grid: Grid) -> IntArray:
    """Classify every cell of ``grid`` into a :class:`DisplayClass`.

    Precedence: fire state (burning, burned) over land use (water, urban, road) over fireline
    over other unburnable cells; the rest is unburned fuel.

    Args:
        grid: Grid with layers of shape ``(H, W)``.

    Returns:
        ``(H, W)`` ``int8`` array of :class:`DisplayClass` codes.
    """
    cats = _base_categories(grid)
    cats[grid.state == int(CellState.BURNED)] = int(DisplayClass.BURNED)
    cats[grid.state == int(CellState.BURNING)] = int(DisplayClass.BURNING)
    return cats


def plot_state(grid: Grid, ax: Axes | None = None, title: str | None = "Fire state") -> Axes:
    """Categorical map of what each cell is: fuel, burning, burned, water, urban, road, fireline, other.

    Retardant (``grid.retardant``, strength in [0, 1]) is overlaid in :data:`RETARDANT_COLOUR`
    with opacity proportional to strength wherever it is > 0. North is up; axes are distances
    from the south-west corner in m or km (from ``grid.dx``, m per cell), with equal aspect.

    Args:
        grid: Grid with layers of shape ``(H, W)``.
        ax: Axes to draw on; a new off-screen figure is created if None.
        title: Axes title, or None for no title.

    Returns:
        The Axes drawn on. The legend lists every class (and retardant if present).
    """
    if ax is None:
        ax = _new_axes()
    cats = classify_grid(grid)
    unit = _show_categories(ax, cats, grid.dx)
    handles = _legend_handles(list(DisplayClass))
    if np.any(grid.retardant > 0):
        extent, _, _ = _extent(grid.shape, grid.dx)
        rgba = np.zeros((*grid.shape, 4), dtype=np.float64)
        rgba[..., :3] = to_rgba(RETARDANT_COLOUR)[:3]
        rgba[..., 3] = 0.7 * np.clip(grid.retardant, 0.0, 1.0)
        ax.imshow(rgba, origin="upper", extent=extent, interpolation="nearest")
        handles.append(Patch(facecolor=RETARDANT_COLOUR, alpha=0.7, edgecolor="#555555", label="Retardant"))
    _place_legend(ax, handles)
    _decorate(ax, unit, title)
    return ax


def plot_arrival(
    result: HasArrival,
    ax: Axes | None = None,
    contours: bool = True,
    n_contours: int = 6,
    title: str | None = "Fire arrival time",
) -> Axes:
    """Map of fire arrival time with a labelled colour bar and optional isochrone contours.

    Time is shown in minutes, or in hours when the last arrival is >= 3 h, using the
    :data:`ARRIVAL_CMAP` sequential map. Wildland that never burned is drawn in a neutral
    grey and non-burnable land use (water, urban, road, fireline) in its categorical colour
    underneath, so barriers stay visible. North up, equal aspect, axes in m or km.

    Args:
        result: Object with ``grid`` (supplies ``dx``, m, and land use) and ``arrival_time``,
            an ``(H, W)`` float array of ignition times in s, NaN where never burned.
        ax: Axes to draw on; a new off-screen figure is created if None.
        contours: Draw ``n_contours`` evenly spaced isochrones (lines of equal arrival time).
        n_contours: Number of isochrone levels if ``contours`` is True.
        title: Axes title, or None for no title.

    Returns:
        The Axes drawn on (its figure also holds the colour bar).
    """
    grid = result.grid
    arrival = np.asarray(result.arrival_time, dtype=np.float64)
    if arrival.shape != grid.shape:
        raise ValueError(f"arrival_time has shape {arrival.shape}, expected {grid.shape}")
    if ax is None:
        ax = _new_axes()
    extent, scale, unit = _extent(grid.shape, grid.dx)

    base = _base_categories(grid)
    base_colours = dict(CATEGORY_COLOURS)
    base_colours[DisplayClass.FUEL] = UNBURNED_COLOUR
    # The categorical fireline yellow would read as a late arrival on viridis; nothing is drawn as
    # "burned" on this map, so black is free and unambiguous for firelines here.
    base_colours[DisplayClass.FIRELINE] = ARRIVAL_FIRELINE_COLOUR
    base_cmap = ListedColormap([base_colours[c] for c in DisplayClass])
    ax.imshow(base, cmap=base_cmap, norm=_CATEGORY_NORM, origin="upper", extent=extent, interpolation="nearest")

    finite = np.isfinite(arrival)
    max_s = float(arrival[finite].max()) if finite.any() else 0.0
    t_scale, t_unit = _time_unit(max_s)
    shown = np.ma.masked_invalid(arrival / t_scale)
    vmax = max(max_s / t_scale, 1e-9)
    image = ax.imshow(
        shown, cmap=ARRIVAL_CMAP, vmin=0.0, vmax=vmax, origin="upper", extent=extent, interpolation="nearest"
    )
    colourbar = _parent(ax).colorbar(image, ax=ax, shrink=0.85)
    colourbar.set_label(f"Arrival time ({t_unit})")

    if contours and n_contours > 0 and finite.sum() >= 4 and max_s > 0 and min(grid.shape) >= 2:
        h, w = grid.shape
        xs = (np.arange(w) + 0.5) * grid.dx / scale
        ys = (h - np.arange(h) - 0.5) * grid.dx / scale
        levels = np.linspace(0.0, max_s / t_scale, n_contours + 2)[1:-1]
        lines = ax.contour(xs, ys, shown, levels=levels, colors="white", linewidths=0.8)
        ax.clabel(lines, fmt="%.3g", fontsize="x-small")

    present = [c for c in DisplayClass if c not in (DisplayClass.FUEL, DisplayClass.BURNING, DisplayClass.BURNED)]
    present = [c for c in present if np.any(base == int(c))]
    if present:
        handles = [
            Patch(facecolor=base_colours[c], edgecolor="#555555", linewidth=0.5, label=CATEGORY_LABELS[c])
            for c in present
        ]
        _place_legend(ax, handles, below=True)
    _decorate(ax, unit, title)
    return ax


def plot_burn_probability(
    stack_or_probability: np.ndarray,
    dx: float,
    ax: Axes | None = None,
    title: str | None = "Burn probability",
) -> Axes:
    """Map of the per-cell probability of burning across an ensemble.

    The colour map (:data:`PROBABILITY_CMAP`) is fixed to [0, 1] so maps from different
    ensembles are comparable. North up, equal aspect, axes in m or km.

    Args:
        stack_or_probability: Either an ``(n, H, W)`` stack of burned masks (bool or values
            in [0, 1]; the mean over axis 0 is plotted) or an ``(H, W)`` probability array
            with values in [0, 1].
        dx: Cell size, m.
        ax: Axes to draw on; a new off-screen figure is created if None.
        title: Axes title, or None for no title.

    Returns:
        The Axes drawn on (its figure also holds the colour bar).

    Raises:
        ValueError: If the array is not 2-D or 3-D, contains NaN or values outside [0, 1],
            or ``dx`` is not positive.
    """
    arr = np.asarray(stack_or_probability)
    if arr.ndim not in (2, 3):
        raise ValueError(f"expected an (n, H, W) stack or an (H, W) probability array, got {arr.ndim} dimensions")
    values = arr.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("burn probability input must be finite")
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        raise ValueError("burn probability input must lie in [0, 1]")
    prob = values.mean(axis=0) if values.ndim == 3 else values
    shape = (int(prob.shape[0]), int(prob.shape[1]))
    extent, _, unit = _extent(shape, dx)  # validates dx
    if ax is None:
        ax = _new_axes()
    image = ax.imshow(
        prob, cmap=PROBABILITY_CMAP, vmin=0.0, vmax=1.0, origin="upper", extent=extent, interpolation="nearest"
    )
    colourbar = _parent(ax).colorbar(image, ax=ax, shrink=0.85)
    colourbar.set_label("Burn probability (fraction of runs, 0-1)")
    _decorate(ax, unit, title)
    return ax


def frames_from_arrival(
    result: HasArrival,
    n_frames: int,
    burning_window_s: float | None = None,
) -> list[tuple[float, IntArray]]:
    """Reconstruct display states at evenly spaced times from a run's arrival times.

    Frame times are ``n_frames`` evenly spaced values from 0 to the last arrival time (all 0
    if nothing burned). At time ``t`` a cell with arrival time ``a`` is:

    * :attr:`DisplayClass.BURNING` if ``t - window < a <= t`` (ignited recently),
    * :attr:`DisplayClass.BURNED` if ``a <= t - window``,
    * otherwise its static class from land use, fireline and fuel (see :func:`classify_grid`).

    The "recent" ``window`` defaults to one frame interval, ``t_last / (n_frames - 1)``, so
    each cell shows as burning in exactly one frame after it ignites. This is a display
    convention, not the simulated residence time. Cells with arrival time 0 (the ignition)
    are burning in the first frame.

    Args:
        result: Object with ``grid`` and ``arrival_time``, an ``(H, W)`` float array of
            ignition times in s, NaN where never burned.
        n_frames: Number of frames, >= 1.
        burning_window_s: Override for the burning window, s (> 0).

    Returns:
        List of ``(t, classes)``: ``t`` in s since the start of the run and ``classes`` an
        ``(H, W)`` ``int8`` array of :class:`DisplayClass` codes.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if burning_window_s is not None and burning_window_s <= 0:
        raise ValueError("burning_window_s must be positive")
    arrival = np.asarray(result.arrival_time, dtype=np.float64)
    if arrival.shape != result.grid.shape:
        raise ValueError(f"arrival_time has shape {arrival.shape}, expected {result.grid.shape}")
    finite = np.isfinite(arrival)
    t_last = float(arrival[finite].max()) if finite.any() else 0.0
    times = np.linspace(0.0, t_last, n_frames)
    if burning_window_s is not None:
        window = burning_window_s
    elif n_frames > 1 and t_last > 0:
        window = t_last / (n_frames - 1)
    else:
        window = np.inf
    base = _base_categories(result.grid)
    # NaN compares False, so never-burned cells keep their static class.
    filled = np.where(finite, arrival, np.inf)
    frames: list[tuple[float, IntArray]] = []
    for t in times:
        cats = base.copy()
        ignited = filled <= t
        cats[ignited & (filled <= t - window)] = int(DisplayClass.BURNED)
        cats[ignited & (filled > t - window)] = int(DisplayClass.BURNING)
        frames.append((float(t), cats))
    return frames


def animate(
    result: HasArrival,
    path: str | Path,
    n_frames: int = 40,
    fps: float = 10.0,
    dpi: float = 100.0,
) -> Path:
    """Write a GIF of the fire's progress reconstructed from arrival times.

    Frames come from :func:`frames_from_arrival`; each is labelled with its simulated time
    (and frame number) and drawn off-screen on an Agg canvas, so no display is needed and
    the global matplotlib backend is untouched. The GIF loops forever.

    Args:
        result: Object with ``grid`` (``dx`` in m) and ``arrival_time``, an ``(H, W)`` float
            array of ignition times in s, NaN where never burned.
        path: Output ``.gif`` path; parent directories are created.
        n_frames: Number of frames, >= 1.
        fps: Frames per second of playback, > 0.
        dpi: Resolution of each frame, dots per inch (frames are 7.5 x 5.5 inches).

    Returns:
        The path written.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    out = Path(path)
    frames = frames_from_arrival(result, n_frames)
    t_last = frames[-1][0]
    _, t_unit = _time_unit(t_last)

    fig = Figure(figsize=(7.5, 5.5), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    extent, _, unit = _extent(result.grid.shape, result.grid.dx)
    image = ax.imshow(
        frames[0][1],
        cmap=_CATEGORY_CMAP,
        norm=_CATEGORY_NORM,
        origin="upper",
        extent=extent,
        interpolation="nearest",
    )
    _place_legend(ax, _legend_handles(list(DisplayClass)))
    # A placeholder title of typical length reserves room for the time label before the layout is frozen.
    _decorate(ax, unit, f"t = {_format_time(t_last, t_unit)}  (frame {len(frames)}/{len(frames)})")
    # Freeze the layout after the first draw so every frame has identical size and placement.
    canvas.draw()
    fig.set_layout_engine("none")

    rgb_frames: list[np.ndarray] = []
    for i, (t, cats) in enumerate(frames):
        image.set_data(cats)
        ax.set_title(f"t = {_format_time(t, t_unit)}  (frame {i + 1}/{len(frames)})")
        canvas.draw()
        rgb_frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())

    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, np.stack(rgb_frames), extension=".gif", duration=1000.0 / fps, loop=0)
    return out


def save_figure(ax_or_fig: Axes | Figure | SubFigure, path: str | Path, dpi: float = 150.0) -> Path:
    """Save the figure holding ``ax_or_fig`` to ``path`` (format from the suffix, e.g. ``.png``).

    Args:
        ax_or_fig: An Axes, Figure or SubFigure; the whole top-level figure is saved.
        path: Output file path; parent directories are created.
        dpi: Resolution, dots per inch.

    Returns:
        The path written.
    """
    out = Path(path)
    if isinstance(ax_or_fig, Axes):
        fig = ax_or_fig.get_figure(root=True)
        if fig is None:
            raise ValueError("Axes is not attached to a figure")
    elif isinstance(ax_or_fig, SubFigure):
        fig = ax_or_fig.get_figure(root=True)
    else:
        fig = ax_or_fig
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    return out
