"""Compiled fire engine (WP 1.7): the per-step update of :mod:`wildfire_sim.sim` in Numba.

Select it with ``engine: numba`` in a scenario (``ScenarioConfig.engine``). It needs the
optional ``fast`` extra (``uv sync --extra fast``); numba is imported only when this engine is
first used, so ``import wildfire_sim`` and the NumPy engine never need it.

**Design.** :func:`wildfire_sim.sim.run` does all the setup (grid, thresholds drawn from the
same seed streams, kernel correction, residence times, weather series, hooks, the initial
ignition) for both engines. :class:`NumbaFireEngine` subclasses the NumPy engine and keeps its
spread-rule cache (:meth:`~wildfire_sim.sim._FireEngine.refresh`: the rule output, link rates
from :func:`~wildfire_sim.spread.base.link_rate_of_spread`, the correction and the
blocked-link mask, all computed by the NumPy code) and its initial-ignition code. It replaces
only :meth:`advance`, the hot per-step update, with one ``@numba.njit`` kernel
(:func:`_step`) working on the same flat per-cell arrays:

1. drop cells a hook extinguished;
2. crossings from the cells burning at the step start (stored progress), then their progress;
3. the in-step cascade, as a Dijkstra search with an indexed binary heap over the cells reached
   in this step (the NumPy engine finds the same shortest-path fixed point by label
   correction);
4. ignition of the cells reached (state, arrival time, flaming clock), their progress and their
   front-normal fireline intensity (the plane fit of :func:`wildfire_sim.sim.front_normal_rate`);
5. burn-out (flaming clocks, live and stalled links, the dormant clock, ``max_source_age_s``).

Every arithmetic expression is the NumPy engine's, element by element, so results agree to
floating-point rounding; only the order in which the cascade visits cells differs. Events are
built in Python from the cells the kernel reports, exactly as the NumPy engine builds them.

Edge detection (WP 1.11) needs nothing here: :func:`wildfire_sim.sim.run` derives
``reached_edge_s`` and the ``"edge"`` event from the arrival times, for both engines alike.

The kernel draws no random numbers. Compiled code is cached on disk (``cache=True``) next to
this file or in ``NUMBA_CACHE_DIR``; when no cache location is writable it compiles in memory.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from wildfire_sim.grid import Grid, Weather
from wildfire_sim.sim import HoldoverTable, ResidenceTimeTable, _FireEngine
from wildfire_sim.spread.base import SpreadRule
from wildfire_sim.types import CellState, FloatArray, IntArray, Stencil
from wildfire_sim.urban import UrbanModel

INSTALL_HINT = (
    "engine 'numba' needs the optional numba package: install the 'fast' extra with "
    "`uv sync --extra fast` (or `pip install 'wildfire-sim[fast]'`), or set `engine: numpy`"
)

_FUEL = int(CellState.FUEL)
_BURNING = int(CellState.BURNING)
_BURNED = int(CellState.BURNED)

_KERNELS: dict[str, Any] | None = None
"""Compiled kernels by name, built on first use by :func:`load_kernels`."""


# ----------------------------------------------------------------------------- kernels (pure Python source)
#
# These are plain Python so they can be read, and run, without numba. load_kernels() compiles
# them and rebinds the module globals _sift_up / _sift_down to the compiled versions before
# compiling _step, which calls them by those names.


def _sift_up_py(heap: IntArray, pos: IntArray, key: FloatArray, i: int) -> None:
    """Move heap entry ``i`` up until its parent's key is not larger (min-heap on ``key[cell]``)."""
    cell = heap[i]
    kc = key[cell]
    while i > 0:
        parent = (i - 1) >> 1
        pc = heap[parent]
        if key[pc] <= kc:
            break
        heap[i] = pc
        pos[pc] = i
        i = parent
    heap[i] = cell
    pos[cell] = i


def _sift_down_py(heap: IntArray, pos: IntArray, key: FloatArray, i: int, size: int) -> None:
    """Move heap entry ``i`` down until no child's key is smaller (heap of ``size`` entries)."""
    cell = heap[i]
    kc = key[cell]
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and key[heap[child + 1]] < key[heap[child]]:
            child += 1
        cc = heap[child]
        if key[cc] >= kc:
            break
        heap[i] = cc
        pos[cc] = i
        i = child
    heap[i] = cell
    pos[cell] = i


_sift_up = _sift_up_py
_sift_down = _sift_down_py


def _step_py(
    burning: IntArray,
    state: np.ndarray,
    burn_timer: np.ndarray,
    intensity: np.ndarray,
    arrival_grid: np.ndarray,
    rate: FloatArray,
    thresholds: FloatArray,
    progress: FloatArray,
    arrival: FloatArray,
    timer: FloatArray,
    timer_init: FloatArray,
    out_intensity: FloatArray,
    blocked: np.ndarray,
    dormant_since: FloatArray,
    holdover: FloatArray,
    front_intensity: FloatArray,
    ros_head: FloatArray,
    heat: FloatArray,
    has_heat: bool,
    speed_scale: FloatArray,
    dx: float,
    dr: IntArray,
    dc: IntArray,
    h: int,
    w: int,
    t: float,
    t_end: float,
    max_age: float,
    best: FloatArray,
    best_v: FloatArray,
    heap: IntArray,
    pos: IntArray,
    touched: IntArray,
) -> tuple[IntArray, IntArray, FloatArray]:
    """One step ``[t, t_end)`` of the fire on flat arrays (cell index ``r * w + c``, ``n = h * w``).

    Args:
        burning: ``(m,)`` int64, cells BURNING at the end of the last step.
        state, burn_timer, intensity, arrival_grid: ``(n,)`` flat views of the grid layers
            ``state`` (int8), ``burn_timer`` (s, float32), ``intensity`` (kW/m, float32) and
            ``arrival_time`` (s, float32); updated in place.
        rate: ``(n, K)`` float64 link progress rate, 1/s (link rate / correction / length, 0 on
            blocked links). Cell-major, the transpose of the NumPy engine's ``(K, n)``, so one
            cell's links are adjacent in memory.
        thresholds: ``(n, K)`` float64 dimensionless crossing thresholds.
        progress: ``(n, K)`` float64 dimensionless link progress; updated in place.
        arrival: ``(n,)`` float64 arrival times, s (NaN = never ignited); updated in place.
        timer: ``(n,)`` float64 remaining flaming time, s; updated in place.
        timer_init: ``(n,)`` float64 flaming time given at ignition, s (``dx / ros_head`` +
            residence time).
        out_intensity: ``(n,)`` float64 head fireline intensity of the rule output, kW/m.
        blocked: ``(n, K)`` bool, True where the link is blocked by barrier geometry (its rate
            is then 0 for good, so it never makes a cell dormant).
        dormant_since: ``(n,)`` float64 time, s, at which each cell became a dormant source;
            NaN while it is not dormant. Updated in place.
        holdover: ``(n,)`` float64 holdover time of each cell's fuel, s.
        front_intensity: ``(n,)`` float64 fireline intensity fixed when a cell ignites, kW/m
            (NaN = never ignited); updated in place.
        ros_head: ``(n,)`` float64 largest directional rate of spread of the rule output, m/s.
        heat: ``(n,)`` float64 heat per unit area of the rule output, kJ/m^2 (read only when
            ``has_heat``).
        has_heat: Whether the rule gave ``heat_per_area``.
        speed_scale: ``(K,)`` float64 link length times front-speed correction, m: ``rate *
            speed_scale`` is the link's rate of spread, m/s.
        dx: Cell size, m.
        dr, dc: ``(K,)`` int64 stencil offsets, rows and columns.
        h, w: Grid shape.
        t, t_end: Step start and end, s.
        max_age: ``max_source_age_s``, s; ``inf`` for no limit.
        best, best_v, heap, pos, touched: ``(n,)`` scratch; ``best`` must be all ``inf`` and
            ``pos`` all ``-1`` on entry, and are left so. ``best_v`` holds the rate, m/s, of the
            link behind each ``best`` entry.

    Returns:
        ``(burning, lit, lit_t)``: int64 cells BURNING after the step; int64 cells ignited in
        this step, ascending; their float64 arrival times, s.
    """
    n_dirs = dr.shape[0]
    inf = np.inf

    # 1. cells a hook extinguished leave the burning set
    src = np.empty(burning.shape[0], dtype=np.int64)
    n_src = 0
    for i in range(burning.shape[0]):
        cell = burning[i]
        if state[cell] == _BURNING:
            src[n_src] = cell
            n_src += 1
        else:
            intensity[cell] = 0.0
            burn_timer[cell] = 0.0

    # 2. crossings from cells burning at the step start, then their progress
    n_touched = 0
    size = 0
    for i in range(n_src):
        s = src[i]
        r = s // w
        c = s - r * w
        a = arrival[s]
        start = max(t, a)
        tau = max(min(t_end, a + max_age) - start, 0.0)
        for k in range(n_dirs):
            tr = r + dr[k]
            tc = c + dc[k]
            if tr < 0 or tr >= h or tc < 0 or tc >= w:
                continue
            rt = rate[s, k]
            if not rt > 0:
                continue
            j = tr * w + tc
            if state[j] != _FUEL:
                continue
            need = max(thresholds[s, k] - progress[s, k], 0.0)
            if rt * tau >= need:
                ct = min(start + need / rt, start + tau)
                if ct == best[j] and rt * speed_scale[k] > best_v[j]:
                    best_v[j] = rt * speed_scale[k]  # same time by a faster link
                if ct < best[j]:
                    if best[j] == inf:
                        touched[n_touched] = j
                        n_touched += 1
                    best[j] = ct
                    best_v[j] = rt * speed_scale[k]
                    if pos[j] < 0:
                        heap[size] = j
                        pos[j] = size
                        size += 1
                        _sift_up(heap, pos, best, size - 1)
                    else:
                        _sift_up(heap, pos, best, pos[j])
        for k in range(n_dirs):
            progress[s, k] += rate[s, k] * tau

    # 3. exact in-step cascade: Dijkstra from the cells reached so far
    while size > 0:
        u = heap[0]
        pos[u] = -2  # settled
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            pos[heap[0]] = 0
            _sift_down(heap, pos, best, 0, size)
        bu = best[u]
        r = u // w
        c = u - r * w
        start = max(t, bu)
        tau = max(min(t_end, bu + max_age) - start, 0.0)
        for k in range(n_dirs):
            tr = r + dr[k]
            tc = c + dc[k]
            if tr < 0 or tr >= h or tc < 0 or tc >= w:
                continue
            rt = rate[u, k]
            if not rt > 0:
                continue
            j = tr * w + tc
            if state[j] != _FUEL:
                continue
            need = thresholds[u, k]
            if rt * tau >= need:
                ct = min(start + need / rt, start + tau)
                if ct == best[j] and rt * speed_scale[k] > best_v[j]:
                    best_v[j] = rt * speed_scale[k]
                if ct < best[j]:
                    if best[j] == inf:
                        touched[n_touched] = j
                        n_touched += 1
                    best[j] = ct
                    best_v[j] = rt * speed_scale[k]
                    if pos[j] == -1:
                        heap[size] = j
                        pos[j] = size
                        size += 1
                        _sift_up(heap, pos, best, size - 1)
                    elif pos[j] >= 0:
                        _sift_up(heap, pos, best, pos[j])

    lit = np.sort(touched[:n_touched])
    lit_t = np.empty(n_touched, dtype=np.float64)
    for i in range(n_touched):
        j = lit[i]
        lit_t[i] = best[j]
        best[j] = inf
        pos[j] = -1

    # 4. ignite the cells reached; their progress starts at 0 at ignition
    for i in range(n_touched):
        j = lit[i]
        tj = lit_t[i]
        state[j] = _BURNING
        arrival[j] = tj
        arrival_grid[j] = tj
        timer[j] = timer_init[j]
        start = max(t, tj)
        tau = max(min(t_end, tj + max_age) - start, 0.0)
        for k in range(n_dirs):
            progress[j, k] = rate[j, k] * tau

    # 4b. front-normal intensity of the new cells: a least-squares plane through the arrival
    # times of each cell and its neighbours that burned no later (the formulas of sim._plane_fit_rate)
    for i in range(n_touched):
        j = lit[i]
        if out_intensity[j] == 0.0:  # a rule without intensity: the cap is 0
            front_intensity[j] = 0.0
            continue
        t0 = arrival[j]
        r = j // w
        c = j - r * w
        n_pts = 1.0
        sx = 0.0
        sy = 0.0
        st = 0.0
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        sxt = 0.0
        syt = 0.0
        for k in range(n_dirs):
            tr = r + dr[k]
            tc = c + dc[k]
            if tr < 0 or tr >= h or tc < 0 or tc >= w:
                continue
            tn = arrival[tr * w + tc]
            if not tn <= t0:  # later, or NaN: not burned yet
                continue
            tau = tn - t0
            x = dc[k] * dx
            y = dr[k] * dx
            n_pts += 1.0
            sx += x
            sy += y
            st += tau
            sxx += x * x
            sxy += x * y
            syy += y * y
            sxt += x * tau
            syt += y * tau
        cxx = sxx - sx * sx / n_pts
        cxy = sxy - sx * sy / n_pts
        cyy = syy - sy * sy / n_pts
        cxt = sxt - sx * st / n_pts
        cyt = syt - sy * st / n_pts
        det = cxx * cyy - cxy * cxy
        head = out_intensity[j]
        if n_pts >= 3.0 and det > 1e-9 * (cxx + cyy) ** 2:
            gx = (cxt * cyy - cyt * cxy) / det
            gy = (cyt * cxx - cxt * cxy) / det
            slowness = np.sqrt(gx * gx + gy * gy)
            normal_rate = 1.0 / slowness if slowness > 0 else inf
        else:
            normal_rate = best_v[j]  # the igniting link's rate
        if normal_rate == inf:
            front_intensity[j] = head
        else:
            if has_heat:
                raw = heat[j] * normal_rate
            elif ros_head[j] > 0:
                raw = head * (normal_rate / ros_head[j])
            else:
                raw = head
            front_intensity[j] = min(raw, head)

    # 5. burn-out over the old sources, then the new cells (the NumPy engine's order)
    out = np.empty(n_src + n_touched, dtype=np.int64)
    n_out = 0
    for i in range(n_src + n_touched):
        s = src[i] if i < n_src else lit[i - n_src]
        a = arrival[s]
        timer[s] = max(timer[s] - (t_end - max(t, a)), 0.0)
        source = False
        stalled = False
        r = s // w
        c = s - r * w
        for k in range(n_dirs):
            tr = r + dr[k]
            tc = c + dc[k]
            if tr < 0 or tr >= h or tc < 0 or tc >= w:
                continue
            if state[tr * w + tc] != _FUEL:
                continue
            if rate[s, k] > 0:
                source = True
                break
            if not blocked[s, k]:
                stalled = True
        young = (t_end - a) < max_age
        source = source and young
        flaming = timer[s] > 0
        # dormant: a stalled link (rate 0 now, not blocked) may come back; wait up to the holdover
        dormant = stalled and young and not source and not flaming
        if dormant:
            if np.isnan(dormant_since[s]):
                dormant_since[s] = t_end
            dormant = not (t_end - dormant_since[s]) >= holdover[s]
        else:
            dormant_since[s] = np.nan
        if source or flaming or dormant:
            burn_timer[s] = timer[s]
            intensity[s] = front_intensity[s] if flaming else 0.0
            out[n_out] = s
            n_out += 1
        else:
            state[s] = _BURNED
            burn_timer[s] = 0.0
            intensity[s] = 0.0
    return out[:n_out].copy(), lit, lit_t


# ----------------------------------------------------------------------------- compilation


def _jit(numba: Any, func: Any) -> Any:
    """``numba.njit(cache=True)(func)``, compiling in memory when no cache location is writable."""
    try:
        return numba.njit(cache=True)(func)
    except RuntimeError:  # "cannot cache function ...: no locator available"
        return numba.njit(cache=False)(func)


def load_kernels() -> dict[str, Any]:
    """Import numba and compile (or load from cache) the kernels; cached for the process.

    Raises:
        ImportError: numba is not installed (message says how to install the ``fast`` extra).
    """
    global _KERNELS
    if _KERNELS is None:
        try:
            numba = importlib.import_module("numba")
        except ImportError as exc:
            raise ImportError(INSTALL_HINT) from exc
        module = globals()
        # helpers first: _step looks them up by these global names when it is compiled
        module["_sift_up"] = _jit(numba, _sift_up_py)
        module["_sift_down"] = _jit(numba, _sift_down_py)
        _KERNELS = {"step": _jit(numba, _step_py)}
    return _KERNELS


def numba_engine_class() -> type[NumbaFireEngine]:
    """:class:`NumbaFireEngine`, after making sure numba is importable (see :func:`load_kernels`)."""
    load_kernels()
    return NumbaFireEngine


# ----------------------------------------------------------------------------- engine


def _flat(layer: np.ndarray) -> np.ndarray:
    """``(H * W,)`` view of a C-contiguous ``(H, W)`` layer (a copy otherwise; see :func:`_write_back`)."""
    return np.ascontiguousarray(layer).reshape(-1)


def _write_back(layer: np.ndarray, flat: np.ndarray) -> None:
    """Copy ``flat`` into ``layer`` when :func:`_flat` had to copy."""
    if not np.may_share_memory(layer, flat):
        layer[...] = flat.reshape(layer.shape)


class NumbaFireEngine(_FireEngine):
    """The NumPy engine with :meth:`advance` replaced by the compiled kernel :func:`_step`.

    Same constructor and state as :class:`wildfire_sim.sim._FireEngine`, except that the
    per-link arrays the kernel reads (``thresholds``, ``progress`` and the link rates) are
    stored cell-major, ``(H * W, K)``, so one cell's links share a cache line. Use it through
    :func:`wildfire_sim.sim.run` with ``engine: numba``.
    """

    def __init__(
        self,
        grid: Grid,
        rule: SpreadRule,
        stencil: Stencil,
        thresholds: FloatArray,
        correction: float,
        residence: ResidenceTimeTable,
        max_source_age_s: float | None,
        record_events: bool,
        holdover: HoldoverTable,
        urban: UrbanModel,
    ) -> None:
        super().__init__(
            grid, rule, stencil, thresholds, correction, residence, max_source_age_s, record_events, holdover, urban
        )
        self._step = load_kernels()["step"]
        n = self.h * self.w
        self.burning = np.empty(0, dtype=np.int64)
        # Per-link arrays are cell-major (n, K) here: the same values, transposed.
        self.thresholds = np.ascontiguousarray(self.thresholds.T)
        self.progress = np.zeros((n, self.k), dtype=np.float64)
        self._rate_nk = np.zeros((n, self.k), dtype=np.float64)
        self._blocked_nk = np.zeros((n, self.k), dtype=np.bool_)
        self._dr = np.ascontiguousarray(stencil.offsets[:, 0], dtype=np.int64)
        self._dc = np.ascontiguousarray(stencil.offsets[:, 1], dtype=np.int64)
        self._max_age = np.inf if max_source_age_s is None else float(max_source_age_s)
        self._timer_init = np.zeros(n, dtype=np.float64)
        self._out_intensity = np.zeros(n, dtype=np.float64)
        self._best = np.full(n, np.inf, dtype=np.float64)
        self._best_v = np.zeros(n, dtype=np.float64)
        self._heat = np.zeros(n, dtype=np.float64)
        self._speed_scale = np.ascontiguousarray(self.link_speed_scale, dtype=np.float64)
        self._heap = np.empty(n, dtype=np.int64)
        self._pos = np.full(n, -1, dtype=np.int64)
        self._touched = np.empty(n, dtype=np.int64)

    def refresh(self, weather: Weather, changed: bool) -> None:
        """As the NumPy engine; also caches the per-cell flaming time and intensity of the new output."""
        before = self.out
        super().refresh(weather, changed)
        if self.out is before:
            return
        out = self.out
        assert out is not None
        grid = self.grid
        # The NumPy engine's ignite() formula, evaluated for every cell at once.
        ros_head = out.ros.max(axis=0)
        crossing = np.divide(grid.dx, ros_head, out=np.zeros_like(ros_head, dtype=np.float64), where=ros_head > 0)
        residence = self.residence.lookup(grid.fuel_model).reshape(-1)
        if self.has_urban:  # urban cells flame longer (their rates are already scaled in ``out``)
            residence = residence * self.residence_scale
        self._timer_init = np.ascontiguousarray(crossing.reshape(-1) + residence)
        self._out_intensity = self.head_intensity
        self._heat = np.zeros(self.h * self.w, dtype=np.float64) if self.heat is None else self.heat
        self._rate_nk = np.ascontiguousarray(self.rate.T)
        self._blocked_nk = np.ascontiguousarray(self.blocked.reshape(self.k, -1).T)
        self.holdover_s = np.ascontiguousarray(self.holdover_s, dtype=np.float64)

    def advance(self, t: float, t_end: float) -> None:
        """Spread over the step ``[t, t_end)`` with the compiled kernel (see :func:`_step_py`)."""
        grid = self.grid
        layers = (grid.state, grid.burn_timer, grid.intensity, grid.arrival_time)
        flats = [_flat(layer) for layer in layers]
        burning, lit, lit_t = self._step(
            np.ascontiguousarray(self.burning, dtype=np.int64),
            *flats,
            self._rate_nk,
            self.thresholds,
            self.progress,
            self.arrival,
            self.timer,
            self._timer_init,
            self._out_intensity,
            self._blocked_nk,
            self.dormant_since,
            self.holdover_s,
            self.front_intensity,
            self.ros_head,
            self._heat,
            self.heat is not None,
            self._speed_scale,
            float(grid.dx),
            self._dr,
            self._dc,
            self.h,
            self.w,
            float(t),
            float(t_end),
            self._max_age,
            self._best,
            self._best_v,
            self._heap,
            self._pos,
            self._touched,
        )
        for layer, flat in zip(layers, flats, strict=True):
            _write_back(layer, flat)
        self.burning = burning
        if self.record_events and lit.size:
            rows, cols = np.divmod(lit, self.w)
            step_events = [
                (float(ti), "spread", int(r), int(c))
                for ti, r, c in zip(lit_t.tolist(), rows.tolist(), cols.tolist(), strict=True)
            ]
            step_events.sort()
            self.events.extend(step_events)


__all__ = ["INSTALL_HINT", "NumbaFireEngine", "load_kernels", "numba_engine_class"]
