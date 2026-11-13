"""Tests for edge detection (WP 1.11, docs/plan.md section 2).

A fire that reaches the edge of its grid can no longer be trusted: the model cannot say what
would have burned beyond it. Every run therefore records whether and when it reached the *edge
band*, the cells within ``stencil.max_offset`` cells of the grid edge. Those are exactly the
cells with a stencil link that leaves the grid, so the band is one cell wide for 8 neighbours
and two for 16.

**The acceptance run.** WP 1.11 asks that ``flat_no_wind`` run for 7200 s reports the time it
reached the edge. With the defaults set after the lattice study (16 neighbours, shape 200) it
does not get there in 7200 s: the band starts 49 cells (1470 m) from the ignition, which a
0.2 m/s front reaches at 7350 s; measured 7403 s. At 7200 s the front is 47 cells out (column
3) and no burned cell has a link out of the grid, so None is the right answer. The tests below therefore
run for 7500 s and check the 7200 s run reports None.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from wildfire_sim import io
from wildfire_sim.cli import app
from wildfire_sim.config import ScenarioConfig, load_scenario
from wildfire_sim.ensemble import run_ensemble
from wildfire_sim.sim import edge_band_mask, first_edge_arrival, run
from wildfire_sim.types import STENCIL_8, STENCIL_16

SCENARIOS = Path(__file__).resolve().parent.parent / "configs" / "scenarios"
runner = CliRunner()

# flat_no_wind: ignition at (50, 50), band = columns/rows 0-1 and 99-100, so 49 cells * 30 m at 0.2 m/s
EDGE_TIME_NOMINAL_S = 49 * 30.0 / 0.2
EDGE_TIME_RTOL = 0.03  # the front-speed tolerance of physics test (f)


@pytest.fixture(params=["numpy", "numba"])
def engine(request: pytest.FixtureRequest) -> str:
    if request.param == "numba" and importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    return request.param


def flat_no_wind(duration: float, engine: str = "numpy") -> ScenarioConfig:
    cfg = load_scenario(SCENARIOS / "flat_no_wind.yaml")
    return cfg.model_copy(update={"duration_s": duration, "engine": engine})


def small(duration: float, *, stencil: int = 16, ensemble: dict[str, Any] | None = None) -> ScenarioConfig:
    """21 x 21 cells, centre ignition, 0.2 m/s: the band is reached after about 20 to 25 minutes."""
    return ScenarioConfig.model_validate(
        {
            "name": "small",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [21, 21]},
            "spread": {"rule": "constant", "params": {"ros": 0.2}},
            "ignition": {"center": True},
            "duration_s": duration,
            "stencil": stencil,
            "ensemble": ensemble or {},
        }
    )


# ----------------------------------------------------------------------------- helpers


def test_edge_band_mask() -> None:
    band = edge_band_mask((6, 7), 2)
    expected = np.ones((6, 7), dtype=bool)
    expected[2:4, 2:5] = False
    np.testing.assert_array_equal(band, expected)
    assert edge_band_mask((5, 5), 1).sum() == 16
    assert not edge_band_mask((5, 5), 0).any()
    assert edge_band_mask((4, 9), 2).all()  # no interior left
    assert edge_band_mask((1, 1), 1).all()
    with pytest.raises(ValueError, match="width"):
        edge_band_mask((5, 5), -1)


def test_edge_band_is_the_cells_with_a_link_leaving_the_grid() -> None:
    for stencil in (STENCIL_8, STENCIL_16):
        h, w = 9, 11
        rows, cols = np.mgrid[0:h, 0:w]
        leaves = np.zeros((h, w), dtype=bool)
        for dr, dc in stencil.offsets.tolist():
            leaves |= (rows + dr < 0) | (rows + dr >= h) | (cols + dc < 0) | (cols + dc >= w)
        np.testing.assert_array_equal(edge_band_mask((h, w), stencil.max_offset), leaves)


def test_first_edge_arrival() -> None:
    arrival = np.full((7, 7), np.nan)
    assert first_edge_arrival(arrival, STENCIL_8) is None
    arrival[3, 3] = 0.0
    arrival[3, 2] = 10.0
    arrival[3, 1] = 20.0
    assert first_edge_arrival(arrival, STENCIL_8) is None  # column 1 is inside for 8 neighbours
    assert first_edge_arrival(arrival, STENCIL_16) == (20.0, 3, 1)
    arrival[3, 0] = 30.0
    arrival[6, 6] = 25.0
    assert first_edge_arrival(arrival, STENCIL_8) == (25.0, 6, 6)
    arrival[0, 5] = 25.0  # a tie goes to the lowest row
    assert first_edge_arrival(arrival, STENCIL_8) == (25.0, 0, 5)
    assert first_edge_arrival(arrival.astype(np.float32), STENCIL_16) == (20.0, 3, 1)


# ----------------------------------------------------------------------------- runs


def test_shipped_flat_no_wind_stays_inside_its_grid(engine: str) -> None:
    """The shipped 5400 s run reports None and has no edge event."""
    result = run(flat_no_wind(5400.0, engine))
    assert result.provenance["reached_edge_s"] is None
    assert all(kind != "edge" for _, kind, _, _ in result.events)
    assert not result.burned_mask[edge_band_mask(result.grid.shape, 2)].any()


def test_flat_no_wind_reports_when_it_reached_the_edge(engine: str) -> None:
    """Acceptance (see the module docstring for why 7500 s and not 7200 s)."""
    result = run(flat_no_wind(7500.0, engine))
    t_edge = result.provenance["reached_edge_s"]
    assert t_edge == pytest.approx(EDGE_TIME_NOMINAL_S, rel=EDGE_TIME_RTOL)
    band = edge_band_mask(result.grid.shape, 2)
    assert t_edge == np.nanmin(result.arrival_time[band])

    edge_events = [e for e in result.events if e[1] == "edge"]
    assert len(edge_events) == 1
    t, _, row, col = edge_events[0]
    assert t == t_edge == result.arrival_time[row, col]
    assert band[row, col] and min(row, col, 100 - row, 100 - col) == 1  # the inner ring of the band comes first
    at = result.events.index(edge_events[0])
    assert result.events[at - 1] == (t, "spread", row, col)
    times = [e[0] for e in result.events]
    assert times == sorted(times)
    json.dumps(result.provenance, allow_nan=False)


def test_flat_no_wind_at_7200_s_has_not_reached_the_edge() -> None:
    """At 7200 s the front is 47 cells out: no burned cell has a link leaving the grid yet."""
    result = run(flat_no_wind(7200.0))
    assert result.provenance["reached_edge_s"] is None
    rows, cols = np.nonzero(result.burned_mask)
    assert min(rows.min(), cols.min()) >= 2 and max(rows.max(), cols.max()) <= 98
    assert min(rows.min(), cols.min()) <= 4  # but it is close: this is the run the plan had in mind


def test_edge_detection_does_not_change_the_fire(engine: str) -> None:
    """Arrival times of the longer run agree bit for bit with the shorter run it extends."""
    short = run(flat_no_wind(5400.0, engine))
    long = run(flat_no_wind(7500.0, engine))
    early = short.burned_mask
    np.testing.assert_array_equal(long.arrival_time[early], short.arrival_time[early])


def test_edge_band_follows_the_stencil() -> None:
    """With 8 neighbours only the outermost ring counts, so the same fire reaches the edge later."""
    t16 = run(small(3600.0, stencil=16), seed=1).provenance["reached_edge_s"]
    t8 = run(small(3600.0, stencil=8), seed=1).provenance["reached_edge_s"]
    assert t16 == pytest.approx(9 * 30.0 / 0.2, rel=0.10)  # from (10, 10) to row or column 1 or 19
    # to row or column 0 or 20. 10 %: the earliest of 80 band cells on a 10-cell run sits below the mean
    assert t8 == pytest.approx(10 * 30.0 / 0.2, rel=0.10)
    assert t8 > t16


def test_edge_is_recorded_without_events() -> None:
    with_events = run(small(3600.0), seed=1)
    without = run(small(3600.0), seed=1, record_events=False)
    assert without.events == []
    assert without.provenance["reached_edge_s"] == with_events.provenance["reached_edge_s"] is not None


def test_ignition_inside_the_edge_band_counts() -> None:
    cfg = small(300.0).model_copy(
        update={"ignition": small(300.0).ignition.model_copy(update={"center": False, "cells": [(0, 5)]})}
    )
    result = run(cfg)
    assert result.provenance["reached_edge_s"] == 0.0
    assert result.events[:2] == [(0.0, "ignition", 0, 5), (0.0, "edge", 0, 5)]


# ----------------------------------------------------------------------------- ensembles


def test_ensemble_records_the_edge_time_of_each_run() -> None:
    cfg = small(1500.0, ensemble={"ros_multiplier_sigma_frac": 0.4})
    result = run_ensemble(cfg, n_runs=8, base_seed=2)
    assert result.reached_edge_s.shape == (8,) and result.reached_edge_s.dtype == np.float64
    reached = np.isfinite(result.reached_edge_s)
    assert 0 < reached.sum() < 8, "the scenario is meant to have runs on both sides"
    assert result.n_reached_edge == reached.sum()
    band = edge_band_mask((21, 21), 2)
    for i in range(8):
        in_band = result.arrival_times[i][band]
        if reached[i]:
            assert np.float32(result.reached_edge_s[i]) == np.nanmin(in_band)
        else:
            assert np.isnan(in_band).all()


def test_ensemble_edge_times_do_not_depend_on_workers() -> None:
    cfg = small(1500.0, ensemble={"ros_multiplier_sigma_frac": 0.4})
    serial = run_ensemble(cfg, n_runs=4, base_seed=2, n_workers=1)
    parallel = run_ensemble(cfg, n_runs=4, base_seed=2, n_workers=2)
    np.testing.assert_array_equal(serial.reached_edge_s, parallel.reached_edge_s)


def test_ensemble_summary_counts_edge_runs() -> None:
    cfg = small(1500.0, ensemble={"ros_multiplier_sigma_frac": 0.4})
    result = run_ensemble(cfg, n_runs=8, base_seed=2)
    summary = io.ensemble_summary(result, 2, 1)
    assert summary["n_reached_edge"] == result.n_reached_edge
    assert summary["first_reached_edge_s"] == np.nanmin(result.reached_edge_s)
    inside = run_ensemble(small(300.0), n_runs=2, base_seed=2)
    summary = io.ensemble_summary(inside, 2, 1)
    assert summary["n_reached_edge"] == 0 and summary["first_reached_edge_s"] is None


# ----------------------------------------------------------------------------- CLI and files

SCENARIO = """\
name: edge
grid: {{kind: synthetic, generator: flat_uniform, shape: [21, 21], dx: 30.0, fuel_model: 1}}
spread: {{rule: constant, params: {{ros: 0.2}}}}
ignition: {{center: true}}
duration_s: {duration}
ensemble: {{n_runs: 8, base_seed: 2, ros_multiplier_sigma_frac: 0.4}}
"""


def _scenario(tmp_path: Path, duration: float) -> Path:
    path = tmp_path / "edge.yaml"
    path.write_text(SCENARIO.format(duration=duration))
    return path


def _flat(output: str) -> str:
    return " ".join(output.split())


def test_cli_run_warns_and_writes_the_edge_time(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(app, ["run", str(_scenario(tmp_path, 3600.0)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "summary.json").read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    assert summary["reached_edge_s"] == provenance["reached_edge_s"] > 0
    assert "Warning" in result.output
    assert "reached the edge of the grid" in _flat(result.output)
    assert "lower bound" in _flat(result.output)
    with (out / "events.csv").open(newline="") as fh:
        edge_rows = [row for row in csv.reader(fh) if row[1] == "edge"]
    assert len(edge_rows) == 1 and float(edge_rows[0][0]) == pytest.approx(summary["reached_edge_s"])


def test_cli_run_is_quiet_when_the_fire_stays_inside(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(app, ["run", str(_scenario(tmp_path, 300.0)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads((out / "summary.json").read_text())["reached_edge_s"] is None
    assert "Warning" not in result.output
    assert "reached grid edge" in _flat(result.output)


def test_cli_ensemble_warns_and_lists_the_runs(tmp_path: Path) -> None:
    out = tmp_path / "ens"
    result = runner.invoke(app, ["ensemble", str(_scenario(tmp_path, 1500.0)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "summary.json").read_text())
    assert 0 < summary["n_reached_edge"] < 8
    assert "Warning" in result.output
    assert f"{summary['n_reached_edge']} of 8 runs reached the edge of the grid" in _flat(result.output)
    with (out / "runs.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    with np.load(out / "ensemble.npz") as data:
        edge = data["reached_edge_s"]
    assert edge.shape == (8,)
    assert np.isfinite(edge).sum() == summary["n_reached_edge"]
    assert summary["first_reached_edge_s"] == pytest.approx(np.nanmin(edge))
    for row, value in zip(rows, edge.tolist(), strict=True):
        if np.isfinite(value):
            assert float(row["reached_edge_s"]) == pytest.approx(value)
        else:
            assert row["reached_edge_s"] == ""


def test_cli_ensemble_is_quiet_when_every_run_stays_inside(tmp_path: Path) -> None:
    out = tmp_path / "ens"
    result = runner.invoke(app, ["ensemble", str(_scenario(tmp_path, 300.0)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Warning" not in result.output
    assert json.loads((out / "summary.json").read_text())["n_reached_edge"] == 0
