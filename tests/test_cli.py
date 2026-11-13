"""Tests for the wildfire-sim CLI.

The ``run``/``ensemble``/``plot`` tests use a 31 x 31-cell, 600 s scenario written to
``tmp_path`` so the file stays fast. Absolute fire sizes are never asserted: the front-speed
correction table is calibrated separately and rescales every rate.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

from wildfire_sim.cli import app
from wildfire_sim.config import load_scenario
from wildfire_sim.grid import Grid
from wildfire_sim.types import CellState

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parent.parent / "configs"

TINY = """\
name: tiny
grid: {kind: synthetic, generator: flat_uniform, shape: [31, 31], dx: 30.0, fuel_model: 1}
spread: {rule: constant, params: {ros: 0.2}}
ignition: {center: true}
dt_s: 30.0
duration_s: 600.0
seed: 3
ensemble: {n_runs: 4, base_seed: 7, ros_multiplier_sigma_frac: 0.3}
"""

RUN_FILES = (
    "grid.npz",
    "arrival_time.npy",
    "events.csv",
    "summary.json",
    "provenance.json",
    "scenario.yaml",
    "final.png",
    "arrival.png",
)
ENSEMBLE_FILES = (
    "ensemble.npz",
    "runs.csv",
    "summary.json",
    "provenance.json",
    "scenario.yaml",
    "burn_probability.png",
    "burned_area.png",
)


def _strict_json(path: Path) -> dict:
    """Parse ``path`` refusing NaN/Infinity, which are not JSON."""

    def reject(token: str) -> None:
        raise ValueError(f"non-strict JSON constant {token} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _assert_user_error(result: Result, fragment: str) -> None:
    """Exit code 1 through ``typer.Exit`` (not an uncaught exception) and ``fragment`` in the output."""
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Traceback" not in result.output
    assert fragment in result.output, result.output


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(TINY)
    return path


@pytest.fixture
def run_dir(tiny: Path, tmp_path: Path) -> Path:
    out = tmp_path / "run"
    result = runner.invoke(app, ["run", str(tiny), "--out", str(out)])
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture
def ensemble_dir(tiny: Path, tmp_path: Path) -> Path:
    out = tmp_path / "ens"
    result = runner.invoke(app, ["ensemble", str(tiny), "--out", str(out)])
    assert result.exit_code == 0, result.output
    return out


# ----------------------------------------------------------------------------- existing commands


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "wildfire-sim" in result.output


def test_validate_ok() -> None:
    result = runner.invoke(app, ["validate", str(CONFIGS / "scenarios" / "flat_no_wind.yaml")])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "101 x 101" in result.output


def test_validate_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "Invalid scenario" in result.output


def test_validate_reports_malformed_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "Invalid scenario" in result.output


def test_validate_reports_grid_build_errors(tmp_path: Path) -> None:
    """Bad generator arguments and missing cached grids exit 1 with a message, not a traceback."""
    for grid in (
        "{kind: synthetic, generator: with_river, params: {no_such_arg: 1}}",
        "{kind: cached, path: does_not_exist.npz}",
    ):
        scenario = tmp_path / "s.yaml"
        scenario.write_text(f"name: x\ngrid: {grid}\nignition: {{center: true}}\n")
        result = runner.invoke(app, ["validate", str(scenario)])
        assert result.exit_code == 1, result.output
        assert "Cannot build grid" in result.output


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("validate", "run", "ensemble", "plot"):
        assert command in result.output


# ----------------------------------------------------------------------------- run


def test_run_writes_every_file(run_dir: Path) -> None:
    for name in RUN_FILES:
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "fire.gif").exists()  # --gif is off by default


def test_run_outputs_agree(run_dir: Path) -> None:
    summary = _strict_json(run_dir / "summary.json")
    provenance = _strict_json(run_dir / "provenance.json")
    arrival = np.load(run_dir / "arrival_time.npy")
    grid = Grid.load(run_dir / "grid.npz")

    assert arrival.dtype == np.float64
    assert arrival.shape == grid.shape == (31, 31)
    burned = int(np.isfinite(arrival).sum())
    assert burned > 1
    assert summary["scenario"] == "tiny"
    assert summary["seed"] == 3 == provenance["seed"]
    assert summary["burned_cells"] == burned
    assert summary["burned_area_m2"] == pytest.approx(burned * grid.dx**2)
    assert summary["burned_area_ha"] == pytest.approx(burned * grid.dx**2 / 1e4)
    assert summary["steps"] == provenance["n_steps"]
    assert summary["simulated_time_s"] == pytest.approx(provenance["final_time_s"])
    assert summary["last_ignition_s"] == pytest.approx(float(np.nanmax(arrival)))
    assert summary["wall_clock_s"] >= 0
    counts = summary["final_state_counts"]
    assert set(counts) == {s.name for s in CellState}
    assert sum(counts.values()) == grid.n_cells
    assert counts["BURNING"] + counts["BURNED"] == burned
    assert provenance["config"]["name"] == "tiny"
    assert "git_commit" in provenance and "versions" in provenance


def test_run_events_csv(run_dir: Path) -> None:
    arrival = np.load(run_dir / "arrival_time.npy")
    with (run_dir / "events.csv").open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["t_s", "kind", "row", "col"]
    body = rows[1:]
    assert len(body) == int(np.isfinite(arrival).sum())
    assert body[0][1] == "ignition"
    for t_s, _, row, col in body:
        assert float(t_s) == pytest.approx(arrival[int(row), int(col)])


def test_run_scenario_yaml_round_trips(run_dir: Path, tmp_path: Path) -> None:
    """The written scenario.yaml is valid and reproduces the run."""
    again = tmp_path / "again"
    result = runner.invoke(app, ["run", str(run_dir / "scenario.yaml"), "--out", str(again)])
    assert result.exit_code == 0, result.output
    np.testing.assert_array_equal(np.load(run_dir / "arrival_time.npy"), np.load(again / "arrival_time.npy"))


def test_run_seeds(tiny: Path, tmp_path: Path) -> None:
    arrays = {}
    for name, seed in (("a", "11"), ("b", "11"), ("c", "12")):
        out = tmp_path / name
        result = runner.invoke(app, ["run", str(tiny), "--seed", seed, "--out", str(out)])
        assert result.exit_code == 0, result.output
        arrays[name] = np.load(out / "arrival_time.npy")
        assert _strict_json(out / "summary.json")["seed"] == int(seed)
    np.testing.assert_array_equal(arrays["a"], arrays["b"])
    assert not np.array_equal(arrays["a"], arrays["c"], equal_nan=True)


def test_run_default_out_dir(tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", str(tiny)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "runs" / "tiny-seed3" / "summary.json").is_file()


def test_run_gif_and_force_removes_stale_gif(tiny: Path, tmp_path: Path) -> None:
    out = tmp_path / "gif"
    result = runner.invoke(app, ["run", str(tiny), "--out", str(out), "--gif", "--frames", "4", "--fps", "5"])
    assert result.exit_code == 0, result.output
    assert (out / "fire.gif").read_bytes()[:6] in (b"GIF89a", b"GIF87a")

    result = runner.invoke(app, ["run", str(tiny), "--out", str(out), "--force"])
    assert result.exit_code == 0, result.output
    assert not (out / "fire.gif").exists()


def test_run_refuses_non_empty_dir_without_force(tiny: Path, tmp_path: Path) -> None:
    out = tmp_path / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    result = runner.invoke(app, ["run", str(tiny), "--out", str(out)])
    _assert_user_error(result, "--force")
    assert not (out / "summary.json").exists()

    result = runner.invoke(app, ["run", str(tiny), "--out", str(out), "--force"])
    assert result.exit_code == 0, result.output
    assert (out / "summary.json").is_file()
    assert (out / "keep.txt").is_file()  # --force overwrites outputs, it does not wipe the directory


def test_run_shipped_flat_no_wind(tmp_path: Path) -> None:
    out = tmp_path / "flat"
    result = runner.invoke(app, ["run", str(CONFIGS / "scenarios" / "flat_no_wind.yaml"), "--out", str(out)])
    assert result.exit_code == 0, result.output
    summary = _strict_json(out / "summary.json")
    assert summary["scenario"] == "flat_no_wind"
    assert summary["burned_cells"] == int(np.isfinite(np.load(out / "arrival_time.npy")).sum())
    assert "flat_no_wind" in result.output


# ----------------------------------------------------------------------------- error paths


@pytest.mark.parametrize("command", ["run", "ensemble"])
def test_invalid_scenario(command: str, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    _assert_user_error(runner.invoke(app, [command, str(bad), "--out", str(tmp_path / "o")]), "Invalid scenario")


@pytest.mark.parametrize("command", ["run", "ensemble"])
def test_missing_scenario_file(command: str, tmp_path: Path) -> None:
    result = runner.invoke(app, [command, str(tmp_path / "nope.yaml"), "--out", str(tmp_path / "o")])
    _assert_user_error(result, "not found")


@pytest.mark.parametrize("command", ["run", "ensemble"])
def test_unknown_rule(command: str, tmp_path: Path) -> None:
    scenario = tmp_path / "s.yaml"
    scenario.write_text(TINY.replace("rule: constant", "rule: no_such_rule"))
    result = runner.invoke(app, [command, str(scenario), "--out", str(tmp_path / "o")])
    _assert_user_error(result, "no_such_rule")


@pytest.mark.parametrize("command", ["run", "ensemble"])
def test_ignition_on_water(command: str, tmp_path: Path) -> None:
    scenario = tmp_path / "s.yaml"
    text = TINY.replace("generator: flat_uniform", "generator: with_river, params: {col: 15, width: 2}")
    scenario.write_text(text.replace("ignition: {center: true}", "ignition: {cells: [[10, 15]]}"))
    result = runner.invoke(app, [command, str(scenario), "--out", str(tmp_path / "o")])
    _assert_user_error(result, "not burnable")
    assert not (tmp_path / "o").exists()  # a failed run leaves no output directory behind


# ----------------------------------------------------------------------------- ensemble


def test_ensemble_writes_every_file(ensemble_dir: Path) -> None:
    for name in ENSEMBLE_FILES:
        assert (ensemble_dir / name).is_file(), name


def test_ensemble_outputs_agree(ensemble_dir: Path) -> None:
    summary = _strict_json(ensemble_dir / "summary.json")
    provenance = _strict_json(ensemble_dir / "provenance.json")
    with np.load(ensemble_dir / "ensemble.npz") as data:
        arrays = {k: data[k] for k in data.files}
    assert {"burn_probability", "arrival_times", "burned_area_m2", "seeds"} <= set(arrays)
    area = arrays["burned_area_m2"]
    assert arrays["arrival_times"].shape == (4, 31, 31)
    assert area.shape == arrays["seeds"].shape == (4,)

    assert summary["n"] == 4 == provenance["n"]
    assert summary["base_seed"] == 7 == provenance["base_seed"]
    assert summary["workers"] == 1
    mean = summary["mean_burned_area_m2"]
    assert mean == pytest.approx(area.mean())
    assert summary["ci95_burned_area_m2"][0] <= mean <= summary["ci95_burned_area_m2"][1]
    assert summary["median_burned_area_m2"] == pytest.approx(np.median(area))
    assert summary["min_burned_area_m2"] == pytest.approx(area.min())
    assert summary["max_burned_area_m2"] == pytest.approx(area.max())
    assert summary["p90_burned_area_m2"] == pytest.approx(np.percentile(area, 90))
    assert provenance["config"]["name"] == "tiny"
    assert provenance["seeds"] == arrays["seeds"].tolist()
    assert "git_commit" in provenance and "versions" in provenance

    with (ensemble_dir / "runs.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert [int(r["seed"]) for r in rows] == arrays["seeds"].tolist()
    assert [int(r["index"]) for r in rows] == [0, 1, 2, 3]
    for row, a in zip(rows, area, strict=True):
        assert float(row["burned_area_m2"]) == pytest.approx(a)
        assert float(row["burned_area_ha"]) == pytest.approx(a / 1e4)
        assert float(row["ros_multiplier"]) > 0
        assert "wind_speed_factor" in row


def test_ensemble_options_override_scenario(tiny: Path, tmp_path: Path) -> None:
    out = tmp_path / "ens"
    result = runner.invoke(app, ["ensemble", str(tiny), "--n", "2", "--seed", "5", "--out", str(out)])
    assert result.exit_code == 0, result.output
    summary = _strict_json(out / "summary.json")
    assert (summary["n"], summary["base_seed"]) == (2, 5)


def test_ensemble_workers_do_not_change_results(tiny: Path, tmp_path: Path) -> None:
    results = []
    for workers in ("1", "2"):
        out = tmp_path / f"w{workers}"
        args = ["ensemble", str(tiny), "--n", "3", "--workers", workers, "--out", str(out)]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        with np.load(out / "ensemble.npz") as data:
            results.append({k: data[k] for k in data.files})
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        np.testing.assert_array_equal(results[0][key], results[1][key])


def test_ensemble_refuses_non_empty_dir(tiny: Path, tmp_path: Path) -> None:
    out = tmp_path / "busy"
    out.mkdir()
    (out / "x").write_text("x")
    _assert_user_error(runner.invoke(app, ["ensemble", str(tiny), "--out", str(out)]), "--force")


def test_ensemble_default_out_dir(tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["ensemble", str(tiny), "--n", "2"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "runs" / "tiny-ensemble" / "summary.json").is_file()


# ----------------------------------------------------------------------------- plot


def test_plot_run_dir(run_dir: Path) -> None:
    for name in ("final.png", "arrival.png"):
        (run_dir / name).unlink()
    result = runner.invoke(app, ["plot", str(run_dir), "--gif", "--frames", "5", "--fps", "4"])
    assert result.exit_code == 0, result.output
    for name in ("final.png", "arrival.png", "fire.gif"):
        assert (run_dir / name).is_file(), name


def test_plot_ensemble_dir(ensemble_dir: Path) -> None:
    for name in ("burn_probability.png", "burned_area.png"):
        (ensemble_dir / name).unlink()
    result = runner.invoke(app, ["plot", str(ensemble_dir)])
    assert result.exit_code == 0, result.output
    for name in ("burn_probability.png", "burned_area.png"):
        assert (ensemble_dir / name).is_file(), name


def test_plot_unrelated_dir(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x")
    _assert_user_error(runner.invoke(app, ["plot", str(tmp_path)]), "not a run or ensemble directory")


def test_plot_missing_dir(tmp_path: Path) -> None:
    _assert_user_error(runner.invoke(app, ["plot", str(tmp_path / "nope")]), "not found")


# ----------------------------------------------------------------------------- wall clock (WP 1.14)


def test_run_records_the_start_time_in_provenance(tmp_path: Path) -> None:
    scenario = tmp_path / "dated.yaml"
    scenario.write_text(TINY + "start_time_utc: '2016-08-13T16:00:00-07:00'\n")
    out = tmp_path / "dated"
    result = runner.invoke(app, ["run", str(scenario), "--out", str(out)])
    assert result.exit_code == 0, result.output
    provenance = _strict_json(out / "provenance.json")
    assert provenance["start_time_utc"] == "2016-08-13T23:00:00+00:00"
    assert load_scenario(out / "scenario.yaml").start_time_utc == datetime(2016, 8, 13, 23, tzinfo=UTC)


def test_run_without_a_start_time_records_none(run_dir: Path) -> None:
    assert _strict_json(run_dir / "provenance.json")["start_time_utc"] is None


def test_naive_start_time_is_a_user_error(tmp_path: Path) -> None:
    scenario = tmp_path / "naive.yaml"
    scenario.write_text(TINY + "start_time_utc: '2016-08-13T16:00:00'\n")
    result = runner.invoke(app, ["run", str(scenario), "--out", str(tmp_path / "x")])
    _assert_user_error(result, "Invalid scenario")
