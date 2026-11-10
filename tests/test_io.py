"""Tests for the run-directory helpers in :mod:`wildfire_sim.io` (no CLI involved)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from wildfire_sim import io, sim
from wildfire_sim.config import ScenarioConfig


@pytest.fixture
def tiny_config() -> ScenarioConfig:
    return ScenarioConfig.model_validate(
        {
            "name": "tiny",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [15, 15]},
            "spread": {"rule": "constant", "params": {"ros": 0.2}},
            "ignition": {"center": True},
            "duration_s": 300.0,
        }
    )


def test_jsonable_replaces_non_finite_with_null() -> None:
    data = {"a": math.nan, "b": [np.float32(np.inf), np.int64(3)], "c": (1.5, -math.inf), "d": np.bool_(True)}
    out = io.jsonable(data)
    assert out == {"a": None, "b": [None, 3], "c": [1.5, None], "d": True}
    json.dumps(out, allow_nan=False)


def test_write_json_is_strict(tmp_path: Path) -> None:
    path = io.write_json(tmp_path / "x.json", {"v": float("nan")})
    assert json.loads(path.read_text()) == {"v": None}


def test_check_output_dir_has_no_side_effects(tmp_path: Path) -> None:
    io.check_output_dir(tmp_path / "missing", force=False)
    assert not (tmp_path / "missing").exists()
    (tmp_path / "f").write_text("x")
    with pytest.raises(io.OutputDirError, match="--force"):
        io.check_output_dir(tmp_path, force=False)
    io.check_output_dir(tmp_path, force=True)
    assert (tmp_path / "f").exists()


def test_prepare_output_dir(tmp_path: Path) -> None:
    new = io.prepare_output_dir(tmp_path / "a" / "b", force=False, owned=io.RUN_OUTPUTS)
    assert new.is_dir()
    assert io.prepare_output_dir(new, force=False, owned=io.RUN_OUTPUTS) == new  # empty is fine

    (new / "summary.json").write_text("{}")
    (new / "mine.txt").write_text("x")
    with pytest.raises(io.OutputDirError, match="--force"):
        io.prepare_output_dir(new, force=False, owned=io.RUN_OUTPUTS)
    io.prepare_output_dir(new, force=True, owned=io.RUN_OUTPUTS)
    assert not (new / "summary.json").exists()
    assert (new / "mine.txt").exists()

    file_path = tmp_path / "file"
    file_path.write_text("x")
    with pytest.raises(io.OutputDirError, match="not a directory"):
        io.prepare_output_dir(file_path, force=True, owned=io.RUN_OUTPUTS)


def test_run_summary_and_round_trip(tiny_config: ScenarioConfig, tmp_path: Path) -> None:
    result = sim.run(tiny_config, seed=1)
    summary = io.write_run(result, tiny_config, tmp_path)
    burned = int(result.burned_mask.sum())
    assert summary["burned_cells"] == burned
    assert summary["burned_area_m2"] == pytest.approx(burned * result.grid.dx**2)

    assert io.dir_kind(tmp_path) == "run"
    saved = io.load_run(tmp_path)
    assert saved.grid == result.grid
    np.testing.assert_array_equal(saved.arrival_time, result.arrival_time)


def test_dir_kind_rejects_other_dirs(tmp_path: Path) -> None:
    with pytest.raises(io.OutputDirError, match="not a run or ensemble directory"):
        io.dir_kind(tmp_path)
    with pytest.raises(io.OutputDirError, match="not found"):
        io.dir_kind(tmp_path / "missing")


def test_burned_area_figure_has_two_panels() -> None:
    fig = io.burned_area_figure(np.array([1e4, 2e4, 4e4]))
    assert len(fig.axes) == 2
