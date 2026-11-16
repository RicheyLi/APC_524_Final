"""Tests for the Morris screening (WP 3.2): ``wildfire_sim.sensitivity`` and ``scripts/sensitivity.py``.

The method is checked on analytic functions with known elementary effects; the acceptance test
screens a small synthetic scenario and requires the deliberately inert input to rank last with
``mu* == 0`` on both outputs. Every run uses the NumPy engine unless numba is installed.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from wildfire_sim.config import configs_dir
from wildfire_sim.sensitivity import (
    OUTPUTS,
    SensitivityInput,
    SensitivityStudy,
    edge_distance_m,
    elementary_effects,
    load_study,
    morris_design,
    morris_indices,
    morris_step,
    parameter_files,
    run_screening,
)
from wildfire_sim.sim import load_holdover_times

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- the design


@pytest.mark.parametrize(("k", "r", "p"), [(1, 3, 2), (5, 10, 4), (12, 7, 6)])
def test_trajectories_move_one_input_by_delta_and_each_input_once(k: int, r: int, p: int) -> None:
    design = morris_design(k, r, p, np.random.default_rng(3))
    assert design.shape == (r, k + 1, k)
    assert design.min() >= 0 and design.max() <= p - 1  # points stay in [0, 1]
    steps = np.diff(design, axis=1)  # (r, k, k)
    assert np.all(np.count_nonzero(steps, axis=2) == 1)  # exactly one input per step
    assert np.all(np.abs(steps).sum(axis=2) == p // 2)  # by delta = p / (2 (p - 1)), in index units
    assert np.all(np.count_nonzero(steps, axis=1) == 1)  # each input changes once per trajectory
    assert morris_step(p) == pytest.approx((p // 2) / (p - 1))


def test_design_is_reproducible_and_uses_both_directions() -> None:
    a = morris_design(6, 20, 4, np.random.default_rng(7))
    b = morris_design(6, 20, 4, np.random.default_rng(7))
    assert np.array_equal(a, b)
    steps = np.diff(a, axis=1).sum(axis=2)
    assert (steps > 0).any() and (steps < 0).any()


@pytest.mark.parametrize("levels", [0, 3, 5])
def test_odd_or_too_few_levels_are_refused(levels: int) -> None:
    with pytest.raises(ValueError, match="even"):
        morris_design(3, 2, levels, np.random.default_rng(0))


# ----------------------------------------------------------------------------- the statistics


def _evaluate(design: np.ndarray, p: int, f: Any) -> np.ndarray:
    x = design / (p - 1)
    return np.array([[f(point) for point in traj] for traj in x])


def test_linear_function_gives_its_coefficients_exactly() -> None:
    """``y = sum(a_i x_i)``: every elementary effect of input i is a_i, so mu = mu* = a_i, sigma = 0."""
    a = np.array([3.0, -2.0, 0.5, 0.0])
    p = 4
    design = morris_design(a.size, 15, p, np.random.default_rng(11))
    ind = morris_indices(design, p, _evaluate(design, p, lambda x: float(a @ x)))
    np.testing.assert_allclose(ind.mu, a, atol=1e-12)
    np.testing.assert_allclose(ind.mu_star, np.abs(a), atol=1e-12)
    np.testing.assert_allclose(ind.sigma, 0.0, atol=1e-12)
    assert ind.mu_star[3] == 0.0 and ind.rank[3] == 4  # the ignored input: exactly 0, ranked last
    assert list(ind.rank) == [1, 2, 3, 4]
    assert np.all(ind.n_valid == 15)


def test_interaction_and_nonlinearity_give_sigma_and_a_cancelling_mu() -> None:
    """``y = x0 x1 + (x2 - 0.5)^2``: x0, x1 interact and x2 is symmetric about 0.5, so sigma > 0;
    x2's signed effects partly cancel in mu while mu* does not; x3 is ignored."""
    p = 4
    design = morris_design(4, 30, p, np.random.default_rng(5))
    y = _evaluate(design, p, lambda x: x[0] * x[1] + (x[2] - 0.5) ** 2)
    ind = morris_indices(design, p, y)
    assert np.all(ind.sigma[:2] > 0.1)
    # x2 steps between 0 and 2/3 (EE = (1/36 - 1/4) / (2/3) = -1/3) or 1/3 and 1 (EE = +1/3)
    assert ind.mu_star[2] == pytest.approx(1.0 / 3.0)
    assert ind.sigma[2] > 0.3
    assert abs(ind.mu[2]) < ind.mu_star[2]
    assert ind.mu_star[3] == 0.0 and ind.sigma[3] == 0.0


def test_elementary_effect_matches_a_hand_computation() -> None:
    design = np.array([[[0, 0], [2, 0], [2, 2]]])  # p = 4: x goes (0, 0) -> (2/3, 0) -> (2/3, 2/3)
    y = np.array([[1.0, 3.0, 2.0]])
    ee = elementary_effects(design, 4, y)
    np.testing.assert_allclose(ee, [[2.0 / (2.0 / 3.0), -1.0 / (2.0 / 3.0)]])
    down = np.array([[[2, 0], [0, 0], [0, 2]]])  # a downward step divides by -delta
    np.testing.assert_allclose(elementary_effects(down, 4, y), [[-3.0, -1.5]])


def test_undefined_outputs_are_left_out_and_counted() -> None:
    p = 4
    design = morris_design(3, 5, p, np.random.default_rng(2))
    y = _evaluate(design, p, lambda x: float(x.sum()))
    y[0, 0] = math.nan  # touches only the first step of trajectory 0
    ind = morris_indices(design, p, y)
    first = int(np.flatnonzero(design[0, 1] - design[0, 0])[0])
    assert ind.n_valid[first] == 4 and ind.n_valid.sum() == 14
    np.testing.assert_allclose(ind.mu_star, 1.0)


def test_a_step_that_moves_two_inputs_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly one input"):
        elementary_effects(np.array([[[0, 0], [2, 2], [2, 2]]]), 4, np.zeros((1, 3)))


# ----------------------------------------------------------------------------- inputs and study file


def test_discrete_levels_and_log_ranges_map_onto_the_grid() -> None:
    stencil = SensitivityInput(target="stencil", levels=[8, 16], units="neighbours", source="test")
    assert [stencil.value(j, 4) for j in range(4)] == [8, 8, 16, 16]
    shape = SensitivityInput(target="arrival_shape", levels=[5, 20, 50, 200], units="-", source="test")
    assert [shape.value(j, 4) for j in range(4)] == [5, 20, 50, 200]
    mult = SensitivityInput(target="ros_multiplier", range=(0.3, 3.0), scale="log", units="-", source="test")
    np.testing.assert_allclose([mult.value(j, 4) for j in range(4)], 0.3 * 10.0 ** (np.arange(4) / 3))


def test_every_morris_step_changes_a_discrete_input() -> None:
    """delta > 1/2 is wider than any bin, so each step moves a discrete input to another level."""
    for p in (2, 4, 6, 8):
        for n_levels in range(2, p + 1):
            spec = SensitivityInput(target="arrival_shape", levels=list(range(1, n_levels + 1)), units="-", source="t")
            values = [spec.value(j, p) for j in range(p)]
            assert sorted(set(values)) == list(range(1, n_levels + 1))  # every level is used
            assert all(values[j] != values[j + p // 2] for j in range(p // 2))


def test_more_discrete_levels_than_grid_levels_are_refused() -> None:
    data = yaml.safe_load((configs_dir() / "sensitivity.yaml").read_text(encoding="utf-8"))
    data["inputs"]["arrival_shape"]["levels"] = [5.0, 10.0, 20.0, 50.0, 200.0]  # 5 levels on a 4-level grid
    with pytest.raises(ValueError, match="more levels"):
        SensitivityStudy.model_validate(data)


def test_committed_study_loads_and_its_head_distances_stay_off_the_edge() -> None:
    study = load_study(configs_dir() / "sensitivity.yaml")
    assert sum(spec.inert for spec in study.inputs.values()) == 1
    for name, scen in study.scenarios.items():
        assert any(study.inputs[i].inert for i in scen.inputs), name
        grid_shape, dx = _shape_dx(scen)
        cell = (grid_shape[0] // 2, grid_shape[1] // 2)
        assert scen.head_distance_m < edge_distance_m(cell, grid_shape, dx)


def _shape_dx(scen: Any) -> tuple[tuple[int, int], float]:
    from wildfire_sim.data import build_grid

    grid = build_grid(scen.scenario.grid)
    return grid.shape, grid.dx


def test_parameter_files_edits_a_copy_and_restores_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WILDFIRE_SIM_CONFIGS", raising=False)
    before = load_holdover_times().default.value
    with parameter_files({"holdover.yaml": {"default.value": 600.0}}):
        assert load_holdover_times().default.value == 600.0
    assert load_holdover_times().default.value == before
    import os

    assert "WILDFIRE_SIM_CONFIGS" not in os.environ


# ----------------------------------------------------------------------------- a real screening


def _small_study(tmp_path: Path, trajectories: int = 4) -> Path:
    """The committed inputs on a 41 x 41 synthetic grid, short runs: cheap enough for every pytest."""
    data = yaml.safe_load((configs_dir() / "sensitivity.yaml").read_text(encoding="utf-8"))
    data["trajectories"] = trajectories
    data["run_seeds"] = [1]
    scen = data["scenarios"]["synthetic"]
    scen["scenario"]["grid"]["shape"] = [41, 41]
    scen["scenario"]["duration_s"] = 7200.0
    scen["window_s"] = 900.0
    scen["head_distance_m"] = 150.0
    scen["inputs"] = ["wind_speed", "moisture_1h", "ros_multiplier", "stencil", "inert_temperature"]
    data["scenarios"] = {"tiny": scen}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_screening_ranks_the_inert_input_last_with_zero_effect(tmp_path: Path) -> None:
    """Acceptance (WP 3.2): the deliberately inert input ranks last with mu* exactly 0 on both outputs."""
    study = load_study(_small_study(tmp_path))
    res = run_screening(study, "tiny", 1)
    inert = res.input_names.index("inert_temperature")
    for out in OUTPUTS:
        ind = res.indices[out]
        assert ind.mu_star[inert] == 0.0 and ind.sigma[inert] == 0.0, out
        assert ind.rank[inert] == ind.rank.max(), out
        assert np.all(np.delete(ind.mu_star, inert) > 0), out
        assert np.all(ind.n_valid == 4), out
    area = res.indices["burned_area_ha"]
    assert area.mu[res.input_names.index("ros_multiplier")] > 0  # a faster rate burns more
    assert res.indices["head_arrival_s"].mu[res.input_names.index("ros_multiplier")] < 0  # and arrives sooner
    assert res.n_edge_in_window == 0

    again = run_screening(study, "tiny", 1)
    for out in OUTPUTS:  # same seed, same runs (NaN in the same places counts as equal)
        np.testing.assert_array_equal(again.output(out), res.output(out))
    np.testing.assert_array_equal(again.indices["burned_area_ha"].mu_star, area.mu_star)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sensitivity_script", REPO / "scripts" / "sensitivity.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_csvs_and_figures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    args = ["--config", str(_small_study(tmp_path, trajectories=2)), "--out", str(out)]
    assert _script().main(args) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "tiny_mu_sigma.png",
        "tiny_mu_star.png",
        "tiny_seed1_indices.csv",
        "tiny_seed1_runs.csv",
    ]
    rows = (out / "tiny_seed1_indices.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "output,input,mu,mu_star,sigma,n_valid,rank,n_edge"
    assert len(rows) == 1 + 2 * 5
    assert len((out / "tiny_seed1_runs.csv").read_text(encoding="utf-8").splitlines()) == 1 + 2 * 6
    printed = capsys.readouterr().out
    assert "| inert_temperature |" in printed and "reached the grid edge within the window" in printed


def test_script_rejects_an_unknown_scenario(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stop:
        _script().main(["--config", str(_small_study(tmp_path)), "--scenario", "nope", "--out", str(tmp_path)])
    assert stop.value.code == 2
