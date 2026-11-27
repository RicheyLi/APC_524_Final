"""Tests for the calibration of WP 3.1 (``wildfire_sim.calibration``, ``scripts/calibrate.py``).

The objective, the synthetic observations, the edge rule and ``configs/calibrated.yaml`` are tested
without optuna; the studies need it (``uv sync --extra opt``) and are skipped otherwise.

The plan's acceptance tests:

* the recovery of known parameters from a synthetic observed fire, with its wind along a grid
  axis and at 13.3, 22.5 and 45 degrees to it (:func:`test_recovers_the_parameters_at_every_head_bearing`,
  where the observed fire is a run of the model, and
  :func:`test_recovers_an_analytic_ellipse_at_every_head_bearing`, where it is the analytic
  ellipse, so the lattice's shape error, which depends on the bearing, is not shared by the
  observed fire);
* a trial in which a run reaches the grid edge is rejected, not scored
  (:func:`test_trials_reaching_the_edge_are_rejected_not_scored`);
* the script runs on the committed crop with 3 trials x 2 seeds
  (:func:`test_script_runs_on_the_committed_crop`).
"""

from __future__ import annotations

import importlib.util
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from wildfire_sim.calibration import (
    CalibratedParams,
    Evaluation,
    RealFire,
    StudyResult,
    Target,
    calibrated_params,
    evaluate,
    load_calibrated,
    load_calibration_settings,
    objective_surface,
    observed_target,
    real_fire,
    run_study,
    synthetic_detections,
    synthetic_target,
    with_params,
    write_calibrated,
)
from wildfire_sim.config import ScenarioConfig, configs_dir
from wildfire_sim.data import build_grid
from wildfire_sim.data.validation import Detection
from wildfire_sim.spread.ellipse import eccentricity
from wildfire_sim.spread.rothermel import RothermelRule
from wildfire_sim.types import get_stencil

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENGINE = "numba" if importlib.util.find_spec("numba") else "numpy"
TRUTH = {"ros_multiplier": 1.5, "lb_scale": 1.25}
BEARINGS = (0.0, 13.3, 22.5, 45.0)


def flat_fire(head_bearing: float, *, size: int = 101, offset: int = 20, window_s: float = 5400.0) -> ScenarioConfig:
    """Flat uniform GR2, a steady 6 m/s wind driving the head toward ``head_bearing`` (degrees).

    The fire is lit ``offset`` cells upwind of the centre. With :data:`TRUTH` the head spreads at
    0.23 m/s and the length-to-breadth ratio is 1 + 1.25 x 0.467 = 1.58, well below the lattice
    cap of 2.1 (16 neighbours); in the 1.5-hour window the head runs about 41 cells.
    """
    c = size // 2
    row = round(c + offset * math.cos(math.radians(head_bearing)))
    col = round(c - offset * math.sin(math.radians(head_bearing)))
    return ScenarioConfig.model_validate(
        {
            "name": f"flat GR2, head {head_bearing:g} deg",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [size, size], "fuel_model": 102},
            "weather": {"series": [{"time_s": 0.0, "wind_speed": 6.0, "wind_direction": (head_bearing + 180.0) % 360}]},
            "spread": {"rule": "rothermel"},
            "ignition": {"cells": [[row, col]]},
            "dt_s": 60.0,
            "duration_s": window_s,
            "engine": ENGINE,
        }
    )


def changed(config: ScenarioConfig, **update: object) -> ScenarioConfig:
    """``config`` with top-level fields replaced, re-validated."""
    return ScenarioConfig.model_validate({**config.model_dump(), **update})


def ellipse_target(config: ScenarioConfig, truth: dict[str, float], footprint_m: float, interval_s: float) -> Target:
    """The analytic fire ellipse of ``truth`` as the observed fire (flat uniform grid, steady weather).

    The ignition is at the rear focus, so a cell at distance ``d`` and angle ``theta`` from the
    head direction ignites at ``d / R(theta)``, ``R(theta) = R_head (1 - e) / (1 - e cos theta)``
    (docs/plan.md section 2.2), with ``R_head`` and ``LB`` from the rule with ``truth``.
    """
    grid = build_grid(config.grid)
    (r0, c0), window = config.ignition.cells[0], config.duration_s
    rule = RothermelRule(ros_multiplier=truth["ros_multiplier"], lb_scale=truth["lb_scale"])
    fire = rule.fire(grid, config.weather.series[0].to_weather(), get_stencil(config.stencil))
    head, bearing = float(fire.ros_head[r0, c0]), float(fire.head_bearing[r0, c0])
    e = float(eccentricity(fire.length_to_breadth[r0, c0]))
    rows, cols = np.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    east, north = (cols - c0) * grid.dx, (r0 - rows) * grid.dx
    theta = np.arctan2(east, north) - math.radians(bearing)
    arrival = np.hypot(east, north) / (head * (1 - e) / (1 - e * np.cos(theta)))
    arrival = np.where(arrival <= window, arrival, np.nan)
    overpasses = np.arange(interval_s, window + 1e-9, interval_s)
    detections = synthetic_detections(arrival, overpasses, 0.0, window, footprint_m, grid.dx)
    return Target(~np.isnan(arrival), tuple(detections), 0.0, window, footprint_m, "analytic ellipse")


def _recover(head_bearing: float, analytic: bool, n_trials: int) -> dict[str, float]:
    """Best parameters of a study on the synthetic fire of :func:`flat_fire` (run in a worker process)."""
    settings = load_calibration_settings()
    config = flat_fire(head_bearing)
    footprint, interval = float(settings.footprint_m.value), float(settings.overpass_interval_s.value)
    if analytic:
        target = ellipse_target(config, TRUTH, footprint, interval)
    else:
        seed = int(settings.truth_seed.value)
        target, _ = synthetic_target(config, TRUTH, seed=seed, overpass_interval_s=interval, footprint_m=footprint)
    study = run_study(
        config,
        target,
        settings.space("rothermel"),
        n_trials=n_trials,
        seeds=settings.seeds(2),
        weights=settings.weights,
        sampler_seed=int(settings.sampler_seed.value),
    )
    assert study.best.objective is not None
    return {**study.best.params, "objective": study.best.objective, "rejected": study.n_rejected}


def _recover_at_every_bearing(analytic: bool, n_trials: int) -> list[dict[str, float]]:
    with ProcessPoolExecutor(len(BEARINGS), mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_recover, BEARINGS, [analytic] * len(BEARINGS), [n_trials] * len(BEARINGS)))


def _check_recovery(found: list[dict[str, float]]) -> None:
    for name, true in TRUTH.items():
        values = [f[name] for f in found]
        for bearing, value in zip(BEARINGS, values, strict=True):
            assert abs(value / true - 1) <= 0.10, f"{name} at {bearing} deg: {value:.3f}, true {true}"
        assert max(values) / min(values) - 1 <= 0.10, f"{name} differs between bearings: {values}"


# ----------------------------------------------------------------------------- settings


def test_settings_hold_the_plan_bounds_and_cite_every_value() -> None:
    s = load_calibration_settings()
    space = s.space("rothermel")
    assert (space["ros_multiplier"].low, space["ros_multiplier"].high, space["ros_multiplier"].log) == (0.3, 3.0, True)
    assert (space["lb_scale"].low, space["lb_scale"].high) == (0.5, 2.0)
    assert set(s.space("alexandridis")) == {"ros_base", "c2"}
    assert s.footprint_m.value == 375.0 and sum(s.weights) > 0
    raw = yaml.safe_load((configs_dir() / "calibration.yaml").read_text(encoding="utf-8"))
    cited = [v for v in raw.values() if isinstance(v, dict) and "value" in v]
    cited += [b for rule in raw["parameters"].values() for b in rule.values()]
    assert cited and all(v["source"] for v in cited)
    assert int(s.truth_seed.value) not in s.seeds()
    with pytest.raises(KeyError, match="constant"):
        s.space("constant")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            '  value: 0.5\n  units: dimensionless\n  source: "PLACEHOLDER',
            '  value: -0.5\n  units: dimensionless\n  source: "PLACEHOLDER',
            "weights",
        ),
        ("  value: 100\n", "  value: 100.5\n", "whole number"),
        ("      low: 0.3\n", "      low: 4.0\n", "low must be < high"),
        ("[-0.2, -0.1, 0.0, 0.1, 0.2]", "[-1.5, 0.0]", "surface_steps"),
    ],
)
def test_settings_reject_bad_values(tmp_path: Path, old: str, new: str, message: str) -> None:
    text = (configs_dir() / "calibration.yaml").read_text(encoding="utf-8")
    assert old in text
    bad = tmp_path / "calibration.yaml"
    bad.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_calibration_settings(bad)


def test_with_params_merges_into_spread_params() -> None:
    config = changed(flat_fire(0.0), spread={"rule": "rothermel", "params": {"lattice_rates": "fitted"}})
    out = with_params(config, {"ros_multiplier": 2, "lb_scale": 0.8})
    assert out.spread.params == {"lattice_rates": "fitted", "ros_multiplier": 2.0, "lb_scale": 0.8}
    assert config.spread.params == {"lattice_rates": "fitted"}


# ----------------------------------------------------------------------------- synthetic observations


def test_synthetic_detections_follow_the_viirs_interval_rule() -> None:
    """``t_detect`` is the first overpass at or after the footprint's earliest arrival, ``t_clear``
    the overpass before it or the ignition; pixels not seen by the window end give nothing."""
    arrival = np.full((40, 40), np.nan)
    arrival[6, 6] = 150.0  # pixel (6, 6): seen at 200, clear at 100
    arrival[18, 20] = 50.0  # pixel (18, 18) sees this cell (2 columns away, inside half a footprint)
    arrival[18, 18] = 260.0
    arrival[30, 30] = 100.0  # exactly at an overpass: detected by it
    arrival[6, 30] = 450.0  # seen only after the window end
    dets = synthetic_detections(arrival, [100.0, 200.0, 300.0, 500.0], 20.0, 400.0, 375.0, 30.0)
    assert dets == [
        Detection(20.0, 100.0, 18, 18),
        Detection(20.0, 100.0, 30, 30),
        Detection(100.0, 200.0, 6, 6),
    ]
    with pytest.raises(ValueError, match="increasing"):
        synthetic_detections(arrival, [200.0, 100.0], 0.0, 400.0, 375.0, 30.0)


def test_truth_scores_zero_against_itself() -> None:
    """The observed fire is a run; the same run is a perfect match: ``J = 1`` and every simulated
    arrival lies in its detection's interval (they were made from it)."""
    config = flat_fire(22.5, size=61, offset=10, window_s=2700.0)
    target, truth = synthetic_target(config, TRUTH, seed=7, overpass_interval_s=900.0, footprint_m=375.0)
    assert target.detections and target.observed.sum() == truth.burned_mask.sum()
    ev = evaluate(config, TRUTH, target, [7], (0.5, 0.5))
    assert ev.objective == 0.0 and ev.overlap == (0.0,) and ev.arrival == (0.0,)
    other = evaluate(config, TRUTH, target, [8, 9], (0.5, 0.5))
    assert other.objective is not None and 0 < other.objective < 0.05  # another seed: only kernel noise
    slow = evaluate(config, {"ros_multiplier": 1.0, "lb_scale": 1.25}, target, [8, 9], (0.5, 0.5))
    assert slow.objective is not None and slow.objective > 5 * other.objective
    assert slow.objective == pytest.approx(
        np.mean([0.5 * o + 0.5 * a for o, a in zip(slow.overlap, slow.arrival, strict=True)])
    )


def test_target_terms_and_validation() -> None:
    config = flat_fire(0.0, size=61, offset=10, window_s=2700.0)
    target, truth = synthetic_target(config, TRUTH, seed=3, overpass_interval_s=900.0, footprint_m=375.0)
    assert target.window_s == 2700.0
    unburned = SimpleNamespace(
        burned_mask=np.zeros_like(truth.burned_mask),
        arrival_time=np.full(truth.burned_mask.shape, np.nan),
        grid=truth.grid,
    )
    overlap, arrival = target.terms(unburned)  # ty: ignore[invalid-argument-type]
    assert overlap == 1.0 and 0 < arrival <= 1.0
    with pytest.raises(ValueError, match="empty"):
        Target(np.zeros((3, 3), bool), (), 0.0, 10.0, 375.0, "x")
    with pytest.raises(ValueError, match="after the ignition"):
        Target(np.ones((3, 3), bool), (), 10.0, 10.0, 375.0, "x")
    with pytest.raises(ValueError, match="after the window end"):
        Target(np.ones((3, 3), bool), (Detection(0.0, 20.0, 1, 1),), 0.0, 10.0, 375.0, "x")
    with pytest.raises(ValueError, match="window"):
        evaluate(changed(config, duration_s=3000.0), TRUTH, target, [1], (0.5, 0.5))


def test_synthetic_truth_must_not_reach_the_edge() -> None:
    with pytest.raises(ValueError, match="grid edge"):
        synthetic_target(flat_fire(0.0, size=31, offset=0), TRUTH, seed=1, overpass_interval_s=900.0, footprint_m=375.0)


# ----------------------------------------------------------------------------- the edge rule


def test_a_run_reaching_the_edge_rejects_the_parameter_set() -> None:
    """The first run to reach the edge rejects the set: no score, and the other seeds are not run."""
    config = flat_fire(0.0, size=61, offset=10, window_s=2700.0)
    target, _ = synthetic_target(config, TRUTH, seed=3, overpass_interval_s=900.0, footprint_m=375.0)
    ev = evaluate(config, {"ros_multiplier": 3.0, "lb_scale": 1.0}, target, [1, 2], (0.5, 0.5))
    assert ev.rejected and ev.objective is None and ev.overlap == () and ev.reached_edge_s is not None
    assert not evaluate(config, TRUTH, target, [1, 2], (0.5, 0.5)).rejected


def test_lb_scale_is_not_identified_above_the_lattice_cap() -> None:
    """In a 12 m/s wind the ratio of GR2 is above the cap of 2.1 for every ``lb_scale`` above 0.83 (``LB`` 2.33), so the
    rule draws the same fire and the objective cannot tell those values apart (docs/calibration.md)."""
    config = changed(
        flat_fire(0.0, size=81, offset=20, window_s=1800.0),
        weather={"series": [{"time_s": 0.0, "wind_speed": 12.0, "wind_direction": 180.0}]},
    )
    target, _ = synthetic_target(config, TRUTH, seed=5, overpass_interval_s=600.0, footprint_m=375.0)
    capped = [
        evaluate(config, {"ros_multiplier": 1.5, "lb_scale": s}, target, [1], (0.5, 0.5)) for s in (1.0, 1.5, 2.0)
    ]
    assert capped[0].objective == capped[1].objective == capped[2].objective
    calm = flat_fire(0.0, size=61, offset=10, window_s=2700.0)
    calm_target, _ = synthetic_target(calm, TRUTH, seed=5, overpass_interval_s=900.0, footprint_m=375.0)
    below = [evaluate(calm, {"ros_multiplier": 1.5, "lb_scale": s}, calm_target, [1], (0.5, 0.5)) for s in (1.0, 1.5)]
    assert below[0].objective != below[1].objective


# ----------------------------------------------------------------------------- objective surface


def test_objective_surface_is_a_clipped_grid_around_the_center() -> None:
    config = flat_fire(0.0, size=61, offset=10, window_s=2700.0)
    target, _ = synthetic_target(config, TRUTH, seed=3, overpass_interval_s=900.0, footprint_m=375.0)
    space = load_calibration_settings().space("rothermel")
    center = {"ros_multiplier": 1.5, "lb_scale": 1.9}
    surface = objective_surface(config, target, center, space, [-0.1, 0.0, 0.1], [1], (0.5, 0.5))
    pairs = {(round(p.params["ros_multiplier"], 6), round(p.params["lb_scale"], 6)) for p in surface}
    assert pairs == {(m, s) for m in (1.35, 1.5, 1.65) for s in (1.71, 1.9, 2.0)}  # 1.9 x 1.1 clipped at 2
    assert len(surface) == 9 and all(p.objective is not None for p in surface)


# ----------------------------------------------------------------------------- real fires


def test_real_fire_says_how_to_build_a_missing_region(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"download_data\.py lake_2024 --observations"):
        real_fire("lake_2024", 24.0, cache_root=tmp_path)


def test_observed_target_needs_daily_perimeters_posted_in_the_window() -> None:
    """Scored against the daily perimeters posted by the window end, never the final perimeter."""
    config = changed(
        flat_fire(0.0, size=21, offset=0, window_s=7200.0),
        start_time_utc="2024-07-05T07:00:00Z",
        ignition={"cells": [(10, 10)], "time_s": 3600.0},
    )
    grid = build_grid(config.grid)
    burned = np.zeros(grid.shape, bool)
    burned[8:12, 8:12] = True
    dets = [Detection(3600.0, 5400.0, 10, 10), Detection(3600.0, 9000.0, 10, 11)]
    obs = SimpleNamespace(daily=["v1"], observed_burn_mask=lambda g, until: burned, detections=lambda g: dets)
    target = observed_target(RealFire("x", config, grid, obs), 375.0)  # ty: ignore[invalid-argument-type]
    assert target.detections == (dets[0],) and target.window_s == 3600.0 and target.observed.sum() == 16
    no_daily = SimpleNamespace(daily=[])
    with pytest.raises(ValueError, match="no daily perimeters"):
        observed_target(RealFire("x", config, grid, no_daily), 375.0)  # ty: ignore[invalid-argument-type]
    none_yet = SimpleNamespace(daily=["v1"], observed_burn_mask=lambda g, until: np.zeros(grid.shape, bool))
    with pytest.raises(ValueError, match="was posted by 2024-07-05 09:00"):
        observed_target(RealFire("x", config, grid, none_yet), 375.0)  # ty: ignore[invalid-argument-type]


# ----------------------------------------------------------------------------- configs/calibrated.yaml


def test_committed_calibrated_yaml_is_an_unmistakable_placeholder() -> None:
    record = load_calibrated()
    assert record.status == "placeholder" and record.provenance is None
    assert record.params == {"ros_multiplier": 1.0, "lb_scale": 1.0} and record.rule == "rothermel"
    assert record.source.startswith("PLACEHOLDER") and "scripts/calibrate.py lake_2024" in record.source
    config = record.apply(flat_fire(0.0))
    assert config.spread.params == {"ros_multiplier": 1.0, "lb_scale": 1.0}
    with pytest.raises(ValueError, match="rule"):
        record.apply(changed(flat_fire(0.0), spread={"rule": "constant"}))
    with pytest.raises(ValueError, match="provenance"):
        CalibratedParams(status="calibrated", source="x", rule="rothermel", params={})


def test_record_round_trips_and_keeps_the_provenance(tmp_path: Path) -> None:
    config = flat_fire(0.0, size=61, offset=10, window_s=2700.0)
    target, _ = synthetic_target(config, TRUTH, seed=3, overpass_interval_s=900.0, footprint_m=375.0)
    space = load_calibration_settings().space("rothermel")
    best = evaluate(config, TRUTH, target, [1], (0.5, 0.5))
    rejected = Evaluation({"ros_multiplier": 3.0, "lb_scale": 1.0}, None, (), (), 1200.0)
    study = StudyResult(best, (rejected, best), (1,), 0)
    record = calibrated_params(
        config, target, study, [best, rejected], space, (0.5, 0.5), status="synthetic", command="x"
    )
    write_calibrated(record, tmp_path / "out" / "calibrated.yaml")
    back = load_calibrated(tmp_path / "out" / "calibrated.yaml")
    assert back == record and back.status == "synthetic" and back.source.startswith("SYNTHETIC")
    p = back.provenance
    assert p is not None and (p.n_trials, p.n_rejected, p.seeds, p.window_hours) == (2, 1, [1], 0.75)
    assert p.objective == best.objective and p.surface[1].objective is None and p.package_version


# ----------------------------------------------------------------------------- the study (optuna)


def test_study_reproduces_and_records_in_storage(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    config = flat_fire(0.0, size=61, offset=10, window_s=2700.0)
    target, _ = synthetic_target(config, TRUTH, seed=3, overpass_interval_s=900.0, footprint_m=375.0)
    space = load_calibration_settings().space("rothermel")
    seen: list[int] = []
    one = run_study(
        config,
        target,
        space,
        n_trials=4,
        seeds=[1],
        weights=(0.5, 0.5),
        sampler_seed=11,
        progress=lambda n, ev: seen.append(n),
    )
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    two = run_study(
        config,
        target,
        space,
        n_trials=4,
        seeds=[1],
        weights=(0.5, 0.5),
        sampler_seed=11,
        storage=storage,
        study_name="s",
    )
    assert one.trials == two.trials and seen == [0, 1, 2, 3] and (tmp_path / "study.db").exists()
    assert one.best.objective == min(t.objective for t in one.trials if t.objective is not None)
    for t in one.trials:
        assert space["ros_multiplier"].low <= t.params["ros_multiplier"] <= space["ros_multiplier"].high


def test_trials_reaching_the_edge_are_rejected_not_scored() -> None:
    """WP 3.1 acceptance. On a small grid many multipliers carry the fire to the edge: those trials
    are counted as rejected and never chosen; if every trial is rejected the study fails."""
    pytest.importorskip("optuna")
    config = flat_fire(0.0, size=61, offset=10, window_s=3600.0)
    target, _ = synthetic_target(config, TRUTH, seed=3, overpass_interval_s=900.0, footprint_m=375.0)
    space = load_calibration_settings().space("rothermel")
    study = run_study(config, target, space, n_trials=12, seeds=[1, 2], weights=(0.5, 0.5), sampler_seed=0)
    assert 0 < study.n_rejected < len(study.trials) == 12
    assert not study.best.rejected and study.best.objective is not None
    scored = [t.objective for t in study.trials if t.objective is not None]
    assert study.best.objective == min(scored) and len(scored) == 12 - study.n_rejected
    fast = {name: b.model_copy(update={"low": 2.5} if name == "ros_multiplier" else {}) for name, b in space.items()}
    with pytest.raises(RuntimeError, match="all 3 trials were rejected"):
        run_study(config, target, fast, n_trials=3, seeds=[1], weights=(0.5, 0.5), sampler_seed=0)


@pytest.mark.slow
def test_recovers_the_parameters_at_every_head_bearing() -> None:
    """WP 3.1 acceptance and milestone M2: the observed fire is a run with known parameters
    (``ros_multiplier`` 1.5, ``lb_scale`` 1.25, truth seed 999, not a calibration seed); a study of
    150 trials x 2 seeds recovers both within 10 % with the head along a grid axis and at 13.3,
    22.5 and 45 degrees to it, and the four recovered values of each agree within 10 %.

    Measured (optuna 5.0, numba engine): the largest error is 2.5 %; across bearings the
    multipliers differ by 2 % and ``lb_scale`` by 4 % (docs/calibration.md)."""
    pytest.importorskip("optuna")
    _check_recovery(_recover_at_every_bearing(analytic=False, n_trials=150))


@pytest.mark.slow
def test_recovers_an_analytic_ellipse_at_every_head_bearing() -> None:
    """The same with the analytic ellipse of the true parameters as the observed fire. A run-made
    observed fire shares the lattice's bearing-dependent shape error with every calibration run,
    so it cancels; against the ellipse it does not, which is the case of a real fire (docs/plan.md
    section 2.4: fitted parameters must not carry the grid orientation).

    Measured: the largest error is 4.0 % (``lb_scale`` at 13.3 degrees); across bearings the
    multipliers differ by 2.7 % and ``lb_scale`` by 7.9 % (docs/calibration.md)."""
    pytest.importorskip("optuna")
    _check_recovery(_recover_at_every_bearing(analytic=True, n_trials=150))


# ----------------------------------------------------------------------------- scripts/calibrate.py


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("calibrate", REPO / "scripts" / "calibrate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_lists_the_regions_and_checks_its_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        _script().main(["--help"])
    assert stop.value.code == 0 and "lake_2024" in capsys.readouterr().out
    for argv, message in (
        ([], "either a region or --synthetic"),
        (["lake_2024", "--truth", "lb_scale=1"], "--synthetic only"),
        (["lake_2024", "--rule", "constant"], "no calibration parameters"),
    ):
        with pytest.raises(SystemExit) as stop:
            _script().main(argv)
        assert stop.value.code == 2 and message in capsys.readouterr().err


def test_script_says_how_to_build_a_missing_region(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _script().main(["lake_2024", "--cache-root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "not built" in out and "download_data.py lake_2024 --observations" in out


@pytest.mark.slow
def test_script_runs_on_the_committed_crop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """WP 3.1 acceptance: the script on the committed 5 km crop of fire A with its HRRR weather,
    3 trials x 2 seeds. The crop has no observations offline (they need a NASA Earthdata token),
    so the "observed" fire is synthetic (``--synthetic``): a run with known parameters, ignited at
    the crop's centre at fire A's ignition time, over a 1-hour window in which even the fastest
    multiplier stays inside the crop."""
    pytest.importorskip("optuna")
    scenario = tmp_path / "crop.yaml"
    scenario.write_text(
        yaml.safe_dump(
            {
                "name": "lake_2024 5 km crop",
                "grid": {"kind": "cached", "path": str(FIXTURES / "lake_2024_5km.npz")},
                "spread": {"rule": "rothermel"},
                "ignition": {"center": True, "time_s": 56880.0},
                "dt_s": 60.0,
                "duration_s": 56880.0 + 3600.0,
                "engine": ENGINE,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "calibrated.yaml"
    argv = ["--synthetic", str(scenario), "--weather", str(FIXTURES / "lake_2024_weather.yaml")]
    argv += ["--truth", "ros_multiplier=1.2", "--truth", "lb_scale=1.1", "--window-hours", "1"]
    argv += ["--trials", "3", "--seeds", "2", "--out", str(out)]
    assert _script().main(argv) == 0
    text = capsys.readouterr().out
    assert "best of 3 trials" in text and "objective surface" in text
    record = load_calibrated(out)
    p = record.provenance
    assert record.status == "synthetic" and p is not None
    assert (p.n_trials, len(p.seeds), p.window_hours) == (3, 2, 1.0) and len(p.surface) == 25
    assert set(record.params) == {"ros_multiplier", "lb_scale"} and p.objective is not None


def test_script_rejects_a_truth_that_names_other_parameters(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scenario = tmp_path / "flat.yaml"
    scenario.write_text(yaml.safe_dump(flat_fire(0.0).model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(SystemExit) as stop:
        _script().main(["--synthetic", str(scenario), "--truth", "ros_multiplier=1.2"])
    assert stop.value.code == 2 and "--truth must give exactly" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stop:
        _script().main(["--synthetic", str(scenario), "--truth", "ros_multiplier"])
    assert stop.value.code == 2 and "NAME=VALUE" in capsys.readouterr().err
