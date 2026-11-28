"""Tests for the validation of WP 3.3 (``wildfire_sim.evaluation``, ``scripts/validate.py``, the notebook).

Real regions are not available offline, so:

* the scoring and the aggregation are checked by hand on small arrays;
* the real-fire path (daily perimeters for fire A, none for fire B) runs on stubbed observations,
  as in ``tests/test_calibration.py``;
* the whole pipeline (ensemble, metrics, tables, figures) runs end to end in the script's
  ``--synthetic`` mode on the committed 5 km crop of fire A with its HRRR weather, where a run
  with known parameters stands in for the observations; its outputs are labelled synthetic.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from wildfire_sim.calibration import (
    CalibratedParams,
    RealFire,
    StudyResult,
    calibrated_params,
    evaluate,
    load_calibrated,
    load_calibration_settings,
    synthetic_target,
    write_calibrated,
)
from wildfire_sim.config import ScenarioConfig, configs_dir
from wildfire_sim.data.validation import Detection
from wildfire_sim.ensemble import run_ensemble
from wildfire_sim.evaluation import (
    Member,
    Observed,
    ParameterRecord,
    RunScore,
    ValidationRuns,
    check_not_tuned_on,
    check_seeds_disjoint,
    day_ends,
    detection_summaries,
    load_validation_settings,
    metric_rows,
    observed_fire,
    run_member,
    score_run,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENGINE = "numba" if importlib.util.find_spec("numba") else "numpy"
SYNTHETIC_ARGS = ["--weather", str(FIXTURES / "lake_2024_weather.yaml"), "--window-hours", "2"]
SYNTHETIC_ARGS += ["--truth", "ros_multiplier=1.2", "--truth", "lb_scale=1.1"]


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate", REPO / "scripts" / "validate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _crop_scenario(path: Path) -> Path:
    """The committed crop of fire A, ignited at its centre at fire A's ignition time (as tests/test_calibration.py)."""
    path.write_text(
        yaml.safe_dump(
            {
                "name": "lake_2024 5 km crop",
                "grid": {"kind": "cached", "path": str(FIXTURES / "lake_2024_5km.npz")},
                "spread": {"rule": "rothermel"},
                "ignition": {"center": True, "time_s": 56880.0},
                "dt_s": 60.0,
                "duration_s": 60480.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def flat_fire(size: int = 41, window_s: float = 1800.0, wind: float = 6.0) -> ScenarioConfig:
    """Flat uniform GR2 lit at the centre, a steady west wind."""
    return ScenarioConfig.model_validate(
        {
            "name": "flat GR2",
            "grid": {"kind": "synthetic", "generator": "flat_uniform", "shape": [size, size], "fuel_model": 102},
            "weather": {"series": [{"time_s": 0.0, "wind_speed": wind, "wind_direction": 270.0}]},
            "spread": {"rule": "rothermel"},
            "ignition": {"center": True},
            "dt_s": 60.0,
            "duration_s": window_s,
            "engine": ENGINE,
        }
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------- settings and checks


def test_settings_cite_every_value_and_cover_both_fires() -> None:
    s = load_validation_settings()
    raw = yaml.safe_load((configs_dir() / "validation.yaml").read_text(encoding="utf-8"))
    cited = [v for v in raw.values() if isinstance(v, dict) and "value" in v]
    cited += list(raw["window_hours"].values()) + list(raw["ensemble"].values())
    assert len(cited) == 14 and all(v["source"] for v in cited)
    assert set(s.window_hours) == {"lake_2024", "silverado_2020"}
    assert s.consensus_threshold.value == 0.5  # docs/plan.md section 7
    cal = load_calibration_settings()
    check_seeds_disjoint(s.seeds(), {*cal.seeds(), int(cal.truth_seed.value)})
    config = s.with_ensemble(flat_fire(), n_runs=7)
    assert config.ensemble.n_runs == 7 and config.ensemble.wind_direction_sigma_deg == 15.0
    assert config.ensemble.ignition_jitter_cells == 0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [("  value: 10000\n", "  value: 50\n", "n_boot"), ("  value: 0.5\n", "  value: 1.5\n", "consensus_threshold")],
)
def test_settings_reject_bad_values(tmp_path: Path, old: str, new: str, message: str) -> None:
    text = (configs_dir() / "validation.yaml").read_text(encoding="utf-8")
    assert old in text
    bad = tmp_path / "validation.yaml"
    bad.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_validation_settings(bad)


def test_seed_and_retuning_guards() -> None:
    with pytest.raises(ValueError, match="1 validation seeds were used"):
        check_seeds_disjoint([5, 6], [6, 7])
    provenance = SimpleNamespace(target="region silverado_2020, 48 h")
    record = SimpleNamespace(provenance=provenance)
    with pytest.raises(ValueError, match="calibrated on silverado_2020 itself"):
        check_not_tuned_on(record, "silverado_2020")  # ty: ignore[invalid-argument-type]
    check_not_tuned_on(record, "lake_2024")  # ty: ignore[invalid-argument-type]
    check_not_tuned_on(load_calibrated(), "silverado_2020")  # a placeholder was tuned on nothing


def test_parameter_record_says_what_the_parameters_are(tmp_path: Path) -> None:
    committed = ParameterRecord.read(load_calibrated(), configs_dir() / "calibrated.yaml")
    assert committed.status == "calibrated" and committed.calibrated_on == "region lake_2024, 48.5 h"
    assert len(committed.sha256) == 64 and committed.warning() is None
    placeholder = ParameterRecord("placeholder", "rothermel", {}, "x", "p", "0" * 64, None)
    assert "NOT CALIBRATED" in (placeholder.warning() or "")
    synthetic = ParameterRecord("synthetic", "rothermel", {}, "x", "p", "0" * 64, "t")
    assert "SYNTHETIC" in (synthetic.warning() or "")
    assert ParameterRecord("calibrated", "rothermel", {}, "x", "p", "0" * 64, "t").warning() is None


# ----------------------------------------------------------------------------- scoring by hand


def test_day_ends() -> None:
    assert day_ends(100.0, 100.0 + 2 * 86400.0) == [100.0 + 86400.0, 100.0 + 2 * 86400.0]
    assert day_ends(0.0, 7200.0) == [7200.0]
    assert day_ends(0.0, 90000.0) == [86400.0, 90000.0]
    with pytest.raises(ValueError, match="after the ignition"):
        day_ends(5.0, 5.0)


def _observed(window: bool = True) -> Observed:
    """6 x 6 cells of 30 m: observed window mask rows 1-2 x cols 1-3 (6 cells), final rows 1-3 x cols 1-3 (9)."""
    window_mask = np.zeros((6, 6), bool)
    window_mask[1:3, 1:4] = True
    final = np.zeros((6, 6), bool)
    final[1:4, 1:4] = True
    early = np.zeros((6, 6), bool)
    early[1, 1:3] = True
    return Observed(
        description="hand-made",
        synthetic=False,
        ignition_s=0.0,
        window_end_s=1000.0,
        footprint_m=30.0,  # half a footprint is 0 cells: the detection's own cell only
        ignition_cell=(1, 1),
        dx=30.0,
        detections=(Detection(0.0, 400.0, 1, 2), Detection(400.0, 800.0, 2, 3), Detection(400.0, 800.0, 5, 5)),
        window_mask=window_mask if window else None,
        final_mask=final,
        final_label="final perimeter (includes suppression)",
        daily=((500.0, early), (1000.0, window_mask)),
    )


def _member(arrival: np.ndarray, edge: float = math.nan, seed: int = 1, capped: float = 0.0) -> Member:
    return Member(seed, {"ros_multiplier": 1.0}, arrival, edge, capped, 2)


def test_score_run_by_hand() -> None:
    arrival = np.full((6, 6), np.nan)
    arrival[1, 1:4] = [0.0, 300.0, 600.0]  # 3 cells
    arrival[2, 1:3] = [700.0, 900.0]  # 2 cells
    arrival[0, 0] = 1200.0  # after the window end: unburned
    score = score_run(_member(arrival), _observed())
    # burned 5 cells, all inside the window mask (6) and the final mask (9)
    assert score.jaccard_window == pytest.approx(5 / 6) and score.sorensen_window == pytest.approx(10 / 11)
    assert score.jaccard_final == pytest.approx(5 / 9) and score.sorensen_final == pytest.approx(10 / 14)
    assert score.burned_area_ha == pytest.approx(5 * 900 / 1e4)
    # detection 1: 300 in (0, 400] -> 0; detection 2: cell (2, 3) never burns -> window end - t_detect = 200;
    # detection 3: never reached -> 200
    assert score.detection_arrival_s[:1] == (300.0,) and math.isnan(score.detection_arrival_s[1])
    assert score.detection_error_s == (0.0, 200.0, 200.0)
    assert score.inside_fraction == pytest.approx(1 / 3) and score.median_error_h == pytest.approx(200 / 3600)
    assert score.area_ratio == (pytest.approx(2 / 2), pytest.approx(5 / 6))  # by 500 s: 2 cells of 2; by 1000 s: 5 of 6
    no_window = score_run(_member(arrival), _observed(window=False))
    assert math.isnan(no_window.jaccard_window) and no_window.jaccard_final == score.jaccard_final


def _score(seed: int, area: float, jac: float, edge: float = math.nan, capped: float = 0.0) -> RunScore:
    return RunScore(
        seed,
        {},
        edge,
        area,
        capped,
        seed,
        jac,
        jac,
        jac,
        jac,
        (100.0, 500.0, math.nan),
        (0.0, 100.0 * seed, 200.0),
        (1.0 + seed, 2.0 + seed),
    )


def test_metrics_leave_out_the_runs_that_reached_the_edge() -> None:
    """Three runs, the second reached the edge: it is counted, listed, kept out of every metric and
    included only in the burned area labelled as a lower bound."""
    observed = _observed()
    runs = (_score(1, 10.0, 0.5, capped=2.0), _score(2, 99.0, 0.0, edge=600.0), _score(3, 20.0, 0.7))
    count = np.zeros((6, 6), np.int64)
    count[1, 1:4] = [2, 1, 0]  # consensus (>= 0.5 of 2 runs): (1, 1) and (1, 2)
    result = ValidationRuns(runs, count, 0.0)
    assert [i for i, _ in result.edge] == [1] and [r.seed for r in result.inside] == [1, 3]
    s = load_validation_settings()
    rows = {
        (r.metric, r.statistic.split(" (")[0].split(",")[0], r.comparison.split(" (")[0]): r
        for r in metric_rows(result, observed, s, rule_caps=True)
    }
    edge = rows[("runs_reaching_edge", "count", "grid edge")]
    assert (edge.value, edge.n_runs) == (1, 3)
    window = "observed burned by the window end"
    assert rows[("jaccard", "consensus mask", window)].value == pytest.approx(2 / 6)
    mean = rows[("jaccard", "mean over runs", window)]
    assert mean.value == pytest.approx(0.6) and mean.n_runs == 2 and mean.lower <= 0.6 <= mean.upper
    dets = "3 VIIRS first detections in the window"
    # per run: mean of (0, 100, 200) s = 100 s and of (0, 300, 200) s = 166.7 s; median 100 s and 200 s
    assert rows[("arrival_error_mean", "mean over runs", dets)].value == pytest.approx((100 + 500 / 3) / 2 / 3600)
    assert rows[("arrival_error_median", "mean over runs", dets)].value == pytest.approx(150 / 3600)
    assert rows[("arrival_inside_fraction", "mean over runs", dets)].value == pytest.approx(1 / 3)
    assert rows[("area_ratio", "mean over runs", "observed burned 0.138889 h after the ignition")].value == 3.0
    inside = rows[("burned_area", "mean", "runs inside the grid")]
    assert inside.value == pytest.approx(15.0) and inside.lower < 15.0 < inside.upper
    lower_bound = rows[("burned_area", "mean", "all runs")]
    assert lower_bound.value == pytest.approx(43.0) and "lower bound" in lower_bound.note
    assert rows[("lb_capped_cell_hours", "mean over runs", "lattice cap")].value == pytest.approx(1.0)
    assert rows[("runs_capped_fraction", "mean over runs", "lattice cap")].value == pytest.approx(0.5)
    assert rows[("breaches", "mean over runs", "breaks")].value == pytest.approx(2.0)  # seeds 1 and 3
    no_cap = metric_rows(result, observed, s, rule_caps=False)
    assert [r.statistic for r in no_cap if r.metric == "lb_capped_cell_hours"] == ["not applicable"]


def test_metrics_with_every_run_at_the_edge_hold_only_the_count() -> None:
    runs = (_score(1, 10.0, 0.5, edge=60.0), _score(2, 12.0, 0.5, edge=90.0))
    result = ValidationRuns(runs, np.zeros((6, 6), np.int64), 0.0)
    assert result.burn_probability is None
    rows = metric_rows(result, _observed(), load_validation_settings(), rule_caps=True)
    assert [(r.metric, r.value) for r in rows] == [("runs_reaching_edge", 2)]


def test_detection_summaries_count_unreached_runs_as_late() -> None:
    observed = _observed()
    runs = []
    for seed, t in enumerate([100.0, 200.0, math.nan, math.nan, math.nan]):
        r = _score(seed + 1, 1.0, 0.5)
        runs.append(replace(r, detection_arrival_s=(t, 450.0, math.nan)))
    summaries = detection_summaries(ValidationRuns(tuple(runs), np.zeros((6, 6), np.int64), 0.0), observed)
    first, second, third = summaries
    assert first.reached_fraction == pytest.approx(0.4) and math.isinf(first.median)  # 3 of 5 never reached
    assert first.q05 == 100.0 and first.median_error_s == pytest.approx(600.0)  # max(0, 1000 - 400)
    assert second.median == 450.0 and second.median_error_s == 0.0
    assert third.reached_fraction == 0.0


# ----------------------------------------------------------------------------- the real-fire path (stubbed)


def _stub_fire(daily: bool) -> RealFire:
    config = flat_fire(size=21, window_s=3600.0 + 2 * 86400.0)
    config = ScenarioConfig.model_validate(
        {
            **config.model_dump(),
            "start_time_utc": "2024-07-05T07:00:00Z",
            "ignition": {"cells": [(10, 10)], "time_s": 3600.0},
        }
    )
    grid = SimpleNamespace(dx=30.0, shape=(21, 21))
    final = np.zeros((21, 21), bool)
    final[5:15, 5:15] = True
    seen = []

    def burned(g: object, until: object = None) -> np.ndarray:
        if until is None:
            return final
        seen.append(until)
        mask = np.zeros((21, 21), bool)
        mask[9:11, 9 : 9 + len(seen)] = True
        return mask

    dets = [Detection(3600.0, 5400.0, 10, 10), Detection(3600.0, 4 * 86400.0, 10, 11)]
    obs = SimpleNamespace(
        daily=["v1"] if daily else [],
        observed_burn_mask=burned,
        detections=lambda g: dets,
        ignition_cell=lambda g: (10, 10),
    )
    return RealFire("lake_2024" if daily else "silverado_2020", config, grid, obs)  # ty: ignore[invalid-argument-type]


def test_observed_fire_with_daily_perimeters() -> None:
    """Fire A: the window mask is the daily perimeters posted by the window end, and each day end has its own."""
    observed = observed_fire(_stub_fire(daily=True), 375.0)
    assert [t for t, _ in observed.daily] == [3600.0 + 86400.0, 3600.0 + 2 * 86400.0]
    assert observed.window_mask is not None and observed.window_mask.sum() == 4  # the second posting (2 x 2)
    assert observed.final_mask.sum() == 100 and len(observed.detections) == 1 and not observed.notes
    assert "includes suppression" in observed.final_label and not observed.synthetic


def test_observed_fire_without_daily_perimeters() -> None:
    """Fire B: no daily perimeters (NIFC's archive starts in 2022), so no window mask and no daily ratio;
    the comparison is the detections and the final perimeter, labelled as including suppression."""
    observed = observed_fire(_stub_fire(daily=False), 375.0)
    assert observed.window_mask is None and observed.daily == ()
    assert "no daily perimeters" in observed.notes[0] and "includes suppression" in observed.final_label
    runs = (_score(1, 10.0, 0.5), _score(2, 12.0, 0.6))
    runs = tuple(replace(r, detection_arrival_s=(5000.0,), detection_error_s=(0.0,), area_ratio=()) for r in runs)
    count = np.zeros((21, 21), np.int64)
    count[8:12, 8:12] = 2
    rows = metric_rows(ValidationRuns(runs, count, 0.0), observed, load_validation_settings(), rule_caps=True)
    comparisons = {r.comparison for r in rows if r.metric == "jaccard"}
    assert comparisons == {"final MTBS perimeter (includes suppression)"}
    assert all("includes suppression" in r.note for r in rows if r.metric in ("jaccard", "sorensen"))
    assert not [r for r in rows if r.metric == "area_ratio"]
    consensus = next(r for r in rows if r.metric == "jaccard" and r.statistic.startswith("consensus"))
    assert consensus.value == pytest.approx(16 / 100)


# ----------------------------------------------------------------------------- the ensemble member


def test_member_is_the_ensemble_member_and_counts_capped_cell_hours() -> None:
    """``run_member`` draws and applies the perturbation exactly as ``run_ensemble`` (bit for bit), and a run
    whose rate the perturbation scales still reports the cell-hours under the lattice cap."""
    config = flat_fire(size=41, window_s=900.0, wind=15.0)  # GR2 in a 15 m/s wind: above the cap of 2.1
    config = load_validation_settings().with_ensemble(config, n_runs=3)
    ensemble = run_ensemble(config, n_runs=3, base_seed=4)
    for i, seed in enumerate(ensemble.seeds):
        member = run_member(config, seed)
        np.testing.assert_array_equal(member.arrival.astype(np.float32), ensemble.arrival_times[i])
        assert member.perturbation == {k: ensemble.perturbations[i][k] for k in member.perturbation}
        assert member.perturbation["ros_multiplier"] != 1.0 and member.lb_capped_cell_hours > 0
    jitter = ScenarioConfig.model_validate({**config.model_dump(), "ensemble": {"ignition_jitter_cells": 1}})
    with pytest.raises(ValueError, match="ignition_jitter_cells"):
        run_member(jitter, 1)


# ----------------------------------------------------------------------------- scripts/validate.py


def test_script_checks_its_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    for argv, message in (
        ([], "exactly one of"),
        (["lake_2024", "--truth", "lb_scale=1"], "--synthetic only"),
        (["lake_2024", "--workers", "0"], ">= 1"),
    ):
        with pytest.raises(SystemExit) as stop:
            _script().main(argv)
        assert stop.value.code == 2 and message in capsys.readouterr().err


def test_script_says_how_to_build_a_missing_region(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _script().main(["lake_2024", "--cache-root", str(tmp_path), "--out", str(tmp_path / "out")]) == 1
    out = capsys.readouterr().out
    assert "not built" in out and "download_data.py lake_2024 --observations" in out
    assert not (tmp_path / "out").exists()


def test_script_refuses_parameters_that_do_not_fit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Another rule's parameters, and parameters tuned on fire B itself, are refused before any data is read."""
    argv = ["--cache-root", str(tmp_path), "--out", str(tmp_path / "out")]
    assert _script().main(["lake_2024", "--rule", "alexandridis", *argv]) == 1
    err = capsys.readouterr().err
    assert "rule 'rothermel'" in err and "calibrate.py lake_2024 --rule alexandridis --out" in err
    data = yaml.safe_load((configs_dir() / "calibrated.yaml").read_text(encoding="utf-8"))
    config = flat_fire(size=31, window_s=1200.0)
    target, _ = synthetic_target(config, data["params"], seed=3, overpass_interval_s=600.0, footprint_m=375.0)
    best = evaluate(config, data["params"], target, [1], (0.5, 0.5))
    record = calibrated_params(
        config,
        target,
        StudyResult(best, (best,), (1,), 0),
        [],
        load_calibration_settings().space("rothermel"),
        (0.5, 0.5),
        status="calibrated",
        command="x",
    )
    assert record.provenance is not None
    on_b = record.model_copy(
        update={"provenance": record.provenance.model_copy(update={"target": "region silverado_2020, 48 h"})}
    )
    write_calibrated(on_b, tmp_path / "on_b.yaml")
    assert _script().main(["silverado_2020", "--calibrated", str(tmp_path / "on_b.yaml"), *argv]) == 1
    assert "calibrated on silverado_2020 itself" in capsys.readouterr().err


def _synthetic_calibration(path: Path) -> Path:
    """A synthetic calibration record with an objective surface (no optuna needed)."""
    config = flat_fire(size=31, window_s=1200.0)
    truth = {"ros_multiplier": 1.2, "lb_scale": 1.1}
    target, _ = synthetic_target(config, truth, seed=3, overpass_interval_s=600.0, footprint_m=375.0)
    surface = [evaluate(config, {"ros_multiplier": m, "lb_scale": 1.1}, target, [1], (0.5, 0.5)) for m in (1.0, 1.2)]
    surface += [evaluate(config, {"ros_multiplier": m, "lb_scale": 1.3}, target, [1], (0.5, 0.5)) for m in (1.0, 1.2)]
    study = StudyResult(surface[1], tuple(surface), (1,), 0)
    space = load_calibration_settings().space("rothermel")
    record = calibrated_params(config, target, study, surface, space, (0.5, 0.5), status="synthetic", command="x")
    write_calibrated(record, path)
    return path


def test_script_end_to_end_on_a_synthetic_fire(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The whole pipeline on the committed crop with a synthetic observed fire: every table and figure is
    written, labelled synthetic, with the parameters' status; one and two workers give identical files."""
    scenario = _crop_scenario(tmp_path / "crop.yaml")
    calibrated = _synthetic_calibration(tmp_path / "calibrated.yaml")
    outs = []
    for workers in (1, 2):
        out = tmp_path / f"w{workers}"
        argv = ["--synthetic", str(scenario), *SYNTHETIC_ARGS, "--runs", "4", "--workers", str(workers)]
        assert _script().main([*argv, "--calibrated", str(calibrated), "--out", str(out)]) == 0
        outs.append(out)
    text = capsys.readouterr()
    assert "SYNTHETIC OBSERVED FIRE - NOT A VALIDATION RESULT" in text.out
    assert "PARAMETERS FROM A SYNTHETIC CALIBRATION" in text.err and "EDGE: 0 of 4 runs" in text.out
    prefix = "synthetic_lake_2024_5_km_crop_rothermel"
    files = {p.name for p in outs[0].iterdir()}
    for suffix in ("summary.json", "metrics.csv", "runs.csv", "detections.csv", "convergence.csv", "maps.npz"):
        assert f"{prefix}_{suffix}" in files
    for figure in ("burn_probability", "arrival", "convergence", "surface"):
        assert f"{prefix}_{figure}.png" in files
    for name in files - {f"{prefix}_summary.json"}:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
    summary = json.loads((outs[0] / f"{prefix}_summary.json").read_text(encoding="utf-8"))
    other = json.loads((outs[1] / f"{prefix}_summary.json").read_text(encoding="utf-8"))
    for record in (summary, other):
        del record["provenance"]["command"]  # names the worker count and the output directory
    assert summary == other
    assert summary["synthetic"] and summary["label"].startswith("SYNTHETIC")
    assert summary["parameters"]["status"] == "synthetic" and summary["parameters"]["retuned"] is False
    assert summary["edge"] == {"n_reached": 0, "n_runs": 4, "m2_no_run_reaches_edge": True, "runs": []}
    assert len(summary["ensemble"]["seeds"]) == 4 and summary["provenance"]["package_version"]
    assert sorted(summary["files"]) == sorted(files)
    metrics = _rows(outs[0] / f"{prefix}_metrics.csv")
    assert metrics[0]["metric"] == "runs_reaching_edge" and metrics[0]["value"] == "0"
    names = {m["metric"] for m in metrics}
    assert {
        "jaccard",
        "sorensen",
        "arrival_inside_fraction",
        "area_ratio",
        "burned_area",
        "lb_capped_cell_hours",
    } <= names
    runs = _rows(outs[0] / f"{prefix}_runs.csv")
    assert [int(r["seed"]) for r in runs] == summary["ensemble"]["seeds"]
    with np.load(outs[0] / f"{prefix}_maps.npz") as maps:
        assert set(maps.files) == {"burn_probability", "consensus", "window_mask", "final_mask"}


def test_script_counts_runs_at_the_edge_and_fails_if_none_stayed_inside(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With parameters three times faster than the synthetic fire every run reaches the edge of a small grid:
    they are listed, nothing is scored, and the exit code is 1; with the placeholder parameters the figures
    are stamped as not calibrated."""
    scenario = tmp_path / "flat.yaml"
    scenario.write_text(yaml.safe_dump(flat_fire(size=31).model_dump(mode="json")), encoding="utf-8")
    fast = CalibratedParams(
        status="placeholder", source="test", rule="rothermel", params={"ros_multiplier": 3.0, "lb_scale": 1.0}
    )
    write_calibrated(fast, tmp_path / "fast.yaml")
    argv = ["--synthetic", str(scenario), "--window-hours", "0.5", "--runs", "3", "--out", str(tmp_path / "out")]
    argv += ["--truth", "ros_multiplier=1", "--truth", "lb_scale=1", "--calibrated", str(tmp_path / "fast.yaml")]
    assert _script().main(argv) == 1
    err = capsys.readouterr().err
    assert "EDGE: 3 of 3 runs reached the grid edge" in err and "every run reached the grid edge" in err
    assert "PARAMETERS NOT CALIBRATED" in err
    summary = json.loads((tmp_path / "out" / "synthetic_flat_gr2_rothermel_summary.json").read_text(encoding="utf-8"))
    assert summary["edge"]["n_reached"] == 3 and not summary["edge"]["m2_no_run_reaches_edge"]
    assert [r["run"] for r in summary["edge"]["runs"]] == [0, 1, 2]
    assert [m["metric"] for m in summary["metrics"]] == ["runs_reaching_edge"]
    assert "PARAMETERS NOT CALIBRATED (placeholder, produced by no study)" in summary["warnings"]
    assert all(r["reached_edge_s"] != "nan" for r in _rows(tmp_path / "out" / "synthetic_flat_gr2_rothermel_runs.csv"))


def test_script_scores_only_the_runs_that_stayed_inside(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The same on a larger grid: two of the three runs reach the edge and one, with a slower draw of the
    inputs, stays inside. Only that one is scored; the edge runs are listed, and enter only the burned area
    over all runs, labelled as a lower bound."""
    scenario = tmp_path / "flat.yaml"
    scenario.write_text(yaml.safe_dump(flat_fire(size=41).model_dump(mode="json")), encoding="utf-8")
    fast = CalibratedParams(
        status="placeholder", source="test", rule="rothermel", params={"ros_multiplier": 3.0, "lb_scale": 1.0}
    )
    write_calibrated(fast, tmp_path / "fast.yaml")
    argv = ["--synthetic", str(scenario), "--window-hours", "0.5", "--runs", "3", "--out", str(tmp_path / "out")]
    argv += ["--truth", "ros_multiplier=1", "--truth", "lb_scale=1", "--calibrated", str(tmp_path / "fast.yaml")]
    assert _script().main(argv) == 0
    assert "EDGE: 2 of 3 runs reached the grid edge" in capsys.readouterr().err
    summary = json.loads((tmp_path / "out" / "synthetic_flat_gr2_rothermel_summary.json").read_text(encoding="utf-8"))
    edge_runs = [r["run"] for r in summary["edge"]["runs"]]
    runs = _rows(tmp_path / "out" / "synthetic_flat_gr2_rothermel_runs.csv")
    assert len(edge_runs) == 2 and [i for i, r in enumerate(runs) if r["reached_edge_s"] != "nan"] == edge_runs
    (inside,) = [r for i, r in enumerate(runs) if i not in edge_runs]
    metrics = {(m["metric"], m["comparison"], m["statistic"]): m for m in summary["metrics"]}
    area = metrics[("burned_area", "runs inside the grid", "mean")]
    assert area["n_runs"] == 1 and area["value"] == pytest.approx(float(inside["burned_area_ha"]), rel=1e-5)
    everything = metrics[("burned_area", "all runs", "mean (lower bound)")]
    assert everything["n_runs"] == 3 and everything["value"] == pytest.approx(
        np.mean([float(r["burned_area_ha"]) for r in runs]), rel=1e-5
    )
    assert all(m["n_runs"] == 1 for k, m in metrics.items() if k[0] in ("jaccard", "sorensen", "arrival_error_mean"))


def test_script_runs_the_alexandridis_rule(tmp_path: Path) -> None:
    """The other rule on the same crop: no lattice cap, so its cell-hours under the cap are not applicable."""
    scenario = _crop_scenario(tmp_path / "crop.yaml")
    data = yaml.safe_load(scenario.read_text(encoding="utf-8"))
    data["spread"] = {"rule": "alexandridis"}
    scenario.write_text(yaml.safe_dump(data), encoding="utf-8")
    record = CalibratedParams(
        status="placeholder", source="test", rule="alexandridis", params={"ros_base": 0.1, "c2": 0.131}
    )
    write_calibrated(record, tmp_path / "alex.yaml")
    argv = ["--synthetic", str(scenario), "--weather", str(FIXTURES / "lake_2024_weather.yaml"), "--window-hours", "1"]
    argv += ["--truth", "ros_base=0.12", "--truth", "c2=0.131", "--runs", "2", "--out", str(tmp_path / "out")]
    assert _script().main([*argv, "--calibrated", str(tmp_path / "alex.yaml")]) == 0
    metrics = _rows(tmp_path / "out" / "synthetic_lake_2024_5_km_crop_alexandridis_metrics.csv")
    capped = [m for m in metrics if m["metric"] == "lb_capped_cell_hours"]
    assert [m["statistic"] for m in capped] == ["not applicable"]


def _summary(rule: str, value: float | None, target: str = "region lake_2024, 48 h") -> dict[str, object]:
    common: dict[str, object] = {"lower": None, "upper": None, "n_runs": 10, "note": ""}
    row = {**common, "metric": "jaccard", "comparison": "final", "statistic": "consensus", "value": value}
    row["units"] = "dimensionless"
    edge = {**common, "metric": "runs_reaching_edge", "comparison": "grid edge", "statistic": "count", "value": 0}
    edge["units"] = "runs"
    return {
        "name": "lake_2024",
        "synthetic": False,
        "target": {"description": target},
        "parameters": {"rule": rule, "status": "calibrated", "path": f"{rule}.yaml"},
        "edge": {"n_reached": 0},
        "metrics": [edge, row] if rule == "rothermel" else [row],
    }


def test_compare_two_rules(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = []
    for rule, value in (("rothermel", 0.6), ("alexandridis", None)):
        paths.append(tmp_path / f"{rule}.json")
        paths[-1].write_text(json.dumps(_summary(rule, value)), encoding="utf-8")
    assert _script().main(["--compare", *map(str, paths), "--out", str(tmp_path)]) == 0
    rows = _rows(tmp_path / "lake_2024_rule_comparison.csv")
    assert len(rows) == 1  # only metrics present in both
    assert rows[0]["rothermel_value"] == "0.6" and rows[0]["alexandridis_value"] == "nan"
    record = json.loads((tmp_path / "lake_2024_rule_comparison.json").read_text(encoding="utf-8"))
    assert set(record["rules"]) == {"rothermel", "alexandridis"}
    paths[1].write_text(json.dumps(_summary("alexandridis", 0.5, target="region lake_2024, 24 h")), encoding="utf-8")
    assert _script().main(["--compare", *map(str, paths), "--out", str(tmp_path)]) == 1
    assert "not of the same target" in capsys.readouterr().err
    assert _script().main(["--compare", str(paths[0]), str(paths[0]), "--out", str(tmp_path)]) == 1


# ----------------------------------------------------------------------------- notebooks/validation.ipynb


def _run_notebook(out: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    notebook = json.loads((REPO / "notebooks" / "validation.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(
        not c.get("outputs") and c.get("execution_count") is None for c in notebook["cells"] if c["cell_type"] == "code"
    )
    monkeypatch.setenv("VALIDATION_OUT", str(out))
    monkeypatch.chdir(REPO / "notebooks")
    printed: list[str] = []
    namespace: dict[str, object] = {"print": lambda *a, **k: printed.append(" ".join(map(str, a)))}
    import matplotlib

    matplotlib.use("Agg")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "validation.ipynb", "exec"), namespace)
    import matplotlib.pyplot as plt

    plt.close("all")
    return "\n".join(printed)


def test_notebook_runs_without_outputs_and_says_what_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = _run_notebook(tmp_path, monkeypatch)
    assert "No validation outputs" in text and "scripts/validate.py lake_2024" in text


def test_notebook_shows_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _crop_scenario(tmp_path / "crop.yaml")
    argv = ["--synthetic", str(scenario), *SYNTHETIC_ARGS, "--runs", "2", "--out", str(tmp_path / "out")]
    assert _script().main(argv) == 0
    text = _run_notebook(tmp_path / "out", monkeypatch)
    assert "SYNTHETIC OBSERVED FIRE" in text and "runs_reaching_edge" in text
