"""Golden-run regression tests (WP 1.9, docs/plan.md section 8).

Three compact scenarios under ``tests/golden/`` (31 x 41 cells, at most a few hundred steps)
have their arrival-time arrays committed at a fixed seed:

* ``flat_no_wind``: constant rule, no wind, 8-neighbour stencil; the whole grid burns.
* ``steady_wind_slope``: Alexandridis rule, 5 m/s west wind, west-facing 0.2 slope; the run
  stops with the head still inside the grid.
* ``river_barrier``: constant rule, a one-cell river, 16-neighbour stencil; only the west side burns.

Each test reruns a scenario and requires the same burned set exactly and every arrival time
within ``rtol = 1e-7`` (plus ``atol = 1e-6`` s for the ignition at t = 0). The runs are
bit-stable on one platform; the slack only covers last-digit differences between platforms
or NumPy builds (the reasoning is in ``tests/golden/regenerate.py``, constant ``RTOL``).

If a golden test fails, the model's results changed. If that is deliberate, regenerate with::

    uv run python tests/golden/regenerate.py

and explain in the pull request why the arrays moved. The ``.json`` file next to each array
records the scenario, seed, kernel correction and package versions of the run that wrote it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from wildfire_sim import sim
from wildfire_sim.config import load_scenario
from wildfire_sim.sim import KernelCorrectionTable
from wildfire_sim.types import FloatArray, Stencil

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _load_helpers() -> ModuleType:
    """Import ``tests/golden/regenerate.py`` (a script, not a package module) by path."""
    spec = importlib.util.spec_from_file_location("golden_regenerate", GOLDEN_DIR / "regenerate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("golden_regenerate", module)
    spec.loader.exec_module(module)
    return module


golden = _load_helpers()
SCENARIOS: tuple[str, ...] = golden.SCENARIOS


def _failure_message(name: str, problems: list[str], meta: dict[str, object], provenance: dict[str, object]) -> str:
    now = {key: provenance.get(key) for key in ("seed", "kernel_correction", "versions", "git_commit")}
    stored = {key: meta.get(key) for key in now}
    return (
        f"golden run {name!r} no longer matches tests/golden/{name}.npz:\n  - "
        + "\n  - ".join(problems)
        + f"\nstored run: {stored}\nthis run:   {now}\n"
        f"If this change is deliberate, regenerate with `{golden.REGENERATE_COMMAND}` and explain in the PR "
        "why the arrival times changed. Otherwise a change under src/ altered the model's results."
    )


@pytest.mark.parametrize("name", SCENARIOS)
def test_golden_run_matches_stored_arrival_times(name: str) -> None:
    result = golden.run_scenario(name)
    expected, meta = golden.load_golden(name)
    problems = golden.compare(result.arrival_time, expected)
    assert not problems, _failure_message(name, problems, meta, result.provenance)


@pytest.mark.parametrize("name", SCENARIOS)
def test_golden_files_record_their_provenance(name: str) -> None:
    """The metadata names the scenario file and seed it came from and the kernel settings used."""
    arrival, meta = golden.load_golden(name)
    config = load_scenario(GOLDEN_DIR / f"{name}.yaml")
    assert meta["scenario"] == f"tests/golden/{name}.yaml"
    assert meta["seed"] == config.seed
    assert meta["stencil"] == config.stencil
    assert meta["arrival_shape"] == config.spread.arrival_shape
    assert meta["kernel_correction"] == sim.load_kernel_corrections().value_for(
        config.stencil, config.spread.arrival_shape
    )
    assert {"numpy", "scipy", "python", "wildfire_sim"} <= set(meta["versions"])
    assert meta["n_burned"] == int((~np.isnan(arrival)).sum())
    assert arrival.dtype == np.float64
    assert list(arrival.shape) == meta["shape"]
    assert (GOLDEN_DIR / f"{name}.npz").stat().st_size < 50_000


def test_golden_scenarios_are_meaningful() -> None:
    """The scenarios exercise what they are named for (guards against regenerating nonsense)."""
    flat, _ = golden.load_golden("flat_no_wind")
    assert not np.isnan(flat).any()  # burns out completely
    wind, _ = golden.load_golden("steady_wind_slope")
    burned_cols = np.nonzero((~np.isnan(wind)).any(axis=0))[0]
    assert burned_cols.max() - 6 > 3 * (6 - burned_cols.min())  # head runs east, downwind and upslope
    assert burned_cols.max() < wind.shape[1] - 1  # still inside the grid
    river, _ = golden.load_golden("river_barrier")
    assert not np.isnan(river[:, :25]).any() and np.isnan(river[:, 25:]).all()


# ----------------------------------------------------------------------------- the test has teeth


def test_changing_the_kernel_correction_fails_the_golden_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raising every front-speed correction by 0.1 % is caught."""
    real = sim.load_kernel_corrections

    def nudged(path: str | Path | None = None) -> KernelCorrectionTable:
        table = real(path)
        entries = [e.model_copy(update={"value": e.value * 1.001}) for e in table.entries]
        return KernelCorrectionTable(entries=entries)

    monkeypatch.setattr(sim, "load_kernel_corrections", nudged)
    expected, _ = golden.load_golden("flat_no_wind")
    problems = golden.compare(golden.run_scenario("flat_no_wind").arrival_time, expected)
    assert problems and "arrival time(s) differ" in problems[-1]


def test_changing_the_link_rate_fails_the_golden_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slowing every link rate by 0.01 % is caught (16-neighbour river scenario).

    The golden landscapes are uniform, so swapping the harmonic link mean for another mean
    would not show; a uniform relative change of every link rate is the smallest kernel change
    every scenario sees.
    """
    real = sim.link_rate_of_spread

    def slowed(ros: FloatArray, stencil: Stencil) -> FloatArray:
        return real(ros, stencil) * (1 - 1e-4)

    monkeypatch.setattr(sim, "link_rate_of_spread", slowed)
    expected, _ = golden.load_golden("river_barrier")
    problems = golden.compare(golden.run_scenario("river_barrier").arrival_time, expected)
    assert problems and "arrival time(s) differ" in problems[-1]


def test_main_prints_one_line_per_scenario(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[bool] = []

    def fake(out_dir: Path = GOLDEN_DIR, force: bool = False) -> list[str]:
        calls.append(force)
        return ["a: unchanged", "b: unchanged"]

    monkeypatch.setattr(golden, "regenerate", fake)
    assert golden.main(["--force"]) == 0
    assert calls == [True]
    assert capsys.readouterr().out == "a: unchanged\nb: unchanged\n"


# ----------------------------------------------------------------------------- the helpers


def test_compare_tolerates_last_digit_noise_only() -> None:
    expected, _ = golden.load_golden("flat_no_wind")
    assert golden.compare(expected * (1 + 1e-12), expected) == []
    assert golden.compare(expected * (1 + 1e-5), expected)
    assert golden.compare(expected[:-1], expected) == [f"shape {expected[:-1].shape} != stored shape {expected.shape}"]


def test_compare_reports_a_changed_burned_set() -> None:
    expected, _ = golden.load_golden("river_barrier")
    actual = expected.copy()
    actual[0, 0] = np.nan
    actual[0, 30] = 1.0
    problems = golden.compare(actual, expected)
    assert problems == ["burned set differs: 1 cell(s) newly burned, 1 cell(s) no longer burned"]


def test_summarize_change() -> None:
    new = np.array([[0.0, 10.0], [np.nan, 5.0]])
    assert golden.summarize_change(None, new) == "new file, 3 burned cell(s)"
    assert golden.summarize_change(new[:1], new).startswith("shape changed")
    old = np.array([[0.0, 12.5], [7.0, np.nan]])
    assert golden.summarize_change(old, new) == "2 cell(s) changed burned status, max arrival-time change 2.5 s"


def test_regenerate_rewrites_only_changed_files(tmp_path: Path) -> None:
    """Into a copy of the golden directory: unchanged arrays are left alone, a changed one is rewritten."""
    for name in SCENARIOS:
        for suffix in (".npz", ".json"):
            shutil.copy(GOLDEN_DIR / f"{name}{suffix}", tmp_path / f"{name}{suffix}")
    stale = golden.load_golden("river_barrier", tmp_path)[0] + 1.0
    np.savez_compressed(tmp_path / "river_barrier.npz", arrival_time=stale)

    lines = golden.regenerate(tmp_path)

    assert lines[0] == "flat_no_wind: unchanged (bit-identical), not rewritten"
    assert lines[2] == (
        "river_barrier: wrote river_barrier.npz; 0 cell(s) changed burned status, max arrival-time change 1 s"
    )
    rewritten, meta = golden.load_golden("river_barrier", tmp_path)
    original, _ = golden.load_golden("river_barrier")
    np.testing.assert_array_equal(rewritten, original)
    assert json.loads((tmp_path / "river_barrier.json").read_text(encoding="utf-8"))["seed"] == meta["seed"]
