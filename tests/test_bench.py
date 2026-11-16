"""Tests for the real-fire mode of scripts/bench.py (WP 3.1); the workload timings are not tested here."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench", REPO / "scripts" / "bench.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_times_a_fire_on_a_saved_grid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "bench.json"
    argv = ["--grid", str(FIXTURES / "lake_2024_5km.npz"), "--weather", str(FIXTURES / "lake_2024_weather.yaml")]
    argv += [
        "--ignition-time-s",
        "56880",
        "--hours",
        "0.25",
        "--repeats",
        "1",
        "--engines",
        "numpy",
        "--json",
        str(out),
    ]
    assert _bench().main(argv) == 0
    assert "one evaluation of the rothermel rule" in capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["fire"]["shape"] == [167, 167] and report["fire"]["hours"] == 0.25
    (result,) = report["results"]
    assert result["engine"] == "numpy" and result["burned_cells"] > 1 and result["reached_edge_h"] is None


def test_fire_mode_arguments(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        _bench().main(["--grid", str(FIXTURES / "lake_2024_5km.npz")])
    assert stop.value.code == 2 and "go together" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stop:
        _bench().main(["--region", "lake_2024", "--grid", "x.npz", "--weather", "y.yaml"])
    assert stop.value.code == 2 and "not both" in capsys.readouterr().err
    assert _bench().main(["--region", "lake_2024", "--cache-root", str(tmp_path)]) == 1
    assert "download_data.py lake_2024 --observations" in capsys.readouterr().err
