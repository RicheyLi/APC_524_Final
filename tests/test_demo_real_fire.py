"""Tests for scripts/demo_real_fire.py.

A full demo run needs a built region and its observations (``scripts/download_data.py <region>
--observations``), which CI does not have; these tests check what runs offline: the regions on
offer, the refusal of an unknown region, and the message when a region is not built yet.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def _demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_real_fire", REPO / "scripts" / "demo_real_fire.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_the_regions(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        _demo().main(["--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "lake_2024" in out and "silverado_2020" in out


def test_rejects_an_unknown_region(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        _demo().main(["no_such_fire"])
    assert stop.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_says_how_to_build_a_missing_region(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _demo().main(["silverado_2020", "--cache-root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "not built" in out and "download_data.py silverado_2020 --observations" in out
