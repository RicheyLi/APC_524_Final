"""Tests for the wildfire-sim CLI."""

from pathlib import Path

from typer.testing import CliRunner

from wildfire_sim.cli import app

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parent.parent / "configs"


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
