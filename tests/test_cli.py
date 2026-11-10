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
