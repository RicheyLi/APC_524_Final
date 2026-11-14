"""Regenerate the golden arrival-time arrays (WP 1.9).

Usage, from the repository root::

    uv run python tests/golden/regenerate.py           # rewrite the files whose arrays changed
    uv run python tests/golden/regenerate.py --force   # rewrite all files (refreshes metadata)

For every scenario ``<name>.yaml`` in this directory the script runs the simulator at the seed
in the file and writes ``<name>.npz`` (key ``arrival_time``: the ``(H, W)`` float64 arrival
times, s, NaN where a cell never burned) and ``<name>.json`` (provenance: scenario file, seed,
kernel correction, package versions, git commit). It prints, per scenario, how many cells
changed burned status and the largest arrival-time change.

A golden file changes only when the model's results change. That must be deliberate: say in
the pull request why the arrays moved. ``tests/test_golden.py`` imports the helpers below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_sim.config import load_scenario
from wildfire_sim.sim import RunResult, run
from wildfire_sim.types import FloatArray

GOLDEN_DIR = Path(__file__).resolve().parent
SCENARIOS: tuple[str, ...] = ("flat_no_wind", "steady_wind_slope", "river_barrier")
REGENERATE_COMMAND = "uv run python tests/golden/regenerate.py"

RTOL = 1e-7
"""Relative tolerance on arrival times (dimensionless). On one platform the runs are bit-stable;
the slack covers a different libm or NumPy build. The link thresholds are float32, so a
last-digit float64 difference in a gamma draw can at worst flip one float32 rounding, a
relative change of 2**-24 (6e-8) in one link's crossing time. 1e-7 absorbs that, while any
deliberate change of the kernel moves arrival times by far more (a 0.1 % change of the
front-speed correction moves them by 1e-3)."""

ATOL = 1e-6
"""Absolute tolerance on arrival times, s; only matters for the ignition cell (time 0)."""


def scenario_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.yaml"


def run_scenario(name: str) -> RunResult:
    """Run golden scenario ``name`` at the seed in its file (events are not recorded)."""
    return run(load_scenario(scenario_path(name)), record_events=False)


def metadata(name: str, result: RunResult) -> dict[str, Any]:
    """What a mismatch needs for diagnosis, taken from the run's provenance."""
    prov = result.provenance
    return {
        "scenario": f"tests/golden/{name}.yaml",
        "seed": prov["seed"],
        "rule": prov["rule"],
        "stencil": prov["stencil"],
        "arrival_shape": prov["arrival_shape"],
        "kernel_correction": prov["kernel_correction"],
        "kernel_correction_source": prov["kernel_correction_source"],
        "versions": prov["versions"],
        "git_commit": prov["git_commit"],
        "n_steps": prov["n_steps"],
        "final_time_s": prov["final_time_s"],
        "n_burned": int(result.burned_mask.sum()),
        "shape": list(result.arrival_time.shape),
    }


def load_golden(name: str, directory: Path = GOLDEN_DIR) -> tuple[FloatArray, dict[str, Any]]:
    """Stored ``(H, W)`` arrival times (s, NaN = never burned) and metadata of scenario ``name``."""
    with np.load(directory / f"{name}.npz", allow_pickle=False) as data:
        arrival = np.asarray(data["arrival_time"], dtype=np.float64)
    meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    return arrival, meta


def compare(actual: FloatArray, expected: FloatArray, rtol: float = RTOL, atol: float = ATOL) -> list[str]:
    """Differences between two arrival-time arrays (s); empty when they match.

    They match when the burned sets (non-NaN cells) are identical and every arrival time agrees
    within ``atol + rtol * |expected|``.
    """
    if actual.shape != expected.shape:
        return [f"shape {actual.shape} != stored shape {expected.shape}"]
    problems: list[str] = []
    burned_a, burned_e = ~np.isnan(actual), ~np.isnan(expected)
    gained = int((burned_a & ~burned_e).sum())
    lost = int((~burned_a & burned_e).sum())
    if gained or lost:
        problems.append(f"burned set differs: {gained} cell(s) newly burned, {lost} cell(s) no longer burned")
    both = burned_a & burned_e
    diff = np.abs(actual[both] - expected[both])
    bad = diff > atol + rtol * np.abs(expected[both])
    if bad.any():
        problems.append(
            f"{int(bad.sum())} arrival time(s) differ beyond rtol={rtol:g}, atol={atol:g} s; "
            f"largest change {float(diff.max()):.6g} s"
        )
    return problems


def summarize_change(old: FloatArray | None, new: FloatArray) -> str:
    """One line: cells whose burned status changed and the largest arrival-time change, s."""
    if old is None:
        return f"new file, {int((~np.isnan(new)).sum())} burned cell(s)"
    if old.shape != new.shape:
        return f"shape changed from {old.shape} to {new.shape}"
    flipped = int((np.isnan(old) != np.isnan(new)).sum())
    both = ~np.isnan(old) & ~np.isnan(new)
    max_change = float(np.abs(new[both] - old[both]).max()) if both.any() else 0.0
    return f"{flipped} cell(s) changed burned status, max arrival-time change {max_change:.6g} s"


def regenerate(out_dir: Path = GOLDEN_DIR, force: bool = False) -> list[str]:
    """Run every scenario and write the files of those whose arrays changed (all if ``force``).

    Returns:
        One summary line per scenario.
    """
    lines: list[str] = []
    for name in SCENARIOS:
        result = run_scenario(name)
        new = result.arrival_time
        npz = out_dir / f"{name}.npz"
        old = load_golden(name, out_dir)[0] if npz.exists() and (out_dir / f"{name}.json").exists() else None
        unchanged = old is not None and np.array_equal(old, new, equal_nan=True)
        if unchanged and not force:
            lines.append(f"{name}: unchanged (bit-identical), not rewritten")
            continue
        np.savez_compressed(npz, arrival_time=new)
        text = json.dumps(metadata(name, result), indent=2, sort_keys=True) + "\n"
        (out_dir / f"{name}.json").write_text(text, encoding="utf-8")
        lines.append(f"{name}: wrote {npz.name}; {summarize_change(old, new)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--force", action="store_true", help="rewrite every file, even if its array is unchanged")
    args = parser.parse_args(argv)
    for line in regenerate(force=args.force):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
