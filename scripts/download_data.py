"""Build the cached study regions (WP 2.6): one command per fire.

Downloads (or reads from the caches) every input of a region in ``configs/regions.yaml`` and writes
``data/cache/<region>/`` (``grid.npz``, ``canopy.npz``, ``weather.yaml``, ``perimeter.geojson``,
``manifest.json``) and the 5 km test crop in ``tests/fixtures/``. See ``docs/data.md``.

Usage (from the repository root, with the geo extra installed: ``uv sync --extra geo``)::

    $env:WILDFIRE_SIM_LANDFIRE_EMAIL = "you@example.edu"   # PowerShell; LANDFIRE needs an email
    uv run python scripts/download_data.py lake_2024
    uv run python scripts/download_data.py --all
    uv run python scripts/download_data.py lake_2024 --verify   # rebuild from cache, compare hashes

The email is only sent to the LANDFIRE Product Service and is never written to disk. A first
build downloads several hundred megabytes and can take tens of minutes (the LANDFIRE job is
queued); later builds read the caches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wildfire_sim.data.cache import (
    FIXTURES_DIR,
    Region,
    build_region,
    email_configured,
    fuel_summary,
    load_regions,
    verify_region,
)
from wildfire_sim.data.landfire import DEFAULT_CACHE, EMAIL_ENV
from wildfire_sim.data.validation import fetch_observations
from wildfire_sim.grid import Grid


def main(argv: list[str] | None = None) -> int:
    regions = load_regions().regions
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"regions to build: {', '.join(regions)}")
    parser.add_argument("--all", action="store_true", help="build every region")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE, help="download cache (default data/cache)")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_DIR, help="where to write the 5 km test crops")
    parser.add_argument("--no-fixtures", action="store_true", help="do not write the test crops")
    parser.add_argument("--verify", action="store_true", help="rebuild from the caches and compare file hashes")
    parser.add_argument(
        "--observations",
        action="store_true",
        help="also fetch the fire's observations (ignition, perimeters, VIIRS; needs a NASA Earthdata token)",
    )
    args = parser.parse_args(argv)

    names = list(regions) if args.all else args.names
    unknown = [n for n in names if n not in regions]
    if not names or unknown:
        parser.error(f"name one or more regions ({', '.join(regions)}) or pass --all; unknown: {unknown}")

    for name in names:
        region = regions[name]
        if args.verify:
            differ = verify_region(name, region, cache_root=args.cache_root)
            print(f"{name}: {'reproduced' if not differ else 'DIFFERENT: ' + ', '.join(differ)}")
            if differ:
                return 1
            continue
        cached = (args.cache_root / f"mtbs-{region.mtbs_fire_id}").exists()
        if not email_configured() and not cached:
            print(f"note: {EMAIL_ENV} is not set; a fresh LANDFIRE download will fail", file=sys.stderr)
        print(f"{name}: {region.description}")
        built = build_region(
            name,
            region,
            cache_root=args.cache_root,
            fixtures_dir=None if args.no_fixtures else args.fixtures,
        )
        g, m = built.grid, built.manifest
        burnable = int(g.burnable_mask().sum())
        print(f"  grid {g.shape[0]} x {g.shape[1]} cells of {g.dx:g} m, {burnable} burnable")
        print(f"  perimeter at least {m['grid']['perimeter_edge_cells']} cells from the edge")
        top = list(fuel_summary(g.fuel_model).items())[:6]
        print("  fuel models (model: cells): " + ", ".join(f"{k}: {v}" for k, v in top))
        print(f"  break cells {int((g.break_width_m > 0).sum())}, assets {g.asset_value.sum():,.0f} USD")
        print(f"  wrote {built.directory}")
        if built.crop is not None:
            print(f"  crop {built.crop} ({built.crop.stat().st_size / 1e6:.2f} MB)")
        if args.observations:
            report_observations(name, region, args.cache_root, built.grid)
    return 0


def report_observations(name: str, region: Region, cache_root: Path, grid: Grid) -> None:
    """Fetch (or read) a region's observations (WP 2.7) and print what they contain."""
    obs = fetch_observations(name, region, cache_root=cache_root, progress=print)
    ign = obs.ignition
    print(f"  ignition {ign.time_utc:%Y-%m-%d %H:%M} UTC at {ign.lat:.5f}, {ign.lon:.5f} ({ign.unique_fire_id})")
    print(f"  daily perimeter versions {len(obs.daily)}; VIIRS overpasses over the region {len(obs.overpasses)}")
    dets = obs.detections(grid)
    widths = [d.t_detect - d.t_clear for d in dets]
    print(
        f"  first detections {len(dets)}; median interval {sorted(widths)[len(widths) // 2] / 3600:.1f} h"
        if dets
        else ""
    )
    print(f"  final perimeter {obs.observed_burn_mask(grid).sum() * grid.dx**2 / 1e4:,.0f} ha")


if __name__ == "__main__":
    raise SystemExit(main())
