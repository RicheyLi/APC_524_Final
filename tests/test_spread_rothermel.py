"""Tests for the native Rothermel surface fire model (WP 2.1a, ``wildfire_sim.spread.rothermel``)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from wildfire_sim.config import configs_dir
from wildfire_sim.grid import Weather
from wildfire_sim.sim import flame_length_from_intensity, load_residence_times
from wildfire_sim.spread.rothermel import (
    FT_PER_MIN_TO_M_PER_S,
    FuelMoisture,
    fuel_bed,
    head_fire,
    herbaceous_cured_fraction,
    load_fuel_models,
    residence_times,
    surface_fire,
    wind_limit,
)

TABLE = load_fuel_models()
SCOTT_BURGAN = [
    *range(101, 110),
    *range(121, 125),
    *range(141, 150),
    *range(161, 166),
    *range(181, 190),
    *range(201, 205),
]

# Scott & Burgan (2005) moisture scenarios (1-h, 10-h, 100-h, live herbaceous, live woody),
# fractions: D1L1 very low dead / fully cured herbaceous, D2L2 low / two-thirds cured, D3L3
# moderate / one-third cured.
MOISTURE = {
    "D1L1": FuelMoisture(0.03, 0.04, 0.05, 0.30, 0.60),
    "D2L2": FuelMoisture(0.06, 0.07, 0.08, 0.60, 0.90),
    "D3L3": FuelMoisture(0.09, 0.10, 0.11, 0.90, 1.20),
}
# (midflame wind, m/s, blowing upslope; slope, rise/run) per case.
WIND_SLOPE = {"D1L1": (2.0, 0.0), "D2L2": (1.0, 0.3), "D3L3": (3.0, 0.6)}

# Head-fire rate of spread (m/s), fireline intensity (kW/m) and flame length (m) from the
# BehavePlus 6 fire behaviour engine: the 'behave' C++ library (github.com/firelab/behave), run
# through its Python binding pyrothermel 0.1.4 with the 'us_standard' preset (doSurfaceRun...
# InDirectionOfMaxSpread, direct midflame wind blowing upslope, standard fuel models), then
# converted to SI: 1 ch/h = 20.1168 m / 3600 s, 1 Btu/ft/s = 3.46146 kW/m, 1 ft = 0.3048 m.
# behave applies Rothermel's wind limit by default; it binds for GR1, GR4, TU1 and TL3 in D3L3.
# (pyrothermel's 'metric' preset passes the heat content in kJ/kg where Btu/lb is expected and
# overstates intensity 2.33-fold, so the reference values come from the US preset.)
BEHAVEPLUS = [
    ("D1L1", "GR1", 0.138253, 137.098, 0.745077),
    ("D1L1", "GR4", 0.629141, 3606.39, 3.3529),
    ("D1L1", "SH2", 0.0540894, 894.994, 1.76606),
    ("D1L1", "SH5", 0.447671, 9582.84, 5.25599),
    ("D1L1", "TU1", 0.0235265, 122.855, 0.708417),
    ("D1L1", "TL3", 0.0114311, 30.5615, 0.373549),
    ("D2L2", "GR1", 0.0423032, 39.1737, 0.418741),
    ("D2L2", "GR4", 0.185469, 991.719, 1.85143),
    ("D2L2", "SH2", 0.00903678, 54.6798, 0.488167),
    ("D2L2", "SH5", 0.167367, 2998.4, 3.0799),
    ("D2L2", "TU1", 0.0089523, 40.3354, 0.424408),
    ("D2L2", "TL3", 0.00556506, 12.4294, 0.246953),
    ("D3L3", "GR1", 0.0247822, 17.8347, 0.291575),
    ("D3L3", "GR4", 0.153485, 288.871, 1.04976),
    ("D3L3", "SH2", 0.0189207, 77.2945, 0.572422),
    ("D3L3", "SH5", 0.42592, 6796.29, 4.48759),
    ("D3L3", "TU1", 0.00445277, 6.31993, 0.180924),
    ("D3L3", "TL3", 0.013644, 28.6689, 0.362724),
]
REL_TOL = 0.02  # WP 2.1a acceptance: within 2 % of BehavePlus; measured agreement is better than 0.02 %


def number(code: str) -> int:
    return int(TABLE.numbers[TABLE.codes.index(code)])


# ----------------------------------------------------------------------------- fuel model table


def test_table_has_the_40_scott_burgan_and_13_anderson_models() -> None:
    assert len(TABLE) == 53
    assert sorted(TABLE.numbers.tolist()) == [*range(1, 14), *SCOTT_BURGAN]
    assert TABLE.codes[TABLE.row(101)] == "GR1" and TABLE.codes[TABLE.row(204)] == "SB4"


def test_table_values_match_scott_and_burgan_table_7() -> None:
    """Spot checks transcribed from Scott & Burgan (2005), Table 7 (loads t/ac, SAV 1/ft, depth ft)."""
    gr1 = TABLE.row(101)
    assert (TABLE.load_1h[gr1], TABLE.load_live_herb[gr1], TABLE.sav_1h[gr1], TABLE.sav_live_herb[gr1]) == (
        0.1,
        0.3,
        2200,
        2000,
    )
    assert (TABLE.depth[gr1], TABLE.mx_dead[gr1], bool(TABLE.dynamic[gr1])) == (0.4, 0.15, True)
    sh5 = TABLE.row(145)
    assert (TABLE.load_1h[sh5], TABLE.load_10h[sh5], TABLE.load_live_woody[sh5], TABLE.sav_1h[sh5]) == (
        3.6,
        2.1,
        2.9,
        750,
    )
    assert (TABLE.depth[sh5], bool(TABLE.dynamic[sh5])) == (6.0, False)
    assert TABLE.heat_dead[TABLE.row(106)] == 9000  # GR6, the one model with a higher heat content
    assert np.all(TABLE.sav_10h == 109) and np.all(TABLE.sav_100h == 30)
    assert np.all(TABLE.particle_density == 32) and np.all(TABLE.mineral_total == 0.0555)
    assert np.all(TABLE.mineral_effective == 0.01)


def test_index_maps_unknown_and_non_burnable_numbers_to_minus_one() -> None:
    idx = TABLE.index(np.array([[101, 0], [91, 183]]))
    assert idx.shape == (2, 2)
    assert idx[0, 1] == -1 and idx[1, 0] == -1
    assert TABLE.numbers[idx[0, 0]] == 101 and TABLE.numbers[idx[1, 1]] == 183
    with pytest.raises(KeyError):
        TABLE.row(99)


def test_shipped_csv_is_the_default(tmp_path: Path) -> None:
    assert (configs_dir() / "fuel_models.csv").exists()
    assert np.array_equal(load_fuel_models(configs_dir() / "fuel_models.csv").numbers, TABLE.numbers)


def write_csv(path: Path, **override: str) -> Path:
    header = (
        "number,code,load_1h,load_10h,load_100h,load_live_herb,load_live_woody,sav_1h,sav_live_herb,"
        "sav_live_woody,depth,mx_dead,heat_dead,heat_live,dynamic,sav_10h,sav_100h,particle_density,"
        "mineral_total,mineral_effective,name"
    )
    defaults = "1,X,0.74,0,0,0,0,3500,0,0,1,12,8000,8000,0,109,30,32,0.0555,0.01,test".split(",")
    values = [override.get(col, val) for col, val in zip(header.split(","), defaults, strict=True)]
    path.write_text("# comment\n" + header + "\n" + ",".join(values) + "\n", encoding="utf-8")
    return path


def test_loader_reads_a_minimal_file(tmp_path: Path) -> None:
    table = load_fuel_models(write_csv(tmp_path / "ok.csv"))
    assert table.numbers.tolist() == [1] and table.mx_dead[0] == pytest.approx(0.12)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mx_dead": "0"}, "mx_dead"),
        ({"load_10h": "-1"}, "load_10h"),
        ({"load_live_woody": "1", "sav_live_woody": "0"}, "sav_live_woody"),
        ({"depth": "0"}, "depth"),
        ({"load_1h": "0"}, "1-h load"),
    ],
)
def test_loader_rejects_bad_values(tmp_path: Path, override: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_fuel_models(write_csv(tmp_path / "bad.csv", **override))


def test_loader_rejects_missing_columns_and_repeats(tmp_path: Path) -> None:
    p = tmp_path / "missing.csv"
    p.write_text("number,code\n1,X\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_fuel_models(p)
    good = write_csv(tmp_path / "dup.csv").read_text(encoding="utf-8")
    p.write_text(good + good.splitlines()[-1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repeat"):
        load_fuel_models(p)
    p.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no fuel models"):
        load_fuel_models(p)


# ----------------------------------------------------------------------------- BehavePlus values


@pytest.mark.parametrize(("case", "code", "ros", "intensity", "flame"), BEHAVEPLUS)
def test_matches_behaveplus(case: str, code: str, ros: float, intensity: float, flame: float) -> None:
    """WP 2.1a acceptance: head ros, intensity and flame length within 2 % of BehavePlus."""
    wind, slope = WIND_SLOPE[case]
    fire = surface_fire(np.array([number(code)]), MOISTURE[case], wind, slope, wind_limit_on=True)
    assert fire.ros[0] == pytest.approx(ros, rel=REL_TOL)
    assert fire.intensity[0] == pytest.approx(intensity, rel=REL_TOL)
    assert fire.flame_length[0] == pytest.approx(flame, rel=REL_TOL)


def test_wind_limit_binds_only_where_behaveplus_says() -> None:
    """Without the wind limit the unlimited cases are unchanged and the limited ones spread faster."""
    for case, code, ros, *_ in BEHAVEPLUS:
        wind, slope = WIND_SLOPE[case]
        fm = np.array([number(code)])
        free = surface_fire(fm, MOISTURE[case], wind, slope)
        capped = surface_fire(fm, MOISTURE[case], wind, slope, wind_limit_on=True)
        expect_limited = case == "D3L3" and code in {"GR1", "GR4", "TU1", "TL3"}
        assert bool(capped.wind_limited[0]) is expect_limited
        assert not free.wind_limited[0]
        if expect_limited:
            assert free.ros[0] > capped.ros[0] * (1 + REL_TOL)
            assert capped.effective_wind[0] == pytest.approx(wind_limit(capped.reaction_intensity)[0])
        else:
            assert free.ros[0] == pytest.approx(ros, rel=REL_TOL)


# ----------------------------------------------------------------------------- physics


def test_no_spread_at_the_moisture_of_extinction() -> None:
    """WP 2.1a acceptance: zero ros once the dead fuel reaches its moisture of extinction.

    For a model without live fuel (TL3, mx 20 %) dead moisture at 20 % stops the fire. For a
    model with live fuel (SH2), the live fuel also has to pass its own extinction moisture.
    """
    tl3 = np.array([183])
    at_mx = FuelMoisture(0.20, 0.20, 0.20, 0.60, 0.90)
    just_below = FuelMoisture(0.199, 0.199, 0.199, 0.60, 0.90)
    assert surface_fire(tl3, at_mx, 5.0, 0.5).ros[0] == 0.0
    assert surface_fire(tl3, just_below, 5.0, 0.5).ros[0] > 0.0
    sh2 = np.array([142])
    wet = FuelMoisture(0.15, 0.15, 0.15, 3.0, 3.0)
    fire = surface_fire(sh2, wet, 5.0, 0.5)
    assert fire.ros[0] == 0.0 and fire.intensity[0] == 0.0 and fire.flame_length[0] == 0.0


def test_non_burnable_cells_get_zeros() -> None:
    fm = np.array([[0, 91], [98, 101]], dtype=np.int16)
    fire = surface_fire(fm, MOISTURE["D2L2"], 3.0, 0.2)
    for name in ("ros", "intensity", "flame_length", "heat_per_area", "residence_time", "phi_wind"):
        values = getattr(fire, name)
        assert values.shape == (2, 2)
        assert np.all(values.ravel()[:3] == 0.0) and values[1, 1] > 0.0, name


def test_ros_rises_with_wind_and_slope_and_falls_with_moisture() -> None:
    fm = np.array(SCOTT_BURGAN)
    d1 = MOISTURE["D1L1"]
    calm = surface_fire(fm, d1, 0.0, 0.0)
    windy = surface_fire(fm, d1, 3.0, 0.0)
    steep = surface_fire(fm, d1, 0.0, 0.5)
    assert np.all(windy.ros > calm.ros) and np.all(steep.ros > calm.ros)
    assert np.allclose(calm.ros, calm.ros_no_wind)
    damp = surface_fire(fm, MOISTURE["D3L3"], 0.0, 0.0)
    assert np.all(damp.ros < calm.ros)


def test_head_values_are_consistent() -> None:
    """``R = R_0 (1 + phi_w + phi_s)``, ``I = H_A R`` (Byram) and flame length from Byram (1959)."""
    fm = np.array(SCOTT_BURGAN)
    fire = surface_fire(fm, MOISTURE["D2L2"], 2.5, 0.4)
    np.testing.assert_allclose(fire.ros, fire.ros_no_wind * (1 + fire.phi_wind + fire.phi_slope), rtol=1e-12)
    np.testing.assert_allclose(fire.intensity, fire.heat_per_area * fire.ros, rtol=1e-12)
    np.testing.assert_allclose(fire.flame_length, flame_length_from_intensity(fire.intensity), rtol=1e-12)
    np.testing.assert_allclose(fire.heat_per_area, fire.reaction_intensity * fire.residence_time, rtol=1e-12)


def test_grid_evaluation_matches_cell_by_cell() -> None:
    """Evaluating once per fuel model and mapping back gives each cell its own answer."""
    rng = np.random.default_rng(0)
    fm = rng.choice([0, 101, 145, 183, 1], size=(12, 9)).astype(np.int16)
    wind = rng.uniform(0, 6, size=fm.shape)
    slope = rng.uniform(0, 0.8, size=fm.shape)
    grid = surface_fire(fm, MOISTURE["D1L1"], wind, slope)
    for r, c in np.ndindex(fm.shape):
        one = surface_fire(fm[r : r + 1, c], MOISTURE["D1L1"], wind[r, c], slope[r, c])
        assert grid.ros[r, c] == pytest.approx(one.ros[0], rel=1e-12, abs=0.0)
        assert grid.intensity[r, c] == pytest.approx(one.intensity[0], rel=1e-12, abs=0.0)


def test_slope_factor_has_rothermels_form() -> None:
    """``phi_s = 5.275 beta^-0.3 tan^2`` (Rothermel 1972, eq. 51): quadruples when the slope doubles."""
    bed = fuel_bed(TABLE, MOISTURE["D2L2"])
    np.testing.assert_allclose(bed.slope_factor(0.6), 4 * bed.slope_factor(0.3), rtol=1e-12)
    np.testing.assert_allclose(bed.slope_factor(1.0), 5.275 * bed.packing_ratio**-0.3, rtol=1e-12)
    assert np.all(bed.slope_factor(0.0) == 0.0) and np.all(bed.wind_factor(0.0) == 0.0)


def test_effective_wind_inverts_the_wind_factor() -> None:
    bed = fuel_bed(TABLE, MOISTURE["D1L1"])
    for u in (0.5, 2.0, 7.0):
        np.testing.assert_allclose(bed.effective_wind(bed.wind_factor(u)), u, rtol=1e-10)
    assert np.all(bed.effective_wind(0.0) == 0.0)


def test_wind_limit_is_point_nine_reaction_intensity() -> None:
    """Rothermel (1972): ``U_max = 0.9 I_R`` with U in ft/min and I_R in Btu/ft^2/min."""
    i_r_btu = 1000.0  # Btu/ft^2/min
    kw = i_r_btu * 1.05505585262 / 0.3048**2 / 60.0
    assert wind_limit(np.array([kw]))[0] == pytest.approx(900.0 * FT_PER_MIN_TO_M_PER_S)


def test_head_fire_without_limit_is_linear_in_phi() -> None:
    bed = fuel_bed(TABLE, MOISTURE["D2L2"], np.array([TABLE.row(101)]))
    fire = head_fire(bed, 3.0)
    assert fire.ros[0] == pytest.approx(4.0 * bed.ros_no_wind[0])
    assert not fire.wind_limited[0]


# ----------------------------------------------------------------------------- dynamic models


def test_cured_fraction_follows_scott_and_burgan() -> None:
    """Fully cured at 30 % live herbaceous moisture or less, fully green from 120 % (rounded line)."""
    cured = herbaceous_cured_fraction(np.array([0.0, 0.30, 0.60, 0.90, 1.21, 2.0]))
    np.testing.assert_allclose(cured, [1.0, 1.0, 0.667, 0.334, 0.0, 0.0], atol=0.001)


def test_curing_moves_herbaceous_load_to_dead_and_speeds_a_grass_fire() -> None:
    """A dynamic grass (GR2) burns faster as its herbaceous fuel cures; a static model does not care."""
    gr2, tl3 = np.array([102]), np.array([183])
    rates = [surface_fire(gr2, FuelMoisture(0.06, 0.07, 0.08, mh, 0.9), 2.0, 0.0).ros[0] for mh in (1.2, 0.9, 0.6, 0.3)]
    assert rates == sorted(rates)
    a = surface_fire(tl3, FuelMoisture(0.06, 0.07, 0.08, 0.3, 0.9), 2.0, 0.0).ros[0]
    b = surface_fire(tl3, FuelMoisture(0.06, 0.07, 0.08, 1.2, 0.9), 2.0, 0.0).ros[0]
    assert a == b


def test_live_moisture_of_extinction_is_at_least_the_dead_one() -> None:
    bed = fuel_bed(TABLE, MOISTURE["D2L2"])
    has_live = (TABLE.load_live_herb + TABLE.load_live_woody) > 0
    assert np.all(bed.mx_live[has_live] >= TABLE.mx_dead[has_live])
    assert np.all(bed.mx_live[~has_live] == 0.0)


def test_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="moisture"):
        surface_fire(np.array([101]), FuelMoisture(-0.01, 0.07, 0.08, 0.6, 0.9), 1.0, 0.0)
    with pytest.raises(ValueError, match="midflame_wind"):
        surface_fire(np.array([101]), MOISTURE["D2L2"], -1.0, 0.0)
    with pytest.raises(ValueError, match="slope"):
        surface_fire(np.array([101]), MOISTURE["D2L2"], 1.0, np.nan)
    with pytest.raises(ValueError, match="rows"):
        fuel_bed(TABLE, MOISTURE["D2L2"], np.array([-1]))


def test_moisture_from_weather() -> None:
    w = Weather(
        moisture_1h=0.05, moisture_10h=0.06, moisture_100h=0.07, moisture_live_herb=0.8, moisture_live_woody=1.1
    )
    assert FuelMoisture.from_weather(w) == FuelMoisture(0.05, 0.06, 0.07, 0.8, 1.1)


# ----------------------------------------------------------------------------- residence time


def test_residence_time_is_andersons_384_over_sigma() -> None:
    bed = fuel_bed(TABLE, MOISTURE["D2L2"])
    np.testing.assert_allclose(bed.residence_time, 384.0 / bed.sigma * 60.0, rtol=1e-12)
    # FM1, one particle class of SAV 3500 1/ft: 384 / 3500 min = 6.58 s
    assert bed.residence_time[TABLE.row(1)] == pytest.approx(384.0 / 3500.0 * 60.0)


def test_shipped_residence_times_match_the_fuel_models() -> None:
    """``configs/residence_time.yaml`` is :func:`residence_times` at the Weather defaults, to 0.05 s."""
    shipped = load_residence_times()
    computed = residence_times()
    assert set(shipped.by_fuel_model) == set(computed)
    for fm, seconds in computed.items():
        assert shipped.by_fuel_model[fm].value == pytest.approx(seconds, abs=0.05), fm


# ----------------------------------------------------------------------------- pyrothermel cross-check

PYRO_CASES = [
    (MOISTURE["D1L1"], 0.0, 0.0),
    (MOISTURE["D2L2"], 2.0, 0.0),
    (MOISTURE["D3L3"], 1.0, 0.3),
    (MOISTURE["D2L2"], 4.0, 0.5),
    (FuelMoisture(0.12, 0.13, 0.14, 1.2, 1.5), 3.0, 0.2),
    (MOISTURE["D1L1"], 15.0, 0.0),
]


@pytest.mark.parametrize(("moisture", "wind", "slope"), PYRO_CASES)
def test_agrees_with_pyrothermel_for_every_model(moisture: FuelMoisture, wind: float, slope: float) -> None:
    """WP 2.1a acceptance: within 2 % of ``pyrothermel`` (the BehavePlus engine) where it is installed.

    Every one of the 53 models, wind blowing upslope, wind limit on as in behave. Measured
    agreement: 0.01 % in ros and intensity, 0.04 % in flame length.
    """
    pyro = pytest.importorskip("pyrothermel")
    mph, ch_h, btu_ft_s = 0.44704, 20.1168 / 3600.0, 1.05505585262 / 0.3048
    mine = surface_fire(TABLE.numbers, moisture, wind, slope, wind_limit_on=True)
    m = (moisture.one_hour, moisture.ten_hour, moisture.hundred_hour, moisture.live_herb, moisture.live_woody)
    for i, fm in enumerate(TABLE.numbers.tolist()):
        run = pyro.PyrothermelRun(
            pyro.FuelModel.from_existing(fm, "us_standard"),
            pyro.MoistureScenario(*(float(x) for x in m)),
            wind / mph,
            units_preset="us_standard",
            slope=math.degrees(math.atan(slope)),
            aspect=0.0,
            wind_direction=0.0,  # behave: wind from the north onto a north aspect blows upslope
        )
        ref = run.run_surface_fire_in_direction_of_max_spread()
        code = TABLE.codes[i]
        assert mine.ros[i] == pytest.approx(ref["spread_rate"] * ch_h, rel=REL_TOL), code
        assert mine.intensity[i] == pytest.approx(ref["fireline_intensity"] * btu_ft_s, rel=REL_TOL), code
        assert mine.flame_length[i] == pytest.approx(ref["flame_length"] * 0.3048, rel=REL_TOL), code
