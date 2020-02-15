"""Area, dateline, and regrid checks against closed forms.

The expected integrals in this file are written out from the spherical band
formula. They do not call `area_weighted_mean` to manufacture the expectation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from forcing_atlas.constants import EARTH_RADIUS_M
from forcing_atlas.errors import GridError
from forcing_atlas.grids import (
    LatLonGrid,
    LongitudeConvention,
    area_weighted_mean,
    coverage_fraction,
    detect_longitude_convention,
    infer_latitude_bounds,
    infer_longitude_bounds,
    normalize_longitude,
    regular_grid,
    unweighted_error,
    unweighted_mean,
)
from forcing_atlas.regrid import covered_integral, remap_field, same_grid_identity_error
from forcing_atlas.regions import get_region, regional_mean
from forcing_atlas.sphere import (
    band_area_m2,
    globe_area_m2,
    great_circle_distance_m,
    hemisphere_area_m2,
    longitude_width_deg,
    rectangle_area_from_excess,
)


def test_longitude_wrap_and_detection() -> None:
    signed = normalize_longitude(np.array([180.0, -180.0, 190.0, -190.0]), LongitudeConvention.NEG180_180)
    assert signed.tolist() == pytest.approx([-180.0, -180.0, -170.0, 170.0])
    positive = normalize_longitude(np.array([-10.0, 360.0, 370.0]), LongitudeConvention.ZERO_360)
    assert positive.tolist() == pytest.approx([350.0, 0.0, 10.0])
    assert detect_longitude_convention(np.array([0.0, 10.0, 200.0])) is LongitudeConvention.ZERO_360
    assert detect_longitude_convention(np.array([-20.0, 10.0])) is LongitudeConvention.NEG180_180
    with pytest.raises(GridError):
        detect_longitude_convention(np.array([-10.0, 200.0]))


def test_dateline_width_and_full_turn() -> None:
    assert longitude_width_deg(350.0, 10.0) == pytest.approx(20.0)
    assert longitude_width_deg(0.0, 360.0) == pytest.approx(360.0)
    assert longitude_width_deg(10.0, 10.0) == pytest.approx(0.0)


def test_hemisphere_area_matches_band_formula() -> None:
    grid = regular_grid(36, 72, name="5deg")
    areas = grid.cell_areas_m2()
    northern = grid.lat_bounds[:, 0] >= 0.0
    assert grid.covers_globe()
    assert abs(float(np.sum(areas[northern])) - hemisphere_area_m2()) / hemisphere_area_m2() < 1e-12
    assert abs(float(np.sum(areas)) - globe_area_m2()) / globe_area_m2() < 1e-12
    assert abs(grid.globe_area_gap_m2()) < 1e-3


def test_girard_matches_band_formula_off_the_pole() -> None:
    # 10 degrees by 20 degrees, centered on the equator. Both edges are proper parallels.
    radius = EARTH_RADIUS_M
    girard = rectangle_area_from_excess(-10.0, 10.0, -5.0, 5.0, radius)
    analytical = radius**2 * (math.sin(math.radians(5.0)) - math.sin(math.radians(-5.0))) * math.radians(20.0)
    assert girard == pytest.approx(analytical, rel=1e-10)
    polar = rectangle_area_from_excess(0.0, 20.0, 80.0, 90.0, radius)
    polar_expected = radius**2 * (math.sin(math.radians(90.0)) - math.sin(math.radians(80.0))) * math.radians(20.0)
    assert polar == pytest.approx(polar_expected, rel=1e-12)
    with pytest.raises(GridError):
        rectangle_area_from_excess(0.0, 0.0, -5.0, 5.0, radius)


def test_great_circle_quarter_meridian() -> None:
    distance = great_circle_distance_m(0.0, 0.0, 0.0, 90.0)
    assert distance == pytest.approx(EARTH_RADIUS_M * math.pi / 2.0, rel=1e-12)


def test_inferred_hadcrut_like_edges() -> None:
    lat = np.arange(-87.5, 90.0, 5.0)
    bounds = infer_latitude_bounds(lat)
    assert bounds[0, 0] == pytest.approx(-90.0)
    assert bounds[-1, 1] == pytest.approx(90.0)
    assert bounds[1, 0] == pytest.approx(-85.0)
    lon = np.arange(2.5, 360.0, 5.0)
    lon_bounds = infer_longitude_bounds(lon, circular=True)
    assert lon_bounds[0, 0] == pytest.approx(0.0)
    assert lon_bounds[0, 1] == pytest.approx(5.0)
    assert longitude_width_deg(float(lon_bounds[-1, 0]), float(lon_bounds[-1, 1])) == pytest.approx(5.0)


def test_branch_cut_cell_area_matches_unwrapped_cell() -> None:
    radius = EARTH_RADIUS_M
    wrapped = LatLonGrid(
        lat=np.array([0.0]),
        lon=np.array([0.0]),
        lat_bounds=np.array([[-10.0, 10.0]]),
        lon_bounds=np.array([[350.0, 10.0]]),
        convention=LongitudeConvention.ZERO_360,
        radius_m=radius,
    )
    plain = LatLonGrid(
        lat=np.array([0.0]),
        lon=np.array([0.0]),
        lat_bounds=np.array([[-10.0, 10.0]]),
        lon_bounds=np.array([[-10.0, 10.0]]),
        convention=LongitudeConvention.NEG180_180,
        radius_m=radius,
    )
    assert wrapped.cell_areas_m2()[0, 0] == pytest.approx(plain.cell_areas_m2()[0, 0])


def test_area_mean_against_hand_weighted_bands() -> None:
    # South cell covers sin-weight 1.5, north cell covers 0.5. Values 0 and 4.
    grid = LatLonGrid(
        lat=np.array([-30.0, 60.0]),
        lon=np.array([180.0]),
        lat_bounds=np.array([[-90.0, 30.0], [30.0, 90.0]]),
        lon_bounds=np.array([[0.0, 360.0]]),
        convention=LongitudeConvention.ZERO_360,
    )
    field = np.array([[0.0], [4.0]])
    expect = (4.0 * 0.5) / 2.0
    assert area_weighted_mean(field, grid.cell_areas_m2())[()] == pytest.approx(expect)
    assert unweighted_mean(field)[()] == pytest.approx(2.0)
    assert unweighted_error(field, grid.cell_areas_m2())[()] == pytest.approx(2.0 - expect)
    # A missing northern cell must not become zero inside the mean.
    missing = np.array([[4.0], [np.nan]])
    assert area_weighted_mean(missing, grid.cell_areas_m2())[()] == pytest.approx(4.0)
    assert coverage_fraction(missing, grid.cell_areas_m2())[()] == pytest.approx(1.5 / 2.0)


def test_constant_field_regrid_and_identity() -> None:
    fine = regular_grid(36, 72)
    coarse = regular_grid(12, 24)
    field = np.full(fine.shape, 5.0)
    # One coarse cell covers a 3 by 3 block of the 5 degree grid. Blank the block.
    field[0:3, 0:3] = np.nan
    remapped = remap_field(field, fine, coarse, policy="intensive")
    finite = np.isfinite(remapped.values)
    assert np.all(np.isclose(remapped.values[finite], 5.0))
    assert np.isnan(remapped.values[0, 0])
    assert remapped.coverage[0, 0] == pytest.approx(0.0)
    intact = np.full(fine.shape, -2.5)
    intact[3, 4] = np.nan
    assert same_grid_identity_error(intact, fine) < 1e-10


def test_hot_cell_integral_is_conserved_on_the_covered_overlap() -> None:
    source = regular_grid(8, 16)
    target = regular_grid(4, 8)
    field = np.zeros(source.shape)
    field[2, 3] = 9.0
    remapped = remap_field(field, source, target, policy="intensive")
    # Covered integral uses finite remapped cells. Because the source cell is
    # fully inside the target tiling, the overlap mass equals the source mass.
    source_mass = covered_integral(field, source)
    target_mass = covered_integral(np.where(remapped.coverage > 0.0, remapped.values, np.nan), target)
    assert target_mass == pytest.approx(source_mass, rel=1e-10)


def test_zero_fill_dilutes_a_partial_cell() -> None:
    source = LatLonGrid(
        lat=np.array([0.0]),
        lon=np.array([0.0]),
        lat_bounds=np.array([[-10.0, 10.0]]),
        lon_bounds=np.array([[0.0, 10.0]]),
        convention=LongitudeConvention.ZERO_360,
    )
    target = LatLonGrid(
        lat=np.array([0.0]),
        lon=np.array([0.0]),
        lat_bounds=np.array([[-10.0, 10.0]]),
        lon_bounds=np.array([[0.0, 20.0]]),
        convention=LongitudeConvention.ZERO_360,
    )
    field = np.array([[4.0]])
    kept = remap_field(field, source, target, policy="intensive")
    diluted = remap_field(field, source, target, policy="zero_fill")
    assert kept.values[0, 0] == pytest.approx(4.0)
    assert kept.coverage[0, 0] == pytest.approx(0.5)
    assert diluted.values[0, 0] == pytest.approx(2.0)
    assert diluted.policy == "zero_fill"


def test_regional_overlap_differs_from_center_on_a_coarse_grid() -> None:
    grid = regular_grid(18, 36)  # 10 degree cells
    field = np.zeros(grid.shape)
    # Everything north of 60 degrees is 8 K. A cell centered at 65 spans 60 to 70.
    for index, center in enumerate(grid.lat):
        if center >= 60.0:
            field[index, :] = 8.0
    arctic = get_region("arctic")
    overlap_mean, overlap_cov = regional_mean(field, grid, arctic, rule="overlap")
    center_mean, _ = regional_mean(field, grid, arctic, rule="center")
    assert overlap_cov == pytest.approx(1.0)
    assert center_mean == pytest.approx(8.0)
    # The 60-70 cell is only partly inside 66.5 N, so the overlap mean is below 8
    # if that cell is 8 and we also have cells south of 66.5 contributing 0? 
    # Field is 0 below center 60, and 8 at centers >= 60. The cell spanning 60-70
    # has center 65, value 8, and part of it (60 to 66.5) is outside the Arctic
    # while still carrying value 8. Overlap therefore still sees 8 everywhere it
    # looks inside the Arctic, because the only intersecting cells are valued 8.
    assert overlap_mean == pytest.approx(8.0)
    # A field that is 8 only poleward of 70, and 0 on the 60-70 cell, separates the rules.
    field[np.where(np.isclose(grid.lat, 65.0))[0][0], :] = 0.0
    overlap_mean, _ = regional_mean(field, grid, arctic, rule="overlap")
    center_mean, _ = regional_mean(field, grid, arctic, rule="center")
    assert center_mean == pytest.approx(8.0)
    assert overlap_mean < center_mean
    # Hand weight: Arctic sin-span is sin(90) - sin(66.5). The 60-70 cell contributes
    # only the slice 66.5 to 70 of zeros, and the rest of the Arctic is 8.
    arctic_sin = math.sin(math.radians(90.0)) - math.sin(math.radians(66.5))
    zero_sin = math.sin(math.radians(70.0)) - math.sin(math.radians(66.5))
    expect = 8.0 * (arctic_sin - zero_sin) / arctic_sin
    assert overlap_mean == pytest.approx(expect, rel=1e-12)


def test_constant_global_mean_ignores_a_hole() -> None:
    grid = regular_grid(12, 24)
    field = np.full(grid.shape, 1.5)
    field[0, :] = np.nan
    mean, coverage = regional_mean(field, grid, get_region("global"), rule="overlap")
    assert mean == pytest.approx(1.5)
    polar_band = band_area_m2(-90.0, grid.lat_bounds[0, 1])
    assert coverage == pytest.approx(1.0 - polar_band / globe_area_m2())
