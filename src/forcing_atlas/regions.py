"""Named regions for drill-down.

A latitude band can be masked by cell center or by exact overlap fraction.
The overlap fraction is the one used when a study quotes an Arctic mean,
because a 5 degree cell centered at 65 degrees is not entirely inside the
Arctic Circle. Center membership is still available and is labeled, so the
difference between the two rules can be shown rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.errors import GridError, MissingDataError
from forcing_atlas.grids import LatLonGrid, area_weighted_mean


@dataclass(frozen=True, slots=True)
class Region:
    region_id: str
    south: float
    north: float
    west: float | None = None
    east: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.north <= self.south:
            raise GridError(f"region {self.region_id} has north <= south")
        if self.south < -90.0 or self.north > 90.0:
            raise GridError(f"region {self.region_id} leaves [-90, 90]")
        if (self.west is None) ^ (self.east is None):
            raise GridError(f"region {self.region_id} needs both longitude edges or neither")


REGIONS: dict[str, Region] = {
    "global": Region("global", -90.0, 90.0, note="Full sphere."),
    "northern_hemisphere": Region("northern_hemisphere", 0.0, 90.0, note="Latitude at or north of the equator."),
    "southern_hemisphere": Region("southern_hemisphere", -90.0, 0.0, note="Latitude at or south of the equator."),
    "arctic": Region("arctic", 66.5, 90.0, note="Latitude at or north of 66.5 degrees. Not a model-native Arctic mask."),
    "antarctic": Region("antarctic", -90.0, -66.5, note="Latitude at or south of -66.5 degrees."),
    "tropics": Region("tropics", -23.5, 23.5, note="Latitude band between 23.5 south and 23.5 north."),
    "nino34": Region(
        "nino34",
        -5.0,
        5.0,
        190.0,
        240.0,
        note="Nino 3.4 box, 5 S to 5 N and 170 W to 120 W, stored on a 0-360 longitude.",
    ),
}


def get_region(region_id: str) -> Region:
    try:
        return REGIONS[region_id]
    except KeyError as exc:
        raise GridError(f"unknown region {region_id!r}") from exc


def _cell_longitude_fraction(grid: LatLonGrid, west: float, east: float) -> np.ndarray:
    """Fraction of each longitude cell inside the eastward interval [west, east).

    The interval may cross the branch cut when east < west. A full-globe
    interval is expressed by a width of 360, which this helper does not
    receive: callers pass None for a zonal mean instead.
    """

    from forcing_atlas.sphere import longitude_width_deg

    width = longitude_width_deg(west, east)
    # Represent the requested interval as pieces in [0, 360).
    start = west % 360.0
    end = start + width
    requested = [(start, min(end, 360.0))]
    if end > 360.0:
        requested.append((0.0, end - 360.0))
    cell_parts_west = grid.lon_bounds[:, 0]
    fractions = np.zeros(grid.nlon, dtype=np.float64)
    for index in range(grid.nlon):
        cell_width = float(grid.longitude_widths_deg()[index])
        cell_start = float(cell_parts_west[index]) % 360.0
        cell_end = cell_start + cell_width
        cell_intervals = [(cell_start, min(cell_end, 360.0))]
        if cell_end > 360.0:
            cell_intervals.append((0.0, cell_end - 360.0))
        overlapped = 0.0
        for c0, c1 in cell_intervals:
            for r0, r1 in requested:
                overlapped += max(0.0, min(c1, r1) - max(c0, r0))
        fractions[index] = overlapped / cell_width
    return fractions


def latitude_overlap_fraction(grid: LatLonGrid, south: float, north: float) -> np.ndarray:
    """Fraction of each latitude cell's sin-weight inside [south, north]."""

    cell_south = grid.lat_bounds[:, 0]
    cell_north = grid.lat_bounds[:, 1]
    lower = np.maximum(cell_south, south)
    upper = np.minimum(cell_north, north)
    overlap = np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower))
    full = np.sin(np.deg2rad(cell_north)) - np.sin(np.deg2rad(cell_south))
    fraction = np.zeros(grid.nlat, dtype=np.float64)
    positive = (upper > lower) & (full > 0.0)
    fraction[positive] = overlap[positive] / full[positive]
    return fraction


def center_mask(grid: LatLonGrid, region: Region) -> np.ndarray:
    """True where the cell center lies inside the region. Boundary cells are all or nothing."""

    lat_ok = (grid.lat >= region.south) & (grid.lat <= region.north)
    if region.west is None or region.east is None:
        return np.broadcast_to(lat_ok[:, None], grid.shape).copy()
    # Compare on a 0-360 ring using the same interval logic as the fractional mask,
    # but test the center point rather than the cell width.
    from forcing_atlas.sphere import longitude_width_deg

    width = longitude_width_deg(region.west, region.east)
    start = region.west % 360.0
    centers = grid.lon % 360.0
    offset = (centers - start) % 360.0
    lon_ok = offset <= width + 1.0e-9
    return lat_ok[:, None] & lon_ok[None, :]


def fractional_weights(grid: LatLonGrid, region: Region) -> np.ndarray:
    """Area weights of the portion of each cell inside the region. Zero if disjoint."""

    lat_fraction = latitude_overlap_fraction(grid, region.south, region.north)
    areas = grid.cell_areas_m2()
    if region.west is None:
        return areas * lat_fraction[:, None]
    lon_fraction = _cell_longitude_fraction(grid, region.west, region.east)
    return areas * lat_fraction[:, None] * lon_fraction[None, :]


def regional_mean(
    field: np.ndarray,
    grid: LatLonGrid,
    region: Region,
    *,
    rule: str = "overlap",
) -> tuple[float, float]:
    """Return (mean, coverage_fraction) for one 2-d field.

    `rule` is `overlap` or `center`. Coverage is the fraction of the region's
    geometric weight that landed on finite field cells.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.shape != grid.shape:
        raise GridError("regional mean expects one field matching the grid")
    if rule == "overlap":
        weights = fractional_weights(grid, region)
    elif rule == "center":
        mask = center_mask(grid, region)
        weights = np.where(mask, grid.cell_areas_m2(), 0.0)
    else:
        raise GridError(f"unknown region rule {rule!r}")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise MissingDataError(f"region {region.region_id} does not intersect the grid")
    valid = np.isfinite(values) & (weights > 0.0)
    retained = float(np.sum(weights[valid]))
    if retained <= 0.0:
        return float("nan"), 0.0
    mean = float(np.sum(values[valid] * weights[valid]) / retained)
    return mean, retained / total


def regional_series(
    field: np.ndarray,
    grid: LatLonGrid,
    region_id: str,
    *,
    rule: str = "overlap",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply `regional_mean` along a leading time axis. Field shape is (time, lat, lon)."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != grid.shape:
        raise GridError("regional series expects (time, lat, lon)")
    region = get_region(region_id)
    means = np.empty(values.shape[0], dtype=np.float64)
    coverage = np.empty(values.shape[0], dtype=np.float64)
    for index in range(values.shape[0]):
        means[index], coverage[index] = regional_mean(values[index], grid, region, rule=rule)
    return means, coverage


def global_mean(field: np.ndarray, grid: LatLonGrid) -> np.ndarray:
    return area_weighted_mean(field, grid.cell_areas_m2())
