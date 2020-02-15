"""Regular latitude-longitude grids, longitude conventions, and area weights.

Centers are not enough for an area. Every grid stored by this project carries
explicit edges. Inferred edges are a convenience for files that publish only
centers, and the inference rule is the midpoint rule with a circular longitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from forcing_atlas.constants import EARTH_RADIUS_M
from forcing_atlas.errors import GridError
from forcing_atlas.sphere import band_area_m2, longitude_width_deg


class LongitudeConvention(Enum):
    ZERO_360 = "0_360"
    NEG180_180 = "-180_180"


def normalize_longitude(lon: np.ndarray, convention: LongitudeConvention) -> np.ndarray:
    """Wrap longitudes into the named interval.

    `0_360` uses [0, 360). `-180_180` uses [-180, 180). The meridian at the
    branch cut is mapped to the lower end of the interval, so 360 and -180
    become 0 and -180. A value of 180 in the signed convention becomes -180.
    """

    array = np.asarray(lon, dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise GridError("longitude contains a non-finite value")
    if convention is LongitudeConvention.ZERO_360:
        wrapped = np.mod(array, 360.0)
        return np.where(np.isclose(wrapped, 360.0), 0.0, wrapped)
    wrapped = np.mod(array + 180.0, 360.0) - 180.0
    # np.mod maps exactly 180 + 180k onto -180 for the signed interval after the shift.
    return np.where(np.isclose(wrapped, 180.0), -180.0, wrapped)


def detect_longitude_convention(lon: np.ndarray, *, atol: float = 1.0e-6) -> LongitudeConvention:
    """Classify a center vector. Mixed signs and values above 180 are ambiguous.

    A vector that stays inside [-180, 180] and contains a negative value is
    signed. A vector that stays inside [0, 360] and contains a value above 180
    is `0_360`. A vector that contains both a negative value and a value above
    180 is rejected. A vector inside [0, 180] is reported as `0_360` because
    that interval is legal in both conventions and the non-negative form is
    the one that does not move the data.
    """

    array = np.asarray(lon, dtype=np.float64)
    if array.size == 0:
        raise GridError("empty longitude vector")
    if np.any(~np.isfinite(array)):
        raise GridError("longitude contains a non-finite value")
    negative = np.any(array < -atol)
    above = np.any(array > 180.0 + atol)
    if negative and above:
        raise GridError("longitude mix of negative values and values above 180 has no single convention")
    if negative:
        return LongitudeConvention.NEG180_180
    return LongitudeConvention.ZERO_360


def _monotonic_increasing(values: np.ndarray, *, circular: bool) -> None:
    if values.size < 1:
        raise GridError("axis is empty")
    if np.any(~np.isfinite(values)):
        raise GridError("axis contains a non-finite value")
    if not circular and np.any(np.diff(values) <= 0.0):
        raise GridError("latitude centers must be strictly increasing from south to north")


def infer_latitude_bounds(lat_centers: np.ndarray) -> np.ndarray:
    """Midpoint edges. Outer edges extrapolate the end spacing and then clamp to the poles.

    Clamping is recorded by the caller via `LatLonGrid.edge_clamped`. A file
    whose published bounds already include the pole should pass those bounds
    instead of inferring them.
    """

    lat = np.asarray(lat_centers, dtype=np.float64)
    _monotonic_increasing(lat, circular=False)
    if np.any(lat < -90.0) or np.any(lat > 90.0):
        raise GridError("latitude center outside [-90, 90]")
    bounds = np.empty((lat.size, 2), dtype=np.float64)
    if lat.size == 1:
        half = 90.0
        bounds[0, 0] = max(-90.0, float(lat[0]) - half)
        bounds[0, 1] = min(90.0, float(lat[0]) + half)
        return bounds
    mids = 0.5 * (lat[:-1] + lat[1:])
    first_width = float(mids[0] - lat[0])
    last_width = float(lat[-1] - mids[-1])
    bounds[0, 0] = float(lat[0]) - first_width
    bounds[0, 1] = float(mids[0])
    bounds[1:, 0] = mids
    bounds[:-1, 1] = mids
    bounds[-1, 0] = float(mids[-1])
    bounds[-1, 1] = float(lat[-1]) + last_width
    bounds[:, 0] = np.maximum(bounds[:, 0], -90.0)
    bounds[:, 1] = np.minimum(bounds[:, 1], 90.0)
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise GridError("inferred latitude bounds collapsed at the pole")
    return bounds


def _eastward_degrees(start: float, end: float) -> float:
    """Degrees traveling east from start to end, in (0, 360]. A zero step is an error."""

    step = (end - start) % 360.0
    if step == 0.0:
        raw = end - start
        if math_isclose_revolution(raw):
            return 360.0
        raise GridError("longitude centers are not separated eastward")
    return float(step)


def math_isclose_revolution(raw: float) -> bool:
    return abs(abs(raw) - 360.0) <= 1.0e-8


def infer_longitude_bounds(lon_centers: np.ndarray, *, circular: bool = True) -> np.ndarray:
    """Midpoint edges in the stored center order.

    Circular axes wrap from the last center back to the first and must tile
    360 degrees. Regional axes do not wrap. Their outer edges extrapolate the
    end spacing. The stored order must already run eastward.
    """

    lon = np.asarray(lon_centers, dtype=np.float64)
    if lon.size < 1 or np.any(~np.isfinite(lon)):
        raise GridError("longitude axis is empty or non-finite")
    bounds = np.empty((lon.size, 2), dtype=np.float64)
    if lon.size == 1:
        if not circular:
            raise GridError("a single longitude center needs a circular axis or explicit bounds")
        bounds[0, 0] = float(lon[0]) - 180.0
        bounds[0, 1] = float(lon[0]) + 180.0
        return bounds
    if circular:
        steps = np.array(
            [_eastward_degrees(float(lon[index]), float(lon[(index + 1) % lon.size])) for index in range(lon.size)],
            dtype=np.float64,
        )
        if abs(float(np.sum(steps)) - 360.0) > 1.0e-6:
            raise GridError(
                f"circular longitude steps sum to {float(np.sum(steps))}, not 360. "
                "Pass circular=False for a regional axis."
            )
        for index in range(lon.size):
            previous = steps[(index - 1) % lon.size]
            bounds[index, 0] = float(lon[index]) - 0.5 * previous
            bounds[index, 1] = float(lon[index]) + 0.5 * steps[index]
        return bounds
    if np.any(np.diff(lon) <= 0.0):
        raise GridError("regional longitude centers must increase eastward without wrapping")
    mids = 0.5 * (lon[:-1] + lon[1:])
    first = float(lon[0] - (mids[0] - lon[0]))
    last = float(lon[-1] + (lon[-1] - mids[-1]))
    bounds[0, 0] = first
    bounds[0, 1] = float(mids[0])
    bounds[1:, 0] = mids
    bounds[:-1, 1] = mids
    bounds[-1, 1] = last
    return bounds


@dataclass(slots=True)
class LatLonGrid:
    """Cell centers and edges in degrees.

    `lat_bounds` is (nlat, 2) as (south, north). `lon_bounds` is (nlon, 2) as
    (west, east). East may be smaller than west modulo 360 when the caller
    stored a branch-cut cell in a wrapped convention. Widths use
    `longitude_width_deg`. Latitude increases south to north. A source file
    that stores north to south is flipped before a grid is built.
    """

    lat: np.ndarray
    lon: np.ndarray
    lat_bounds: np.ndarray
    lon_bounds: np.ndarray
    convention: LongitudeConvention
    radius_m: float = EARTH_RADIUS_M
    name: str = ""

    def __post_init__(self) -> None:
        self.lat = np.asarray(self.lat, dtype=np.float64)
        self.lon = np.asarray(self.lon, dtype=np.float64)
        self.lat_bounds = np.asarray(self.lat_bounds, dtype=np.float64)
        self.lon_bounds = np.asarray(self.lon_bounds, dtype=np.float64)
        if self.lat.ndim != 1 or self.lon.ndim != 1:
            raise GridError("lat and lon centers must be 1-d")
        if self.lat_bounds.shape != (self.lat.size, 2):
            raise GridError("lat_bounds shape does not match latitude centers")
        if self.lon_bounds.shape != (self.lon.size, 2):
            raise GridError("lon_bounds shape does not match longitude centers")
        if self.radius_m <= 0.0:
            raise GridError("radius must be positive")
        if np.any(np.diff(self.lat) <= 0.0):
            raise GridError("latitude centers must increase south to north")
        if np.any(self.lat_bounds[:, 1] <= self.lat_bounds[:, 0]):
            raise GridError("each latitude cell needs north > south")
        if np.any(self.lat_bounds[:, 0] < -90.0) or np.any(self.lat_bounds[:, 1] > 90.0):
            raise GridError("latitude bounds outside [-90, 90]")
        if np.any(self.lat < self.lat_bounds[:, 0]) or np.any(self.lat > self.lat_bounds[:, 1]):
            raise GridError("a latitude center lies outside its cell")
        widths = self.longitude_widths_deg()
        if np.any(widths <= 0.0) or np.any(widths > 360.0):
            raise GridError("longitude cell width must lie in (0, 360]")

    @property
    def nlat(self) -> int:
        return int(self.lat.size)

    @property
    def nlon(self) -> int:
        return int(self.lon.size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nlat, self.nlon)

    def longitude_widths_deg(self) -> np.ndarray:
        widths = np.empty(self.nlon, dtype=np.float64)
        for index in range(self.nlon):
            widths[index] = longitude_width_deg(float(self.lon_bounds[index, 0]), float(self.lon_bounds[index, 1]))
        return widths

    def cell_areas_m2(self) -> np.ndarray:
        """Spherical cell areas. Missing data are not involved. Every cell has a positive area."""

        dsin = np.sin(np.deg2rad(self.lat_bounds[:, 1])) - np.sin(np.deg2rad(self.lat_bounds[:, 0]))
        if np.any(dsin <= 0.0):
            raise GridError("latitude bounds produced a nonpositive sin difference")
        widths = np.deg2rad(self.longitude_widths_deg())
        areas = (self.radius_m ** 2) * dsin[:, None] * widths[None, :]
        return areas

    def covers_globe(self, *, atol_deg: float = 1.0e-6, pole_atol_deg: float = 1.0e-6) -> bool:
        width = float(np.sum(self.longitude_widths_deg()))
        lat_span_ok = self.lat_bounds[0, 0] <= -90.0 + pole_atol_deg and self.lat_bounds[-1, 1] >= 90.0 - pole_atol_deg
        return abs(width - 360.0) <= atol_deg and lat_span_ok

    def globe_area_gap_m2(self) -> float:
        """Difference between 4 pi R^2 and the sum of cell areas. Zero on a full sphere."""

        return 4.0 * np.pi * self.radius_m ** 2 - float(np.sum(self.cell_areas_m2()))


def regular_grid(
    nlat: int,
    nlon: int,
    *,
    convention: LongitudeConvention = LongitudeConvention.ZERO_360,
    radius_m: float = EARTH_RADIUS_M,
    name: str = "",
) -> LatLonGrid:
    """Global grid with equal longitude spacing and equal latitude spacing, edges on the poles.

    Latitude centers sit in the middle of bands that tile [-90, 90]. This is
    the HadCRUT-style 5 degree arrangement when nlat=36 and nlon=72.
    """

    if nlat < 1 or nlon < 1:
        raise GridError("grid dimensions must be positive")
    lat_edges = np.linspace(-90.0, 90.0, nlat + 1)
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lat_bounds = np.column_stack([lat_edges[:-1], lat_edges[1:]])
    if convention is LongitudeConvention.ZERO_360:
        lon_edges = np.linspace(0.0, 360.0, nlon + 1)
        lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lon_bounds = np.column_stack([lon_edges[:-1], lon_edges[1:]])
    else:
        lon_edges = np.linspace(-180.0, 180.0, nlon + 1)
        lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lon_bounds = np.column_stack([lon_edges[:-1], lon_edges[1:]])
    return LatLonGrid(lat, lon, lat_bounds, lon_bounds, convention, radius_m, name)


def flip_latitude_if_needed(
    lat: np.ndarray,
    field: np.ndarray,
    lat_bounds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return south-to-north latitude, field, and bounds.

    The field's last-but-one axis is latitude. A north-to-south axis is flipped.
    A non-monotonic axis is rejected rather than sorted, because sorting would
    silently reorder a scrambled file.
    """

    centers = np.asarray(lat, dtype=np.float64)
    values = np.asarray(field, dtype=np.float64)
    if values.shape[-2] != centers.size:
        raise GridError("field latitude axis does not match lat centers")
    if np.all(np.diff(centers) > 0.0):
        bounds = None if lat_bounds is None else np.asarray(lat_bounds, dtype=np.float64).copy()
        return centers.copy(), values.copy(), bounds
    if np.all(np.diff(centers) < 0.0):
        flipped = np.flip(values, axis=-2).copy()
        flipped_lat = centers[::-1].copy()
        flipped_bounds = None
        if lat_bounds is not None:
            raw = np.asarray(lat_bounds, dtype=np.float64)[::-1].copy()
            # After reversal the edge pair may be (north, south). Store (south, north).
            flipped_bounds = np.column_stack([np.minimum(raw[:, 0], raw[:, 1]), np.maximum(raw[:, 0], raw[:, 1])])
        return flipped_lat, flipped, flipped_bounds
    raise GridError("latitude axis is not monotonic")


def area_weighted_mean(field: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Mean of finite cells. Weights of non-finite cells are dropped, not zero-filled into the value.

    `field` ends in (nlat, nlon). `areas` is (nlat, nlon) and must be finite
    and positive. The result has the leading shape of `field`.
    """

    values = np.asarray(field, dtype=np.float64)
    weight = np.asarray(areas, dtype=np.float64)
    if values.shape[-2:] != weight.shape:
        raise GridError(
            f"field spatial shape {values.shape[-2:]} does not match areas {weight.shape}"
        )
    if np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
        raise GridError("areas must be finite and positive")
    valid = np.isfinite(values)
    weighted = np.where(valid, values * weight, 0.0)
    retained = np.where(valid, weight, 0.0)
    numerator = np.sum(weighted, axis=(-2, -1))
    denominator = np.sum(retained, axis=(-2, -1))
    mean = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=mean, where=denominator > 0.0)
    return mean


def unweighted_mean(field: np.ndarray) -> np.ndarray:
    """Arithmetic mean of finite cells. This is the diagnostic, not the global mean."""

    values = np.asarray(field, dtype=np.float64)
    valid = np.isfinite(values)
    count = np.sum(valid, axis=(-2, -1))
    total = np.sum(np.where(valid, values, 0.0), axis=(-2, -1))
    mean = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean


def unweighted_error(field: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Unweighted mean minus area-weighted mean, on the same finite cells."""

    return unweighted_mean(field) - area_weighted_mean(field, areas)


def coverage_fraction(field: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Fraction of the area in `areas` whose field value is finite."""

    values = np.asarray(field, dtype=np.float64)
    weight = np.asarray(areas, dtype=np.float64)
    if values.shape[-2:] != weight.shape:
        raise GridError("coverage field and areas disagree in shape")
    valid = np.isfinite(values)
    retained = np.sum(np.where(valid, weight, 0.0), axis=(-2, -1))
    total = np.sum(weight, axis=(-2, -1))
    fraction = np.full(retained.shape, np.nan, dtype=np.float64)
    np.divide(retained, total, out=fraction, where=total > 0.0)
    return fraction


def hemisphere_area_from_grid(grid: LatLonGrid, *, northern: bool) -> float:
    """Sum of cell areas whose latitude interval lies wholly in one hemisphere.

    Cells that straddle the equator are excluded. `band_area_m2` is the
    analytical comparison for a grid that tiles the hemisphere with such cells.
    """

    areas = grid.cell_areas_m2()
    if northern:
        mask = grid.lat_bounds[:, 0] >= 0.0
    else:
        mask = grid.lat_bounds[:, 1] <= 0.0
    return float(np.sum(areas[mask, :]))


def analytical_band_area(lat_south: float, lat_north: float, radius_m: float = EARTH_RADIUS_M) -> float:
    return band_area_m2(lat_south, lat_north, radius_m)
