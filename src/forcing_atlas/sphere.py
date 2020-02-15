"""Spherical geometry that does not assume a latitude-longitude rectangle.

Cell areas on a regular grid use the closed sin(latitude) formula in
`grids.py`. This module is the independent path: unit vectors, great-circle
turning, and Girard's theorem for a spherical polygon. The two paths must
agree on a lat-lon rectangle. A disagreement is a bug in one of them, not a
license to average the answers.
"""

from __future__ import annotations

import math

import numpy as np

from forcing_atlas.constants import EARTH_RADIUS_M
from forcing_atlas.errors import GridError


def radians(degrees: np.ndarray | float) -> np.ndarray:
    return np.deg2rad(np.asarray(degrees, dtype=np.float64))


def lonlat_to_unit(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Map longitude and latitude in degrees to unit vectors (..., 3).

    The frame is right-handed: x through (lon 0, lat 0), y through (lon 90, lat 0),
    z through the north pole. Latitude outside [-90, 90] is rejected.
    """

    lon = np.asarray(lon_deg, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)
    if lon.shape != lat.shape:
        raise GridError("longitude and latitude shapes differ")
    if np.any(lat < -90.0) or np.any(lat > 90.0):
        raise GridError("latitude outside [-90, 90]")
    if np.any(~np.isfinite(lon)) or np.any(~np.isfinite(lat)):
        raise GridError("longitude and latitude must be finite")
    lam = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    cos_phi = np.cos(phi)
    vectors = np.stack((cos_phi * np.cos(lam), cos_phi * np.sin(lam), np.sin(phi)), axis=-1)
    return vectors


def unit_to_lonlat(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(vectors, dtype=np.float64)
    if array.shape[-1] != 3:
        raise GridError("unit vectors must end with a length-3 axis")
    norm = np.linalg.norm(array, axis=-1)
    if np.any(norm <= 0.0):
        raise GridError("zero vector has no longitude")
    unit = array / norm[..., None]
    lat = np.rad2deg(np.arcsin(np.clip(unit[..., 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(unit[..., 1], unit[..., 0]))
    return lon, lat


def great_circle_distance_m(
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
    radius_m: float = EARTH_RADIUS_M,
) -> float:
    """Central angle via atan2 of the cross and dot product, then times radius."""

    if radius_m <= 0.0:
        raise GridError("radius must be positive")
    a = lonlat_to_unit(np.array(lon0), np.array(lat0))
    b = lonlat_to_unit(np.array(lon1), np.array(lat1))
    cross = np.linalg.norm(np.cross(a, b))
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = math.atan2(cross, dot)
    return angle * radius_m


def spherical_excess(lon_deg: np.ndarray, lat_deg: np.ndarray) -> float:
    """Signed spherical excess of a simple polygon, in steradians.

    This is the boundary integral used for a geodetic ring:

        sum (lon[i+1] - lon[i]) * (2 + sin lat[i] + sin lat[i+1]) / 2

    with angles in radians. It is Girard's excess for a simple polygon that
    does not contain a pole in its interior in a way that reverses the ring,
    and it is a different code path from the cell-area loop in `grids.py`.
    Vertices are in order and are not repeated at the end. Consecutive
    duplicates are rejected. Longitude must already be continuous: a ring
    that jumps by nearly 360 degrees is a branch-cut bug in the caller.
    """

    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64).reshape(-1))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64).reshape(-1))
    if lon.size != lat.size or lon.size < 3:
        raise GridError("a spherical polygon needs at least three vertices")
    if np.any(lat < -math.pi / 2.0) or np.any(lat > math.pi / 2.0):
        raise GridError("polygon latitude outside the sphere")
    total = 0.0
    count = int(lon.size)
    for index in range(count):
        nxt = (index + 1) % count
        dlon = float(lon[nxt] - lon[index])
        if abs(dlon) > math.pi * 1.5:
            raise GridError("polygon longitude jumps by more than 270 degrees; split the ring at the branch cut")
        if abs(dlon) < 1e-15 and abs(float(lat[nxt] - lat[index])) < 1e-15:
            raise GridError("polygon has a repeated vertex")
        total += dlon * (2.0 + math.sin(float(lat[index])) + math.sin(float(lat[nxt])))
    return 0.5 * total


def spherical_polygon_area_m2(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    radius_m: float = EARTH_RADIUS_M,
) -> float:
    """Absolute area of the smaller interior when the excess magnitude is below 2 pi.

    Polygons larger than a hemisphere are rejected. Climate cells are much
    smaller than that, and the absolute excess would otherwise pick the
    complement if the winding were reversed.
    """

    if radius_m <= 0.0:
        raise GridError("radius must be positive")
    excess = spherical_excess(lon_deg, lat_deg)
    magnitude = abs(excess)
    if magnitude >= 2.0 * math.pi:
        raise GridError(
            f"spherical excess {magnitude} steradians is at least a hemisphere; split the polygon"
        )
    return magnitude * radius_m * radius_m


def rectangle_area_from_excess(
    lon_west: float,
    lon_east: float,
    lat_south: float,
    lat_north: float,
    radius_m: float = EARTH_RADIUS_M,
) -> float:
    """Area of one lat-lon rectangle via Girard, after unwrapping longitude.

    `lon_east` may be numerically smaller than `lon_west` when the cell crosses
    the branch cut. The width must be in (0, 360].
    """

    width = longitude_width_deg(lon_west, lon_east)
    if width <= 0.0 or width > 360.0:
        raise GridError(f"longitude width {width} is outside (0, 360]")
    if lat_north <= lat_south:
        raise GridError("north edge must be north of the south edge")
    if lat_south < -90.0 or lat_north > 90.0:
        raise GridError("rectangle latitude is outside the sphere")
    # Walk the boundary with continuous longitude so the polygon does not jump.
    west = lon_west
    east = lon_west + width
    lons = np.array([west, east, east, west], dtype=np.float64)
    lats = np.array([lat_south, lat_south, lat_north, lat_north], dtype=np.float64)
    return spherical_polygon_area_m2(lons, lats, radius_m)


def longitude_width_deg(lon_west: float, lon_east: float) -> float:
    """Positive eastward width in degrees, allowing a branch-cut crossing.

    A width of 360 describes the whole parallel. A width of 0 is rejected by
    callers that need an area. This helper itself returns 0 when the edges
    coincide modulo 360, and 360 only when the caller passes a full turn by
    setting east = west + 360 numerically. Passing identical edges yields 0.
    """

    if not math.isfinite(lon_west) or not math.isfinite(lon_east):
        raise GridError("longitude edges must be finite")
    raw = lon_east - lon_west
    # A raw turn of exactly one revolution is the whole parallel. The modulo
    # operator collapses that turn to 0, which would look like coincident edges.
    if math.isclose(abs(raw), 360.0, abs_tol=1e-8):
        return 360.0
    return float(raw % 360.0)


def band_area_m2(lat_south: float, lat_north: float, radius_m: float = EARTH_RADIUS_M) -> float:
    """Exact area of a full latitude band. This is the analytical hemisphere check."""

    if lat_north <= lat_south:
        raise GridError("band north must exceed band south")
    if lat_south < -90.0 or lat_north > 90.0:
        raise GridError("band outside [-90, 90]")
    if radius_m <= 0.0:
        raise GridError("radius must be positive")
    return (
        radius_m
        * radius_m
        * (math.sin(math.radians(lat_north)) - math.sin(math.radians(lat_south)))
        * 2.0
        * math.pi
    )


def hemisphere_area_m2(radius_m: float = EARTH_RADIUS_M) -> float:
    return band_area_m2(-90.0, 0.0, radius_m)


def globe_area_m2(radius_m: float = EARTH_RADIUS_M) -> float:
    return 4.0 * math.pi * radius_m * radius_m
