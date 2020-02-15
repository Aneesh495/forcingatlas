"""First-order conservative overlap between regular latitude-longitude grids.

The overlap of two lat-lon rectangles separates into a latitude factor in
sin(phi) and a longitude factor. That is exact on the sphere used by this
project. It is not a bilinear interpolant.

Two policies exist because they answer different questions:

- `intensive` keeps the overlap-weighted mean and leaves a target cell missing
  when no finite source cell overlaps it. Uncovered area is not treated as zero.
- `zero_fill` divides the overlapped mass by the full target cell area. That
  conserves a global integral only by writing explicit zeros into the gaps.
  It is available so tests can show the dilution. It is not the default and
  the export path does not call it.

Extensive densities use the intensive weights. The integral over the covered
overlap is unchanged. A target cell reports its coverage fraction next to the
value so a later average can decide whether the cell is usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.errors import GridError
from forcing_atlas.grids import LatLonGrid
from forcing_atlas.sphere import longitude_width_deg


@dataclass(frozen=True, slots=True)
class RemapResult:
    values: np.ndarray
    coverage: np.ndarray
    policy: str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


def _latitude_overlap_sin(source: LatLonGrid, target: LatLonGrid) -> np.ndarray:
    """(nlat_target, nlat_source) overlap of sin(latitude)."""

    south_s = source.lat_bounds[:, 0]
    north_s = source.lat_bounds[:, 1]
    south_t = target.lat_bounds[:, 0]
    north_t = target.lat_bounds[:, 1]
    lower = np.maximum(south_t[:, None], south_s[None, :])
    upper = np.minimum(north_t[:, None], north_s[None, :])
    overlap = np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower))
    overlap = np.where(upper > lower, overlap, 0.0)
    return overlap


def _split_longitude_intervals(bounds: np.ndarray) -> list[list[tuple[float, float]]]:
    """Each cell becomes one or two intervals inside [0, 360)."""

    pieces: list[list[tuple[float, float]]] = []
    for index in range(bounds.shape[0]):
        west = float(bounds[index, 0])
        width = longitude_width_deg(west, float(bounds[index, 1]))
        start = west % 360.0
        end = start + width
        if end <= 360.0:
            pieces.append([(start, end)])
        else:
            pieces.append([(start, 360.0), (0.0, end - 360.0)])
    return pieces


def _interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    lower = max(start_a, start_b)
    upper = min(end_a, end_b)
    if upper <= lower:
        return 0.0
    return upper - lower


def _longitude_overlap_deg(source: LatLonGrid, target: LatLonGrid) -> np.ndarray:
    """(nlon_target, nlon_source) overlap in degrees, after splitting the branch cut."""

    source_parts = _split_longitude_intervals(source.lon_bounds)
    target_parts = _split_longitude_intervals(target.lon_bounds)
    overlap = np.zeros((target.nlon, source.nlon), dtype=np.float64)
    for target_index, target_intervals in enumerate(target_parts):
        for source_index, source_intervals in enumerate(source_parts):
            total = 0.0
            for target_start, target_end in target_intervals:
                for source_start, source_end in source_intervals:
                    total += _interval_overlap(target_start, target_end, source_start, source_end)
            overlap[target_index, source_index] = total
    return overlap


def _weights(source: LatLonGrid, target: LatLonGrid) -> tuple[np.ndarray, np.ndarray]:
    if source.radius_m != target.radius_m:
        raise GridError("regridding two radii is a different geometry and is refused")
    lat = _latitude_overlap_sin(source, target)
    lon = np.deg2rad(_longitude_overlap_deg(source, target))
    return lat, lon


def remap_field(
    field: np.ndarray,
    source: LatLonGrid,
    target: LatLonGrid,
    *,
    policy: str = "intensive",
) -> RemapResult:
    """Remap a field whose last two axes are (lat, lon) on `source`.

    `policy` is `intensive` or `zero_fill`. Any other name is rejected.
    Non-finite source cells do not contribute weight.
    """

    if policy not in {"intensive", "zero_fill"}:
        raise GridError(f"unknown remap policy {policy!r}")
    values = np.asarray(field, dtype=np.float64)
    if values.shape[-2:] != source.shape:
        raise GridError(
            f"field spatial shape {values.shape[-2:]} does not match source grid {source.shape}"
        )
    lat_w, lon_w = _weights(source, target)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    mask = valid.astype(np.float64)
    leading = values.shape[:-2]
    flat = filled.reshape((-1, source.nlat, source.nlon))
    flat_mask = mask.reshape((-1, source.nlat, source.nlon))
    out = np.empty((flat.shape[0], target.nlat, target.nlon), dtype=np.float64)
    coverage = np.empty_like(out)
    # Target geometric factors for the zero-fill denominator and the coverage fraction.
    target_sin = np.sin(np.deg2rad(target.lat_bounds[:, 1])) - np.sin(np.deg2rad(target.lat_bounds[:, 0]))
    target_lon = np.deg2rad(target.longitude_widths_deg())
    full = target_sin[:, None] * target_lon[None, :]
    for sample in range(flat.shape[0]):
        # numerator_jt_it = sum_js sum_is value * lat[jt, js] * lon[it, is]
        temp = lat_w @ flat[sample]
        numerator = temp @ lon_w.T
        temp_mask = lat_w @ flat_mask[sample]
        covered = temp_mask @ lon_w.T
        if policy == "intensive":
            sample_out = np.full(numerator.shape, np.nan, dtype=np.float64)
            np.divide(numerator, covered, out=sample_out, where=covered > 0.0)
            sample_cov = np.zeros(numerator.shape, dtype=np.float64)
            np.divide(covered, full, out=sample_cov, where=full > 0.0)
        else:
            sample_out = np.zeros(numerator.shape, dtype=np.float64)
            np.divide(numerator, full, out=sample_out, where=full > 0.0)
            sample_cov = np.zeros(numerator.shape, dtype=np.float64)
            np.divide(covered, full, out=sample_cov, where=full > 0.0)
            # A target cell with no overlap stays zero under this explicit policy.
            # Coverage still says the cell was empty, so a reader can see the zero.
        out[sample] = sample_out
        coverage[sample] = sample_cov
    return RemapResult(
        out.reshape(leading + (target.nlat, target.nlon)),
        coverage.reshape(leading + (target.nlat, target.nlon)),
        policy,
    )


def covered_integral(field: np.ndarray, grid: LatLonGrid) -> float:
    """Area integral of finite cells. Non-finite cells are omitted, not zeroed."""

    values = np.asarray(field, dtype=np.float64)
    if values.shape != grid.shape:
        raise GridError("integral expects a single field on the grid")
    areas = grid.cell_areas_m2()
    valid = np.isfinite(values)
    if not np.any(valid):
        raise GridError("integral has no finite cells")
    return float(np.sum(values[valid] * areas[valid]))


def same_grid_identity_error(field: np.ndarray, grid: LatLonGrid) -> float:
    """Max absolute difference after a round trip onto the same grid.

    Finite cells must come back unchanged. Missing cells must stay missing.
    """

    remapped = remap_field(field, grid, grid, policy="intensive")
    values = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(finite & ~np.isfinite(remapped.values)):
        raise GridError("identity remap dropped a finite cell")
    if np.any(~finite & np.isfinite(remapped.values)):
        raise GridError("identity remap filled a missing cell")
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(remapped.values[finite] - values[finite])))
