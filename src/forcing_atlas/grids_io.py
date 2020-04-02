"""Read a HadCRUT-style NetCDF anomaly grid without filling missing cells.

xarray may apply a fill value on its own. This reader checks `_FillValue` and
`missing_value`, forces those codes to NaN, and refuses a file whose decoded
field has replaced every fill with zero. Latitude is returned south to north.
The time axis is monthly, and the annual series is weighted by Gregorian
month length.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from forcing_atlas.calendars import CalendarKind, days_in_year, month_length
from forcing_atlas.errors import ContractError
from forcing_atlas.grids import LatLonGrid, LongitudeConvention, area_weighted_mean, flip_latitude_if_needed


@dataclass(slots=True)
class GriddedAnomaly:
    grid: LatLonGrid
    years: np.ndarray
    annual: np.ndarray
    annual_coverage: np.ndarray
    monthly_time: np.ndarray
    monthly: np.ndarray
    global_annual: np.ndarray
    attributes: dict[str, str]

    @property
    def ntime(self) -> int:
        return int(self.annual.shape[0])


def _decode_hadcrut(path: Path) -> xr.Dataset:
    if not path.is_file():
        raise ContractError(f"missing grid file {path}")
    # mask_and_scale keeps fill values out of the array. We still check.
    dataset = xr.open_dataset(path, mask_and_scale=True, decode_times=True)
    return dataset


def read_hadcrut_grid(path: Path, *, name: str) -> GriddedAnomaly:
    dataset = _decode_hadcrut(path)
    try:
        if "tas_mean" not in dataset:
            raise ContractError(f"{path.name} has no tas_mean variable")
        variable = dataset["tas_mean"]
        fill = variable.attrs.get("_FillValue", variable.encoding.get("_FillValue"))
        missing = variable.attrs.get("missing_value", variable.encoding.get("missing_value"))
        values = np.asarray(variable.values, dtype=np.float64)
        for sentinel in (fill, missing):
            if sentinel is None:
                continue
            try:
                code = float(sentinel)
            except (TypeError, ValueError):
                continue
            if np.isfinite(code):
                values = np.where(values == code, np.nan, values)
        if np.any(np.isinf(values)):
            raise ContractError(f"{path.name} contains infinite temperatures")
        lat = np.asarray(dataset["latitude"].values, dtype=np.float64)
        lon = np.asarray(dataset["longitude"].values, dtype=np.float64)
        lat_bounds = np.asarray(dataset["latitude_bnds"].values, dtype=np.float64)
        lon_bounds = np.asarray(dataset["longitude_bnds"].values, dtype=np.float64)
        lat, values, lat_bounds = flip_latitude_if_needed(lat, values, lat_bounds)
        if lat_bounds is None:
            raise ContractError("latitude bounds disappeared during the flip")
        # Longitude bounds are stored with the longitude axis, which was not flipped.
        convention = LongitudeConvention.NEG180_180
        if np.any(lon < -180.0) or np.any(lon > 180.0):
            raise ContractError(f"{path.name} longitude is outside [-180, 180]")
        grid = LatLonGrid(lat, lon, lat_bounds, lon_bounds, convention, name=name)
        times = dataset["time"].values
        # numpy datetime64[ns]
        months = times.astype("datetime64[M]")
        years = months.astype("datetime64[Y]").astype(int) + 1970
        month_index = (months.astype(int) % 12) + 1
        unique_years = np.unique(years)
        annual_stack = []
        coverage_stack = []
        year_list = []
        for year in unique_years.tolist():
            mask = years == year
            sample = values[mask]
            sample_months = month_index[mask]
            order = np.argsort(sample_months)
            sample = sample[order]
            sample_months = sample_months[order]
            if sample.shape[0] != sample_months.size:
                raise ContractError("month alignment failed")
            # Weight by Gregorian month length. Missing months stay missing.
            year_days = days_in_year(int(year), CalendarKind.PROLEPTIC_GREGORIAN)
            weighted = np.zeros(sample.shape[1:], dtype=np.float64)
            covered_days = np.zeros(sample.shape[1:], dtype=np.float64)
            seen = set()
            for local, month in enumerate(sample_months.tolist()):
                if month in seen:
                    raise ContractError(f"duplicate month {year}-{month}")
                seen.add(month)
                length = float(month_length(int(year), int(month), CalendarKind.PROLEPTIC_GREGORIAN))
                finite = np.isfinite(sample[local])
                weighted += np.where(finite, sample[local] * length, 0.0)
                covered_days += np.where(finite, length, 0.0)
            coverage = covered_days / year_days
            annual = np.full(sample.shape[1:], np.nan, dtype=np.float64)
            complete = covered_days >= year_days - 1e-6
            np.divide(weighted, covered_days, out=annual, where=complete)
            annual_stack.append(annual)
            coverage_stack.append(coverage)
            year_list.append(int(year))
        annual = np.stack(annual_stack, axis=0)
        coverage = np.stack(coverage_stack, axis=0)
        areas = grid.cell_areas_m2()
        global_annual = area_weighted_mean(annual, areas)
        attributes = {key: str(value) for key, value in dataset.attrs.items()}
        return GriddedAnomaly(
            grid=grid,
            years=np.asarray(year_list, dtype=np.int64),
            annual=annual,
            annual_coverage=coverage,
            monthly_time=times.astype("datetime64[ns]"),
            monthly=values,
            global_annual=global_annual,
            attributes=attributes,
        )
    finally:
        dataset.close()


def assert_ogl_licence(attributes: dict[str, str]) -> None:
    licence = attributes.get("licence", "")
    if "Open Government Licence v3.0" not in licence:
        raise ContractError(f"HadCRUT licence attribute does not state OGL v3.0: {licence!r}")
