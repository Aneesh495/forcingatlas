"""Global means of the fetched ssp245 fields.

Each series is an anomaly against the first twenty years of that same file.
That baseline is not 1961-1990, and the values are not observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from forcing_atlas.grids import LatLonGrid, LongitudeConvention, area_weighted_mean, flip_latitude_if_needed


def _grid_from_dataset(dataset: xr.Dataset) -> tuple[np.ndarray, LatLonGrid]:
    lat_name = "lat" if "lat" in dataset.coords else "latitude"
    lon_name = "lon" if "lon" in dataset.coords else "longitude"
    lat = np.asarray(dataset[lat_name].values, dtype=np.float64)
    lon = np.asarray(dataset[lon_name].values, dtype=np.float64)
    field = np.asarray(dataset["tas"].values, dtype=np.float64)
    lat_bounds = None
    if f"{lat_name}_bnds" in dataset:
        lat_bounds = np.asarray(dataset[f"{lat_name}_bnds"].values, dtype=np.float64)
    lon_bounds = None
    if f"{lon_name}_bnds" in dataset:
        lon_bounds = np.asarray(dataset[f"{lon_name}_bnds"].values, dtype=np.float64)
    lat, field, lat_bounds = flip_latitude_if_needed(lat, field, lat_bounds)
    if lat_bounds is None:
        edges = np.empty(lat.size + 1)
        edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
        edges[0] = max(-90.0, lat[0] - (edges[1] - lat[0]))
        edges[-1] = min(90.0, lat[-1] + (lat[-1] - edges[-2]))
        lat_bounds = np.column_stack([edges[:-1], edges[1:]])
    if lon_bounds is None:
        step = float(np.median(np.diff(np.sort(lon))))
        lon_bounds = np.column_stack([lon - step / 2.0, lon + step / 2.0])
    convention = LongitudeConvention.NEG180_180 if np.any(lon < 0) else LongitudeConvention.ZERO_360
    grid = LatLonGrid(lat, lon, lat_bounds, lon_bounds, convention, name=dataset.attrs.get("source_id", ""))
    return field, grid


def global_anomaly_series(path: Path, *, baseline_years: int = 20) -> dict[str, object]:
    dataset = xr.open_dataset(path)
    try:
        field, grid = _grid_from_dataset(dataset)
        weights = grid.cell_areas_m2()
        series = area_weighted_mean(field, weights)
        if series.size <= baseline_years:
            raise ValueError(f"{path.name} is shorter than the baseline window")
        baseline = float(np.nanmean(series[:baseline_years]))
        anomaly = series - baseline
        time = dataset["time"]
        calendar = str(time.attrs.get("calendar") or time.encoding.get("calendar") or "missing")
        # Annual files are one sample per model year. The first sample is 2015
        # when the filename says 2015-2100. The time encoding is recorded, and
        # the year index is the file order plus 2015 only for that filename pattern.
        if "2015-2100" in path.name:
            years = np.arange(2015, 2015 + anomaly.size)
        else:
            years = np.arange(anomaly.size)
        return {
            "file": path.name,
            "calendar": calendar,
            "baseline": f"first {baseline_years} years of this file",
            "years": years,
            "anomaly_k": anomaly,
            "semantics": "source_simulation",
        }
    finally:
        dataset.close()


def load_ssp245_means(directory: Path, *, baseline_years: int = 5) -> list[dict[str, object]]:
    paths = sorted(directory.glob("tas_Ayr_*_ssp245_*.nc"))
    return [global_anomaly_series(path, baseline_years=baseline_years) for path in paths]
