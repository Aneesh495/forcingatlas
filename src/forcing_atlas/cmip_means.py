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


def years_from_time(stamps) -> np.ndarray:
    """Calendar years from a decoded CMIP time axis.

    numpy datetime64 has no `.year` attribute. cftime objects do. A failed
    decode must not be relabeled as year 0.
    """

    values = np.asarray(stamps)
    if values.size == 0:
        return np.array([], dtype=np.int64)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[Y]").astype(np.int64) + 1970
    if hasattr(values[0], "year"):
        return np.array([int(stamp.year) for stamp in values], dtype=np.int64)
    raise ValueError("time axis did not decode to calendar years")


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
        years = years_from_time(time.values)
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


def control_versus_ssp245(directory: Path) -> list[dict[str, object]]:
    """Compare each model's piControl spread with its ssp245 change.

    The piControl year labels stay on the model branch. They are not shifted
    onto 1850-2100. The statistic is the standard deviation of the global
    annual mean after removing that segment's own mean.
    """

    rows = []
    controls = {path.name.split("_")[2]: path for path in directory.glob("tas_Ayr_*_piControl_*.nc")}
    forced = {path.name.split("_")[2]: path for path in directory.glob("tas_Ayr_*_ssp245_*.nc")}
    for model in sorted(set(controls) & set(forced)):
        control = global_anomaly_series(controls[model], baseline_years=5)
        scenario = global_anomaly_series(forced[model], baseline_years=5)
        rows.append(
            {
                "model": model,
                "piControl_years": [int(control["years"][0]), int(control["years"][-1])],
                "piControl_std_k": float(np.std(control["anomaly_k"])),
                "ssp245_years": [int(scenario["years"][0]), int(scenario["years"][-1])],
                "ssp245_end_anomaly_k": float(scenario["anomaly_k"][-1]),
                "calendar": scenario["calendar"],
            }
        )
    return rows
