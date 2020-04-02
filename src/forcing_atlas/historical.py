"""Align the pinned instrumental series with the assessed forcing.

This is the observational target for the energy-balance hindcast. The HadCRUT
summary is the published global mean, which is the average of the hemispheres,
not the area-weighted mean of the non-infilled grid. Those are different
numbers and both are kept. Forcing is the sum of the IGCC component columns,
checked against the aggregate total.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forcing_atlas.catalog import sources_by_id
from forcing_atlas.errors import ContractError, MissingDataError
from forcing_atlas.readers import (
    compare_component_and_aggregate_totals,
    read_energy_inventory,
    read_erf_aggregates,
    read_erf_components,
    read_hadcrut_summary,
    read_pages2k_summary,
    read_berkeley_land_ocean,
)
from forcing_atlas.timebase import align_annual, window_mean


@dataclass(slots=True)
class HistoricalBundle:
    years: np.ndarray
    hadcrut_anomaly_k: np.ndarray
    hadcrut_lower_k: np.ndarray
    hadcrut_upper_k: np.ndarray
    forcing_w_m2: np.ndarray
    forcing_co2_w_m2: np.ndarray
    forcing_aerosol_w_m2: np.ndarray
    forcing_solar_w_m2: np.ndarray
    forcing_volcanic_w_m2: np.ndarray
    spinup_forcing_w_m2: np.ndarray
    spinup_years: np.ndarray
    component_aggregate_max_diff: float
    root: Path


def _read(root: Path, dataset_id: str) -> str:
    source = sources_by_id()[dataset_id]
    path = root / source.file_name
    if not path.is_file():
        raise MissingDataError(f"missing {path}. Run the acquirer before the historical bundle.")
    return path.read_text()


def load_historical_bundle(root: Path) -> HistoricalBundle:
    hadcrut_source = sources_by_id()["hadcrut5-analysis-global-annual"]
    component_source = sources_by_id()["igcc-erf-components-1750-2023"]
    aggregate_source = sources_by_id()["igcc-erf-aggregates-1750-2023"]
    hadcrut = read_hadcrut_summary(_read(root, hadcrut_source.dataset_id), hadcrut_source)
    components = read_erf_components(_read(root, component_source.dataset_id), component_source)
    aggregates = read_erf_aggregates(_read(root, aggregate_source.dataset_id), aggregate_source)
    comparison = compare_component_and_aggregate_totals(components, aggregates, tolerance_w_m2=1.0e-4)
    if comparison["within_tolerance"] != 1.0:
        raise ContractError(
            f"ERF component sum and aggregate total differ by {comparison['max_abs_diff_w_m2']} W m-2"
        )
    years, temperature, forcing = align_annual(
        hadcrut.years, hadcrut.values, components.years, components.values
    )
    _y, _t, co2 = align_annual(hadcrut.years, hadcrut.values, components.years, components.extras["erf_co2"])
    _y, _t, aerosol = align_annual(hadcrut.years, hadcrut.values, components.years, components.extras["erf_aerosol"])
    _y, _t, solar = align_annual(hadcrut.years, hadcrut.values, components.years, components.extras["erf_solar"])
    _y, _t, volcanic = align_annual(hadcrut.years, hadcrut.values, components.years, components.extras["erf_volcanic"])
    spinup_mask = components.years < int(years[0])
    lower_map = {int(year): float(value) for year, value in zip(hadcrut.years, hadcrut.extras["lower_2_5"], strict=True)}
    upper_map = {int(year): float(value) for year, value in zip(hadcrut.years, hadcrut.extras["upper_97_5"], strict=True)}
    return HistoricalBundle(
        years=years,
        hadcrut_anomaly_k=temperature,
        hadcrut_lower_k=np.array([lower_map[int(year)] for year in years]),
        hadcrut_upper_k=np.array([upper_map[int(year)] for year in years]),
        forcing_w_m2=forcing,
        forcing_co2_w_m2=co2,
        forcing_aerosol_w_m2=aerosol,
        forcing_solar_w_m2=solar,
        forcing_volcanic_w_m2=volcanic,
        spinup_forcing_w_m2=np.asarray(components.values[spinup_mask], dtype=np.float64),
        spinup_years=np.asarray(components.years[spinup_mask], dtype=np.int64),
        component_aggregate_max_diff=comparison["max_abs_diff_w_m2"],
        root=root,
    )


def rebase_to_window(years: np.ndarray, values: np.ndarray, start: int, end: int) -> np.ndarray:
    offset = window_mean(years, values, start, end)
    return np.asarray(values, dtype=np.float64) - offset


def load_context_series(root: Path) -> dict[str, object]:
    """Berkeley and PAGES 2k records for the cross-product and proxy studies."""

    berkeley_source = sources_by_id()["berkeley-land-ocean-complete"]
    pages_source = sources_by_id()["pages2k-2019-ensemble-summary"]
    inventory_source = sources_by_id()["igcc-energy-inventory"]
    berkeley = read_berkeley_land_ocean(_read(root, berkeley_source.dataset_id), berkeley_source)
    pages = read_pages2k_summary(_read(root, pages_source.dataset_id), pages_source)
    inventory = read_energy_inventory(_read(root, inventory_source.dataset_id), inventory_source)
    return {"berkeley": berkeley, "pages": pages, "inventory": inventory}
