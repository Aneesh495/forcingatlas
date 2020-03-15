"""Turn a declared set of inputs into one forcing series.

The adapter refuses combinations that would count the same contribution
twice, and it refuses a GWP emissions series as if it were a carbon mass.
A natural-only series is solar plus volcanic and nothing else. It is not a
hist-nat simulation. Missing components are not filled with zero unless the
caller passes an explicit zero array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.carbon import excess_gtc_to_ppm, integrate_emissions, zero_state
from forcing_atlas.constants import CO2_ERF_DOUBLING_AR6_W_M2
from forcing_atlas.conversions import co2_erf_from_concentration, co2_mass_to_carbon_mass, reject_gwp_as_carbon_mass
from forcing_atlas.errors import DoubleCountError, ScenarioError, UnitError
from forcing_atlas.units import get_unit
from forcing_atlas.variables import get_contract


@dataclass(slots=True)
class ForcingBuild:
    years: np.ndarray
    total_w_m2: np.ndarray
    parts: dict[str, np.ndarray]
    assumptions: tuple[str, ...]
    kind: str


def _as_years(years: np.ndarray) -> np.ndarray:
    time = np.asarray(years)
    if time.ndim != 1 or time.size < 2:
        raise ScenarioError("a forcing series needs at least two years")
    if np.any(~np.isfinite(time)) or np.any(np.diff(time.astype(np.float64)) <= 0.0):
        raise ScenarioError("forcing years must be finite and strictly increasing")
    return time.astype(np.int64)


def _series(values: np.ndarray | None, years: np.ndarray, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != years.shape:
        raise ScenarioError(f"{name} length {array.shape} does not match years {years.shape}")
    if np.any(~np.isfinite(array)):
        raise ScenarioError(f"{name} contains a non-finite value. Missing forcing is not replaced with zero.")
    return array


def _require_unit(unit_name: str, variable: str) -> None:
    unit = get_unit(unit_name)
    get_contract(variable).require(unit, "forcing adapter")


def build_forcing(
    years: np.ndarray,
    *,
    erf_total: np.ndarray | None = None,
    erf_co2: np.ndarray | None = None,
    erf_non_co2: np.ndarray | None = None,
    erf_aerosol: np.ndarray | None = None,
    erf_solar: np.ndarray | None = None,
    erf_volcanic: np.ndarray | None = None,
    erf_other: np.ndarray | None = None,
    co2_ppm: np.ndarray | None = None,
    co2_emissions_gtc_per_year: np.ndarray | None = None,
    co2_emissions_gtco2_per_year: np.ndarray | None = None,
    gwp_emissions_gtco2e_per_year: np.ndarray | None = None,
    co2_reference_ppm: float = 278.3,
    doubling_erf_w_m2: float = CO2_ERF_DOUBLING_AR6_W_M2,
    natural_only: bool = False,
) -> ForcingBuild:
    """Assemble one total forcing from exactly one consistent description."""

    time = _as_years(years)
    if gwp_emissions_gtco2e_per_year is not None:
        reject_gwp_as_carbon_mass(get_unit("GtCO2e/yr"))
    _require_unit("W/m^2", "erf_total")
    assumptions: list[str] = []
    if natural_only and any(
        item is not None
        for item in (erf_total, erf_co2, erf_non_co2, erf_aerosol, erf_other, co2_ppm, co2_emissions_gtc_per_year, co2_emissions_gtco2_per_year)
    ):
        raise ScenarioError("natural_only accepts only solar and volcanic series")
    if natural_only:
        solar = _series(erf_solar, time, "erf_solar")
        volcanic = _series(erf_volcanic, time, "erf_volcanic")
        if solar is None or volcanic is None:
            raise ScenarioError("natural_only needs both solar and volcanic series")
        total = solar + volcanic
        assumptions.append(
            "Natural-only means the supplied solar and volcanic effective radiative forcing. "
            "It is not a hist-nat simulation and it is not an observation."
        )
        return ForcingBuild(time, total, {"erf_solar": solar, "erf_volcanic": volcanic}, tuple(assumptions), "natural_only")

    component_flags = [erf_co2, erf_non_co2, erf_aerosol, erf_solar, erf_volcanic, erf_other, co2_ppm, co2_emissions_gtc_per_year, co2_emissions_gtco2_per_year]
    has_component = any(item is not None for item in component_flags)
    if erf_total is not None and has_component:
        raise DoubleCountError("erf_total cannot be combined with components or with a CO2 concentration or emission")
    if erf_total is not None:
        total = _series(erf_total, time, "erf_total")
        assert total is not None
        assumptions.append("Total effective radiative forcing was supplied. No component was added.")
        return ForcingBuild(time, total, {"erf_total": total}, tuple(assumptions), "prescribed_total")

    if co2_emissions_gtc_per_year is not None and co2_emissions_gtco2_per_year is not None:
        raise DoubleCountError("pass CO2 emissions either as GtC/yr or as GtCO2/yr, not both")
    if co2_emissions_gtc_per_year is not None and co2_ppm is not None:
        raise DoubleCountError("CO2 emissions and CO2 concentration both determine the carbon burden")
    if co2_emissions_gtco2_per_year is not None and co2_ppm is not None:
        raise DoubleCountError("CO2 emissions and CO2 concentration both determine the carbon burden")
    if erf_co2 is not None and (co2_ppm is not None or co2_emissions_gtc_per_year is not None or co2_emissions_gtco2_per_year is not None):
        raise DoubleCountError("CO2 forcing cannot be combined with a CO2 concentration or emission")
    if erf_co2 is not None and erf_non_co2 is not None and any(item is not None for item in (erf_aerosol, erf_solar, erf_volcanic, erf_other)):
        raise DoubleCountError("erf_non_co2 already includes the non-CO2 components")

    emissions = None
    if co2_emissions_gtc_per_year is not None:
        _require_unit("GtC/yr", "co2_emissions")
        emissions = _series(co2_emissions_gtc_per_year, time, "co2_emissions_gtc_per_year")
        assumptions.append("CO2 mass emissions in GtC/yr drive the four-pool carbon module.")
    elif co2_emissions_gtco2_per_year is not None:
        _require_unit("GtCO2/yr", "co2_emissions")
        raw = _series(co2_emissions_gtco2_per_year, time, "co2_emissions_gtco2_per_year")
        assert raw is not None
        emissions = co2_mass_to_carbon_mass(raw)
        assumptions.append("GtCO2/yr was converted to GtC/yr by 12/44 before the carbon module.")

    parts: dict[str, np.ndarray] = {}
    if emissions is not None:
        _time, excess = integrate_emissions(time.astype(np.float64), emissions, state=zero_state())
        ppm = excess_gtc_to_ppm(excess, co2_reference_ppm)
        parts["co2_ppm"] = ppm
        parts["erf_co2"] = co2_erf_from_concentration(ppm, co2_reference_ppm, doubling_erf_w_m2=doubling_erf_w_m2)
        assumptions.append(
            f"CO2 forcing uses F2x = {doubling_erf_w_m2} W m-2 and reference {co2_reference_ppm} ppm. "
            "The carbon pools start at zero excess, so the reference concentration is the initial burden."
        )
    elif co2_ppm is not None:
        _require_unit("ppm_CO2", "co2_concentration")
        ppm = _series(co2_ppm, time, "co2_ppm")
        assert ppm is not None
        if np.any(ppm <= 0.0):
            raise UnitError("CO2 concentration must be positive")
        parts["co2_ppm"] = ppm
        parts["erf_co2"] = co2_erf_from_concentration(ppm, co2_reference_ppm, doubling_erf_w_m2=doubling_erf_w_m2)
        assumptions.append(
            f"CO2 concentration was converted with F2x = {doubling_erf_w_m2} W m-2. Emissions were not also applied."
        )
    elif erf_co2 is not None:
        parts["erf_co2"] = _series(erf_co2, time, "erf_co2")

    if erf_non_co2 is not None:
        parts["erf_non_co2"] = _series(erf_non_co2, time, "erf_non_co2")
    else:
        for key, values in (
            ("erf_aerosol", erf_aerosol),
            ("erf_solar", erf_solar),
            ("erf_volcanic", erf_volcanic),
            ("erf_other", erf_other),
        ):
            series = _series(values, time, key)
            if series is not None:
                parts[key] = series
    if not parts:
        raise ScenarioError("no forcing input was provided")
    # co2_ppm is carried for provenance. It is not added as if it were a flux.
    additive = [series for key, series in parts.items() if key != "co2_ppm" and series is not None]
    if not additive:
        raise ScenarioError("the build produced a concentration and no forcing")
    total = np.sum(np.vstack(additive), axis=0)
    assumptions.append("The total is the sum of the forcing parts. The sign of each part is the sign that was supplied.")
    return ForcingBuild(time, total, parts, tuple(assumptions), "components")
