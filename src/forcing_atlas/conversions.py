"""Explicit physical conversions that are not unit identities.

Each function records the assumption in its name. Callers that want a
different factor pass it in. None of these functions accept a warming
potential or a radiative forcing as if it were a concentration.
"""

from __future__ import annotations

import math

import numpy as np

from forcing_atlas.constants import (
    CO2_ERF_DOUBLING_AR6_W_M2,
    EARTH_RADIUS_M,
    GTC_PER_PPM_CO2,
    JULIAN_YEAR_SECONDS,
    KG_C_PER_KG_CO2,
    KG_CO2_PER_KG_C,
    MYHRE_1998_ALPHA_W_M2,
)
from forcing_atlas.errors import UnitError
from forcing_atlas.units import Accounting, Species, Unit, get_unit


def earth_surface_area_m2(radius_m: float = EARTH_RADIUS_M) -> float:
    if radius_m <= 0.0:
        raise UnitError("Earth radius must be positive")
    return 4.0 * math.pi * radius_m * radius_m


def zj_per_year_to_w_m2(
    zj_per_year: np.ndarray,
    *,
    radius_m: float = EARTH_RADIUS_M,
    year_seconds: float = JULIAN_YEAR_SECONDS,
) -> np.ndarray:
    """Convert an energy-uptake rate in ZJ per Julian year to W m-2.

    The input is a rate of change of heat content, not the heat content itself.
    """

    area = earth_surface_area_m2(radius_m)
    joule_per_year = np.asarray(zj_per_year, dtype=np.float64) * 1.0e21
    watts = joule_per_year / year_seconds
    return watts / area


def w_m2_to_zj_per_year(
    flux: np.ndarray,
    *,
    radius_m: float = EARTH_RADIUS_M,
    year_seconds: float = JULIAN_YEAR_SECONDS,
) -> np.ndarray:
    area = earth_surface_area_m2(radius_m)
    joule = np.asarray(flux, dtype=np.float64) * area * year_seconds
    return joule / 1.0e21


def co2_mass_to_carbon_mass(gtco2: np.ndarray) -> np.ndarray:
    """Convert GtCO2 to GtC by stoichiometry, 12/44."""

    return np.asarray(gtco2, dtype=np.float64) * KG_C_PER_KG_CO2


def carbon_mass_to_co2_mass(gtc: np.ndarray) -> np.ndarray:
    return np.asarray(gtc, dtype=np.float64) * KG_CO2_PER_KG_C


def ppm_co2_to_gtc(ppm: np.ndarray, *, gtc_per_ppm: float = GTC_PER_PPM_CO2) -> np.ndarray:
    """Convert a CO2 mole fraction in ppm to atmospheric carbon mass.

    `gtc_per_ppm` is the declared airborne-carbon factor, default 2.124 GtC ppm-1.
    """

    if gtc_per_ppm <= 0.0:
        raise UnitError("gtc_per_ppm must be positive")
    return np.asarray(ppm, dtype=np.float64) * gtc_per_ppm


def gtc_to_ppm_co2(gtc: np.ndarray, *, gtc_per_ppm: float = GTC_PER_PPM_CO2) -> np.ndarray:
    if gtc_per_ppm <= 0.0:
        raise UnitError("gtc_per_ppm must be positive")
    return np.asarray(gtc, dtype=np.float64) / gtc_per_ppm


def co2_erf_from_concentration(
    concentration_ppm: np.ndarray,
    reference_ppm: float,
    *,
    doubling_erf_w_m2: float = CO2_ERF_DOUBLING_AR6_W_M2,
) -> np.ndarray:
    """Logarithmic CO2 effective radiative forcing.

    F = (F_2x / ln 2) * ln(C / C_ref). The default F_2x is the AR6 central
    assessed ERF of 3.93 W m-2. This is not the Myhre 5.35 coefficient unless
    the caller sets doubling_erf_w_m2 to 5.35 * ln(2).
    """

    if reference_ppm <= 0.0:
        raise UnitError("CO2 reference concentration must be positive")
    if doubling_erf_w_m2 <= 0.0:
        raise UnitError("CO2 doubling ERF must be positive")
    conc = np.asarray(concentration_ppm, dtype=np.float64)
    if np.any(conc[np.isfinite(conc)] <= 0.0):
        raise UnitError("CO2 concentration must be positive where it is finite")
    alpha = doubling_erf_w_m2 / math.log(2.0)
    with np.errstate(invalid="ignore"):
        return alpha * np.log(conc / reference_ppm)


def myhre_1998_co2_forcing(concentration_ppm: np.ndarray, reference_ppm: float) -> np.ndarray:
    """F = 5.35 ln(C/C_ref), kept as a named comparator rather than the default."""

    if reference_ppm <= 0.0:
        raise UnitError("CO2 reference concentration must be positive")
    conc = np.asarray(concentration_ppm, dtype=np.float64)
    return MYHRE_1998_ALPHA_W_M2 * np.log(conc / reference_ppm)


def rainfall_mm_day_to_kg_m2_s(mm_per_day: np.ndarray, *, water_density_kg_m3: float = 1000.0) -> np.ndarray:
    """Depth rate to mass flux. The density is an assumption, default pure water."""

    if water_density_kg_m3 <= 0.0:
        raise UnitError("water density must be positive")
    # 1 mm day-1 of water at 1000 kg m-3 is 1 kg m-2 day-1.
    per_day = np.asarray(mm_per_day, dtype=np.float64) * (water_density_kg_m3 / 1000.0)
    return per_day / 86400.0


def reject_gwp_as_carbon_mass(unit: Unit) -> None:
    if unit.accounting is Accounting.GWP_CO2E:
        raise UnitError(
            f"{unit.name} is a GWP100 equivalent and cannot enter the carbon-cycle mass balance"
        )
    if unit.species not in (Species.CARBON, Species.CO2):
        raise UnitError(f"carbon-cycle mass input cannot use species {unit.species.value}")


def anomaly_unit_for_absolute(unit: Unit) -> Unit:
    if unit.name == "K":
        return get_unit("K_anomaly")
    if unit.name == "degC":
        return get_unit("degC_anomaly")
    raise UnitError(f"{unit.name} is not an absolute temperature with a declared anomaly twin")
