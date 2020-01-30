"""Variable contracts.

Emissions, concentrations, radiative forcing, temperature, heat content,
rainfall, and sea level do not substitute for one another. The registry is
the check the forcing adapter and the readers share.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from forcing_atlas.errors import ContractError, UnitError
from forcing_atlas.units import Unit, get_unit


class VariableKind(Enum):
    EMISSION = "emission"
    CONCENTRATION = "concentration"
    RADIATIVE_FORCING = "radiative_forcing"
    TEMPERATURE = "temperature"
    HEAT_CONTENT = "heat_content"
    ENERGY_UPTAKE_RATE = "energy_uptake_rate"
    RAINFALL = "rainfall"
    SEA_LEVEL = "sea_level"
    COVERAGE = "coverage"
    HEAT_CAPACITY = "heat_capacity"
    FEEDBACK = "feedback"


class FieldExtent(Enum):
    INTENSIVE = "intensive"
    EXTENSIVE_DENSITY = "extensive_density"
    EXTENSIVE_TOTAL = "extensive_total"
    GLOBAL_SCALAR = "global_scalar"


@dataclass(frozen=True, slots=True)
class VariableContract:
    name: str
    kind: VariableKind
    extent: FieldExtent
    allowed_units: tuple[str, ...]
    description: str
    sign_convention: str

    def accepts(self, unit: Unit) -> bool:
        return unit.name in self.allowed_units

    def require(self, unit: Unit, purpose: str) -> None:
        if not self.accepts(unit):
            raise UnitError(
                f"{purpose}: variable {self.name} ({self.kind.value}) rejects unit {unit.name}. "
                f"Allowed: {', '.join(self.allowed_units)}"
            )


def _c(
    name: str,
    kind: VariableKind,
    extent: FieldExtent,
    units: tuple[str, ...],
    description: str,
    sign_convention: str,
) -> VariableContract:
    for unit_name in units:
        get_unit(unit_name)
    return VariableContract(name, kind, extent, units, description, sign_convention)


CONTRACTS: dict[str, VariableContract] = {}


def _add(contract: VariableContract) -> None:
    if contract.name in CONTRACTS:
        raise ContractError(f"duplicate variable {contract.name}")
    CONTRACTS[contract.name] = contract


_add(_c(
    "tas_absolute",
    VariableKind.TEMPERATURE,
    FieldExtent.INTENSIVE,
    ("K", "degC"),
    "Near-surface air temperature, absolute.",
    "Positive is warmer on the absolute scale.",
))
_add(_c(
    "tas_anomaly",
    VariableKind.TEMPERATURE,
    FieldExtent.INTENSIVE,
    ("K_anomaly", "degC_anomaly"),
    "Near-surface air temperature anomaly against a named baseline.",
    "Positive is warmer than the baseline.",
))
_add(_c(
    "sst_anomaly",
    VariableKind.TEMPERATURE,
    FieldExtent.INTENSIVE,
    ("K_anomaly", "degC_anomaly"),
    "Sea-surface temperature anomaly. Not air temperature.",
    "Positive is warmer than the baseline.",
))
_add(_c(
    "co2_emissions",
    VariableKind.EMISSION,
    FieldExtent.GLOBAL_SCALAR,
    ("GtC/yr", "GtCO2/yr"),
    "CO2 mass flux into the atmosphere. Not a GWP equivalent and not a forcing.",
    "Positive emits carbon to the atmosphere.",
))
_add(_c(
    "co2e_gwp100_emissions",
    VariableKind.EMISSION,
    FieldExtent.GLOBAL_SCALAR,
    ("GtCO2e/yr",),
    "GWP100 carbon-dioxide-equivalent emission rate. Forbidden as carbon-cycle input.",
    "Positive is a positive GWP100 emission. This is an accounting sign, not a mass.",
))
_add(_c(
    "so2_emissions",
    VariableKind.EMISSION,
    FieldExtent.GLOBAL_SCALAR,
    ("MtSO2/yr",),
    "Sulfur dioxide mass flux. Not an aerosol radiative forcing.",
    "Positive emits SO2.",
))
_add(_c(
    "co2_concentration",
    VariableKind.CONCENTRATION,
    FieldExtent.GLOBAL_SCALAR,
    ("ppm_CO2",),
    "Well-mixed CO2 dry-air mole fraction.",
    "Positive concentration. Zero is not a valid atmospheric state in this contract.",
))
_add(_c(
    "ch4_concentration",
    VariableKind.CONCENTRATION,
    FieldExtent.GLOBAL_SCALAR,
    ("ppb_CH4",),
    "Well-mixed CH4 dry-air mole fraction.",
    "Positive concentration.",
))
_add(_c(
    "n2o_concentration",
    VariableKind.CONCENTRATION,
    FieldExtent.GLOBAL_SCALAR,
    ("ppb_N2O",),
    "Well-mixed N2O dry-air mole fraction.",
    "Positive concentration.",
))
_add(_c(
    "erf_total",
    VariableKind.RADIATIVE_FORCING,
    FieldExtent.GLOBAL_SCALAR,
    ("W/m^2",),
    "Effective radiative forcing, sum of declared components since a stated reference.",
    "Positive forcing warms the climate system in the energy-balance sign convention.",
))
_add(_c(
    "erf_co2",
    VariableKind.RADIATIVE_FORCING,
    FieldExtent.GLOBAL_SCALAR,
    ("W/m^2",),
    "CO2 component of effective radiative forcing.",
    "Positive CO2 forcing is a warming contribution. Do not also pass a CO2 concentration into the same adapter call.",
))
_add(_c(
    "erf_nonco2",
    VariableKind.RADIATIVE_FORCING,
    FieldExtent.GLOBAL_SCALAR,
    ("W/m^2",),
    "All non-CO2 effective radiative forcing, anthropogenic and natural as declared by the caller.",
    "Positive warms.",
))
_add(_c(
    "erf_aerosol",
    VariableKind.RADIATIVE_FORCING,
    FieldExtent.GLOBAL_SCALAR,
    ("W/m^2",),
    "Aerosol effective radiative forcing, radiation and cloud interactions as declared.",
    "Negative values are a cooling contribution. Do not flip the sign on ingest.",
))
_add(_c(
    "erf_volcanic",
    VariableKind.RADIATIVE_FORCING,
    FieldExtent.GLOBAL_SCALAR,
    ("W/m^2",),
    "Volcanic effective radiative forcing.",
    "Stratospheric aerosol eruptions are negative in the AR6 and IGCC tables.",
))
_add(_c(
    "erf_solar",
    VariableKind.RADIATIVE_FORCING,
    FieldExtent.GLOBAL_SCALAR,
    ("W/m^2",),
    "Solar effective radiative forcing relative to the stated reference, not total solar irradiance.",
    "Positive is a warming contribution relative to the reference.",
))
_add(_c(
    "erf_other",
    VariableKind.RADIATIVE_FORCING,
    FieldExtent.GLOBAL_SCALAR,
    ("W/m^2",),
    "Residual forcing components kept separate from CO2, aerosol, volcanic, and solar.",
    "Positive warms.",
))
_add(_c(
    "heat_content",
    VariableKind.HEAT_CONTENT,
    FieldExtent.EXTENSIVE_TOTAL,
    ("ZJ", "J", "J/m^2"),
    "Heat content. The IGCC file named earth_energy_imbalance.csv is this kind when the unit is ZJ.",
    "Positive is a gain of heat relative to the file's own reference.",
))
_add(_c(
    "energy_uptake_rate",
    VariableKind.ENERGY_UPTAKE_RATE,
    FieldExtent.EXTENSIVE_DENSITY,
    ("W/m^2",),
    "Planetary energy uptake rate. Derived from heat content only by an explicit time difference.",
    "Positive is heat entering the climate system.",
))
_add(_c(
    "precipitation",
    VariableKind.RAINFALL,
    FieldExtent.INTENSIVE,
    ("mm/day", "kg/m^2/s"),
    "Precipitation flux. Depth rate and mass flux convert only with a declared density.",
    "Positive is precipitation, not a rainfall anomaly, unless the series was rebased and renamed.",
))
_add(_c(
    "sea_level",
    VariableKind.SEA_LEVEL,
    FieldExtent.INTENSIVE,
    ("m", "mm"),
    "Sea level. Absolute and anomaly products must say which in the dataset note.",
    "Positive is a higher sea surface.",
))
_add(_c(
    "coverage_fraction",
    VariableKind.COVERAGE,
    FieldExtent.INTENSIVE,
    ("fraction",),
    "Fraction of a domain with finite source values. Not a temperature.",
    "One is complete coverage. Zero is no coverage. Missing coverage is NaN, not zero, when the mask itself is unknown.",
))
_add(_c(
    "heat_capacity_coeff",
    VariableKind.HEAT_CAPACITY,
    FieldExtent.GLOBAL_SCALAR,
    ("W yr/m^2/K", "J/m^2/K"),
    "Layer heat capacity in the two-reservoir model.",
    "Positive capacity. A nonpositive capacity is rejected by the solver.",
))
_add(_c(
    "feedback_parameter",
    VariableKind.FEEDBACK,
    FieldExtent.GLOBAL_SCALAR,
    ("W/m^2/K",),
    "Net radiative feedback lambda, or the exchange coefficient gamma.",
    "Positive lambda is a stabilizing negative temperature feedback in the equation C dT/dt = F - lambda T.",
))


def get_contract(name: str) -> VariableContract:
    try:
        return CONTRACTS[name]
    except KeyError as exc:
        raise ContractError(f"unknown variable {name!r}") from exc


# Pairs that must never be coerced into each other by a reader or an adapter.
_FORBIDDEN_SUBSTITUTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("co2_emissions", "erf_co2"),
        ("co2_emissions", "erf_total"),
        ("co2e_gwp100_emissions", "co2_emissions"),
        ("co2e_gwp100_emissions", "co2_concentration"),
        ("co2e_gwp100_emissions", "erf_co2"),
        ("co2_concentration", "erf_co2"),
        ("co2_concentration", "tas_anomaly"),
        ("so2_emissions", "erf_aerosol"),
        ("heat_content", "energy_uptake_rate"),
        ("heat_content", "erf_total"),
        ("erf_volcanic", "tas_anomaly"),
        ("precipitation", "tas_anomaly"),
        ("sea_level", "heat_content"),
        ("coverage_fraction", "tas_anomaly"),
        ("sst_anomaly", "tas_anomaly"),
    }
)


def assert_not_substituted(source_variable: str, target_variable: str) -> None:
    """Raise if a reader or adapter tries to relabel one variable as another."""

    pair = (source_variable, target_variable)
    if pair in _FORBIDDEN_SUBSTITUTIONS or (target_variable, source_variable) in _FORBIDDEN_SUBSTITUTIONS:
        left = get_contract(source_variable)
        right = get_contract(target_variable)
        raise ContractError(
            f"refusing to substitute {source_variable} ({left.kind.value}) "
            f"for {target_variable} ({right.kind.value})"
        )
    if source_variable == target_variable:
        return
    left = get_contract(source_variable)
    right = get_contract(target_variable)
    if left.kind is not right.kind:
        raise ContractError(
            f"refusing to coerce {source_variable} ({left.kind.value}) "
            f"into {target_variable} ({right.kind.value})"
        )


def temperature_anomaly_baseline_required(variable: str, baseline_id: str | None) -> None:
    contract = get_contract(variable)
    if contract.name in {"tas_anomaly", "sst_anomaly"} and not baseline_id:
        raise ContractError(f"{variable} requires a baseline_id")
    if contract.kind is not VariableKind.TEMPERATURE and baseline_id:
        if contract.name != "sea_level":
            raise ContractError(f"{variable} does not take a temperature baseline")
