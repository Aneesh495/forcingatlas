"""Dimensioned units for climate quantities.

Temperature anomalies are not absolute temperatures. A warming potential is
not a carbon mass. Parts per million of CO2 are not watts per square metre.
Conversions that need a physical assumption live in explicit functions in
`forcing_atlas.conversions`, not in the unit table.

SI value = numeric * scale + offset.
The offset is nonzero only for absolute Celsius.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from forcing_atlas.constants import (
    ABSOLUTE_ZERO_OFFSET_C,
    DAY360_YEAR_SECONDS,
    DAY_SECONDS,
    JULIAN_YEAR_SECONDS,
    KG_PER_GT,
    KG_PER_MT,
    NOLEAP_YEAR_SECONDS,
)
from forcing_atlas.errors import UnitError


class Accounting(Enum):
    PHYSICAL = "physical"
    ABSOLUTE_TEMPERATURE = "absolute_temperature"
    TEMPERATURE_ANOMALY = "temperature_anomaly"
    GWP_CO2E = "gwp_co2e"


class Species(Enum):
    NONE = "none"
    CARBON = "C"
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    SO2 = "SO2"
    BC = "BC"
    WATER = "H2O"


@dataclass(frozen=True, slots=True)
class Dimension:
    """Exponent vector on (length, mass, time, temperature, amount)."""

    length: int = 0
    mass: int = 0
    time: int = 0
    temperature: int = 0
    amount: int = 0

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.length + other.length,
            self.mass + other.mass,
            self.time + other.time,
            self.temperature + other.temperature,
            self.amount + other.amount,
        )

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(
            self.length - other.length,
            self.mass - other.mass,
            self.time - other.time,
            self.temperature - other.temperature,
            self.amount - other.amount,
        )

    def __pow__(self, exponent: int) -> Dimension:
        if not isinstance(exponent, int):
            raise UnitError("dimension powers must be integers")
        return Dimension(
            self.length * exponent,
            self.mass * exponent,
            self.time * exponent,
            self.temperature * exponent,
            self.amount * exponent,
        )

    def is_dimensionless(self) -> bool:
        return self == Dimension()

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.length, self.mass, self.time, self.temperature, self.amount)


@dataclass(frozen=True, slots=True)
class Unit:
    name: str
    dimension: Dimension
    scale: float
    offset: float
    accounting: Accounting
    species: Species = Species.NONE
    note: str = ""

    def si_from_numeric(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        return array * self.scale + self.offset

    def numeric_from_si(self, si: np.ndarray) -> np.ndarray:
        array = np.asarray(si, dtype=np.float64)
        return (array - self.offset) / self.scale


def _temperature() -> Dimension:
    return Dimension(temperature=1)


def _mass() -> Dimension:
    return Dimension(mass=1)


def _time() -> Dimension:
    return Dimension(time=1)


def _length() -> Dimension:
    return Dimension(length=1)


def _power() -> Dimension:
    # watt = kg m^2 s^-3
    return Dimension(mass=1, length=2, time=-3)


def _energy() -> Dimension:
    return Dimension(mass=1, length=2, time=-2)


def _flux() -> Dimension:
    # W m^-2 = kg s^-3
    return Dimension(mass=1, time=-3)


def _heat_capacity_coeff() -> Dimension:
    # J m^-2 K^-1 = kg s^-2 K^-1
    return Dimension(mass=1, time=-2, temperature=-1)


def _feedback() -> Dimension:
    # W m^-2 K^-1 = kg s^-3 K^-1
    return Dimension(mass=1, time=-3, temperature=-1)


def _rate(dimension: Dimension) -> Dimension:
    return dimension / _time()


_DIMLESS = Dimension()

UNITS: dict[str, Unit] = {}


def _register(unit: Unit) -> Unit:
    if unit.name in UNITS:
        raise UnitError(f"duplicate unit name {unit.name}")
    if unit.offset != 0.0 and unit.accounting is not Accounting.ABSOLUTE_TEMPERATURE:
        raise UnitError("only absolute temperature may carry an offset")
    if unit.scale == 0.0:
        raise UnitError("unit scale must be nonzero")
    UNITS[unit.name] = unit
    return unit


def get_unit(name: str) -> Unit:
    try:
        return UNITS[name]
    except KeyError as exc:
        raise UnitError(f"unknown unit {name!r}") from exc


def _u(
    name: str,
    dimension: Dimension,
    scale: float,
    accounting: Accounting = Accounting.PHYSICAL,
    species: Species = Species.NONE,
    offset: float = 0.0,
    note: str = "",
) -> Unit:
    return _register(
        Unit(
            name=name,
            dimension=dimension,
            scale=scale,
            offset=offset,
            accounting=accounting,
            species=species,
            note=note,
        )
    )


_u("1", _DIMLESS, 1.0, note="dimensionless identity")
_u("fraction", _DIMLESS, 1.0, note="dimensionless fraction on [0, 1] by convention")
_u("ppm", _DIMLESS, 1.0e-6, note="mole fraction, species not yet attached")
_u("ppb", _DIMLESS, 1.0e-9, note="mole fraction, species not yet attached")
_u("ppt", _DIMLESS, 1.0e-12, note="mole fraction, species not yet attached")

_u("K", _temperature(), 1.0, Accounting.ABSOLUTE_TEMPERATURE, note="absolute thermodynamic temperature")
_u(
    "degC",
    _temperature(),
    1.0,
    Accounting.ABSOLUTE_TEMPERATURE,
    offset=ABSOLUTE_ZERO_OFFSET_C,
    note="absolute Celsius",
)
_u("K_anomaly", _temperature(), 1.0, Accounting.TEMPERATURE_ANOMALY, note="temperature anomaly in kelvin")
_u(
    "degC_anomaly",
    _temperature(),
    1.0,
    Accounting.TEMPERATURE_ANOMALY,
    note="temperature anomaly in deg C, identical increment to kelvin",
)

_u("m", _length(), 1.0)
_u("mm", _length(), 1.0e-3)
_u("kg", _mass(), 1.0)
_u("Gt", _mass(), KG_PER_GT, note="gigatonne, species not attached")
_u("Mt", _mass(), KG_PER_MT, note="megatonne, species not attached")
_u("GtC", _mass(), KG_PER_GT, species=Species.CARBON, note="gigatonnes of carbon")
_u("GtCO2", _mass(), KG_PER_GT, species=Species.CO2, note="gigatonnes of carbon dioxide")
_u("MtSO2", _mass(), KG_PER_MT, species=Species.SO2, note="megatonnes of sulfur dioxide")
_u("kg_H2O", _mass(), 1.0, species=Species.WATER)

_u("s", _time(), 1.0)
_u("day", _time(), DAY_SECONDS)
_u("yr", _time(), JULIAN_YEAR_SECONDS, note="Julian year of 365.25 days, EBM clock")
_u("yr_365", _time(), NOLEAP_YEAR_SECONDS, note="noleap year")
_u("yr_360", _time(), DAY360_YEAR_SECONDS, note="360-day year")

_u("W", _power(), 1.0)
_u("J", _energy(), 1.0)
_u("ZJ", _energy(), 1.0e21, note="zettajoule, 1e21 joule")
_u("W/m^2", _flux(), 1.0, note="energy flux density, including effective radiative forcing")
_u("J/m^2", _energy() / (_length() ** 2), 1.0)
_u("W yr/m^2/K", _heat_capacity_coeff(), JULIAN_YEAR_SECONDS, note="EBM heat capacity on the Julian-year clock")
_u("J/m^2/K", _heat_capacity_coeff(), 1.0)
_u("W/m^2/K", _feedback(), 1.0, note="radiative feedback or heat-exchange coefficient")

_u("GtC/yr", _rate(_mass()), KG_PER_GT / JULIAN_YEAR_SECONDS, species=Species.CARBON)
_u("GtCO2/yr", _rate(_mass()), KG_PER_GT / JULIAN_YEAR_SECONDS, species=Species.CO2)
_u("MtSO2/yr", _rate(_mass()), KG_PER_MT / JULIAN_YEAR_SECONDS, species=Species.SO2)
_u(
    "GtCO2e/yr",
    _rate(_mass()),
    KG_PER_GT / JULIAN_YEAR_SECONDS,
    accounting=Accounting.GWP_CO2E,
    species=Species.CO2,
    note="GWP100 carbon-dioxide equivalent. Not a carbon mass flux.",
)
_u("kg/m^2/s", _rate(_mass()) / (_length() ** 2), 1.0, species=Species.WATER, note="precipitation mass flux")
_u(
    "mm/day",
    _length() / _time(),
    1.0e-3 / DAY_SECONDS,
    note="rainfall depth rate. Equivalence to kg m^-2 s^-1 needs a declared water density.",
)
_u("m/yr", _length() / _time(), 1.0 / JULIAN_YEAR_SECONDS, note="sea-level rate on the Julian year")


def mixing_ratio(species: Species, prefix: str) -> Unit:
    """Return a species-tagged mole fraction. prefix is ppm, ppb, or ppt."""

    if species is Species.NONE:
        raise UnitError("a mixing ratio needs a species")
    base = get_unit(prefix)
    name = f"{prefix}_{species.value}"
    existing = UNITS.get(name)
    if existing is not None:
        return existing
    return _u(
        name,
        base.dimension,
        base.scale,
        species=species,
        note=f"{prefix} of {species.value}",
    )


# Common greenhouse-gas mixing ratios used by the IGCC concentration table.
for _species, _prefix in (
    (Species.CO2, "ppm"),
    (Species.CH4, "ppb"),
    (Species.N2O, "ppb"),
):
    mixing_ratio(_species, _prefix)


def same_physical_dimension(left: Unit, right: Unit) -> bool:
    return left.dimension == right.dimension


def convertible(src: Unit, dst: Unit) -> bool:
    if src.accounting != dst.accounting:
        return False
    if src.dimension != dst.dimension:
        return False
    if src.species != dst.species:
        return False
    return True


def convert_numeric(value: np.ndarray, src: Unit, dst: Unit) -> np.ndarray:
    """Convert an array. Missing values stay missing. Nothing is filled with zero."""

    if not convertible(src, dst):
        raise UnitError(
            f"cannot convert {src.name} ({src.accounting.value}, {src.species.value}) "
            f"to {dst.name} ({dst.accounting.value}, {dst.species.value})"
        )
    array = np.asarray(value, dtype=np.float64)
    si = src.si_from_numeric(array)
    out = dst.numeric_from_si(si)
    # Preserve NaN exactly rather than letting an offset create a finite number.
    return np.where(np.isfinite(array), out, np.nan)


def require_accounting(unit: Unit, accounting: Accounting, purpose: str) -> None:
    if unit.accounting is not accounting:
        raise UnitError(
            f"{purpose} requires {accounting.value}, got {unit.name} ({unit.accounting.value})"
        )


def require_unit_name(unit: Unit, *names: str, purpose: str) -> None:
    if unit.name not in names:
        raise UnitError(f"{purpose} requires one of {names}, got {unit.name}")
