"""Typed failures for contracts, numerics, acquisition, and verification.

Callers branch on the exception class. Messages name the rejected quantity
and the rule. They do not substitute a default value.
"""

from __future__ import annotations


class ForcingAtlasError(Exception):
    """Base class for errors raised by this project."""


class ContractError(ForcingAtlasError):
    """A dataset, variable, or pathway violates a declared contract."""


class UnitError(ContractError):
    """Dimensions, species, or temperature kinds cannot be combined."""


class CalendarError(ContractError):
    """A date or time axis is illegal on the declared calendar."""


class GridError(ContractError):
    """Cell bounds, longitude convention, or regridding weights are illegal."""


class MissingDataError(ContractError):
    """A required value is missing and the operation refuses to invent one."""


class DoubleCountError(ContractError):
    """Two inputs represent the same physical contribution."""


class BaselineError(ContractError):
    """Anomaly baselines differ, or a baseline window has no support."""


class SemanticsError(ContractError):
    """Observation, proxy, simulation, and emulation were mixed without a rule."""


class LicenseError(ContractError):
    """A source license does not allow the requested operation."""


class BudgetError(ForcingAtlasError):
    """The local byte cap or a stage budget would be exceeded."""


class AcquisitionError(ForcingAtlasError):
    """A download, checksum, or resume failed."""


class CancelledError(ForcingAtlasError):
    """A cancellation token fired during a bounded operation."""


class CalibrationError(ForcingAtlasError):
    """A fit did not meet its declared numerical conditions."""


class AbstentionError(ForcingAtlasError):
    """An emulator input is outside the supported training range."""


class ScenarioError(ContractError):
    """A pathway is internally inconsistent or asks for an unsupported layer."""


class VerificationError(ForcingAtlasError):
    """Independent verification rejected an evidence bundle."""


class ReplayError(VerificationError):
    """A bounded replay did not reproduce a saved quantity."""


class ProfileError(ForcingAtlasError):
    """A full-acceptance profile was requested without its required inputs."""
