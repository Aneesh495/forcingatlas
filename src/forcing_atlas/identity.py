"""Dataset identity and the four-way semantic split.

Observations, observational analyses, proxy reconstructions, source
simulations, and original emulations are different claims. Helpers below
refuse the mixes that would let a proxy or a model run stand in for a
thermometer reading without an explicit study rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from forcing_atlas.errors import SemanticsError


class Semantics(Enum):
    OBSERVATION = "observation"
    OBSERVATIONAL_ANALYSIS = "observational_analysis"
    PROXY_RECONSTRUCTION = "proxy_reconstruction"
    SOURCE_SIMULATION = "source_simulation"
    EMULATION = "emulation"
    FORCING_ASSESSMENT = "forcing_assessment"
    EMISSION_INVENTORY = "emission_inventory"
    ENERGY_INVENTORY = "energy_inventory"


class MemberRole(Enum):
    """How an ensemble axis may be used in a sample size.

    Uncertainty members of one observational analysis are not independent
    climate models. Initial-condition members of one model are not independent
    model families. Only `model_family` counts as an independent-by-design
    model unit, and only after the caller has already grouped by source id.
    """

    SINGLE = "single"
    UNCERTAINTY_ENSEMBLE = "uncertainty_ensemble"
    INITIAL_CONDITION = "initial_condition"
    MODEL_FAMILY = "model_family"


class TimeLabel(Enum):
    """How a numeric year stamp relates to the interval it summarizes."""

    YEAR_START = "year_start"
    YEAR_MIDPOINT = "year_midpoint"
    BOUNDED_INTERVAL = "bounded_interval"
    INSTANT = "instant"
    MODEL_BRANCH_YEAR = "model_branch_year"


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    baseline_id: str
    start_year: int
    end_year: int
    description: str

    def __post_init__(self) -> None:
        if self.end_year < self.start_year:
            raise SemanticsError(
                f"baseline {self.baseline_id} ends before it starts "
                f"({self.start_year} to {self.end_year})"
            )


# Windows used by the pinned instrumental products. These are definitions,
# not results.
BASELINES: dict[str, BaselineSpec] = {
    "1961-1990": BaselineSpec(
        "1961-1990",
        1961,
        1990,
        "HadCRUT5 anomaly baseline and the PAGES2k stated reference.",
    ),
    "1951-1980": BaselineSpec(
        "1951-1980",
        1951,
        1980,
        "Berkeley Earth land-ocean anomaly baseline.",
    ),
    "1850-1900": BaselineSpec(
        "1850-1900",
        1850,
        1900,
        "Early-industrial window used as a warming reference, not as a preindustrial control run.",
    ),
    "preindustrial_1750": BaselineSpec(
        "preindustrial_1750",
        1750,
        1750,
        "Forcing assessments in the IGCC ERF tables are changes since 1750.",
    ),
}


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset_id: str
    semantics: Semantics
    source_id: str
    title: str
    version: str
    variable: str
    unit: str
    license_id: str
    url: str
    citation: str
    experiment: str | None = None
    member_id: str | None = None
    member_role: MemberRole = MemberRole.SINGLE
    grid_label: str | None = None
    calendar: str | None = None
    baseline_id: str | None = None
    time_label: TimeLabel = TimeLabel.YEAR_START
    checksum_sha256: str | None = None
    revision: str | None = None
    file_name: str | None = None
    license_note: str = ""
    unit_note: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.source_id:
            raise SemanticsError("dataset_id and source_id are required")
        if self.semantics is Semantics.SOURCE_SIMULATION and not self.experiment:
            raise SemanticsError(
                f"{self.dataset_id} is a source simulation and needs an experiment id"
            )
        if self.semantics is Semantics.PROXY_RECONSTRUCTION and self.member_role is MemberRole.MODEL_FAMILY:
            raise SemanticsError("a proxy ensemble is not a climate-model family")


def assert_observation_target(identity: DatasetIdentity, purpose: str) -> None:
    """Targets of an observational comparison must be instrumental products."""

    allowed = {Semantics.OBSERVATION, Semantics.OBSERVATIONAL_ANALYSIS}
    if identity.semantics not in allowed:
        raise SemanticsError(
            f"{purpose} cannot use {identity.dataset_id} "
            f"({identity.semantics.value}) as an observational target"
        )


def assert_not_proxy_as_thermometer(identity: DatasetIdentity) -> None:
    if identity.semantics is Semantics.PROXY_RECONSTRUCTION:
        raise SemanticsError(
            f"{identity.dataset_id} is a proxy reconstruction and is not a thermometer series"
        )


def assert_simulation_not_used_as_observation(identity: DatasetIdentity, purpose: str) -> None:
    if identity.semantics in {Semantics.SOURCE_SIMULATION, Semantics.EMULATION}:
        raise SemanticsError(
            f"{purpose} cannot treat {identity.dataset_id} ({identity.semantics.value}) as observed"
        )


def assert_comparable_pair(left: DatasetIdentity, right: DatasetIdentity, purpose: str) -> None:
    """Refuse pairs that answer different questions unless the caller uses a study gate.

    This guard is for direct overlays that share a color scale and a sentence.
    Studies that intentionally place a proxy beside an instrumental series call
    `allow_context_pair` instead and must keep the badges.
    """

    blocked = {
        (Semantics.PROXY_RECONSTRUCTION, Semantics.OBSERVATION),
        (Semantics.PROXY_RECONSTRUCTION, Semantics.OBSERVATIONAL_ANALYSIS),
        (Semantics.OBSERVATION, Semantics.PROXY_RECONSTRUCTION),
        (Semantics.OBSERVATIONAL_ANALYSIS, Semantics.PROXY_RECONSTRUCTION),
        (Semantics.SOURCE_SIMULATION, Semantics.OBSERVATION),
        (Semantics.SOURCE_SIMULATION, Semantics.OBSERVATIONAL_ANALYSIS),
        (Semantics.OBSERVATION, Semantics.SOURCE_SIMULATION),
        (Semantics.OBSERVATIONAL_ANALYSIS, Semantics.SOURCE_SIMULATION),
        (Semantics.EMULATION, Semantics.OBSERVATION),
        (Semantics.EMULATION, Semantics.OBSERVATIONAL_ANALYSIS),
        (Semantics.OBSERVATION, Semantics.EMULATION),
        (Semantics.OBSERVATIONAL_ANALYSIS, Semantics.EMULATION),
    }
    pair = (left.semantics, right.semantics)
    if pair in blocked:
        raise SemanticsError(
            f"{purpose} cannot directly compare {left.dataset_id} ({left.semantics.value}) "
            f"with {right.dataset_id} ({right.semantics.value})"
        )


def allow_context_pair(left: DatasetIdentity, right: DatasetIdentity) -> str:
    """Return the badge sentence required when a study shows unlike semantics together."""

    return (
        f"{left.dataset_id} is {left.semantics.value}; "
        f"{right.dataset_id} is {right.semantics.value}. "
        "They are shown as context and are not interchangeable measurements."
    )


def independent_model_key(identity: DatasetIdentity) -> str:
    """Key used to count independent-by-design model families.

    Initial-condition members collapse to the source id. Observational
    uncertainty members are not models and raise.
    """

    if identity.semantics is not Semantics.SOURCE_SIMULATION:
        raise SemanticsError(
            f"{identity.dataset_id} is not a source simulation and has no model-family key"
        )
    if identity.member_role is MemberRole.UNCERTAINTY_ENSEMBLE:
        raise SemanticsError("observational uncertainty members are not model families")
    return identity.source_id


def shared_observation_dependence(left: DatasetIdentity, right: DatasetIdentity) -> str | None:
    """Describe known shared inputs. Independence is not assumed from the product name.

    Berkeley Earth land-ocean uses a reinterpolated HadSST ocean field. HadCRUT5
    blends CRUTEM land air temperature with HadSST4. The ocean component is a
    shared ancestor. Land processing differs.
    """

    sources = {left.source_id, right.source_id}
    if sources == {"HadCRUT5", "BerkeleyEarth"}:
        return (
            "HadCRUT5 and Berkeley Earth land-ocean both depend on HadSST for the ocean. "
            "Their land analyses differ. Agreement over the ocean is not two independent observations."
        )
    return None
