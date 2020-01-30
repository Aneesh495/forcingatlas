"""Semantic and variable-contract guards."""

from __future__ import annotations

import pytest

from forcing_atlas.errors import ContractError, SemanticsError
from forcing_atlas.identity import (
    DatasetIdentity,
    MemberRole,
    Semantics,
    TimeLabel,
    allow_context_pair,
    assert_comparable_pair,
    assert_not_proxy_as_thermometer,
    assert_observation_target,
    independent_model_key,
    shared_observation_dependence,
)
from forcing_atlas.variables import assert_not_substituted, get_contract, temperature_anomaly_baseline_required


def _identity(**overrides: object) -> DatasetIdentity:
    payload: dict[str, object] = {
        "dataset_id": "example",
        "semantics": Semantics.OBSERVATIONAL_ANALYSIS,
        "source_id": "HadCRUT5",
        "title": "Example",
        "version": "0",
        "variable": "tas_anomaly",
        "unit": "degC_anomaly",
        "license_id": "see-source",
        "url": "https://example.invalid",
        "citation": "Example",
    }
    payload.update(overrides)
    return DatasetIdentity(**payload)  # type: ignore[arg-type]


def test_proxy_is_not_a_thermometer() -> None:
    proxy = _identity(
        dataset_id="pages2k",
        semantics=Semantics.PROXY_RECONSTRUCTION,
        source_id="PAGES2k",
        variable="tas_anomaly",
    )
    with pytest.raises(SemanticsError):
        assert_not_proxy_as_thermometer(proxy)
    with pytest.raises(SemanticsError):
        assert_observation_target(proxy, "hindcast target")


def test_direct_compare_blocks_proxy_and_simulation() -> None:
    obs = _identity(dataset_id="hadcrut")
    proxy = _identity(
        dataset_id="pages2k",
        semantics=Semantics.PROXY_RECONSTRUCTION,
        source_id="PAGES2k",
    )
    sim = _identity(
        dataset_id="canesm5-ssp245",
        semantics=Semantics.SOURCE_SIMULATION,
        source_id="CanESM5",
        experiment="ssp245",
        member_id="r1i1p1f1",
        member_role=MemberRole.INITIAL_CONDITION,
    )
    with pytest.raises(SemanticsError):
        assert_comparable_pair(obs, proxy, "map overlay")
    with pytest.raises(SemanticsError):
        assert_comparable_pair(obs, sim, "map overlay")
    sentence = allow_context_pair(obs, proxy)
    assert "proxy_reconstruction" in sentence
    assert "not interchangeable" in sentence


def test_model_family_key_collapses_members() -> None:
    member_a = _identity(
        dataset_id="a",
        semantics=Semantics.SOURCE_SIMULATION,
        source_id="MIROC6",
        experiment="ssp245",
        member_id="r1i1p1f1",
        member_role=MemberRole.INITIAL_CONDITION,
    )
    member_b = _identity(
        dataset_id="b",
        semantics=Semantics.SOURCE_SIMULATION,
        source_id="MIROC6",
        experiment="ssp245",
        member_id="r2i1p1f1",
        member_role=MemberRole.INITIAL_CONDITION,
    )
    assert independent_model_key(member_a) == independent_model_key(member_b) == "MIROC6"


def test_shared_hadsst_dependence_is_stated() -> None:
    had = _identity(source_id="HadCRUT5")
    bek = _identity(dataset_id="berkeley", source_id="BerkeleyEarth", baseline_id="1951-1980")
    note = shared_observation_dependence(had, bek)
    assert note is not None
    assert "HadSST" in note


def test_forbidden_substitutions() -> None:
    with pytest.raises(ContractError):
        assert_not_substituted("co2_emissions", "erf_co2")
    with pytest.raises(ContractError):
        assert_not_substituted("co2e_gwp100_emissions", "co2_emissions")
    with pytest.raises(ContractError):
        assert_not_substituted("heat_content", "energy_uptake_rate")
    with pytest.raises(ContractError):
        assert_not_substituted("so2_emissions", "erf_aerosol")
    # Same kind may be discussed together only when not on the forbidden list.
    assert get_contract("erf_co2").kind == get_contract("erf_aerosol").kind
    assert_not_substituted("erf_co2", "erf_aerosol")


def test_anomaly_requires_baseline() -> None:
    with pytest.raises(ContractError):
        temperature_anomaly_baseline_required("tas_anomaly", None)
    temperature_anomaly_baseline_required("tas_anomaly", "1961-1990")
    temperature_anomaly_baseline_required("erf_total", None)


def test_time_label_enum_exists() -> None:
    assert TimeLabel.MODEL_BRANCH_YEAR.value == "model_branch_year"
