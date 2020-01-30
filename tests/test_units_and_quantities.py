"""Unit identities and the refusals around them."""

from __future__ import annotations

import numpy as np
import pytest

from forcing_atlas.constants import ABSOLUTE_ZERO_OFFSET_C, KG_PER_GT
from forcing_atlas.conversions import (
    carbon_mass_to_co2_mass,
    co2_erf_from_concentration,
    co2_mass_to_carbon_mass,
    myhre_1998_co2_forcing,
    ppm_co2_to_gtc,
    rainfall_mm_day_to_kg_m2_s,
    reject_gwp_as_carbon_mass,
    w_m2_to_zj_per_year,
    zj_per_year_to_w_m2,
)
from forcing_atlas.errors import BaselineError, UnitError
from forcing_atlas.quantities import Quantity, anomaly_from_absolute, rebase_anomaly
from forcing_atlas.units import Accounting, convert_numeric, get_unit


def test_celsius_absolute_offset() -> None:
    converted = convert_numeric(np.array([0.0, 100.0]), get_unit("degC"), get_unit("K"))
    assert converted[0] == pytest.approx(ABSOLUTE_ZERO_OFFSET_C)
    assert converted[1] == pytest.approx(100.0 + ABSOLUTE_ZERO_OFFSET_C)


def test_anomaly_is_not_absolute() -> None:
    with pytest.raises(UnitError):
        convert_numeric(np.array([1.0]), get_unit("degC_anomaly"), get_unit("K"))
    with pytest.raises(UnitError):
        convert_numeric(np.array([1.0]), get_unit("K"), get_unit("degC_anomaly"))


def test_anomaly_celsius_and_kelvin_match() -> None:
    converted = convert_numeric(np.array([1.5, np.nan]), get_unit("degC_anomaly"), get_unit("K_anomaly"))
    assert converted[0] == pytest.approx(1.5)
    assert np.isnan(converted[1])


def test_mass_scale() -> None:
    grams = convert_numeric(np.array([1.0]), get_unit("Gt"), get_unit("kg"))
    assert grams[0] == pytest.approx(KG_PER_GT)


def test_species_block_carbon_and_sulfur() -> None:
    with pytest.raises(UnitError):
        convert_numeric(np.array([1.0]), get_unit("GtC"), get_unit("MtSO2"))


def test_gwp_is_not_carbon_mass() -> None:
    with pytest.raises(UnitError):
        Quantity(np.array([36.0]), get_unit("GtCO2e/yr")) + Quantity(np.array([1.0]), get_unit("GtC/yr"))
    with pytest.raises(UnitError):
        reject_gwp_as_carbon_mass(get_unit("GtCO2e/yr"))


def test_stoichiometry_and_ppm_factor() -> None:
    assert co2_mass_to_carbon_mass(np.array([44.0]))[0] == pytest.approx(12.0)
    assert carbon_mass_to_co2_mass(np.array([12.0]))[0] == pytest.approx(44.0)
    assert ppm_co2_to_gtc(np.array([1.0]))[0] == pytest.approx(2.124)


def test_doubling_forcing_matches_declared_f2x() -> None:
    forcing = co2_erf_from_concentration(np.array([560.0]), 280.0, doubling_erf_w_m2=3.93)
    assert forcing[0] == pytest.approx(3.93)
    myhre = myhre_1998_co2_forcing(np.array([560.0]), 280.0)
    assert myhre[0] == pytest.approx(5.35 * np.log(2.0))
    assert myhre[0] != pytest.approx(forcing[0])


def test_energy_rate_round_trip() -> None:
    flux = np.array([1.0, np.nan])
    back = zj_per_year_to_w_m2(w_m2_to_zj_per_year(flux))
    assert back[0] == pytest.approx(1.0)
    assert np.isnan(back[1])


def test_rainfall_density_is_explicit() -> None:
    # 1 mm/day of water at 1000 kg m-3 is 1 kg m-2 day-1.
    flux = rainfall_mm_day_to_kg_m2_s(np.array([1.0]))
    assert flux[0] == pytest.approx(1.0 / 86400.0)


def test_absolute_difference_is_anomaly() -> None:
    warm = Quantity(np.array([20.0]), get_unit("degC"))
    cool = Quantity(np.array([10.0]), get_unit("degC"))
    delta = warm - cool
    assert delta.unit.accounting is Accounting.TEMPERATURE_ANOMALY
    assert delta.values[0] == pytest.approx(10.0)


def test_cannot_add_two_absolute_temperatures() -> None:
    with pytest.raises(UnitError):
        Quantity(np.array([20.0]), get_unit("degC")) + Quantity(np.array([1.0]), get_unit("K"))


def test_anomaly_plus_absolute() -> None:
    absolute = Quantity(np.array([15.0]), get_unit("degC"))
    anomaly = Quantity(np.array([1.25]), get_unit("degC_anomaly"), baseline_id="1961-1990")
    combined = absolute + anomaly
    assert combined.unit.name == "degC"
    assert combined.values[0] == pytest.approx(16.25)


def test_baseline_mismatch_is_refused() -> None:
    left = Quantity(np.array([0.2]), get_unit("K_anomaly"), baseline_id="1961-1990")
    right = Quantity(np.array([0.1]), get_unit("degC_anomaly"), baseline_id="1951-1980")
    with pytest.raises(BaselineError):
        _ = left + right


def test_rebase_changes_id_and_values() -> None:
    series = Quantity(np.array([0.0, 1.0, 2.0]), get_unit("degC_anomaly"), baseline_id="1961-1990")
    shifted = rebase_anomaly(series, window_mean=1.0, new_baseline_id="1850-1900")
    assert shifted.baseline_id == "1850-1900"
    assert shifted.values.tolist() == pytest.approx([-1.0, 0.0, 1.0])


def test_anomaly_from_absolute_tags_baseline() -> None:
    absolute = Quantity(np.array([15.0, 16.0]), get_unit("degC"))
    climatology = Quantity(np.array([15.0, 15.0]), get_unit("degC"))
    anomaly = anomaly_from_absolute(absolute, climatology, "1951-1980")
    assert anomaly.baseline_id == "1951-1980"
    assert anomaly.values.tolist() == pytest.approx([0.0, 1.0])
    assert anomaly.unit.name == "degC_anomaly"


def test_nan_survives_conversion() -> None:
    values = convert_numeric(np.array([np.nan, 1.0]), get_unit("ZJ"), get_unit("J"))
    assert np.isnan(values[0])
    assert values[1] == pytest.approx(1.0e21)
