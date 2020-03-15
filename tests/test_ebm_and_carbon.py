"""Closed forms for the two-box model and the carbon pools.

The Taylor exponential and the one-box formula in this file are not imported
from the solver. The carbon airborne fraction at infinite time is the
permanent coefficient, checked by integrating a pulse rather than by reading
the coefficient back out of the same expression.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from forcing_atlas.carbon import (
    AIRBORNE_PERMANENT,
    apply_pulse,
    coefficient_sum,
    impulse_response,
    integrate_emissions,
    step_constant_emission,
    zero_state,
)
from forcing_atlas.ebm import (
    InitialAnomaly,
    TwoBoxParameters,
    energy_storage_w_yr_m2,
    equilibrium,
    equilibrium_climate_sensitivity,
    forward_euler_step_limit,
    integrate,
    matrix_exponential,
    net_uptake_w_m2,
    one_box_surface,
    step_response,
    system_matrix,
    transient_climate_response,
)
from forcing_atlas.errors import ContractError, DoubleCountError, ScenarioError
from forcing_atlas.forcing_adapter import build_forcing


def _expm_series(matrix: np.ndarray, years: float, squarings: int = 8, terms: int = 24) -> np.ndarray:
    scaled = matrix * (years / (2**squarings))
    total = np.eye(2)
    term = np.eye(2)
    for k in range(1, terms + 1):
        term = term @ scaled / k
        total = total + term
    for _ in range(squarings):
        total = total @ total
    return total


def _stable() -> TwoBoxParameters:
    return TwoBoxParameters(
        lambda_w_m2_k=1.2,
        gamma_w_m2_k=0.7,
        c_surface_w_yr_m2_k=8.0,
        c_deep_w_yr_m2_k=100.0,
    )


def test_eigen_exponential_matches_the_series() -> None:
    matrix = system_matrix(_stable())
    for years in (0.0, 0.5, 4.0, 40.0):
        got = matrix_exponential(matrix, years)
        expect = _expm_series(matrix, years)
        assert got == pytest.approx(expect, abs=1e-8)


def test_one_box_limit_matches_the_scalar_solution() -> None:
    parameters = TwoBoxParameters(1.5, 0.0, 8.0, 100.0)
    years = np.linspace(0.0, 50.0, 11)
    forcing = 3.0
    response = step_response(parameters, forcing, years)
    scalar = (forcing / 1.5) * (1.0 - np.exp(-1.5 * years / 8.0))
    assert response[:, 0] == pytest.approx(scalar, abs=1e-8)
    assert response[:, 1] == pytest.approx(0.0, abs=1e-8)
    assert one_box_surface(parameters, forcing, years) == pytest.approx(scalar)


def test_pure_integrator() -> None:
    parameters = TwoBoxParameters(0.0, 0.0, 10.0, 50.0)
    years = np.array([0.0, 2.0, 5.0])
    response = step_response(parameters, 4.0, years)
    assert response[:, 0] == pytest.approx(4.0 * years / 10.0)
    assert response[:, 1] == pytest.approx(0.0)


def _independent_step(parameters: TwoBoxParameters, forcing: float, years: np.ndarray) -> np.ndarray:
    """Trajectory from the contract matrix and a Taylor exponential, not the solver's eigen path."""

    surface = parameters.c_surface_w_yr_m2_k
    deep = parameters.c_deep_w_yr_m2_k
    lam = parameters.lambda_w_m2_k
    gamma = parameters.gamma_w_m2_k
    matrix = np.array(
        [[-(lam + gamma) / surface, gamma / surface], [gamma / deep, -gamma / deep]],
        dtype=np.float64,
    )
    target = np.array([forcing / lam, forcing / lam])
    out = np.empty((years.size, 2))
    for index, instant in enumerate(years.tolist()):
        out[index] = target + _expm_series(matrix, float(instant)) @ (np.zeros(2) - target)
    return out


def test_equilibrium_and_energy_identity() -> None:
    parameters = _stable()
    forcing = 3.93
    final = equilibrium(parameters, forcing)
    assert final[0] == pytest.approx(final[1])
    assert final[0] == pytest.approx(forcing / parameters.lambda_w_m2_k)
    years = np.linspace(0.0, 80.0, 17)
    response = step_response(parameters, forcing, years)
    independent = _independent_step(parameters, forcing, years)
    assert response == pytest.approx(independent, abs=1e-6)
    # One-box storage equals capacity times the scalar solution, which is also
    # the integral of F exp(-lambda t / C).
    one = TwoBoxParameters(1.5, 0.0, 8.0, 100.0)
    one_years = np.array([0.0, 3.0, 12.0, 40.0])
    one_response = step_response(one, 3.0, one_years)
    scalar = (3.0 / 1.5) * (1.0 - np.exp(-1.5 * one_years / 8.0))
    assert energy_storage_w_yr_m2(one, one_response) == pytest.approx(8.0 * scalar, abs=1e-8)
    assert net_uptake_w_m2(one, one_response[:, 0], np.full(one_years.shape, 3.0))[0] == pytest.approx(3.0)


def test_forward_euler_rejects_an_unstable_step_and_backward_euler_does_not() -> None:
    parameters = _stable()
    limit = forward_euler_step_limit(parameters)
    forcing = np.full(5, 2.0)
    with pytest.raises(ContractError):
        integrate(parameters, forcing, limit, method="forward_euler")
    _times, states = integrate(parameters, np.full(30, 2.0), dt_years=limit * 0.99, method="backward_euler")
    assert np.all(np.isfinite(states))
    # A long explicit-stable run and the eigen stepper stay close on a smooth step.
    fine = 0.25
    steps = int(40 / fine)
    eigen_t, eigen_s = integrate(parameters, np.full(steps, 1.0), fine, method="eigen")
    back_t, back_s = integrate(parameters, np.full(steps, 1.0), fine, method="backward_euler")
    assert eigen_t[-1] == pytest.approx(back_t[-1])
    assert back_s[-1, 0] == pytest.approx(eigen_s[-1, 0], abs=0.02)


def test_negative_capacity_and_initial_note() -> None:
    with pytest.raises(ContractError):
        TwoBoxParameters(1.0, 0.5, -1.0, 10.0).validate()
    state = InitialAnomaly(0.2, -0.1, note="spun up from 1750")
    assert "1750" in state.note
    response = step_response(_stable(), 0.0, np.array([0.0, 10.0]), initial=state)
    assert response[0, 0] == pytest.approx(0.2)


def test_ecs_is_f2x_over_lambda() -> None:
    parameters = _stable()
    assert equilibrium_climate_sensitivity(parameters, 3.93) == pytest.approx(3.93 / 1.2)
    # TCR of a strong-feedback, high-capacity box is below ECS. This is a sign check, not a tuned result.
    tcr = transient_climate_response(parameters)
    assert 0.0 < tcr < equilibrium_climate_sensitivity(parameters)


def test_carbon_coefficients_and_pulse_limit() -> None:
    assert coefficient_sum() == pytest.approx(1.0)
    state = apply_pulse(zero_state(), 1.0)
    assert state.excess_gtc == pytest.approx(1.0)
    # A long decay leaves the permanent fraction. The test integrates the ODE,
    # it does not evaluate the closed impulse formula that the package also exposes.
    current = state
    elapsed = 0.0
    while elapsed < 20000.0:
        current = step_constant_emission(current, 0.0, 50.0)
        elapsed += 50.0
    assert current.excess_gtc == pytest.approx(AIRBORNE_PERMANENT, abs=1e-4)
    assert impulse_response(np.array([0.0]))[0] == pytest.approx(1.0)
    assert impulse_response(np.array([1.0e9]))[0] == pytest.approx(AIRBORNE_PERMANENT, abs=1e-6)


def test_emission_rate_exact_step() -> None:
    # One year of 2 GtC/yr adds 2 * a0 to the permanent pool and the decay pools
    # have not fully lost their share.
    updated = step_constant_emission(zero_state(), 2.0, 1.0)
    assert updated.pools_gtc[0] == pytest.approx(2.0 * AIRBORNE_PERMANENT)
    years, excess = integrate_emissions(np.array([0.0, 1.0, 2.0]), np.array([2.0, 0.0, 0.0]))
    assert years[-1] == 2.0
    assert excess[1] == pytest.approx(updated.excess_gtc)
    assert excess[2] < excess[1]


def test_adapter_rejects_double_counts_and_gwp() -> None:
    years = np.array([2000, 2001])
    with pytest.raises(DoubleCountError):
        build_forcing(years, erf_total=np.array([1.0, 1.0]), erf_co2=np.array([0.5, 0.5]))
    with pytest.raises(DoubleCountError):
        build_forcing(years, co2_ppm=np.array([400.0, 401.0]), erf_co2=np.array([1.0, 1.0]))
    with pytest.raises(DoubleCountError):
        build_forcing(years, co2_emissions_gtc_per_year=np.array([1.0, 1.0]), co2_ppm=np.array([400.0, 401.0]))
    with pytest.raises(Exception):
        build_forcing(years, gwp_emissions_gtco2e_per_year=np.array([36.0, 36.0]), erf_solar=np.zeros(2))


def test_doubling_concentration_is_the_declared_forcing() -> None:
    years = np.array([0, 1])
    built = build_forcing(years, co2_ppm=np.array([560.0, 560.0]), co2_reference_ppm=280.0, doubling_erf_w_m2=3.93)
    assert built.total_w_m2 == pytest.approx(3.93)


def test_emissions_warm_the_box_and_aerosol_cools_it() -> None:
    years = np.arange(0, 51)
    emissions = np.full(years.shape, 10.0)
    co2_only = build_forcing(years, co2_emissions_gtc_per_year=emissions, co2_reference_ppm=280.0)
    aerosol = np.linspace(0.0, -1.0, years.size)
    both = build_forcing(
        years,
        co2_emissions_gtc_per_year=emissions,
        erf_aerosol=aerosol,
        co2_reference_ppm=280.0,
    )
    assert np.all(co2_only.parts["co2_ppm"][1:] > 280.0)
    parameters = _stable()
    _t1, warm = integrate(parameters, co2_only.total_w_m2[:-1], 1.0, method="eigen")
    _t2, cool = integrate(parameters, both.total_w_m2[:-1], 1.0, method="eigen")
    assert warm[-1, 0] > cool[-1, 0] > 0.0


def test_natural_only_does_not_invent_a_greenhouse_term() -> None:
    years = np.array([1991, 1992, 1993])
    solar = np.array([0.05, 0.05, 0.04])
    volcanic = np.array([-0.1, -2.0, -0.4])
    built = build_forcing(years, erf_solar=solar, erf_volcanic=volcanic, natural_only=True)
    assert built.kind == "natural_only"
    assert built.total_w_m2[1] == pytest.approx(-1.95)
    assert "hist-nat" in built.assumptions[0]
    with pytest.raises(ScenarioError):
        build_forcing(years, erf_solar=solar, erf_volcanic=volcanic, erf_co2=np.ones(3), natural_only=True)


def test_missing_forcing_is_not_zero() -> None:
    years = np.array([2000, 2001])
    with pytest.raises(ScenarioError):
        build_forcing(years, erf_total=np.array([1.0, np.nan]))
    with pytest.raises(ScenarioError):
        build_forcing(years)
