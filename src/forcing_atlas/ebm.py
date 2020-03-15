"""Two-reservoir energy-balance model.

The equation and the closed forms are fixed in `docs/NUMERICAL_CONTRACTS.md`.
This file is the solver. It is a linear anomaly model with one surface box
and one deep box. It is not FaIR, MAGICC, or a CMIP emulator. Comparable
published quantities are equilibrium climate sensitivity, the transient
climate response at year 70 of a 1 percent per year CO2 ramp, and the heat
uptake F - lambda Ts. The carbon cycle and aerosol chemistry are outside
this module.

Time is in Julian years of 365.25 days. Heat capacities are in
W yr m-2 K-1 on that clock. Forcing is in W m-2. Temperature is an anomaly
in kelvin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.constants import CO2_ERF_DOUBLING_AR6_W_M2
from forcing_atlas.errors import CalibrationError, ContractError


@dataclass(frozen=True, slots=True)
class TwoBoxParameters:
    lambda_w_m2_k: float
    gamma_w_m2_k: float
    c_surface_w_yr_m2_k: float
    c_deep_w_yr_m2_k: float

    def validate(self) -> None:
        for name, value in (
            ("c_surface", self.c_surface_w_yr_m2_k),
            ("c_deep", self.c_deep_w_yr_m2_k),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ContractError(f"{name} must be finite and positive, got {value}")
        for name, value in (("lambda", self.lambda_w_m2_k), ("gamma", self.gamma_w_m2_k)):
            if not np.isfinite(value) or value < 0.0:
                raise ContractError(f"{name} must be finite and nonnegative, got {value}")
        if self.lambda_w_m2_k == 0.0 and self.gamma_w_m2_k != 0.0:
            raise ContractError("lambda = 0 with a nonzero exchange has no finite equilibrium and is rejected")


@dataclass(frozen=True, slots=True)
class InitialAnomaly:
    surface_k: float = 0.0
    deep_k: float = 0.0
    note: str = "zero anomaly at the start of the forcing window"

    def as_array(self) -> np.ndarray:
        if not np.isfinite(self.surface_k) or not np.isfinite(self.deep_k):
            raise ContractError("initial anomalies must be finite")
        return np.array([self.surface_k, self.deep_k], dtype=np.float64)


def system_matrix(parameters: TwoBoxParameters) -> np.ndarray:
    parameters.validate()
    surface = parameters.c_surface_w_yr_m2_k
    deep = parameters.c_deep_w_yr_m2_k
    lam = parameters.lambda_w_m2_k
    gamma = parameters.gamma_w_m2_k
    return np.array(
        [
            [-(lam + gamma) / surface, gamma / surface],
            [gamma / deep, -gamma / deep],
        ],
        dtype=np.float64,
    )


def forcing_column(parameters: TwoBoxParameters) -> np.ndarray:
    parameters.validate()
    return np.array([1.0 / parameters.c_surface_w_yr_m2_k, 0.0], dtype=np.float64)


def eigenvalues(parameters: TwoBoxParameters) -> np.ndarray:
    """Eigenvalues sorted from slowest decay (least negative) to fastest."""

    values = np.linalg.eigvals(system_matrix(parameters))
    if np.any(np.abs(values.imag) > 1.0e-8):
        raise ContractError(f"two-box eigenvalues are not real: {values}")
    real = np.real(values)
    if np.any(real > 1.0e-10):
        raise ContractError(f"two-box eigenvalue has a positive real part: {real}")
    order = np.argsort(-real)  # least negative first
    return real[order]


def stiffness_ratio(parameters: TwoBoxParameters) -> float:
    """|s_fast| / |s_slow| when both are nonzero. A single finite mode returns 1."""

    values = eigenvalues(parameters)
    magnitudes = np.abs(values[np.abs(values) > 1.0e-12])
    if magnitudes.size == 0:
        return 1.0
    if magnitudes.size == 1:
        return 1.0
    return float(np.max(magnitudes) / np.min(magnitudes))


def forward_euler_step_limit(parameters: TwoBoxParameters) -> float:
    """Largest stable forward-Euler step, in years.

    The limit is 2 / |s| for the eigenvalue of largest magnitude. A zero
    eigenvalue, as in the decoupled deep box, does not set the limit. The
    pure integrator has no decaying mode and returns infinity because the
    update is exact for constant forcing at any step.
    """

    values = eigenvalues(parameters)
    decaying = np.abs(values[values < -1.0e-12])
    if decaying.size == 0:
        return float("inf")
    return float(2.0 / np.max(decaying))


def equilibrium(parameters: TwoBoxParameters, forcing_w_m2: float) -> np.ndarray:
    parameters.validate()
    if parameters.lambda_w_m2_k == 0.0:
        raise ContractError("the pure integrator has no finite equilibrium")
    if not np.isfinite(forcing_w_m2):
        raise ContractError("forcing must be finite")
    level = forcing_w_m2 / parameters.lambda_w_m2_k
    return np.array([level, level], dtype=np.float64)


def matrix_exponential(matrix: np.ndarray, years: float) -> np.ndarray:
    """exp(A t) from the eigen-decomposition. t = 0 returns the identity."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (2, 2):
        raise ContractError("matrix exponential expects the 2x2 two-box matrix")
    if years == 0.0:
        return np.eye(2, dtype=np.float64)
    if years < 0.0 or not np.isfinite(years):
        raise ContractError("matrix exponential time must be nonnegative and finite")
    values, vectors = np.linalg.eig(array)
    if np.any(np.abs(values.imag) > 1.0e-8):
        raise ContractError("refusing a complex eigen-expansion for this thermal system")
    try:
        inverse = np.linalg.inv(vectors)
    except np.linalg.LinAlgError as exc:
        raise ContractError("two-box eigenbasis is singular") from exc
    diagonal = np.diag(np.exp(np.real(values) * years))
    transformed = vectors @ diagonal @ inverse
    if np.max(np.abs(transformed.imag)) > 1.0e-8:
        raise ContractError("eigen-expansion produced a non-real exponential")
    return np.real(transformed)


def step_response(parameters: TwoBoxParameters, forcing_w_m2: float, years: np.ndarray, initial: InitialAnomaly | None = None) -> np.ndarray:
    """Exact response to a forcing that switches on at year 0 and stays constant.

    The returned array has shape (n, 2) for (surface, deep). `years` are times
    after the step. Negative times are rejected.
    """

    parameters.validate()
    time = np.asarray(years, dtype=np.float64)
    if time.ndim != 1 or np.any(time < 0.0) or np.any(~np.isfinite(time)):
        raise ContractError("step response years must be a nonnegative finite vector")
    start = InitialAnomaly() if initial is None else initial
    state0 = start.as_array()
    matrix = system_matrix(parameters)
    column = forcing_column(parameters)
    if parameters.lambda_w_m2_k == 0.0:
        # Ts = Ts0 + F t / Cs, Td stays at Td0 because gamma is also 0.
        out = np.repeat(state0[None, :], time.size, axis=0)
        out[:, 0] = state0[0] + forcing_w_m2 * time / parameters.c_surface_w_yr_m2_k
        return out
    target = equilibrium(parameters, forcing_w_m2)
    # Particular solution x_eq, homogeneous from (state0 - x_eq).
    # x(t) = x_eq + exp(At) (x0 - x_eq)
    out = np.empty((time.size, 2), dtype=np.float64)
    delta0 = state0 - target
    for index, instant in enumerate(time.tolist()):
        out[index] = target + matrix_exponential(matrix, float(instant)) @ delta0
    # The deep box is not forced directly. A constant forcing from a zero state
    # reaches (F/lambda, F/lambda). The column is checked so a refactor that
    # drops it fails this identity: A x_eq + b F = 0.
    residual = matrix @ target + column * forcing_w_m2
    if np.max(np.abs(residual)) > 1.0e-8:
        raise ContractError(f"equilibrium does not satisfy A x + b F = 0, residual {residual}")
    return out


def one_box_surface(parameters: TwoBoxParameters, forcing_w_m2: float, years: np.ndarray, initial_k: float = 0.0) -> np.ndarray:
    """Closed surface solution when gamma is 0. Used by tests and by the one-box limit."""

    parameters.validate()
    if parameters.gamma_w_m2_k != 0.0:
        raise ContractError("one_box_surface requires gamma = 0")
    time = np.asarray(years, dtype=np.float64)
    lam = parameters.lambda_w_m2_k
    capacity = parameters.c_surface_w_yr_m2_k
    if lam == 0.0:
        return initial_k + forcing_w_m2 * time / capacity
    equilibrium_k = forcing_w_m2 / lam
    rate = lam / capacity
    return equilibrium_k + (initial_k - equilibrium_k) * np.exp(-rate * time)


def energy_storage_w_yr_m2(parameters: TwoBoxParameters, state: np.ndarray) -> np.ndarray:
    """Cs Ts + Cd Td, in W yr m-2. Multiply by the Julian year to get J m-2."""

    values = np.asarray(state, dtype=np.float64)
    if values.shape[-1] != 2:
        raise ContractError("state must end with (surface, deep)")
    return (
        parameters.c_surface_w_yr_m2_k * values[..., 0]
        + parameters.c_deep_w_yr_m2_k * values[..., 1]
    )


def net_uptake_w_m2(parameters: TwoBoxParameters, surface_k: np.ndarray, forcing_w_m2: np.ndarray) -> np.ndarray:
    """F - lambda Ts. The exchange term is not part of the planetary budget."""

    return np.asarray(forcing_w_m2, dtype=np.float64) - parameters.lambda_w_m2_k * np.asarray(surface_k, dtype=np.float64)


def integrate(
    parameters: TwoBoxParameters,
    forcing_w_m2: np.ndarray,
    dt_years: float,
    *,
    method: str = "backward_euler",
    initial: InitialAnomaly | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Step a piecewise-constant forcing.

    `forcing_w_m2[i]` is held over the interval that ends at time (i+1) dt.
    Returned times have length n+1 and include 0. States have shape (n+1, 2).
    """

    parameters.validate()
    forcing = np.asarray(forcing_w_m2, dtype=np.float64)
    if forcing.ndim != 1 or forcing.size < 1 or np.any(~np.isfinite(forcing)):
        raise ContractError("forcing series must be finite and one-dimensional")
    if dt_years <= 0.0 or not np.isfinite(dt_years):
        raise ContractError("time step must be positive and finite")
    if method not in {"forward_euler", "backward_euler", "eigen"}:
        raise ContractError(f"unknown integrator {method}")
    if method == "forward_euler":
        limit = forward_euler_step_limit(parameters)
        if dt_years >= limit:
            raise ContractError(
                f"forward Euler step {dt_years} years is not below the stability limit {limit} years"
            )
    matrix = system_matrix(parameters)
    column = forcing_column(parameters)
    state = InitialAnomaly().as_array() if initial is None else initial.as_array()
    count = forcing.size
    times = np.arange(count + 1, dtype=np.float64) * dt_years
    states = np.empty((count + 1, 2), dtype=np.float64)
    states[0] = state
    identity = np.eye(2)
    for index, force in enumerate(forcing.tolist()):
        if method == "forward_euler":
            state = state + dt_years * (matrix @ state + column * force)
        elif method == "backward_euler":
            left = identity - dt_years * matrix
            right = state + dt_years * column * force
            state = np.linalg.solve(left, right)
        else:
            # Exact on a piecewise-constant step: x1 = exp(A dt) x0 + A^{-1} (exp(A dt) - I) b F
            # when A is invertible. The pure integrator is handled explicitly.
            transition = matrix_exponential(matrix, dt_years)
            if parameters.lambda_w_m2_k == 0.0:
                state = np.array(
                    [
                        state[0] + force * dt_years / parameters.c_surface_w_yr_m2_k,
                        state[1],
                    ]
                )
            else:
                target = equilibrium(parameters, force)
                state = target + transition @ (state - target)
        if not np.all(np.isfinite(state)):
            raise CalibrationError(f"integrator {method} produced a non-finite state at step {index}")
        states[index + 1] = state
    return times, states


def equilibrium_climate_sensitivity(parameters: TwoBoxParameters, doubling_erf_w_m2: float = CO2_ERF_DOUBLING_AR6_W_M2) -> float:
    parameters.validate()
    if parameters.lambda_w_m2_k <= 0.0:
        raise ContractError("ECS is undefined when lambda is not positive")
    if doubling_erf_w_m2 <= 0.0:
        raise ContractError("doubling ERF must be positive")
    return float(doubling_erf_w_m2 / parameters.lambda_w_m2_k)


def one_percent_forcing(years: np.ndarray, *, doubling_erf_w_m2: float = CO2_ERF_DOUBLING_AR6_W_M2) -> np.ndarray:
    """Forcing of a 1 percent per year CO2 increase, F(t) = (F2x / ln 2) * t * ln(1.01)."""

    time = np.asarray(years, dtype=np.float64)
    if np.any(time < 0.0):
        raise ContractError("1 percent ramp years cannot be negative")
    alpha = doubling_erf_w_m2 / np.log(2.0)
    return alpha * time * np.log(1.01)


def transient_climate_response(
    parameters: TwoBoxParameters,
    *,
    doubling_erf_w_m2: float = CO2_ERF_DOUBLING_AR6_W_M2,
    year: float = 70.0,
) -> float:
    """Surface anomaly at `year` of a 1 percent per year ramp, default year 70."""

    # Sample the ramp finely enough that the eigen stepper's piecewise-constant
    # assumption is a small fraction of the 70-year signal. The forcing at the
    # left edge of each year is used, which is a stated discretization.
    steps = int(np.ceil(year))
    forcing = one_percent_forcing(np.arange(steps, dtype=np.float64), doubling_erf_w_m2=doubling_erf_w_m2)
    _times, states = integrate(parameters, forcing, 1.0, method="eigen")
    if steps != year:
        raise ContractError("TCR year must be an integer in this discretization")
    return float(states[-1, 0])
