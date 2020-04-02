"""Calibrate the two-box model without looking at the holdout window.

The search is a coarse log-space grid followed by a finite-difference
Levenberg-Marquardt polish. A particle cloud around the polish supplies an
explicit uncertainty distribution. Weights are Gaussian in the training
residual. The baseline used to compare the model with an anomaly series has
to sit inside the training years. Holdout temperatures are not part of the
objective, the baseline mean, or the particle weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.ebm import InitialAnomaly, TwoBoxParameters, integrate
from forcing_atlas.errors import CalibrationError, ContractError


@dataclass(frozen=True, slots=True)
class CalibrationWindow:
    train_start: int
    train_end: int
    baseline_start: int
    baseline_end: int
    holdout_start: int
    holdout_end: int

    def __post_init__(self) -> None:
        if not (self.train_start <= self.baseline_start <= self.baseline_end <= self.train_end):
            raise ContractError("the anomaly baseline must lie inside the training window")
        if self.holdout_start <= self.train_end:
            raise ContractError("holdout must start after the training window")
        if self.holdout_end < self.holdout_start:
            raise ContractError("holdout ends before it starts")


@dataclass(slots=True)
class CalibrationResult:
    parameters: TwoBoxParameters
    window: CalibrationWindow
    train_rmse_k: float
    holdout_rmse_k: float
    baseline_offset_k: float
    years: np.ndarray
    simulated_anomaly_k: np.ndarray
    observed_anomaly_k: np.ndarray
    particle_parameters: np.ndarray
    particle_weights: np.ndarray
    sigma_k: float
    evaluations: int
    note: str

    def weighted_lambda(self) -> tuple[float, float, float]:
        return _weighted_interval(self.particle_parameters[:, 0], self.particle_weights)

    def weighted_ecs(self, doubling_erf_w_m2: float = 3.93) -> tuple[float, float, float]:
        ecs = doubling_erf_w_m2 / self.particle_parameters[:, 0]
        return _weighted_interval(ecs, self.particle_weights)


def _weighted_interval(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(keep):
        raise CalibrationError("no weighted particles were finite")
    values = values[keep]
    weights = weights[keep]
    weights = weights / np.sum(weights)
    order = np.argsort(values)
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative = cumulative / cumulative[-1]
    mean = float(np.sum(values * weights) / np.sum(weights))

    def quantile(level: float) -> float:
        index = int(np.searchsorted(cumulative, level, side="left"))
        index = min(index, ordered.size - 1)
        return float(ordered[index])

    return mean, quantile(0.05), quantile(0.95)


# Hard prior used by the search. It is not updated from the holdout.
# lambda in W m-2 K-1, capacities in W yr m-2 K-1.
PARAMETER_PRIOR = {
    "lambda_w_m2_k": (0.5, 2.5),
    "gamma_w_m2_k": (0.2, 2.0),
    "c_surface_w_yr_m2_k": (4.0, 20.0),
    "c_deep_w_yr_m2_k": (50.0, 300.0),
}


def _parameters_from_theta(theta: np.ndarray) -> TwoBoxParameters:
    lam, gamma, surface, deep = np.exp(np.asarray(theta, dtype=np.float64))
    parameters = TwoBoxParameters(float(lam), float(gamma), float(surface), float(deep))
    parameters.validate()
    values = {
        "lambda_w_m2_k": parameters.lambda_w_m2_k,
        "gamma_w_m2_k": parameters.gamma_w_m2_k,
        "c_surface_w_yr_m2_k": parameters.c_surface_w_yr_m2_k,
        "c_deep_w_yr_m2_k": parameters.c_deep_w_yr_m2_k,
    }
    for name, (low, high) in PARAMETER_PRIOR.items():
        if not low <= values[name] <= high:
            raise ContractError(f"{name} = {values[name]} is outside the prior [{low}, {high}]")
    return parameters


def _simulate(parameters: TwoBoxParameters, forcing: np.ndarray, spinup: np.ndarray | None) -> np.ndarray:
    series = forcing if spinup is None else np.concatenate([spinup, forcing])
    _times, states = integrate(parameters, series, 1.0, method="eigen", initial=InitialAnomaly())
    # Drop the spinup years. The returned anomaly is aligned with `forcing`.
    end_states = states[1:, 0]
    return end_states[end_states.size - forcing.size :]


def _anomaly(simulated: np.ndarray, years: np.ndarray, window: CalibrationWindow) -> tuple[np.ndarray, float]:
    baseline = (years >= window.baseline_start) & (years <= window.baseline_end) & np.isfinite(simulated)
    if int(np.count_nonzero(baseline)) < (window.baseline_end - window.baseline_start + 1) * 0.8:
        raise CalibrationError("the model baseline window is not covered")
    offset = float(np.mean(simulated[baseline]))
    return simulated - offset, offset


def _rmse(residual: np.ndarray) -> float:
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


class _Objective:
    def __init__(
        self,
        years: np.ndarray,
        observed: np.ndarray,
        forcing: np.ndarray,
        window: CalibrationWindow,
        spinup: np.ndarray | None,
    ) -> None:
        self.years = years
        self.observed = observed
        self.forcing = forcing
        self.window = window
        self.spinup = spinup
        self.train = (years >= window.train_start) & (years <= window.train_end) & np.isfinite(observed)
        self.holdout = (years >= window.holdout_start) & (years <= window.holdout_end) & np.isfinite(observed)
        if int(np.count_nonzero(self.train)) < 20:
            raise CalibrationError("fewer than 20 finite training years")
        self.evaluations = 0

    def residuals(self, theta: np.ndarray) -> np.ndarray:
        parameters = _parameters_from_theta(theta)
        simulated = _simulate(parameters, self.forcing, self.spinup)
        anomaly, _offset = _anomaly(simulated, self.years, self.window)
        self.evaluations += 1
        return anomaly[self.train] - self.observed[self.train]

    def sse(self, theta: np.ndarray) -> float:
        residual = self.residuals(theta)
        return float(np.dot(residual, residual))


def _grid_search(objective: _Objective) -> np.ndarray:
    lambdas = np.array([0.6, 0.9, 1.2, 1.8, 2.4])
    gammas = np.array([0.4, 0.8, 1.3])
    surfaces = np.array([5.0, 8.0, 12.0])
    deeps = np.array([70.0, 120.0, 200.0])
    best_theta = None
    best_sse = float("inf")
    for lam in lambdas:
        for gamma in gammas:
            for surface in surfaces:
                for deep in deeps:
                    theta = np.log(np.array([lam, gamma, surface, deep]))
                    score = objective.sse(theta)
                    if score < best_sse:
                        best_sse = score
                        best_theta = theta
    if best_theta is None:
        raise CalibrationError("grid search did not evaluate a parameter")
    return best_theta


def _levenberg(objective: _Objective, theta: np.ndarray, *, iterations: int = 12, damping: float = 1.0e-2) -> np.ndarray:
    current = theta.copy()
    current_sse = objective.sse(current)
    step = np.array([0.05, 0.05, 0.08, 0.08])
    lam = damping
    for _ in range(iterations):
        residual = objective.residuals(current)
        jacobian = np.empty((residual.size, current.size))
        for index in range(current.size):
            bumped = current.copy()
            bumped[index] += step[index]
            jacobian[:, index] = (objective.residuals(bumped) - residual) / step[index]
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        try:
            delta = np.linalg.solve(normal + lam * np.diag(np.diag(normal) + 1e-8), gradient)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        candidate = current - delta
        if np.any(~np.isfinite(candidate)):
            lam *= 10.0
            continue
        try:
            candidate_sse = objective.sse(candidate)
        except ContractError:
            lam *= 10.0
            continue
        if candidate_sse < current_sse:
            current = candidate
            current_sse = candidate_sse
            lam = max(lam / 3.0, 1.0e-6)
        else:
            lam *= 4.0
    return current


def _particles(objective: _Objective, theta: np.ndarray, sigma_k: float, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    # Perturb in log space. The spread is a declared prior width, not a result.
    spread = np.array([0.15, 0.2, 0.15, 0.25])
    draws = theta + generator.normal(0.0, 1.0, size=(count, theta.size)) * spread
    parameters = np.empty((count, 4))
    weights = np.empty(count)
    for index in range(count):
        try:
            unpacked = _parameters_from_theta(draws[index])
            residual = objective.residuals(draws[index])
        except (ContractError, CalibrationError):
            parameters[index] = np.nan
            weights[index] = 0.0
            continue
        parameters[index] = (
            unpacked.lambda_w_m2_k,
            unpacked.gamma_w_m2_k,
            unpacked.c_surface_w_yr_m2_k,
            unpacked.c_deep_w_yr_m2_k,
        )
        sse = float(np.dot(residual, residual))
        weights[index] = np.exp(-0.5 * sse / (sigma_k**2 * residual.size))
    if not np.any(weights > 0.0):
        raise CalibrationError("every particle was rejected")
    weights = weights / np.sum(weights)
    return parameters, weights


def calibrate_two_box(
    years: np.ndarray,
    observed_anomaly_k: np.ndarray,
    forcing_w_m2: np.ndarray,
    window: CalibrationWindow,
    *,
    sigma_k: float = 0.1,
    particles: int = 48,
    seed: int = 11,
    spinup_forcing_w_m2: np.ndarray | None = None,
) -> CalibrationResult:
    """Fit on `window` and report the holdout residual.

    `sigma_k` is the preregistered annual residual scale used only for particle
    weights. It is not adjusted to the holdout.
    """

    time = np.asarray(years, dtype=np.int64)
    observed = np.asarray(observed_anomaly_k, dtype=np.float64)
    forcing = np.asarray(forcing_w_m2, dtype=np.float64)
    if time.shape != observed.shape or time.shape != forcing.shape:
        raise ContractError("years, observations, and forcing must share a shape")
    if np.any(np.diff(time) != 1):
        raise ContractError("calibration expects consecutive calendar years")
    if sigma_k <= 0.0:
        raise ContractError("sigma_k must be positive")
    spinup = None if spinup_forcing_w_m2 is None else np.asarray(spinup_forcing_w_m2, dtype=np.float64)
    if spinup is not None and (spinup.ndim != 1 or np.any(~np.isfinite(spinup))):
        raise ContractError("spinup forcing must be a finite vector")
    objective = _Objective(time, observed, forcing, window, spinup)
    theta = _levenberg(objective, _grid_search(objective))
    parameters = _parameters_from_theta(theta)
    simulated = _simulate(parameters, forcing, spinup)
    anomaly, offset = _anomaly(simulated, time, window)
    train_residual = anomaly[objective.train] - observed[objective.train]
    holdout_residual = anomaly[objective.holdout] - observed[objective.holdout]
    cloud, weights = _particles(objective, theta, sigma_k, particles, seed)
    return CalibrationResult(
        parameters=parameters,
        window=window,
        train_rmse_k=_rmse(train_residual),
        holdout_rmse_k=_rmse(holdout_residual),
        baseline_offset_k=offset,
        years=time,
        simulated_anomaly_k=anomaly,
        observed_anomaly_k=observed,
        particle_parameters=cloud,
        particle_weights=weights,
        sigma_k=sigma_k,
        evaluations=objective.evaluations,
        note=(
            "Initial state is a zero anomaly at the start of the spinup forcing, or at the "
            "first scored year when no spinup is supplied. The simulated series is then "
            "shifted to the training-era baseline. Holdout years do not enter the objective, "
            "the baseline, or the weights. Parameters are confined to PARAMETER_PRIOR. "
            "A training optimum on the prior boundary is not rewritten after seeing the holdout."
        ),
    )
