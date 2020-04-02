"""Holdout isolation for the two-box fit, plus baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forcing_atlas.baselines import linear_trend_forecast, persistence_forecast, ridge_linear_response, training_mean_forecast
from forcing_atlas.calibrate import CalibrationWindow, calibrate_two_box
from forcing_atlas.ebm import InitialAnomaly, TwoBoxParameters, integrate
from forcing_atlas.historical import load_historical_bundle


def _synthetic(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = np.arange(1900, 1980)
    forcing = np.linspace(0.2, 1.6, years.size)
    parameters = TwoBoxParameters(1.2, 0.8, 8.0, 120.0)
    _times, states = integrate(parameters, forcing, 1.0, method="eigen", initial=InitialAnomaly())
    simulated = states[1:, 0]
    baseline = (years >= 1920) & (years <= 1949)
    observed = simulated - float(np.mean(simulated[baseline]))
    generator = np.random.default_rng(seed)
    observed = observed + generator.normal(0.0, 0.01, size=observed.shape)
    return years, observed, forcing


def test_holdout_temperatures_do_not_move_the_parameters() -> None:
    years, observed, forcing = _synthetic()
    window = CalibrationWindow(1900, 1959, 1920, 1949, 1960, 1979)
    first = calibrate_two_box(years, observed, forcing, window, particles=8, sigma_k=0.1)
    corrupted = observed.copy()
    corrupted[years >= 1960] += 4.0
    second = calibrate_two_box(years, corrupted, forcing, window, particles=8, sigma_k=0.1)
    assert first.parameters.lambda_w_m2_k == pytest.approx(second.parameters.lambda_w_m2_k, rel=1e-8, abs=1e-8)
    assert first.parameters.c_deep_w_yr_m2_k == pytest.approx(second.parameters.c_deep_w_yr_m2_k, rel=1e-6, abs=1e-6)
    assert first.train_rmse_k < 0.05
    assert second.holdout_rmse_k > first.holdout_rmse_k


def test_noise_free_lambda_is_recovered() -> None:
    years = np.arange(1900, 1980)
    forcing = np.linspace(0.3, 2.0, years.size)
    truth = TwoBoxParameters(1.2, 0.8, 8.0, 120.0)
    _times, states = integrate(truth, forcing, 1.0, method="eigen")
    simulated = states[1:, 0]
    window = CalibrationWindow(1900, 1969, 1920, 1949, 1970, 1979)
    baseline = (years >= 1920) & (years <= 1949)
    observed = simulated - float(np.mean(simulated[baseline]))
    result = calibrate_two_box(years, observed, forcing, window, particles=6)
    assert result.parameters.lambda_w_m2_k == pytest.approx(1.2, abs=0.15)
    assert result.holdout_rmse_k < 0.05


def test_baselines_do_not_fit_the_holdout() -> None:
    years = np.arange(1950, 1980)
    observed = 0.01 * (years - 1950)
    trend = linear_trend_forecast(years, observed, 1950, 1969, 1979)
    assert trend.holdout_rmse < 1e-8
    # A slope estimated on the holdout would still fit, so corrupt the training slope's
    # implication by checking the training mean instead, which cannot follow the ramp.
    mean = training_mean_forecast(years, observed, 1950, 1969, 1979)
    assert mean.holdout_rmse > mean.train_rmse
    persist = persistence_forecast(years, observed, 1969, 1979)
    assert np.isnan(persist.prediction[0])
    assert persist.prediction[1] == pytest.approx(observed[0])


def test_ridge_penalty_ignores_the_holdout() -> None:
    years = np.arange(80)
    forcing = np.linspace(0.0, 2.0, years.size)
    target = 0.4 * forcing + 0.1 * np.roll(forcing, 1)
    target[0] = 0.0
    prediction, penalty = ridge_linear_response(forcing, target, train_end_index=60, validation_start_index=45, lags=2)
    corrupted = target.copy()
    corrupted[60:] += 10.0
    prediction_2, penalty_2 = ridge_linear_response(
        forcing, corrupted, train_end_index=60, validation_start_index=45, lags=2
    )
    assert penalty == penalty_2
    assert np.isnan(prediction[0])
    assert prediction[1:60] == pytest.approx(prediction_2[1:60], abs=1e-8)


def test_real_hadcrut_forcing_alignment_and_fit() -> None:
    root = Path("data/raw")
    if not (root / "HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv").is_file():
        pytest.skip("pinned HadCRUT file is not in data/raw")
    bundle = load_historical_bundle(root)
    assert bundle.component_aggregate_max_diff < 1.0e-4
    assert bundle.years[0] == 1850
    assert bundle.years[-1] >= 2020
    # The component file continues while HadCRUT starts in 1850. Forcing before
    # 1850 is not in this aligned bundle. The fit below uses the overlap only,
    # with a zero initial anomaly at 1850, which the result note has to admit.
    window = CalibrationWindow(1850, 1994, 1961, 1990, 1995, int(bundle.years[-1]))
    assert bundle.spinup_years[0] == 1750
    assert bundle.spinup_years[-1] == 1849
    result = calibrate_two_box(
        bundle.years,
        bundle.hadcrut_anomaly_k,
        bundle.forcing_w_m2,
        window,
        particles=12,
        sigma_k=0.1,
        spinup_forcing_w_m2=bundle.spinup_forcing_w_m2,
    )
    assert result.train_rmse_k < 0.2
    assert np.isfinite(result.holdout_rmse_k)
    assert 0.5 <= result.parameters.lambda_w_m2_k <= 2.5
    mean_lambda, low, high = result.weighted_lambda()
    assert low <= mean_lambda <= high


def test_low_feedback_fits_training_and_misses_the_holdout() -> None:
    """A smaller lambda can win on 1850-1994 and then run warm after 1994.

    The numbers are computed from fixed parameters, not from the optimizer, so
    this does not retune the fit. It is the reason the search keeps a prior and
    still publishes the holdout.
    """

    root = Path("data/raw")
    if not (root / "HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv").is_file():
        pytest.skip("pinned HadCRUT file is not in data/raw")
    from forcing_atlas.ebm import InitialAnomaly, TwoBoxParameters, integrate

    bundle = load_historical_bundle(root)
    full = np.concatenate([bundle.spinup_forcing_w_m2, bundle.forcing_w_m2])

    def rmse(lam: float, start: int, end: int) -> float:
        parameters = TwoBoxParameters(lam, 0.7, 8.0, 100.0)
        _times, states = integrate(parameters, full, 1.0, method="eigen", initial=InitialAnomaly())
        simulated = states[1:, 0][-bundle.years.size :]
        baseline = (bundle.years >= 1961) & (bundle.years <= 1990)
        anomaly = simulated - float(np.mean(simulated[baseline]))
        mask = (bundle.years >= start) & (bundle.years <= end)
        delta = anomaly[mask] - bundle.hadcrut_anomaly_k[mask]
        return float(np.sqrt(np.mean(delta**2)))

    assert rmse(0.5, 1850, 1994) < rmse(1.8, 1850, 1994)
    assert rmse(0.5, 1995, 2023) > rmse(1.8, 1995, 2023)
