"""The runnable historical and conditional products behind the atlas.

Every series here names its semantics. Proxy values are not copied into the
instrumental column. Future years are emulation under a stated forcing path,
and they do not carry a gridded field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forcing_atlas.baselines import linear_trend_forecast, persistence_forecast, training_mean_forecast
from forcing_atlas.calibrate import CalibrationWindow, calibrate_two_box
from forcing_atlas.ebm import InitialAnomaly, TwoBoxParameters, integrate
from forcing_atlas.forcing_adapter import build_forcing
from forcing_atlas.historical import load_context_series, load_historical_bundle, rebase_to_window
from forcing_atlas.timebase import align_annual


PRIMARY_WINDOW = CalibrationWindow(1850, 1994, 1961, 1990, 1995, 2023)


@dataclass(slots=True)
class GlobalProduct:
    years: np.ndarray
    hadcrut: np.ndarray
    hadcrut_lower: np.ndarray
    hadcrut_upper: np.ndarray
    berkeley_rebased: np.ndarray
    berkeley_years: np.ndarray
    pages_years: np.ndarray
    pages_median: np.ndarray
    pages_low: np.ndarray
    pages_high: np.ndarray
    pages_instrument: np.ndarray
    ebm_historical: np.ndarray
    ebm_natural: np.ndarray
    natural_years: np.ndarray
    parameters: TwoBoxParameters
    train_rmse: float
    holdout_rmse: float
    persistence_holdout_rmse: float
    trend_holdout_rmse: float
    mean_holdout_rmse: float
    lambda_interval: tuple[float, float, float]
    notes: tuple[str, ...]


def _replay(parameters: TwoBoxParameters, forcing: np.ndarray, years: np.ndarray) -> np.ndarray:
    _times, states = integrate(parameters, forcing, 1.0, method="eigen", initial=InitialAnomaly())
    simulated = states[1:, 0]
    if simulated.size != years.size:
        raise RuntimeError("replay length does not match years")
    baseline = (years >= 1961) & (years <= 1990)
    return simulated - float(np.mean(simulated[baseline]))


def build_global_product(root: Path, *, particles: int = 24) -> GlobalProduct:
    bundle = load_historical_bundle(root)
    context = load_context_series(root)
    # The published summary currently runs past the forcing file. Score the overlap.
    overlap_end = min(PRIMARY_WINDOW.holdout_end, int(bundle.years[-1]))
    window = CalibrationWindow(
        PRIMARY_WINDOW.train_start,
        PRIMARY_WINDOW.train_end,
        PRIMARY_WINDOW.baseline_start,
        PRIMARY_WINDOW.baseline_end,
        PRIMARY_WINDOW.holdout_start,
        overlap_end,
    )
    mask = bundle.years <= overlap_end
    years = bundle.years[mask]
    observed = bundle.hadcrut_anomaly_k[mask]
    forcing = bundle.forcing_w_m2[mask]
    fitted = calibrate_two_box(
        years,
        observed,
        forcing,
        window,
        particles=particles,
        sigma_k=0.1,
        spinup_forcing_w_m2=bundle.spinup_forcing_w_m2,
    )
    full_forcing = np.concatenate([bundle.spinup_forcing_w_m2, forcing])
    full_years = np.concatenate([bundle.spinup_years, years])
    historical = _replay(fitted.parameters, full_forcing, full_years)
    # Keep the scored years for the atlas trace. The spinup is in the same replay
    # so the 1850 state is not reset.
    historical_scored = historical[full_years >= years[0]]
    natural = build_forcing(
        full_years,
        erf_solar=np.concatenate([_component(root, "erf_solar", bundle.spinup_years), bundle.forcing_solar_w_m2[mask]]),
        erf_volcanic=np.concatenate(
            [_component(root, "erf_volcanic", bundle.spinup_years), bundle.forcing_volcanic_w_m2[mask]]
        ),
        natural_only=True,
    )
    natural_anomaly = _replay(fitted.parameters, natural.total_w_m2, full_years)
    berkeley = context["berkeley"]["air_above_ice"]
    berkeley_years = berkeley.years[np.isfinite(berkeley.values)]
    berkeley_values = berkeley.values[np.isfinite(berkeley.values)]
    berkeley_1961 = rebase_to_window(berkeley_years, berkeley_values, 1961, 1990)
    pages = context["pages"]["proxy"]
    instrument = context["pages"]["instrumental_target"]
    common_years, _bvals, _hvals = align_annual(berkeley_years, berkeley_1961, years, observed)
    del common_years
    persistence = persistence_forecast(years, observed, window.train_end, window.holdout_end)
    trend = linear_trend_forecast(years, observed, window.train_start, window.train_end, window.holdout_end)
    mean_baseline = training_mean_forecast(years, observed, window.train_start, window.train_end, window.holdout_end)
    return GlobalProduct(
        years=years,
        hadcrut=observed,
        hadcrut_lower=bundle.hadcrut_lower_k[mask],
        hadcrut_upper=bundle.hadcrut_upper_k[mask],
        berkeley_rebased=berkeley_1961,
        berkeley_years=berkeley_years,
        pages_years=pages.years,
        pages_median=pages.values,
        pages_low=pages.extras["p2_5"],
        pages_high=pages.extras["p97_5"],
        pages_instrument=instrument.values,
        ebm_historical=historical_scored,
        ebm_natural=natural_anomaly[full_years >= years[0]],
        natural_years=years,
        parameters=fitted.parameters,
        train_rmse=fitted.train_rmse_k,
        holdout_rmse=fitted.holdout_rmse_k,
        persistence_holdout_rmse=persistence.holdout_rmse,
        trend_holdout_rmse=trend.holdout_rmse,
        mean_holdout_rmse=mean_baseline.holdout_rmse,
        lambda_interval=fitted.weighted_lambda(),
        notes=tuple(natural.assumptions) + (fitted.note,),
    )


def _component(root: Path, key: str, years: np.ndarray) -> np.ndarray:
    """Read one forcing component on the spinup years from the component file.

    The historical bundle already checked the component sum. This lookup only
    repeats those years so the natural-only path uses the same file.
    """

    from forcing_atlas.catalog import sources_by_id
    from forcing_atlas.readers import read_erf_components

    source = sources_by_id()["igcc-erf-components-1750-2023"]
    record = read_erf_components((root / source.file_name).read_text(), source)
    lookup = {int(year): float(value) for year, value in zip(record.years, record.extras[key], strict=True)}
    return np.array([lookup[int(year)] for year in years], dtype=np.float64)


def conditional_extension(
    parameters: TwoBoxParameters,
    history_years: np.ndarray,
    history_forcing: np.ndarray,
    *,
    end_year: int = 2100,
) -> dict[str, np.ndarray]:
    """Continue one global emulation under two declared forcing paths.

    `history_forcing` includes the spinup, so the state is not reset in 1850.
    `stabilize` holds the last forcing. `decline` moves it linearly to the
    1990 forcing by `end_year`. Extended years have no regional field.
    """

    years = np.asarray(history_years, dtype=np.int64)
    forcing = np.asarray(history_forcing, dtype=np.float64)
    last = int(years[-1])
    if end_year <= last:
        raise ValueError("extension must pass the last assessed year")
    if int(np.count_nonzero(years == 1990)) != 1:
        raise ValueError("decline needs a 1990 forcing value")
    future_years = np.arange(last + 1, end_year + 1)
    last_force = float(forcing[-1])
    force_1990 = float(forcing[years == 1990][0])
    fraction = (future_years - last) / (end_year - last)
    stabilize_future = np.full(future_years.shape, last_force)
    decline_future = last_force + (force_1990 - last_force) * fraction
    extended_years = np.concatenate([years, future_years])
    stabilize = _replay(parameters, np.concatenate([forcing, stabilize_future]), extended_years)
    decline = _replay(parameters, np.concatenate([forcing, decline_future]), extended_years)
    return {
        "years": extended_years,
        "stabilize_k": stabilize,
        "decline_k": decline,
        "future_start": np.array([last + 1]),
    }
