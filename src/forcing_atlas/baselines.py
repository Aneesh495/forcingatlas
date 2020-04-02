"""Statistical baselines fit only on the training window.

Persistence and a linear trend are global-mean comparators. Pattern scaling
and ridge regression are spatial. The ridge penalty is chosen on a late slice
of the training window, never on the holdout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.errors import ContractError
from forcing_atlas.grids import area_weighted_mean


@dataclass(slots=True)
class SeriesBaseline:
    name: str
    years: np.ndarray
    prediction: np.ndarray
    train_rmse: float
    holdout_rmse: float


def _rmse(years: np.ndarray, prediction: np.ndarray, observed: np.ndarray, start: int, end: int) -> float:
    mask = (years >= start) & (years <= end) & np.isfinite(prediction) & np.isfinite(observed)
    if int(np.count_nonzero(mask)) == 0:
        return float("nan")
    delta = prediction[mask] - observed[mask]
    return float(np.sqrt(np.mean(delta**2)))


def persistence_forecast(years: np.ndarray, observed: np.ndarray, train_end: int, holdout_end: int) -> SeriesBaseline:
    """Next-year prediction equals the previous observed year.

    The previous year may be a holdout observation when the target year is
    also in the holdout. That is the persistence forecast. It is not a fit.
    """

    time = np.asarray(years, dtype=np.int64)
    values = np.asarray(observed, dtype=np.float64)
    prediction = np.full(time.shape, np.nan)
    prediction[1:] = values[:-1]
    return SeriesBaseline(
        "persistence_lag1",
        time,
        prediction,
        _rmse(time, prediction, values, int(time[0]) + 1, train_end),
        _rmse(time, prediction, values, train_end + 1, holdout_end),
    )


def training_mean_forecast(years: np.ndarray, observed: np.ndarray, train_start: int, train_end: int, holdout_end: int) -> SeriesBaseline:
    time = np.asarray(years, dtype=np.int64)
    values = np.asarray(observed, dtype=np.float64)
    train = (time >= train_start) & (time <= train_end) & np.isfinite(values)
    if int(np.count_nonzero(train)) < 5:
        raise ContractError("training-mean baseline needs five finite years")
    level = float(np.mean(values[train]))
    prediction = np.full(time.shape, level)
    return SeriesBaseline(
        "training_mean",
        time,
        prediction,
        _rmse(time, prediction, values, train_start, train_end),
        _rmse(time, prediction, values, train_end + 1, holdout_end),
    )


def linear_trend_forecast(years: np.ndarray, observed: np.ndarray, train_start: int, train_end: int, holdout_end: int) -> SeriesBaseline:
    """Ordinary least squares of temperature on year, training years only."""

    time = np.asarray(years, dtype=np.float64)
    values = np.asarray(observed, dtype=np.float64)
    train = (time >= train_start) & (time <= train_end) & np.isfinite(values)
    if int(np.count_nonzero(train)) < 5:
        raise ContractError("trend baseline needs five finite training years")
    design = np.column_stack([np.ones(int(np.count_nonzero(train))), time[train]])
    coefficient, *_ = np.linalg.lstsq(design, values[train], rcond=None)
    prediction = coefficient[0] + coefficient[1] * time
    years_i = time.astype(np.int64)
    return SeriesBaseline(
        "linear_trend",
        years_i,
        prediction,
        _rmse(years_i, prediction, values, train_start, train_end),
        _rmse(years_i, prediction, values, train_end + 1, holdout_end),
    )


def pattern_scale(
    years: np.ndarray,
    field: np.ndarray,
    global_mean: np.ndarray,
    areas: np.ndarray,
    train_start: int,
    train_end: int,
) -> np.ndarray:
    """Regress each cell on the global mean using training years only.

    The returned pattern has the field's spatial shape. A prediction is
    `pattern * global_mean_forecast`. Cells with no finite training pairs stay NaN.
    """

    time = np.asarray(years, dtype=np.int64)
    values = np.asarray(field, dtype=np.float64)
    signal = np.asarray(global_mean, dtype=np.float64)
    if values.shape[0] != time.size or signal.shape != time.shape:
        raise ContractError("pattern scaling years, field, and global mean disagree")
    if values.shape[1:] != areas.shape:
        raise ContractError("pattern scaling areas do not match the field")
    train = (time >= train_start) & (time <= train_end)
    pattern = np.full(values.shape[1:], np.nan)
    x = signal[train]
    for lat in range(values.shape[1]):
        for lon in range(values.shape[2]):
            y = values[train, lat, lon]
            finite = np.isfinite(x) & np.isfinite(y)
            if int(np.count_nonzero(finite)) < 10:
                continue
            design = np.column_stack([np.ones(int(np.count_nonzero(finite))), x[finite]])
            coefficient, *_ = np.linalg.lstsq(design, y[finite], rcond=None)
            pattern[lat, lon] = coefficient[1]
    # A global-mean regression of an area-weighted field should have an area-weighted
    # slope near 1. That check is available to the caller via the returned pattern.
    _ = area_weighted_mean(pattern, areas)
    return pattern


def ridge_linear_response(
    forcing: np.ndarray,
    target: np.ndarray,
    train_end_index: int,
    validation_start_index: int,
    *,
    lags: int = 3,
    penalties: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> tuple[np.ndarray, float]:
    """Ridge map from lagged forcing to a scalar target.

    Rows before `lags` are dropped. The penalty minimizes validation SSE on
    `[validation_start_index, train_end_index)`. Indices at or above
    `train_end_index` are not used to choose the penalty or the coefficients.
    """

    force = np.asarray(forcing, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    if force.shape != values.shape or force.ndim != 1:
        raise ContractError("ridge forcing and target must be flat and aligned")
    if not 0 < validation_start_index < train_end_index <= force.size:
        raise ContractError("ridge validation slice must sit inside the training prefix")
    if lags < 1:
        raise ContractError("ridge lags must be positive")
    design = np.column_stack([force[lags - 1 - lag : force.size - lag] for lag in range(lags)])
    design = np.column_stack([np.ones(design.shape[0]), design])
    aligned_target = values[lags - 1 :]
    # Row i of design predicts year index (lags - 1 + i).
    def row_index(year_index: int) -> int:
        return year_index - (lags - 1)

    train_rows = np.arange(0, row_index(validation_start_index))
    valid_rows = np.arange(row_index(validation_start_index), row_index(train_end_index))
    if train_rows.size < lags + 2 or valid_rows.size < 2:
        raise ContractError("ridge split is too short")
    best_penalty = None
    best_score = float("inf")
    best_coefficient = None
    for penalty in penalties:
        coefficient = _ridge_solve(design[train_rows], aligned_target[train_rows], penalty)
        prediction = design[valid_rows] @ coefficient
        residual = prediction - aligned_target[valid_rows]
        finite = residual[np.isfinite(residual)]
        score = float(np.mean(finite**2)) if finite.size else float("inf")
        if score < best_score:
            best_score = score
            best_penalty = penalty
            best_coefficient = coefficient
    if best_coefficient is None or best_penalty is None:
        raise ContractError("ridge search failed")
    # Refit on the whole training prefix with the chosen penalty.
    fit_rows = np.arange(0, row_index(train_end_index))
    coefficient = _ridge_solve(design[fit_rows], aligned_target[fit_rows], best_penalty)
    full = np.full(force.shape, np.nan)
    full[lags - 1 :] = design @ coefficient
    return full, float(best_penalty)


def _ridge_solve(design: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    finite = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    matrix = design[finite]
    gram = matrix.T @ matrix
    regularizer = penalty * np.eye(gram.shape[0])
    regularizer[0, 0] = 0.0  # do not penalize the intercept
    return np.linalg.solve(gram + regularizer, matrix.T @ target[finite])
