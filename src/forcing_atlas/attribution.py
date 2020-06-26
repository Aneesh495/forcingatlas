"""Scaling factors for emulator fingerprints.

The columns are single-forcing trajectories of this project's two-box model.
They are not DAMIP simulations. The estimand is the set of coefficients in

    observed anomaly = beta_co2 * co2_fingerprint
                     + beta_aerosol * aerosol_fingerprint
                     + beta_natural * natural_fingerprint
                     + residual

fitted by least squares on the training years only. A time-shuffled observation
is the negative control. Feature importance is not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.errors import ContractError


@dataclass(slots=True)
class FingerprintFit:
    names: tuple[str, ...]
    beta: np.ndarray
    train_r2: float
    holdout_rmse: float
    condition_number: float
    train_years: int
    holdout_years: int


def _design(columns: dict[str, np.ndarray], names: tuple[str, ...], mask: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(columns[name], dtype=np.float64)[mask] for name in names])


def fit_scaling(
    years: np.ndarray,
    observed: np.ndarray,
    columns: dict[str, np.ndarray],
    names: tuple[str, ...],
    *,
    train_end: int,
    holdout_start: int,
) -> FingerprintFit:
    time = np.asarray(years, dtype=np.int64)
    target = np.asarray(observed, dtype=np.float64)
    if any(columns[name].shape != time.shape for name in names):
        raise ContractError("fingerprint columns must match the observation years")
    train = (time <= train_end) & np.isfinite(target)
    holdout = (time >= holdout_start) & np.isfinite(target)
    for name in names:
        train = train & np.isfinite(columns[name])
        holdout = holdout & np.isfinite(columns[name])
    if int(np.count_nonzero(train)) < len(names) + 5:
        raise ContractError("not enough finite training years for a fingerprint fit")
    design = _design(columns, names, train)
    beta, *_ = np.linalg.lstsq(design, target[train], rcond=None)
    fitted = design @ beta
    residual = target[train] - fitted
    total = target[train] - float(np.mean(target[train]))
    ss_tot = float(np.dot(total, total))
    r2 = 1.0 - float(np.dot(residual, residual)) / ss_tot if ss_tot > 0 else float("nan")
    gram = design.T @ design
    condition = float(np.linalg.cond(gram))
    if int(np.count_nonzero(holdout)) == 0:
        holdout_rmse = float("nan")
    else:
        hold_design = _design(columns, names, holdout)
        delta = target[holdout] - hold_design @ beta
        holdout_rmse = float(np.sqrt(np.mean(delta**2)))
    return FingerprintFit(names, np.asarray(beta, dtype=np.float64), r2, holdout_rmse, condition, int(np.count_nonzero(train)), int(np.count_nonzero(holdout)))


def shuffle_observation(observed: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = np.asarray(observed, dtype=np.float64).copy()
    finite = np.isfinite(values)
    values[finite] = generator.permutation(values[finite])
    return values
