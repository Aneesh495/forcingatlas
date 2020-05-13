"""Three-seed spatial fit on the non-infilled HadCRUT grid.

Training years stop at 1994. 1995 onward is scored and is not an index the
loss sees. Pattern scaling and a uniform global-trend field are fit on the
same training years. If the emulator does not beat the uniform trend, that
miss is part of the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forcing_atlas.baselines import linear_trend_forecast
from forcing_atlas.emulator import TrainConfig, new_emulator, predict, support_radius, train_emulator
from forcing_atlas.errors import ContractError
from forcing_atlas.grids import area_weighted_mean
from forcing_atlas.grids_io import read_hadcrut_grid
from forcing_atlas.historical import load_historical_bundle

SEEDS = (11, 29, 47)
TRAIN_END = 1994
HOLDOUT_START = 1995
LAGS = 4


def _lag_window(forcing: np.ndarray, lags: int) -> np.ndarray:
    window = np.zeros((forcing.shape[0], lags, forcing.shape[1]), dtype=np.float64)
    for lag in range(lags):
        if lag == 0:
            window[:, 0, :] = forcing
        else:
            window[lag:, lag, :] = forcing[:-lag]
    return window


def _indices(years: np.ndarray, start: int, end: int, *, minimum_index: int) -> np.ndarray:
    chosen = [index for index, year in enumerate(years.tolist()) if start <= int(year) <= end and index >= minimum_index]
    return np.asarray(chosen, dtype=np.int64)


def assert_disjoint(train_index: np.ndarray, holdout_index: np.ndarray) -> None:
    overlap = set(train_index.tolist()) & set(holdout_index.tolist())
    if overlap:
        raise ContractError(f"training index leaks into the holdout: {sorted(overlap)[:8]}")


def spatial_rmse(prediction: np.ndarray, target: np.ndarray, areas: np.ndarray, index: np.ndarray) -> float:
    if index.size == 0:
        return float("nan")
    pred = prediction[index]
    obs = target[index]
    valid = np.isfinite(pred) & np.isfinite(obs)
    weight = np.where(valid, areas, 0.0)
    denominator = float(np.sum(weight))
    if denominator <= 0.0:
        return float("nan")
    square = np.where(valid, (pred - obs) ** 2, 0.0)
    return float(np.sqrt(np.sum(square * areas) / denominator))


def _pattern_forecast(field: np.ndarray, global_mean: np.ndarray, train_index: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Cell-wise regression on the training global mean, applied to `signal` at every year."""

    prediction = np.full(field.shape, np.nan, dtype=np.float64)
    x = global_mean[train_index]
    for lat in range(field.shape[1]):
        for lon in range(field.shape[2]):
            y = field[train_index, lat, lon]
            finite = np.isfinite(x) & np.isfinite(y)
            if int(np.count_nonzero(finite)) < 10:
                continue
            design = np.column_stack([np.ones(int(np.count_nonzero(finite))), x[finite]])
            coefficient, *_ = np.linalg.lstsq(design, y[finite], rcond=None)
            prediction[:, lat, lon] = coefficient[0] + coefficient[1] * signal
    return prediction


def run_grid_campaign(
    root: Path,
    output: Path,
    *,
    epochs: int = 15,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, object]:
    grid = read_hadcrut_grid(root / "HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc", name="noninfilled")
    bundle = load_historical_bundle(root)
    year_to_force = {int(year): index for index, year in enumerate(bundle.years.tolist())}
    keep = [index for index, year in enumerate(grid.years.tolist()) if int(year) in year_to_force]
    years = grid.years[keep]
    field = grid.annual[keep]
    areas = grid.grid.cell_areas_m2()
    weights = areas / np.sum(areas)
    components = []
    for name in ("forcing_co2_w_m2", "forcing_aerosol_w_m2", "forcing_solar_w_m2", "forcing_volcanic_w_m2"):
        series = getattr(bundle, name)
        components.append(np.array([series[year_to_force[int(year)]] for year in years], dtype=np.float64))
    forcing = np.column_stack(components)
    window = _lag_window(forcing, LAGS)
    train_index = _indices(years, int(years[0]), TRAIN_END, minimum_index=LAGS - 1)
    holdout_index = _indices(years, HOLDOUT_START, int(years[-1]), minimum_index=0)
    assert_disjoint(train_index, holdout_index)
    latitude = grid.grid.lat / 90.0
    longitude = grid.grid.lon / 180.0
    global_mean = area_weighted_mean(field, areas)
    trend = linear_trend_forecast(years, global_mean, int(years[0]), TRAIN_END, int(years[-1]))
    uniform = np.repeat(trend.prediction[:, None, None], field.shape[1], axis=1)
    uniform = np.repeat(uniform, field.shape[2], axis=2)
    patterned = _pattern_forecast(field, global_mean, train_index, trend.prediction)
    inside = support_radius(forcing[train_index], forcing)
    seed_rows = []
    predictions = []
    for seed in seeds:
        model = new_emulator(forcing.shape[1], LAGS, field.shape[1], field.shape[2], seed)
        trained = train_emulator(
            model,
            window,
            forcing,
            field,
            train_index,
            latitude,
            longitude,
            weights,
            TrainConfig(seed=seed, epochs=epochs, lr=0.02, conservation_weight=0.2),
            checkpoint=output.parent / "checkpoints" / f"emulator-seed-{seed}.pt",
        )
        predicted = predict(model, window, forcing, latitude, longitude, weights)
        predicted = predicted.copy()
        predicted[~inside] = np.nan
        predictions.append(predicted)
        seed_rows.append(
            {
                "seed": seed,
                "initial_loss": trained.initial_loss,
                "final_loss": trained.final_loss,
                "parameter_delta": trained.parameter_delta,
                "holdout_rmse_k": spatial_rmse(predicted, field, areas, holdout_index),
                "train_rmse_k": spatial_rmse(predicted, field, areas, train_index),
            }
        )
    ensemble = np.nanmean(np.stack(predictions, axis=0), axis=0)
    result = {
        "format": "forcing-atlas-grid-campaign/1",
        "grid": "HadCRUT.5.1.0.0 non-infilled annual",
        "semantics": "observation",
        "baseline": "1961-1990",
        "train_end": TRAIN_END,
        "holdout_start": HOLDOUT_START,
        "train_years": [int(years[index]) for index in train_index.tolist()],
        "holdout_years": [int(years[index]) for index in holdout_index.tolist()],
        "epochs": epochs,
        "seeds": seed_rows,
        "uniform_trend_holdout_rmse_k": spatial_rmse(uniform, field, areas, holdout_index),
        "pattern_trend_holdout_rmse_k": spatial_rmse(patterned, field, areas, holdout_index),
        "ensemble_holdout_rmse_k": spatial_rmse(ensemble, field, areas, holdout_index),
        "abstained_holdout_years": [int(years[index]) for index in holdout_index.tolist() if not bool(inside[index])],
        "beats_uniform_trend": None,
    }
    best = min(row["holdout_rmse_k"] for row in seed_rows)
    result["beats_uniform_trend"] = bool(best < result["uniform_trend_holdout_rmse_k"])
    result["note"] = (
        "Holdout years are absent from train_years. "
        "A false beats_uniform_trend value means the learned field did not beat a latitude-uniform trend."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result
