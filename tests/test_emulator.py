"""The emulator learns on the rows it is given and not on the others."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from forcing_atlas.emulator import SpatialEmulator, new_emulator, predict, support_radius, train_emulator, TrainConfig


def _problem(seed: int = 3):
    generator = np.random.default_rng(seed)
    ntime, nlat, nlon, n_forcing, lags = 24, 6, 8, 2, 3
    forcing = generator.normal(size=(ntime, n_forcing))
    window = np.stack([np.vstack([np.zeros((lag, n_forcing)), forcing[: ntime - lag]]) for lag in range(lags)], axis=1)
    # window shape (time, lags, forcing) with zero pad at the start
    window = np.zeros((ntime, lags, n_forcing))
    for lag in range(lags):
        window[lag:, lag, :] = forcing[: ntime - lag]
    latitude = np.linspace(-1.0, 1.0, nlat)
    longitude = np.linspace(-1.0, 1.0, nlon)
    lat_grid = latitude[:, None]
    pattern = 1.0 + 0.3 * lat_grid
    target = forcing[:, 0][:, None, None] * pattern + 0.05 * generator.normal(size=(ntime, nlat, nlon))
    target[0:2] = np.nan
    weights = np.cos(np.linspace(-1, 1, nlat))[:, None] * np.ones((1, nlon))
    weights = weights / weights.sum()
    return window, forcing, target, latitude, longitude, weights


def test_loss_does_not_broadcast_time_against_space() -> None:
    import torch
    from forcing_atlas.emulator import masked_mse

    field = torch.zeros(4, 1, 3, 5)
    target = torch.ones(4, 3, 5)
    mask = torch.ones(4, 3, 5, dtype=torch.bool)
    loss = masked_mse(field, target, mask)
    assert float(loss) == pytest.approx(1.0)


def test_training_changes_parameters_and_lowers_loss(tmp_path) -> None:
    window, forcing, target, latitude, longitude, weights = _problem()
    model = SpatialEmulator(2, 3, 6, 8)
    before = predict(model, window, forcing, latitude, longitude, weights)
    result = train_emulator(
        model,
        window,
        forcing,
        target,
        train_index=np.arange(4, 18),
        latitude=latitude,
        longitude=longitude,
        area_weights=weights,
        config=TrainConfig(seed=11, epochs=25, lr=0.05),
        checkpoint=tmp_path / "seed11.pt",
    )
    after = predict(model, window, forcing, latitude, longitude, weights)
    assert result.parameter_delta > 0.0
    assert result.final_loss < result.initial_loss
    assert not np.allclose(before, after)
    assert (tmp_path / "seed11.pt").is_file()


def test_three_seeds_are_not_identical() -> None:
    window, forcing, target, latitude, longitude, weights = _problem(9)
    outputs = []
    for seed in (11, 29, 47):
        model = new_emulator(2, 3, 6, 8, seed)
        train_emulator(
            model,
            window,
            forcing,
            target,
            train_index=np.arange(4, 18),
            latitude=latitude,
            longitude=longitude,
            area_weights=weights,
            config=TrainConfig(seed=seed, epochs=8, lr=0.05),
        )
        outputs.append(predict(model, window, forcing, latitude, longitude, weights))
    assert not np.allclose(outputs[0], outputs[1])
    assert not np.allclose(outputs[1], outputs[2])


def test_abstention_radius_flags_a_distant_forcing() -> None:
    train = np.array([[0.0, 0.0], [0.1, -0.1], [0.0, 0.2], [-0.2, 0.0]])
    query = np.array([[0.0, 0.0], [8.0, 8.0]])
    inside = support_radius(train, query, quantile=0.9)
    assert bool(inside[0]) is True
    assert bool(inside[1]) is False


def test_checkpoint_reloads() -> None:
    window, forcing, target, latitude, longitude, weights = _problem(1)
    model = SpatialEmulator(2, 3, 6, 8)
    train_emulator(
        model,
        window,
        forcing,
        target,
        np.arange(6, 20),
        latitude,
        longitude,
        weights,
        TrainConfig(seed=5, epochs=3, lr=0.01),
    )
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fresh = SpatialEmulator(2, 3, 6, 8)
    fresh.load_state_dict(state)
    left = predict(model, window[:2], forcing[:2], latitude, longitude, weights)
    right = predict(fresh, window[:2], forcing[:2], latitude, longitude, weights)
    assert np.allclose(left, right)
