"""Spatial emulator with a global response and a regional residual.

The global head reads a short forcing window and emits one temperature
anomaly. A learned pattern carries that anomaly across the grid. A shallow
convolution reads the current forcing, broadcast spatially, plus latitude and
longitude channels, and adds a residual whose area-weighted mean is removed.
The conservation term in the loss penalizes a residual that tries to sneak a
global mean past the global head.

Training indices are chosen by the caller. This module does not decide that a
year is eligible. It does refuse a batch whose mask is entirely empty.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from forcing_atlas.errors import CalibrationError, ContractError


class SpatialEmulator(nn.Module):
    def __init__(self, n_forcing: int, lags: int, nlat: int, nlon: int) -> None:
        super().__init__()
        if n_forcing < 1 or lags < 1 or nlat < 2 or nlon < 2:
            raise ContractError("emulator dimensions must be positive and the grid at least 2 by 2")
        self.n_forcing = n_forcing
        self.lags = lags
        self.nlat = nlat
        self.nlon = nlon
        self.global_head = nn.Sequential(
            nn.Linear(n_forcing * lags, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.pattern = nn.Parameter(torch.zeros(1, 1, nlat, nlon))
        self.residual_net = nn.Sequential(
            nn.Conv2d(n_forcing + 2, 8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(
        self,
        forcing_window: torch.Tensor,
        forcing_now: torch.Tensor,
        latitude: torch.Tensor,
        longitude: torch.Tensor,
        area_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(field, global_response)`.

        `forcing_window` is (batch, lags, n_forcing). `forcing_now` is
        (batch, n_forcing). `area_weights` is (nlat, nlon) and sums to 1.
        """

        batch = forcing_window.shape[0]
        global_response = self.global_head(forcing_window.reshape(batch, -1))
        pattern = self.pattern - torch.sum(self.pattern * area_weights, dim=(-2, -1), keepdim=True)
        lat = latitude.view(1, 1, self.nlat, 1).expand(batch, 1, self.nlat, self.nlon)
        lon = longitude.view(1, 1, 1, self.nlon).expand(batch, 1, self.nlat, self.nlon)
        spatial_forcing = forcing_now[:, :, None, None].expand(batch, self.n_forcing, self.nlat, self.nlon)
        residual_in = torch.cat([spatial_forcing, lat, lon], dim=1)
        residual = self.residual_net(residual_in)
        residual = residual - torch.sum(residual * area_weights, dim=(-2, -1), keepdim=True)
        field = global_response[:, :, None, None] * (1.0 + pattern) + residual
        return field, global_response


def conservation_penalty(field: torch.Tensor, global_response: torch.Tensor, area_weights: torch.Tensor) -> torch.Tensor:
    area_mean = torch.sum(field * area_weights, dim=(-2, -1))
    return torch.mean((area_mean - global_response.squeeze(-1)) ** 2)


def masked_mse(field: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error on cells where `mask` is true. Empty masks raise.

    A field shaped (batch, 1, lat, lon) is the network output. The target is
    (batch, lat, lon). The singleton channel is removed here so the batch axis
    is not broadcast against latitude.
    """

    if field.dim() == target.dim() + 1 and field.shape[1] == 1:
        field = field.squeeze(1)
    if field.shape != target.shape or mask.shape != target.shape:
        raise CalibrationError(
            f"emulator loss shapes field {tuple(field.shape)} target {tuple(target.shape)} mask {tuple(mask.shape)}"
        )
    weights = mask.to(field.dtype)
    denom = torch.sum(weights)
    if float(denom) <= 0.0:
        raise CalibrationError("emulator batch has no finite target cells")
    delta = (field - target) ** 2
    return torch.sum(delta * weights) / denom


@dataclass(slots=True)
class TrainConfig:
    seed: int
    epochs: int = 40
    lr: float = 1.0e-2
    conservation_weight: float = 1.0
    device: str = "cpu"


@dataclass(slots=True)
class TrainResult:
    seed: int
    initial_loss: float
    final_loss: float
    parameter_delta: float
    epochs_run: int
    checkpoint: Path | None


def _seed_everything(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return generator


def new_emulator(n_forcing: int, lags: int, nlat: int, nlon: int, seed: int) -> SpatialEmulator:
    """Construct a model whose initial weights come from `seed`.

    Seeding inside the training loop is too late: the layers have already
    drawn their weights. Campaigns call this helper once per seed.
    """

    _seed_everything(seed)
    return SpatialEmulator(n_forcing, lags, nlat, nlon)


def train_emulator(
    model: SpatialEmulator,
    forcing_window: np.ndarray,
    forcing_now: np.ndarray,
    target: np.ndarray,
    train_index: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    area_weights: np.ndarray,
    config: TrainConfig,
    checkpoint: Path | None = None,
) -> TrainResult:
    """Train `model` from its current weights on `train_index` only.

    `target` may contain NaN. Those cells are masked. `train_index` is the
    only set of time rows the loss sees.
    """

    if config.epochs < 1:
        raise ContractError("epochs must be positive")
    _seed_everything(config.seed)
    device = torch.device(config.device)
    model.to(device)
    window = torch.as_tensor(forcing_window, dtype=torch.float32, device=device)
    now = torch.as_tensor(forcing_now, dtype=torch.float32, device=device)
    observed = torch.as_tensor(np.nan_to_num(target, nan=0.0), dtype=torch.float32, device=device)
    mask = torch.as_tensor(np.isfinite(target), dtype=torch.bool, device=device)
    lat = torch.as_tensor(latitude, dtype=torch.float32, device=device)
    lon = torch.as_tensor(longitude, dtype=torch.float32, device=device)
    weights = torch.as_tensor(area_weights, dtype=torch.float32, device=device)
    index = torch.as_tensor(np.asarray(train_index, dtype=np.int64), dtype=torch.long, device=device)
    if index.numel() == 0:
        raise ContractError("training index is empty")
    before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    initial = None
    final = None
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        field, global_response = model(window[index], now[index], lat, lon, weights)
        data_loss = masked_mse(field, observed[index], mask[index])
        conserve = conservation_penalty(field, global_response, weights)
        loss = data_loss + config.conservation_weight * conserve
        if not torch.isfinite(loss):
            raise CalibrationError(f"emulator loss became non-finite at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach())
        if initial is None:
            initial = value
        final = value
    after = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    delta = float(torch.linalg.norm(after - before))
    if checkpoint is not None:
        _atomic_state(checkpoint, model, config)
    return TrainResult(config.seed, float(initial), float(final), delta, config.epochs, checkpoint)


def _atomic_state(path: Path, model: SpatialEmulator, config: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {"state_dict": model.state_dict(), "seed": config.seed, "epochs": config.epochs},
        temporary,
    )
    # torch.save closes the file. Replace is atomic on the same filesystem.
    os.replace(temporary, path)


@torch.no_grad()
def predict(
    model: SpatialEmulator,
    forcing_window: np.ndarray,
    forcing_now: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    area_weights: np.ndarray,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    field, _global = model(
        torch.as_tensor(forcing_window, dtype=torch.float32, device=device),
        torch.as_tensor(forcing_now, dtype=torch.float32, device=device),
        torch.as_tensor(latitude, dtype=torch.float32, device=device),
        torch.as_tensor(longitude, dtype=torch.float32, device=device),
        torch.as_tensor(area_weights, dtype=torch.float32, device=device),
    )
    return field.squeeze(1).cpu().numpy()


def support_radius(training_forcing: np.ndarray, query_forcing: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """True where a query lies inside the training cloud's robust radius.

    The radius is the declared quantile of distances from the training mean,
    in the standardized forcing space. Points outside are an abstention, not
    a filled field.
    """

    train = np.asarray(training_forcing, dtype=np.float64)
    query = np.asarray(query_forcing, dtype=np.float64)
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    train_distance = np.linalg.norm((train - center) / scale, axis=1)
    limit = float(np.quantile(train_distance, quantile))
    query_distance = np.linalg.norm((query - center) / scale, axis=1)
    return query_distance <= limit * 1.05
