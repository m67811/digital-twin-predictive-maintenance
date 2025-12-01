"""Simple Physics-Informed Neural Network example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn


class SimplePINN(nn.Module):
    """Fully connected PINN for 2D state (temperature, mixing)."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Input concatenation: [temp, mix, u]
        z = torch.cat([x, u], dim=-1)
        return self.net(z)


@dataclass
class PINNConfig:
    """Training hyperparameters for the PINN."""
    lr: float = 1e-3
    epochs: int = 200
    lambda_physics: float = 1.0
    device: str = "cpu"


def physics_residual(x: torch.Tensor, u: torch.Tensor, params: dict) -> torch.Tensor:
    """Thermal ODE residual used in physics loss."""
    temp = x[:, 0]
    mix = x[:, 1]
    heat_input = u[:, 0]
    dtemp_dt = (
        params["heat_transfer_coeff"] * heat_input
        + 0.0
        - params["loss_coeff"] * (temp - params["ambient_temp"])
        - 0.1 * mix
    ) / params["thermal_capacity"]
    dmix_dt = (heat_input - mix) / max(params["mixing_time_constant"], 1e-3)
    return torch.stack([dtemp_dt, dmix_dt], dim=1)


def train_pinn(
    model: SimplePINN,
    x: np.ndarray,
    u: np.ndarray,
    x_next: np.ndarray,
    dt: float,
    params: dict,
    config: PINNConfig,
) -> SimplePINN:
    """Train a small PINN to fit state transitions while respecting the ODE."""
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()

    x_t = torch.tensor(x[:-1], dtype=torch.float32, device=config.device)
    u_t = torch.tensor(u[: len(x) - 1], dtype=torch.float32, device=config.device)
    x_tp1 = torch.tensor(x_next[1:], dtype=torch.float32, device=config.device)

    for _ in range(config.epochs):
        optimizer.zero_grad()
        pred_dx = model(x_t, u_t)
        x_pred = x_t + pred_dx * dt
        loss_data = mse(x_pred, x_tp1)
        phys = physics_residual(x_t, u_t, params)
        loss_phys = mse(pred_dx, phys)
        loss = loss_data + config.lambda_physics * loss_phys
        loss.backward()
        optimizer.step()
    return model
