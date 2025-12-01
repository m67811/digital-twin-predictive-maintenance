"""Residual neural network models (LSTM/GRU) for hybrid modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ResidualRNN(nn.Module):
    """Sequence model that predicts residuals from recent measurements."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        rnn_type: Literal["lstm", "gru"] = "lstm",
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class ResidualDataset(Dataset):
    """Construct sliding windows for residual learning."""

    def __init__(self, y: np.ndarray, residuals: np.ndarray, seq_len: int = 10) -> None:
        assert len(y) == len(residuals)
        self.y = y.astype(np.float32)
        self.residuals = residuals.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.y) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.y[idx : idx + self.seq_len]
        target = self.residuals[idx + self.seq_len - 1]
        return torch.from_numpy(window), torch.from_numpy(target)


@dataclass
class ResidualTrainingConfig:
    """Hyperparameters for residual model training."""
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    seq_len: int = 10
    device: str = "cpu"


def train_residual_model(
    model: ResidualRNN,
    y: np.ndarray,
    residuals: np.ndarray,
    config: ResidualTrainingConfig,
) -> ResidualRNN:
    """Train residual model on sliding windows."""
    dataset = ResidualDataset(y, residuals, seq_len=config.seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    model.to(config.device)

    model.train()
    for _ in range(config.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(config.device), yb.to(config.device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
