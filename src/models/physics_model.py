"""Physics-based model wrapper."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from src.simulation.process_model import ThermalProcessModel


class PhysicsModel:
    """Wrapper around the process model that provides prediction utilities."""

    def __init__(self, process: ThermalProcessModel) -> None:
        self.process = process

    def predict_next(
        self, x: np.ndarray, u: np.ndarray, d: np.ndarray | None, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state and output."""
        return self.process.step(x, u, d, dt)

    def rollout(
        self, x0: np.ndarray, inputs: np.ndarray, disturbances: np.ndarray | None, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the simulator in open-loop."""
        return self.process.simulate(x0, inputs, disturbances, dt)

    def update_params(self, theta: Dict[str, float]) -> None:
        """Update process parameters on the fly."""
        self.process.params = self.process.params.__class__(**self.process._merge_params(theta))

    def hybrid_predict(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray | None,
        dt: float,
        residual_model,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend physics prediction with residual correction."""
        physics_pred, _ = self.predict_next(x, u, d, dt)
        with torch.no_grad():
            residual_input = torch.tensor(x[None, None, :], dtype=torch.float32)
            residual = residual_model(residual_input).cpu().numpy().squeeze()
        return alpha * physics_pred + (1 - alpha) * (physics_pred + residual)
