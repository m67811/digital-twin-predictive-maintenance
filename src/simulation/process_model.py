"""Nonlinear process model and simulator for a generic thermal process."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class ProcessParameters:
    """Physical parameters for the thermal process."""

    thermal_capacity: float = 5000.0
    loss_coeff: float = 15.0
    heat_transfer_coeff: float = 0.8
    ambient_temp: float = 300.0
    mixing_time_constant: float = 30.0


class ThermalProcessModel:
    """Nonlinear thermal process with two states: temperature and mixing lag."""

    def __init__(
        self,
        params: ProcessParameters | None = None,
        process_noise_std: float = 0.2,
        measurement_noise_std: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        self.params = params or ProcessParameters()
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.rng = np.random.default_rng(random_state)

    def f(
        self, x: Array, u: Array, d: Array | None, theta: Dict[str, float] | None = None
    ) -> Array:
        """Continuous-time dynamics dx/dt = f(x,u,d,theta)."""
        params = self._merge_params(theta)
        temp, mix_state = x
        heat_input = u[0]
        disturbance_heat = 0.0 if d is None else d[0]

        # Energy balance with loss to ambient and internal mixing dynamics.
        dtemp_dt = (
            params["heat_transfer_coeff"] * heat_input
            + disturbance_heat
            - params["loss_coeff"] * (temp - params["ambient_temp"])
            - 0.1 * mix_state
        ) / params["thermal_capacity"]

        dmix_dt = (heat_input - mix_state) / max(params["mixing_time_constant"], 1e-3)
        return np.array([dtemp_dt, dmix_dt], dtype=float)

    def g(self, x: Array) -> Array:
        """Measurement model y = g(x)."""
        temp, mix_state = x
        return np.array([temp, mix_state], dtype=float)

    def step(
        self,
        x: Array,
        u: Array,
        d: Array | None,
        dt: float,
        theta: Dict[str, float] | None = None,
    ) -> Tuple[Array, Array]:
        """Discrete-time propagation using forward Euler with process noise."""
        dx = self.f(x, u, d, theta)
        noise = self.rng.normal(scale=self.process_noise_std, size=x.shape)
        x_next = x + dx * dt + noise
        y = self.g(x_next) + self.rng.normal(
            scale=self.measurement_noise_std, size=x.shape
        )
        return x_next, y

    def simulate(
        self,
        x0: Array,
        inputs: Array,
        disturbances: Array | None,
        dt: float,
    ) -> Tuple[Array, Array]:
        """Simulate over a horizon."""
        n_steps = inputs.shape[0]
        states = np.zeros((n_steps + 1, len(x0)))
        outputs = np.zeros((n_steps + 1, len(x0)))
        states[0] = x0
        outputs[0] = self.g(x0)
        for k in range(n_steps):
            d_k = None if disturbances is None else disturbances[k]
            states[k + 1], outputs[k + 1] = self.step(
                states[k], inputs[k], d_k, dt
            )
        return states, outputs

    def _merge_params(self, theta: Dict[str, float] | None) -> Dict[str, float]:
        p = self.params.__dict__.copy()
        if theta:
            p.update(theta)
        return p


def simple_input_profile(n_steps: int, heat_level: float = 800.0) -> Array:
    """Generate a simple control input profile."""
    base = np.linspace(0.6, 1.0, n_steps) * heat_level
    profile = base + 30 * np.sin(np.linspace(0, 3 * math.pi, n_steps))
    return profile.reshape(-1, 1)


def random_disturbances(n_steps: int, std: float = 50.0, seed: int | None = None) -> Array:
    """Draw random disturbances to mimic feed or ambient fluctuations."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=std, size=(n_steps, 1))
