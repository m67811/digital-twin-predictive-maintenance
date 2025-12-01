"""Fault and degradation scenario helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class FaultProfile:
    """Defines how a fault evolves over time."""

    start_step: int
    duration: int
    magnitude: float

    def window(self, n_steps: int) -> Array:
        mask = np.zeros(n_steps)
        end = min(self.start_step + self.duration, n_steps)
        mask[self.start_step:end] = 1.0
        return mask


def sensor_drift(signal: Array, profile: FaultProfile) -> Array:
    """Adds a drifting bias to a sensor signal."""
    n_steps = signal.shape[0]
    drift = profile.window(n_steps) * profile.magnitude
    return signal + drift.reshape(-1, 1)


def efficiency_loss(inputs: Array, profile: FaultProfile) -> Array:
    """Reduces effective input power to mimic fouling or blockage."""
    n_steps = inputs.shape[0]
    loss_factor = 1.0 - 0.2 * profile.window(n_steps)
    return inputs * loss_factor.reshape(-1, 1)


def intermittent_spikes(signal: Array, probability: float = 0.01, scale: float = 5.0) -> Array:
    """Injects random spikes to emulate outliers."""
    rng = np.random.default_rng()
    spikes = rng.random(signal.shape[0]) < probability
    noise = rng.normal(scale=scale, size=signal.shape[0])
    return signal + (spikes * noise).reshape(-1, 1)
