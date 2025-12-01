"""AI-assisted maintenance policy using digital twin residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class AIResult:
    """Results of AI-assisted maintenance simulation for one unit."""
    failures: int
    total_uptime: float
    total_downtime: float
    energy_consumption: float
    defect_rate: float
    maintenance_cost: float
    events: List[Tuple[str, float, float]]


def simulate_ai_policy(
    horizon: int,
    base_failure_rate: float = 0.01,
    repair_time_mean: float = 1.5,
    energy_rate: float = 0.9,
    defect_base: float = 0.015,
    anomaly_threshold: float = 1.5,
    rng: np.random.Generator | None = None,
) -> AIResult:
    """Simulate condition-based maintenance triggered by anomaly score."""
    rng = rng or np.random.default_rng()
    time = 0
    failures = 0
    downtime = 0.0
    energy = 0.0
    maintenance_cost = 0.0
    events: List[Tuple[str, float, float]] = []
    defect_events = 0
    samples = 0

    while time < horizon:
        # Anomaly score derived from hypothetical residual magnitude.
        anomaly_score = abs(rng.normal(scale=0.7))
        hazard = base_failure_rate * (1.0 + 0.5 * anomaly_score)
        if anomaly_score > anomaly_threshold:
            # Trigger predictive maintenance earlier
            downtime += 0.5
            maintenance_cost += 40.0
            events.append(("predictive", time, anomaly_score))
            time += 0.5
            continue

        if rng.random() < hazard:
            failures += 1
            repair_time = rng.exponential(scale=repair_time_mean)
            downtime += repair_time
            events.append(("failure", time, anomaly_score))
            time += repair_time
            continue

        energy += energy_rate
        defect_events += 1 if rng.random() < defect_base else 0
        samples += 1
        time += 1

    availability = (horizon - downtime) / max(horizon, 1e-6)
    defect_rate = defect_events / max(samples, 1e-6)
    return AIResult(
        failures=failures,
        total_uptime=horizon - downtime,
        total_downtime=downtime,
        energy_consumption=energy,
        defect_rate=defect_rate,
        maintenance_cost=maintenance_cost,
        events=events,
    )
