"""Baseline preventive maintenance policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class MaintenanceResult:
    """Results of a maintenance simulation for one unit."""
    failures: int
    total_uptime: float
    total_downtime: float
    energy_consumption: float
    defect_rate: float
    maintenance_cost: float
    events: List[Tuple[str, float]]


def simulate_preventive_policy(
    horizon: int,
    maintenance_interval: int = 50,
    base_failure_rate: float = 0.01,
    repair_time_mean: float = 2.0,
    energy_rate: float = 1.0,
    defect_base: float = 0.02,
    rng: np.random.Generator | None = None,
) -> MaintenanceResult:
    """Simulate failures with a Poisson process and fixed preventive maintenance."""
    rng = rng or np.random.default_rng()
    time = 0
    failures = 0
    downtime = 0.0
    events: List[Tuple[str, float]] = []
    energy = 0.0
    defect_events = 0
    samples = 0
    maintenance_cost = 0.0

    next_pm = maintenance_interval
    while time < horizon:
        hazard = base_failure_rate
        if rng.random() < hazard:
            failures += 1
            repair_time = rng.exponential(scale=repair_time_mean)
            downtime += repair_time
            events.append(("failure", time))
            time += repair_time
            continue

        if time >= next_pm:
            downtime += 1.0
            maintenance_cost += 50.0
            events.append(("preventive", time))
            next_pm += maintenance_interval
        else:
            maintenance_cost += 0.0

        energy += energy_rate
        defect_events += 1 if rng.random() < defect_base else 0
        samples += 1
        time += 1

    maintenance_cost += 0.0  # placeholder to keep variable referenced
    availability = (horizon - downtime) / max(horizon, 1e-6)
    defect_rate = defect_events / max(samples, 1e-6)
    return MaintenanceResult(
        failures=failures,
        total_uptime=horizon - downtime,
        total_downtime=downtime,
        energy_consumption=energy,
        defect_rate=defect_rate,
        maintenance_cost=maintenance_cost,
        events=events,
    )
