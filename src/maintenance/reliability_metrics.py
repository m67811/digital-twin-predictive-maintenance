"""Reliability metrics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class ReliabilitySummary:
    """Aggregate reliability metrics for a population."""
    mtbf: float
    mttr: float
    availability: float
    energy: float
    defect_rate: float
    maintenance_cost: float


def compute_mtbf(total_uptime: float, failures: int) -> float:
    """Mean time between failures."""
    return total_uptime / failures if failures > 0 else total_uptime


def compute_mttr(total_downtime: float, failures: int) -> float:
    """Mean time to repair."""
    return total_downtime / failures if failures > 0 else 0.0


def compute_availability(mtbf: float, mttr: float) -> float:
    """Steady-state availability."""
    return mtbf / (mtbf + mttr) if (mtbf + mttr) > 0 else 1.0


def summarize_population(results: Iterable) -> ReliabilitySummary:
    """Aggregate metrics for a population of simulated units."""
    total_uptime = sum(r.total_uptime for r in results)
    total_downtime = sum(r.total_downtime for r in results)
    total_failures = sum(r.failures for r in results)
    energy = sum(r.energy_consumption for r in results)
    defect_rate = np.mean([r.defect_rate for r in results]) if results else 0.0
    cost = sum(r.maintenance_cost for r in results)

    mtbf = compute_mtbf(total_uptime, total_failures)
    mttr = compute_mttr(total_downtime, total_failures)
    availability = compute_availability(mtbf, mttr)
    return ReliabilitySummary(
        mtbf=mtbf, mttr=mttr, availability=availability, energy=energy, defect_rate=defect_rate, maintenance_cost=cost
    )
