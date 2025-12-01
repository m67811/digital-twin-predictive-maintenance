"""Experiment design for control vs. AI-assisted maintenance and modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_pipeline import acquisition, preprocessing
from src.data_pipeline.features import extract_features
from src.maintenance.policy_ai import AIResult, simulate_ai_policy
from src.maintenance.policy_control import MaintenanceResult, simulate_preventive_policy
from src.maintenance.reliability_metrics import ReliabilitySummary, summarize_population
from src.simulation.process_model import ThermalProcessModel, random_disturbances, simple_input_profile


@dataclass
class ExperimentConfig:
    """Configuration for population experiments."""
    horizon: int = 200
    dt: float = 1.0
    seed: int = 42
    n_control: int = 20
    n_experiment: int = 20
    process_noise_std: float = 0.2
    measurement_noise_std: float = 0.5


class ExperimentDesign:
    """Creates populations and runs simulations."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def simulate_process(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Run one process simulation and return raw and feature data."""
        process = ThermalProcessModel(
            process_noise_std=self.cfg.process_noise_std,
            measurement_noise_std=self.cfg.measurement_noise_std,
            random_state=self.cfg.seed,
        )
        inputs = simple_input_profile(self.cfg.horizon)
        disturbances = random_disturbances(self.cfg.horizon, seed=self.cfg.seed)
        x0 = np.array([500.0, 0.0])
        states, outputs = process.simulate(x0, inputs, disturbances, self.cfg.dt)
        sensor_df = acquisition.generate_sensor_stream(outputs, rng=self.rng)
        sensor_df = preprocessing.interpolate_missing(sensor_df)
        sensor_df = preprocessing.moving_average_filter(sensor_df, window=5)
        sensor_df, _ = preprocessing.standardize(sensor_df)
        feature_df = extract_features(sensor_df, window=20, fs=1.0 / self.cfg.dt)
        return states, outputs, sensor_df, feature_df, inputs

    def run_population(self) -> Tuple[Dict[str, ReliabilitySummary], List[dict]]:
        """Simulate control and AI groups and capture unit-level metrics."""
        control_results: List[MaintenanceResult] = []
        ai_results: List[AIResult] = []
        unit_records: List[dict] = []
        for i in range(self.cfg.n_control):
            res = simulate_preventive_policy(horizon=self.cfg.horizon, rng=self.rng)
            control_results.append(res)
            unit_records.append(
                {
                    "group": "control",
                    "unit": f"c{i}",
                    "failures": res.failures,
                    "uptime": res.total_uptime,
                    "downtime": res.total_downtime,
                    "mtbf": res.total_uptime / res.failures if res.failures else res.total_uptime,
                    "mttr": res.total_downtime / res.failures if res.failures else 0.0,
                    "availability": (res.total_uptime)
                    / max(res.total_uptime + res.total_downtime, 1e-6),
                    "energy": res.energy_consumption,
                    "defect_rate": res.defect_rate,
                    "maintenance_cost": res.maintenance_cost,
                }
            )
        for i in range(self.cfg.n_experiment):
            res = simulate_ai_policy(horizon=self.cfg.horizon, rng=self.rng)
            ai_results.append(res)
            unit_records.append(
                {
                    "group": "experiment",
                    "unit": f"e{i}",
                    "failures": res.failures,
                    "uptime": res.total_uptime,
                    "downtime": res.total_downtime,
                    "mtbf": res.total_uptime / res.failures if res.failures else res.total_uptime,
                    "mttr": res.total_downtime / res.failures if res.failures else 0.0,
                    "availability": (res.total_uptime)
                    / max(res.total_uptime + res.total_downtime, 1e-6),
                    "energy": res.energy_consumption,
                    "defect_rate": res.defect_rate,
                    "maintenance_cost": res.maintenance_cost,
                }
            )

        return (
            {
                "control": summarize_population(control_results),
                "experiment": summarize_population(ai_results),
            },
            unit_records,
        )
