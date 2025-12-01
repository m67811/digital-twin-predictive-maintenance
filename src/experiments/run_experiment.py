"""CLI entry point to run simulation + analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.reporting import save_text_summary
from src.analysis.statistics import compare_groups
from src.experiments.experiment_design import ExperimentConfig, ExperimentDesign
from src.models.nn_residual import ResidualRNN, ResidualTrainingConfig, train_residual_model
from src.models.pinn import PINNConfig, SimplePINN, train_pinn
from src.simulation.process_model import ProcessParameters
from src.visualization.plots import plot_availability, plot_energy_timeseries, plot_metric_boxplot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run digital twin experiment.")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--n_control", type=int, default=20)
    parser.add_argument("--n_experiment", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="figures")
    return parser.parse_args()


def ensure_dirs(results_dir: Path, figures_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        horizon=args.horizon,
        n_control=args.n_control,
        n_experiment=args.n_experiment,
        seed=args.seed,
    )
    torch.manual_seed(cfg.seed)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dirs(results_dir, figures_dir)

    design = ExperimentDesign(cfg)
    states, outputs, sensor_df, feature_df, inputs = design.simulate_process()

    # Train residual model on synthetic residuals (measurement - true state).
    residuals = outputs - states
    model = ResidualRNN(input_dim=outputs.shape[1], hidden_dim=32, rnn_type="lstm")
    train_residual_model(
        model,
        y=outputs.astype(np.float32),
        residuals=residuals.astype(np.float32),
        config=ResidualTrainingConfig(epochs=2, seq_len=5, batch_size=16),
    )

    # Train a tiny PINN on the same synthetic data.
    pinn = SimplePINN()
    train_pinn(
        pinn,
        x=states,
        u=inputs,
        x_next=states,
        dt=cfg.dt,
        params=ProcessParameters().__dict__,
        config=PINNConfig(epochs=50),
    )

    pop_summary, unit_records = design.run_population()

    # Prepare tables
    metrics_df = pd.DataFrame(unit_records)
    metrics_df.to_csv(results_dir / "unit_metrics.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"group": "control", **pop_summary["control"].__dict__},
            {"group": "experiment", **pop_summary["experiment"].__dict__},
        ]
    )
    summary_df.to_csv(results_dir / "metrics_summary.csv", index=False)

    # Statistical comparison on MTBF and availability
    stats_results = compare_groups(metrics_df)
    with open(results_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2)

    # Save raw sensor and feature data
    sensor_df.to_csv(results_dir / "sensor_stream.csv", index=False)
    feature_df.to_csv(results_dir / "features.csv", index=False)

    # Generate plots
    plot_availability(summary_df, figures_dir / "availability.png")
    plot_metric_boxplot(metrics_df, figures_dir / "mtbf_boxplot.png", metric="mtbf")
    plot_energy_timeseries(outputs[:, 0], figures_dir / "energy_series.png")

    save_text_summary(pop_summary, stats_results, results_dir / "summary.txt")
    print(f"Experiment complete. Results saved to {results_dir}, figures to {figures_dir}.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
