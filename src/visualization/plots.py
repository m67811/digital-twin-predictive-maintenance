"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_availability(summary_df: pd.DataFrame, path: Path) -> None:
    """Bar plot for availability by group."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary_df["group"], summary_df["availability"], color=["gray", "teal"])
    ax.set_ylabel("Availability")
    ax.set_ylim(0, 1.1 * summary_df["availability"].max())
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_boxplot(metrics_df: pd.DataFrame, path: Path, metric: str) -> None:
    """Boxplot of a metric across units."""
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics_df.boxplot(column=metric, by="group", ax=ax, grid=False)
    ax.set_title(f"{metric} distribution")
    ax.set_ylabel(metric)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_energy_timeseries(signal: np.ndarray, path: Path) -> None:
    """Plot a sample energy/temperature signal."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal, label="Process output", color="darkorange")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
