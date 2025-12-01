"""Preprocessing utilities: resampling, interpolation, filtering, normalization."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import signal


def resample_timeseries(df: pd.DataFrame, freq: str = "1S") -> pd.DataFrame:
    """Resample to a common rate using forward-fill for timestamps."""
    resampled = (
        df.set_index("timestamp")
        .resample(freq)
        .mean()
        .interpolate(method="time")
        .reset_index()
    )
    return resampled


def interpolate_missing(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    """Fill missing values via interpolation."""
    numeric = df.select_dtypes(include=["float", "int"])
    filled = numeric.interpolate(method=method).fillna(method="bfill").fillna(method="ffill")
    df.update(filled)
    return df


def moving_average_filter(df: pd.DataFrame, window: int = 5, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Apply a causal moving average filter."""
    df = df.copy()
    cols = list(columns) if columns else df.select_dtypes(include=["float", "int"]).columns
    for col in cols:
        df[col] = df[col].rolling(window=window, min_periods=1).mean()
    return df


def chebyshev_filter(
    df: pd.DataFrame, cutoff_hz: float = 0.05, order: int = 3, fs: float = 1.0
) -> pd.DataFrame:
    """Low-pass Chebyshev filter for noise suppression."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    b, a = signal.cheby1(order, 0.5, cutoff_hz / (0.5 * fs))
    for col in numeric_cols:
        df[col] = signal.filtfilt(b, a, df[col].to_numpy())
    return df


def standardize(df: pd.DataFrame, columns: Iterable[str] | None = None) -> Tuple[pd.DataFrame, dict]:
    """Z-score normalization."""
    df = df.copy()
    cols = list(columns) if columns else df.select_dtypes(include=["float", "int"]).columns
    stats = {}
    for col in cols:
        mean = df[col].mean()
        std = df[col].std() or 1.0
        df[col] = (df[col] - mean) / std
        stats[col] = {"mean": mean, "std": std}
    return df, stats
