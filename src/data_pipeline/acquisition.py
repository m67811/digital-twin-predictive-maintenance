"""Mock data acquisition: sensors -> PLC/SCADA -> edge -> message broker -> TSDB."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def generate_sensor_stream(
    outputs: np.ndarray,
    start_time: Optional[datetime] = None,
    sampling_seconds: int = 1,
    missing_rate: float = 0.02,
    outlier_rate: float = 0.005,
    noise_std: float = 0.05,
    sensor_names: Optional[Iterable[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Create a mock multi-sensor time series."""
    rng = rng or np.random.default_rng()
    start = start_time or datetime.utcnow()
    n_steps, n_sensors = outputs.shape
    times = [start + timedelta(seconds=i * sampling_seconds) for i in range(n_steps)]
    names = list(sensor_names) if sensor_names else [f"y{i}" for i in range(n_sensors)]

    noisy = outputs + rng.normal(scale=noise_std, size=outputs.shape)
    data = pd.DataFrame(noisy, columns=names)
    data.insert(0, "timestamp", times)

    data = inject_missing(data, missing_rate, rng)
    data = inject_outliers(data, outlier_rate, scale=6.0, rng=rng)
    return data


def inject_missing(df: pd.DataFrame, missing_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Randomly drop a percentage of samples."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    mask = rng.random((df.shape[0], len(numeric_cols))) < missing_rate
    df.loc[:, numeric_cols] = np.where(mask, np.nan, df[numeric_cols])
    return df


def inject_outliers(
    df: pd.DataFrame, outlier_rate: float, scale: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Inject occasional spikes as outliers."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    mask = rng.random(df[numeric_cols].shape) < outlier_rate
    spikes = rng.normal(scale=scale, size=df[numeric_cols].shape)
    df.loc[:, numeric_cols] = df[numeric_cols] + mask * spikes
    return df
