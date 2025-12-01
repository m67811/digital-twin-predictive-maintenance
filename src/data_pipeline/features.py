"""Feature extraction utilities."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def crest_factor(x: np.ndarray) -> float:
    denom = rms(x) or 1e-6
    return float(np.max(np.abs(x)) / denom)


def time_domain_features(x: np.ndarray) -> Dict[str, float]:
    """Compute simple statistics."""
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "var": float(np.var(x)),
        "rms": rms(x),
        "kurtosis": float(np.mean(((x - np.mean(x)) / (np.std(x) or 1e-6)) ** 4)),
        "crest_factor": crest_factor(x),
    }


def frequency_domain_features(x: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """FFT-based features: spectral centroid and dominant frequency."""
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x))
    total = np.sum(spectrum) or 1e-6
    centroid = float(np.sum(freqs * spectrum) / total)
    dom_idx = int(np.argmax(spectrum))
    dom_freq = float(freqs[dom_idx]) if dom_idx < len(freqs) else 0.0
    spectral_energy = float(np.sum(np.square(spectrum)))
    return {
        "spectral_centroid": centroid,
        "dominant_freq": dom_freq,
        "spectral_energy": spectral_energy,
    }


def extract_features(df: pd.DataFrame, window: int = 20, fs: float = 1.0) -> pd.DataFrame:
    """Slide over the time series and compute features per window."""
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    feature_rows = []
    for start in range(0, len(df) - window + 1, window):
        end = start + window
        row = {"timestamp": df["timestamp"].iloc[end - 1]}
        for col in numeric_cols:
            segment = df[col].iloc[start:end].to_numpy()
            feats = time_domain_features(segment)
            feats.update({f: v for f, v in frequency_domain_features(segment, fs).items()})
            row.update({f"{col}_{k}": v for k, v in feats.items()})
        feature_rows.append(row)
    return pd.DataFrame(feature_rows)
