"""Statistical analysis for control vs. experimental groups."""

from __future__ import annotations

from typing import Dict

import math

import numpy as np
import pandas as pd
from scipy import stats


def normality_test(values: np.ndarray) -> Dict[str, float]:
    """Shapiro-Wilk normality test."""
    stat, p = stats.shapiro(values) if len(values) >= 3 else (np.nan, np.nan)
    return {"stat": float(stat), "p_value": float(p)}


def compare_groups(metrics_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Perform group comparison for MTBF, availability, and energy."""
    results: Dict[str, Dict[str, float]] = {}
    for metric in ["mtbf", "availability", "energy"]:
        control = metrics_df.loc[metrics_df["group"] == "control", metric].to_numpy()
        exp = metrics_df.loc[metrics_df["group"] == "experiment", metric].to_numpy()
        ctrl_norm = normality_test(control)
        exp_norm = normality_test(exp)
        if (ctrl_norm["p_value"] > 0.05) and (exp_norm["p_value"] > 0.05):
            stat, p = stats.ttest_ind(control, exp, equal_var=False)
            test_name = "t_test"
        else:
            stat, p = stats.mannwhitneyu(control, exp, alternative="two-sided")
            test_name = "mann_whitney"
        ci_low, ci_high = confidence_interval(exp, control)
        results[metric] = {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(p),
            "control_mean": float(np.mean(control)),
            "experiment_mean": float(np.mean(exp)),
            "confidence_interval_low": float(ci_low),
            "confidence_interval_high": float(ci_high),
        }
    return results


def confidence_interval(exp: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """CI for difference of means (independent samples)."""
    mean_diff = float(np.mean(exp) - np.mean(control))
    var = np.var(exp, ddof=1) / max(len(exp), 1) + np.var(control, ddof=1) / max(len(control), 1)
    std = math.sqrt(var) if var > 0 else 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    half = z * std
    return mean_diff - half, mean_diff + half
