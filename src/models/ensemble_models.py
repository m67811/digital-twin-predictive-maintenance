"""Classical ensemble models for baseline comparisons."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Fit a random forest regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    return model


def train_gradient_boosting(X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
    """Fit a gradient boosting regressor."""
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X, y)
    return model


def train_xgboost(X: np.ndarray, y: np.ndarray) -> Any:
    """Fit an XGBoost regressor (optional dependency)."""
    if XGBRegressor is None:
        raise ImportError("xgboost not installed.")
    model = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8
    )
    model.fit(X, y)
    return model
