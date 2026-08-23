"""
Per-floor hall-call demand forecasting.

A gradient-boosted model over the features in `src/lifts/features.py`,
predicting how many hall calls each floor will generate in each direction in
each slot of the day. Trained with a Poisson objective because the target is a
count of button presses — most floor-slots see zero or one call, and a squared-
error objective on that distribution both predicts negative demand and gets
pulled around by the handful of busy lobby slots.

The forecast is the input the placement optimizer runs on: predict tomorrow's
demand for all 33 floors and 24 slots tonight, optimize the parking plan
against it, hand the plan to the controller before the morning peak.

Evaluated on a chronological hold-out — the last N days of the log, never
seen in training — and reported as WMAPE and bias, both aggregated to the
level the decision actually uses (calls per floor per slot), because a model
can look excellent on a building-wide total while being useless per floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.lifts.building import BuildingConfig
from src.lifts.features import FEATURE_COLUMNS, TARGET_COLUMN

try:  # LightGBM is the intended model; sklearn keeps the module importable without it
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised only where lightgbm is absent
    from sklearn.ensemble import HistGradientBoostingRegressor

    _HAS_LIGHTGBM = False


def temporal_split(featured: pd.DataFrame, test_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split: the last `test_days` of the log are held out."""
    cutoff = featured["slot_start"].max().normalize() - pd.Timedelta(days=test_days - 1)
    return featured[featured["slot_start"] < cutoff], featured[featured["slot_start"] >= cutoff]


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted MAPE — total absolute error over total actual volume.

    Unlike plain MAPE this stays defined when a floor-slot has zero calls,
    which most of them do overnight."""
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denominator)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Signed relative error — positive means the model over-forecasts."""
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return float("nan")
    return float((predicted - actual).sum() / denominator)


@dataclass
class CallDemandForecaster:
    """Gradient-boosted Poisson regression on (slot, floor, direction) call counts."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 40
    random_state: int = 42
    model: Optional[object] = None

    def fit(self, train: pd.DataFrame) -> "CallDemandForecaster":
        X = train[FEATURE_COLUMNS]
        y = train[TARGET_COLUMN]
        if _HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                objective="poisson",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
            )
        else:  # pragma: no cover
            self.model = HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.num_leaves,
                min_samples_leaf=self.min_child_samples,
                random_state=self.random_state,
            )
        self.model.fit(X, y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        return np.clip(self.model.predict(frame[FEATURE_COLUMNS]), 0.0, None)

    def evaluate(self, test: pd.DataFrame) -> dict:
        """WMAPE and bias at the floor-slot level the optimizer consumes, plus
        the same metrics on building-wide slot totals for context."""
        predicted = self.predict(test)
        actual = test[TARGET_COLUMN].to_numpy(dtype=float)

        totals = (
            test.assign(predicted=predicted)
            .groupby("slot_start", observed=True)[[TARGET_COLUMN, "predicted"]]
            .sum()
        )
        return {
            "wmape_floor_slot": wmape(actual, predicted),
            "bias_floor_slot": bias(actual, predicted),
            "wmape_building_slot": wmape(
                totals[TARGET_COLUMN].to_numpy(), totals["predicted"].to_numpy()
            ),
            "bias_building_slot": bias(
                totals[TARGET_COLUMN].to_numpy(), totals["predicted"].to_numpy()
            ),
            "n_rows": len(test),
        }

    def feature_importance(self) -> pd.Series:
        """Which features the model actually leans on, most important first."""
        if self.model is None:
            raise RuntimeError("call fit() before feature_importance()")
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:  # pragma: no cover - sklearn fallback path
            raise RuntimeError("the fitted model does not expose feature importances")
        return pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)

    def predict_demand_matrix(
        self, frame: pd.DataFrame, building: BuildingConfig
    ) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Forecast a (n_slots, n_floors) demand matrix for the optimizer, summing
        the up and down predictions for each floor.
        """
        predicted = frame.assign(predicted=self.predict(frame))
        wide = (
            predicted.groupby(["slot_start", "floor"], observed=True)["predicted"]
            .sum()
            .unstack("floor")
            .reindex(columns=range(building.n_floors), fill_value=0.0)
            .fillna(0.0)
            .sort_index()
        )
        return pd.DatetimeIndex(wide.index), wide.to_numpy(dtype=float)
