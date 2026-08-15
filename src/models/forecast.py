"""
Demand/price forecasting model — Phase 1.

Trains a LightGBM regressor on the calendar + lag + rolling features from
src/features/build_features.py and scores it with the metrics that matter
for a dispatch decision:

  WMAPE  weighted mean absolute percentage error — sum|err| / sum|actual|.
         Preferred over plain MAPE because hourly load/price series contain
         near-zero values that make MAPE explode.
  bias   mean signed error as a fraction of mean actual. A model can have
         good WMAPE and still systematically over- or under-forecast, which
         shows up downstream as a battery that consistently charges too
         early or too late.
  MAE / RMSE  absolute-scale errors, in the target's own units.

The price series this produces plugs straight into optimize_dispatch() in
src/optimization/battery_dispatch.py — that function takes a pd.Series of
hourly prices and does not care whether they are historical or forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "n_estimators": 300,
    "verbosity": -1,
}


@dataclass
class ForecastModel:
    """A trained LightGBM regressor plus the feature contract it was fit on."""

    booster: object
    feature_cols: list[str]
    target_col: str
    params: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict the target for `df`, which must carry the training features."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"missing feature column(s) at predict time: {missing}")
        preds = self.booster.predict(df[self.feature_cols])
        return pd.Series(np.asarray(preds), index=df.index, name=f"{self.target_col}_forecast")

    def feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Gain-based feature importance, most important first."""
        gains = self.booster.feature_importances_
        imp = (
            pd.DataFrame({"feature": self.feature_cols, "gain": gains})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        # Normalised share makes the numbers comparable across regimes, which
        # have different training-set sizes and therefore different raw gains.
        total = imp["gain"].sum()
        imp["share"] = imp["gain"] / total if total > 0 else 0.0
        return imp.head(top_n) if top_n else imp


def train_lightgbm(
    train: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    params: Optional[dict] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> ForecastModel:
    """
    Fit a LightGBM regressor.

    `sample_weight` is what the recency-weighted backtest regime uses to make
    recent hours count for more than old ones.
    """
    import lightgbm as lgb

    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("feature_cols is empty — nothing to train on")

    merged = {**DEFAULT_PARAMS, **(params or {})}
    model = lgb.LGBMRegressor(**merged)
    model.fit(train[feature_cols], train[target_col], sample_weight=sample_weight)
    return ForecastModel(
        booster=model,
        feature_cols=feature_cols,
        target_col=target_col,
        params=merged,
    )


def wmape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Weighted MAPE: sum|actual - predicted| / sum|actual|."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.abs(actual).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(actual - predicted).sum() / denom)


def bias(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """
    Mean signed error as a fraction of mean actual.

    Positive means the model over-forecasts on average.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.abs(actual).mean()
    if denom == 0:
        return float("nan")
    return float((predicted - actual).mean() / denom)


def evaluate(
    model: ForecastModel,
    test: pd.DataFrame,
    target_col: Optional[str] = None,
) -> dict[str, float]:
    """Score a trained model on a held-out frame."""
    target_col = target_col or model.target_col
    actual = test[target_col]
    predicted = model.predict(test)
    errors = predicted.to_numpy() - actual.to_numpy()
    return {
        "wmape": wmape(actual, predicted),
        "bias": bias(actual, predicted),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "n_test": int(len(test)),
    }


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    params: Optional[dict] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[ForecastModel, dict[str, float]]:
    """Convenience wrapper: fit on `train`, score on `test`."""
    model = train_lightgbm(train, target_col, feature_cols, params, sample_weight)
    return model, evaluate(model, test, target_col)


def seasonal_naive_baseline(
    train: pd.DataFrame, test: pd.DataFrame, target_col: str, season_hours: int = 168
) -> dict[str, float]:
    """
    Score the "same hour last week" baseline on the same test set.

    Any forecasting claim is meaningless without this: a gradient-boosted
    model that cannot beat seasonal-naive on hourly grid data is not adding
    anything. Reported alongside the model in every backtest regime.
    """
    history = pd.concat([train[[target_col]], test[[target_col]]]).sort_index()
    naive = history[target_col].shift(season_hours).reindex(test.index)

    valid = naive.notna()
    if not valid.any():
        return {"wmape": float("nan"), "bias": float("nan"), "n_test": 0}

    actual = test.loc[valid, target_col]
    predicted = naive[valid]
    return {
        "wmape": wmape(actual, predicted),
        "bias": bias(actual, predicted),
        "n_test": int(valid.sum()),
    }
