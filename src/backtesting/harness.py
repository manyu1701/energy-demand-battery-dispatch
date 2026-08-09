"""
Multi-regime backtesting harness — Phase 1.

A single train/test split tells you almost nothing about a demand model:
pick a lucky window and any model looks good. This harness scores the same
model under five regimes that stress different failure modes, and reports
WMAPE + bias for each, next to a seasonal-naive baseline.

  temporal          chronological split — the honest default. Train on the
                    past, predict the future.
  random            random holdout across the whole history. Optimistic by
                    construction (the model sees hours from either side of
                    every test hour); the gap between this and `temporal`
                    measures how much the model depends on interpolating
                    within known conditions.
  recency_weighted  chronological split, but training rows are exponentially
                    down-weighted by age. Tests whether the series has drifted.
  short_horizon     chronological, evaluated only on the next N hours — the
                    horizon a day-ahead dispatch decision actually needs.
  feature_ablation  chronological, refit with one feature group removed at a
                    time. Shows which features are load-bearing rather than
                    decorative.

Every regime returns (train, test) frames drawn from an already-built
feature matrix, so no regime can reintroduce lookahead that the feature
builder was careful to exclude.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_GROUPS, feature_columns
from src.models.forecast import seasonal_naive_baseline, train_and_evaluate

REGIMES = ("temporal", "random", "recency_weighted", "short_horizon", "feature_ablation")


def _check_test_size(test_size: float) -> None:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")


def temporal_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last `test_size` fraction of hours is the test set."""
    _check_test_size(test_size)
    df = df.sort_index()
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]


def random_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random holdout across the full history (optimistic — see module docstring)."""
    _check_test_size(test_size)
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < test_size
    return df.loc[~mask].sort_index(), df.loc[mask].sort_index()


def recency_weighted_split(
    df: pd.DataFrame, test_size: float = 0.2, half_life_days: float = 30.0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Chronological split plus exponential sample weights on the training rows.

    A row `d` days before the end of the training window gets weight
    0.5 ** (d / half_life_days), so with the default a 30-day-old hour counts
    half as much as the most recent one.
    """
    train, test = temporal_split(df, test_size)
    if len(train) == 0:
        return train, test, np.array([])

    age_days = (train.index.max() - train.index).total_seconds() / 86_400
    weights = np.power(0.5, age_days / half_life_days)
    return train, test, np.asarray(weights, dtype=float)


def short_horizon_split(
    df: pd.DataFrame, horizon_hours: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split where the test set is only the next `horizon_hours`.

    This is the split that matches the operational question — given
    everything known up to now, how good is the next day's forecast?
    """
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive, got {horizon_hours}")
    if horizon_hours >= len(df):
        raise ValueError(
            f"horizon_hours ({horizon_hours}) must be smaller than the dataset ({len(df)} rows)"
        )
    df = df.sort_index()
    return df.iloc[:-horizon_hours], df.iloc[-horizon_hours:]


def feature_ablation_splits(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    groups: Optional[Sequence[str]] = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """
    One entry per ablation: the same chronological split, with a reduced
    feature list that drops one group.

    Returns {label: (train, test, feature_cols)}, including an "all_features"
    entry as the reference point.
    """
    groups = list(groups if groups is not None else FEATURE_GROUPS)
    train, test = temporal_split(df, test_size)

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]] = {
        "all_features": (train, test, feature_columns(df, target_col, groups))
    }
    for dropped in groups:
        kept = [g for g in groups if g != dropped]
        cols = feature_columns(df, target_col, kept)
        if not cols:
            continue  # nothing left to train on; skip rather than raise
        out[f"without_{dropped}"] = (train, test, cols)
    return out


def evaluate_all_regimes(
    df: pd.DataFrame,
    target_col: str = "demand_mw",
    feature_cols: Optional[Sequence[str]] = None,
    test_size: float = 0.2,
    horizon_hours: int = 24,
    half_life_days: float = 30.0,
    seed: int = 42,
    params: Optional[dict] = None,
    trainer: Callable = train_and_evaluate,
) -> pd.DataFrame:
    """
    Run every regime and return one row per regime.

    Columns: regime, wmape, bias, mae, rmse, n_train, n_test, plus
    baseline_wmape and skill_vs_naive (1 - model_wmape / baseline_wmape;
    positive means the model beats "same hour last week").
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("evaluate_all_regimes expects a feature matrix with a DatetimeIndex")

    cols = list(feature_cols) if feature_cols is not None else feature_columns(df, target_col)
    rows: list[dict] = []

    def _score(label: str, train: pd.DataFrame, test: pd.DataFrame, use_cols: list[str], weight=None):
        _, metrics = trainer(train, test, target_col, use_cols, params, weight)
        baseline = seasonal_naive_baseline(train, test, target_col)
        skill = (
            1 - metrics["wmape"] / baseline["wmape"]
            if baseline["wmape"] and not np.isnan(baseline["wmape"])
            else float("nan")
        )
        rows.append(
            {
                "regime": label,
                "wmape": metrics["wmape"],
                "bias": metrics["bias"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "n_train": len(train),
                "n_test": metrics["n_test"],
                "baseline_wmape": baseline["wmape"],
                "skill_vs_naive": skill,
                "n_features": len(use_cols),
            }
        )

    train, test = temporal_split(df, test_size)
    _score("temporal", train, test, cols)

    train, test = random_split(df, test_size, seed)
    _score("random", train, test, cols)

    train, test, weights = recency_weighted_split(df, test_size, half_life_days)
    _score("recency_weighted", train, test, cols, weights)

    train, test = short_horizon_split(df, horizon_hours)
    _score(f"short_horizon_{horizon_hours}h", train, test, cols)

    for label, (tr, te, ablated_cols) in feature_ablation_splits(df, target_col, test_size).items():
        if label == "all_features":
            continue  # identical to the temporal regime already reported above
        _score(f"ablation_{label}", tr, te, ablated_cols)

    return pd.DataFrame(rows)
