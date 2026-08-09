"""
Feature engineering for the demand/price forecasting model (Phase 1).

Three groups of features, kept separable so the backtesting harness can
ablate them independently:

  calendar  hour / day-of-week / month / weekend, plus cyclical sin-cos
            encodings so the model doesn't have to learn that hour 23 and
            hour 0 are adjacent.
  lag       the target's own value 24h and 168h (one week) ago — the single
            strongest predictor for hourly grid demand.
  rolling   trailing mean and standard deviation over the last 24h/168h,
            shifted by one hour so no feature ever sees the current value.
  weather   exogenous temperature columns, when present in the input frame.

Every feature is causal: it is computed only from information available
strictly before the timestamp being predicted, so a backtest cannot leak
the future into the training set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAGS = (24, 168)
DEFAULT_ROLLING_WINDOWS = (24, 168)

# Feature-group registry, consumed by the feature_ablation regime in
# src/backtesting/harness.py. Keys are group names; values are the prefixes
# used to match generated column names.
FEATURE_GROUPS = {
    "calendar": ("hour", "dayofweek", "month", "is_weekend", "dayofyear", "sin_", "cos_"),
    "lag": ("lag_",),
    "rolling": ("roll_",),
    "weather": ("temperature_c", "temp_"),
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/day-of-week/month/weekend columns from a DatetimeIndex."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = df.index.dayofweek >= 5

    # Cyclical encodings: hour-of-day and day-of-week wrap around, and a raw
    # integer column forces the tree to spend splits rediscovering that.
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(
    df: pd.DataFrame, target_col: str, lags: tuple[int, ...] = DEFAULT_LAGS
) -> pd.DataFrame:
    """Add `target_col` shifted back by each lag (in hours)."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Add trailing mean and standard deviation of `target_col` over each window.

    The series is shifted by one hour before rolling, so the window ends at
    t-1 and never includes the value being predicted.
    """
    df = df.copy()
    shifted = df[target_col].shift(1)
    for window in windows:
        df[f"roll_mean_{window}h"] = shifted.rolling(window, min_periods=window).mean()
        df[f"roll_std_{window}h"] = shifted.rolling(window, min_periods=window).std()
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive heating/cooling degree signals from temperature, if available.

    Grid demand responds to temperature in a V shape around a comfort point
    (~18 C): both cold and hot hours raise load. Splitting that into two
    one-sided columns gives the model a monotone signal on each side instead
    of one folded, non-monotone one.
    """
    if "temperature_c" not in df.columns:
        return df
    df = df.copy()
    df["temp_heating_deg"] = (18 - df["temperature_c"]).clip(lower=0)
    df["temp_cooling_deg"] = (df["temperature_c"] - 18).clip(lower=0)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "demand_mw",
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Build the full modelling frame for `target_col`.

    Returns the input columns plus every generated feature. The first
    max(lags + windows) rows have undefined lag/rolling values; with
    `dropna=True` (the default) they are dropped, so the returned frame is
    directly trainable.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("build_feature_matrix expects a DatetimeIndex")
    if target_col not in df.columns:
        raise KeyError(f"target column {target_col!r} not in dataframe: {list(df.columns)}")

    out = add_calendar_features(df)
    out = add_lag_features(out, target_col, lags)
    out = add_rolling_features(out, target_col, windows)
    out = add_weather_features(out)

    if dropna:
        out = out.dropna()
    return out


def feature_columns(df: pd.DataFrame, target_col: str, groups: list[str] | None = None) -> list[str]:
    """
    Select the model-input columns from a built feature matrix.

    Excludes the target itself and any other raw series that would leak the
    answer (e.g. price when forecasting demand is fine, but the target's own
    contemporaneous value is not). Pass `groups` to restrict the selection to
    specific FEATURE_GROUPS keys — this is what the ablation regime uses.
    """
    unknown = set(groups or ()) - set(FEATURE_GROUPS)
    if unknown:
        raise KeyError(f"unknown feature group(s): {sorted(unknown)}")

    prefixes: tuple[str, ...] = tuple(
        prefix
        for name in (groups if groups is not None else FEATURE_GROUPS)
        for prefix in FEATURE_GROUPS[name]
    )
    return [
        col
        for col in df.columns
        if col != target_col and col.startswith(prefixes)
    ]
