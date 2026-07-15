"""
Feature engineering for the demand/price forecasting model (Phase 1).

Currently provides the calendar features needed for a first LightGBM
baseline. Planned additions for Phase 1 (see README roadmap):
  - lag features (24h, 168h)
  - rolling means / rolling volatility
  - holiday calendar join
"""
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/day-of-week/month/weekend columns from a DatetimeIndex."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek >= 5
    return df
