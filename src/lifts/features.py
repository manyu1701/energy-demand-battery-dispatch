"""
Feature engineering for the hall-call demand model.

One row per (slot, floor, direction). The features fall into three groups:

  calendar   where in the day and week the slot sits, encoded cyclically so
             23:00 and 00:00 are neighbours rather than opposite extremes
  spatial    which floor this is, how far it sits from the entry level, and
             how many people live on it
  lag        what this same floor and direction did at this same time
             yesterday, a week ago, and on average over the last four weeks

The lag features carry most of the signal: a floor's traffic is strongly
self-similar day to day, which is what makes a per-floor forecast worth having
over a single building-wide average curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.lifts.building import BuildingConfig

FEATURE_COLUMNS = [
    "slot_of_day",
    "hour",
    "dayofweek",
    "is_weekend",
    "is_holiday",
    "slot_sin",
    "slot_cos",
    "dow_sin",
    "dow_cos",
    "floor",
    "floor_fraction",
    "is_entry_floor",
    "distance_from_entry",
    "population",
    "is_up",
    "lag_1d",
    "lag_7d",
    "roll_mean_28d",
]

TARGET_COLUMN = "calls"


def build_features(
    slots: pd.DataFrame, building: BuildingConfig, slot_minutes: int = 60
) -> pd.DataFrame:
    """
    Add model features to a slot table from `aggregate_to_slots()`.

    Rows whose lag features reach back before the start of the log carry NaN
    there; LightGBM handles missing values natively, so they are left in place
    rather than dropped — dropping them would throw away the first four weeks
    of every history.
    """
    slots_per_day = 1440 // slot_minutes
    df = slots.sort_values(["floor", "direction", "slot_start"]).copy()

    ts = df["slot_start"]
    df["slot_of_day"] = (ts.dt.hour * 60 + ts.dt.minute) // slot_minutes
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0
    df["is_holiday"] = df["is_holiday"].astype(int)

    df["slot_sin"] = np.sin(2 * np.pi * df["slot_of_day"] / slots_per_day)
    df["slot_cos"] = np.cos(2 * np.pi * df["slot_of_day"] / slots_per_day)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    entry = np.array(list(building.entry_floors))
    population = building.population()
    df["floor_fraction"] = df["floor"] / max(building.n_floors - 1, 1)
    df["is_entry_floor"] = df["floor"].isin(entry).astype(int)
    df["distance_from_entry"] = df["floor"].map(
        lambda f: int(np.min(np.abs(entry - f)))
    )
    df["population"] = df["floor"].map(lambda f: population[f])
    df["is_up"] = (df["direction"] == "up").astype(int)

    by_series = df.groupby(["floor", "direction"], observed=True)[TARGET_COLUMN]
    df["lag_1d"] = by_series.shift(slots_per_day)
    df["lag_7d"] = by_series.shift(7 * slots_per_day)
    # Mean of this slot-of-day over the previous 28 days: shift first so the
    # window can only ever see days strictly before the row being described.
    df["roll_mean_28d"] = (
        by_series.shift(slots_per_day)
        .groupby([df["floor"], df["direction"], df["slot_of_day"]], observed=True)
        .transform(lambda s: s.rolling(28, min_periods=3).mean())
    )

    return df.sort_values(["slot_start", "floor", "direction"], ignore_index=True)


def build_future_frame(
    history: pd.DataFrame,
    date: str | pd.Timestamp,
    building: BuildingConfig,
    slot_minutes: int = 60,
    holidays: tuple = (),
) -> pd.DataFrame:
    """
    Build the feature rows for a day that hasn't happened yet.

    The target day's own `calls` are unknown (and set to NaN), but its lag
    features are all drawn from history, so the whole day can be forecast in
    one shot the night before — which is what a parking schedule needs, since
    it must be handed to the controller in advance.
    """
    date = pd.Timestamp(date).normalize()
    slots = pd.date_range(date, date + pd.Timedelta(days=1), freq=f"{slot_minutes}min", inclusive="left")
    holiday_dates = {pd.Timestamp(d).date() for d in holidays}
    future = pd.MultiIndex.from_product(
        [slots, range(building.n_floors), ("up", "down")],
        names=["slot_start", "floor", "direction"],
    ).to_frame(index=False)
    future[TARGET_COLUMN] = np.nan
    future["is_holiday"] = future["slot_start"].dt.date.isin(holiday_dates)

    combined = pd.concat([history, future], ignore_index=True)
    featured = build_features(combined, building, slot_minutes)
    return featured[featured["slot_start"].isin(slots)].reset_index(drop=True)
