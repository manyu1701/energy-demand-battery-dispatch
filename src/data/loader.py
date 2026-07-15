"""
Data loading for the energy demand + battery dispatch project.

Two modes:
- generate_synthetic_sample(): fully synthetic hourly demand/price/temperature
  data with realistic daily/weekly seasonality, used for local development
  and tests before real data is wired in.
- load_real_data(): loads a real hourly load/price CSV placed in data/raw/
  (see README for dataset sources — Open Power System Data or EIA).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def generate_synthetic_sample(n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generate an hourly time series of demand (MW), price (currency/MWh),
    and temperature (deg C) with a double-peak daily shape, weekday/weekend
    effect, and a temperature-driven heating/cooling load component.
    """
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")

    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values
    is_weekend = dow >= 5

    # Double-peak daily demand shape (morning + evening commute/usage), damped on weekends
    daily_shape = (
        30 * np.exp(-((hour - 8) ** 2) / 8)
        + 45 * np.exp(-((hour - 19) ** 2) / 10)
        + 60
    )
    weekend_factor = np.where(is_weekend, 0.85, 1.0)

    temperature = (
        15
        + 8 * np.sin(2 * np.pi * (timestamps.dayofyear.values / 365))
        + rng.normal(0, 2, n_hours)
    )
    # Both heating and cooling raise demand relative to an ~18C comfort point
    temp_effect = 0.6 * np.abs(temperature - 18)

    noise = rng.normal(0, 4, n_hours)
    demand_mw = (daily_shape * weekend_factor + temp_effect + noise).clip(min=10)

    # Simple merit-order proxy: price rises with demand, plus independent noise
    price = 20 + 0.9 * (demand_mw - demand_mw.min()) + rng.normal(0, 5, n_hours)
    price = price.clip(min=5)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": demand_mw.round(2),
            "price_eur_mwh": price.round(2),
            "temperature_c": temperature.round(1),
            "is_weekend": is_weekend,
        }
    ).set_index("timestamp")


def load_real_data(filename: str = "load_and_price.csv") -> pd.DataFrame:
    """
    Load a real hourly load/price CSV from data/raw/.

    Get real data from one of:
      - Open Power System Data: https://data.open-power-system-data.org/time_series/
        (no account needed, direct CSV download, CC-BY licensed)
      - EIA hourly demand: https://api.eia.gov (free API key required)

    Expected columns after your own light renaming: timestamp, demand_mw,
    price_eur_mwh (rename to match whatever units/currency your source uses).
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download real data (see docstring for sources) "
            f"and save it as data/raw/{filename}, or use generate_synthetic_sample() "
            "for local development."
        )
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
