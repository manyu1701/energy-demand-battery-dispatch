"""
Data loading for the energy demand + battery dispatch project.

Three modes:
- generate_synthetic_sample(): small, fast, fully synthetic hourly series
  with daily/weekly seasonality. Used by the tests and the standalone demo.
- generate_hourly_dataset(): a full year of hourly demand, price, weather and
  renewable generation with the structure a forecasting model actually has to
  cope with — annual seasonality, holidays, autocorrelated residuals, price
  spikes and negative prices. This is what ships as the committed sample
  dataset (see scripts/generate_sample_data.py) and what the dashboard shows.
- load_sample_dataset() / load_real_data(): read a CSV back off disk, from
  data/sample/ (committed, synthetic) or data/raw/ (your own real download —
  see README for sources: Open Power System Data or EIA).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
SAMPLE_FILENAME = "hourly_energy_sample.csv"

# Fixed-date public holidays, as (month, day). Deliberately a small
# fixed-date set — the point is to give the model a "this day behaves like a
# Sunday even though it is a Tuesday" signal, not to be a complete calendar.
HOLIDAYS_MONTH_DAY = ((1, 1), (5, 1), (7, 4), (11, 1), (12, 25), (12, 26))

# Inflexible generation (nuclear / CHP / run-of-river) that stays online even
# when it is uneconomic. Once net load falls below this, supply is structurally
# long and prices go negative — see generate_hourly_dataset().
MUST_RUN_FLOOR_MW = 550.0


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


def _ar1(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw an AR(1) process x[t] = rho * x[t-1] + eps, scaled so its stationary
    standard deviation is `sigma`.

    Real load and weather residuals are strongly autocorrelated: a hot hour is
    followed by another hot hour. Independent noise would make lag features
    useless and flatter the model, so the generator produces persistence.
    """
    innovation_sd = sigma * np.sqrt(1 - rho**2)
    eps = rng.normal(0, innovation_sd, n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def _is_holiday(timestamps: pd.DatetimeIndex) -> np.ndarray:
    md = list(zip(timestamps.month.values, timestamps.day.values))
    holidays = set(HOLIDAYS_MONTH_DAY)
    return np.array([pair in holidays for pair in md])


def generate_hourly_dataset(
    n_days: int = 365,
    start: str = "2024-01-01",
    seed: int = 20240101,
) -> pd.DataFrame:
    """
    Generate a full year of hourly grid data with realistic structure.

    Columns
    -------
    demand_mw        total system load
    price_eur_mwh    day-ahead style price, can go negative
    temperature_c    dry-bulb temperature
    solar_mw         solar generation (zero at night)
    wind_mw          wind generation
    net_load_mw      demand - solar - wind ("duck curve" residual load)
    is_weekend       Saturday/Sunday flag
    is_holiday       fixed-date public holiday flag

    The structure the forecasting model has to learn:
      * annual temperature cycle plus a diurnal cycle
      * double-peak weekday load shape, damped on weekends and holidays
      * V-shaped temperature response around an ~18 C comfort point
      * AR(1) load residual, so recent hours carry information beyond the
        calendar (this is what makes lag/rolling features earn their place)
      * a slow upward demand trend (electrification)
      * price set by merit order on *net* load, not raw demand, with a convex
        scarcity term, a fuel-cost random walk, and negative prices in hours
        when renewables alone over-serve the load
    """
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24
    timestamps = pd.date_range(start, periods=n_hours, freq="h")

    hour = timestamps.hour.values.astype(float)
    doy = timestamps.dayofyear.values.astype(float)
    dow = timestamps.dayofweek.values
    is_weekend = dow >= 5
    is_holiday = _is_holiday(timestamps)
    # A holiday behaves like a weekend regardless of which weekday it lands on
    off_day = is_weekend | is_holiday

    # --- Weather -----------------------------------------------------------
    # Annual cycle peaking in late July (day ~205), plus a diurnal swing
    # peaking mid-afternoon, plus persistent synoptic weather noise.
    annual_temp = 12 + 11 * np.sin(2 * np.pi * (doy - 115) / 365.25)
    diurnal_temp = 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    temperature = annual_temp + diurnal_temp + _ar1(n_hours, rho=0.97, sigma=3.0, rng=rng)

    # --- Demand ------------------------------------------------------------
    # Double-peak weekday shape: morning ramp + evening peak. Weekends and
    # holidays keep the evening peak but lose most of the morning one.
    morning_peak = 320 * np.exp(-((hour - 8) ** 2) / 6)
    evening_peak = 430 * np.exp(-((hour - 19) ** 2) / 8)
    overnight_base = 1150
    daily_shape = overnight_base + np.where(off_day, 0.35, 1.0) * morning_peak + evening_peak
    weekend_level = np.where(off_day, 0.88, 1.0)

    heating = 26 * np.clip(18 - temperature, 0, None)
    cooling = 34 * np.clip(temperature - 18, 0, None)

    trend = 1 + 0.02 * (np.arange(n_hours) / max(n_hours - 1, 1))  # ~2%/yr electrification
    load_residual = _ar1(n_hours, rho=0.90, sigma=45.0, rng=rng)

    demand_mw = (daily_shape * weekend_level * trend + heating + cooling + load_residual)
    demand_mw = demand_mw.clip(min=200)

    # --- Renewables --------------------------------------------------------
    # Solar: a daylight bell curve whose width and height follow the season,
    # multiplied by a persistent cloud factor.
    daylight_hours = 12 + 4 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    solar_noon_offset = np.abs(hour - 13)
    solar_shape = np.clip(1 - (solar_noon_offset / (daylight_hours / 2)) ** 2, 0, None)
    seasonal_solar_capacity = 420 * (0.45 + 0.55 * np.clip(np.sin(np.pi * (doy - 80) / 365.25), 0, None))
    cloud_factor = np.clip(0.75 + _ar1(n_hours, rho=0.95, sigma=0.22, rng=rng), 0.05, 1.0)
    solar_mw = (seasonal_solar_capacity * solar_shape * cloud_factor).clip(min=0)

    # Wind: persistent, mildly anti-correlated with summer, occasionally near zero
    wind_base = 260 - 60 * np.sin(2 * np.pi * (doy - 115) / 365.25)
    wind_mw = np.clip(wind_base + _ar1(n_hours, rho=0.985, sigma=150.0, rng=rng), 0, 900)

    net_load_mw = demand_mw - solar_mw - wind_mw

    # --- Price -------------------------------------------------------------
    # Merit order on net load: a linear ramp up the supply stack, plus a
    # convex scarcity term that bites once net load runs into peaking plant.
    fuel_cost = 32 + np.cumsum(rng.normal(0, 0.06, n_hours))  # slow random walk
    scarcity_ref = np.quantile(net_load_mw, 0.85)
    scarcity = 0.00022 * np.clip(net_load_mw - scarcity_ref, 0, None) ** 2

    price = fuel_cost + 0.045 * net_load_mw + scarcity + rng.normal(0, 3.5, n_hours)

    # Negative prices. These do not need net load to go below zero: once net
    # load drops under the inflexible must-run floor (nuclear, CHP, hydro that
    # cannot cheaply shut down), those units bid negative to stay online rather
    # than curtail, and the clearing price follows them down. Roughly 2% of
    # hours here, which is the order of magnitude seen in high-renewable
    # European markets.
    surplus = np.clip(MUST_RUN_FLOOR_MW - net_load_mw, 0, None)
    price = price - 0.55 * surplus
    price = np.where(surplus > 0, np.minimum(price, -0.5), price)

    # A handful of genuine scarcity spikes (~0.15% of hours), weighted toward
    # already-tight hours — these are what a battery earns its money on.
    tight = net_load_mw > np.quantile(net_load_mw, 0.95)
    spike_draw = rng.random(n_hours) < 0.03
    spike_hours = tight & spike_draw
    price[spike_hours] += rng.uniform(120, 420, spike_hours.sum())

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": demand_mw.round(2),
            "price_eur_mwh": price.round(2),
            "temperature_c": temperature.round(2),
            "solar_mw": solar_mw.round(2),
            "wind_mw": wind_mw.round(2),
            "net_load_mw": net_load_mw.round(2),
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
        }
    ).set_index("timestamp")


def load_sample_dataset(filename: str = SAMPLE_FILENAME) -> pd.DataFrame:
    """
    Load the committed synthetic sample dataset from data/sample/.

    Unlike data/raw/, this file is checked into the repository, so a fresh
    clone can run the whole pipeline and the dashboard with no downloads.
    Regenerate it with `python scripts/generate_sample_data.py`.
    """
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate it with: python scripts/generate_sample_data.py"
        )
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")


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
