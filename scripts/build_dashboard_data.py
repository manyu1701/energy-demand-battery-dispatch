"""
Run the whole pipeline and export the JSON artifacts the Node dashboard reads.

    data -> features -> forecast -> backtest -> dispatch -> data/dashboard/*.json

The dashboard never runs Python at request time: this script is the only place
the model is fitted, so the frontend stays a thin viewer over committed
artifacts and a fresh clone can `npm start` with no Python environment at all.

Re-run this after changing the model, the features, or the sample dataset:
    python scripts/build_dashboard_data.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.harness import evaluate_all_regimes, temporal_split
from src.data.loader import load_sample_dataset
from src.features.build_features import build_feature_matrix, feature_columns
from src.models.forecast import evaluate, seasonal_naive_baseline, train_lightgbm
from src.optimization.battery_dispatch import (
    BatteryConfig,
    fixed_rule_dispatch,
    rolling_horizon_dispatch,
    settle_at_actual_prices,
    summarize_dispatch,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dashboard"
TEST_SIZE = 0.2
DEMO_WEEK_HOURS = 168

# The battery the headline numbers describe: a 20 MWh / 5 MW system, i.e. a
# 4-hour duration unit, which is the most common grid-scale configuration.
DEFAULT_BATTERY = BatteryConfig(
    capacity_mwh=20.0,
    max_power_mw=5.0,
    round_trip_efficiency=0.88,
    initial_soc_mwh=10.0,
)

# Sensitivity sweep: same energy capacity, different power ratings (i.e.
# different storage durations), since duration is the design choice that
# decides whether a battery can chase daily spreads or only short peaks.
SWEEP_CONFIGS = [
    ("1h duration (20 MWh / 20 MW)", BatteryConfig(20.0, 20.0, 0.88, 10.0)),
    ("2h duration (20 MWh / 10 MW)", BatteryConfig(20.0, 10.0, 0.88, 10.0)),
    ("4h duration (20 MWh / 5 MW)", BatteryConfig(20.0, 5.0, 0.88, 10.0)),
    ("8h duration (20 MWh / 2.5 MW)", BatteryConfig(20.0, 2.5, 0.88, 10.0)),
]


def _clean(value):
    """Make numpy/pandas values JSON-safe; NaN and inf become null."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else round(value, 6)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _write(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(_clean(payload), indent=1) + "\n")
    print(f"  wrote {path.relative_to(path.parents[2])} ({path.stat().st_size / 1024:.0f} KB)")
    return path


def _timestamps(index: pd.DatetimeIndex) -> list[str]:
    return [t.isoformat() for t in index]


def _span(index: pd.DatetimeIndex) -> dict:
    return {"start": index.min().isoformat(), "end": index.max().isoformat(), "hours": len(index)}


def build_forecast_artifact(feature_matrix: pd.DataFrame, target: str, unit: str) -> dict:
    """Train on the chronological train split, score and export the test window."""
    cols = feature_columns(feature_matrix, target)
    train, test = temporal_split(feature_matrix, TEST_SIZE)

    model = train_lightgbm(train, target, cols)
    predicted = model.predict(test)
    metrics = evaluate(model, test, target)
    baseline = seasonal_naive_baseline(train, test, target)

    residuals = predicted.to_numpy() - test[target].to_numpy()
    # Error by hour-of-day exposes the failure mode that matters for dispatch:
    # a model can look fine on average and still be worst at the evening peak,
    # which is exactly when the battery is deciding whether to discharge.
    by_hour = (
        pd.DataFrame({"hour": test.index.hour, "abs_err": np.abs(residuals)})
        .groupby("hour")["abs_err"]
        .mean()
    )

    return {
        "target": target,
        "unit": unit,
        "metrics": metrics,
        "baseline": baseline,
        "skill_vs_naive": 1 - metrics["wmape"] / baseline["wmape"] if baseline["wmape"] else None,
        "train_span": _span(train.index),
        "test_span": _span(test.index),
        "feature_count": len(cols),
        "series": {
            "timestamps": _timestamps(test.index),
            "actual": test[target].tolist(),
            "predicted": predicted.tolist(),
        },
        "error_by_hour": {"hour": by_hour.index.tolist(), "mae": by_hour.tolist()},
        "feature_importance": model.feature_importance(top_n=15).to_dict("records"),
    }, predicted


def build_dispatch_artifact(
    actual_prices: pd.Series, forecast_prices: pd.Series, run_sweep: bool
) -> dict:
    """
    Compare three dispatch strategies over the forecast test window.

    perfect_foresight  optimize against the prices that actually occurred —
                       the unreachable ceiling, and the denominator for
                       "how much of the available value did we capture?"
    forecast_driven    optimize against forecast prices, then settle the
                       resulting schedule at actual prices. This is the real
                       strategy: it is what you could have run on the day.
    fixed_rule         charge overnight, discharge at the evening peak, on the
                       clock, ignoring prices. The no-model baseline.
    """
    config = DEFAULT_BATTERY
    strategies: dict[str, dict] = {}

    perfect = rolling_horizon_dispatch(actual_prices, config)
    strategies["perfect_foresight"] = {
        "label": "Perfect foresight",
        "description": "Optimized against the prices that actually occurred. Unreachable ceiling.",
        "summary": summarize_dispatch(perfect, config),
        "schedule": perfect,
    }

    on_forecast = rolling_horizon_dispatch(forecast_prices, config)
    settled = settle_at_actual_prices(on_forecast, actual_prices)
    strategies["forecast_driven"] = {
        "label": "Forecast-driven",
        "description": "Optimized against forecast prices, settled at actual prices. The runnable strategy.",
        "summary": summarize_dispatch(settled, config),
        "schedule": settled,
    }

    fixed = fixed_rule_dispatch(actual_prices, config)
    strategies["fixed_rule"] = {
        "label": "Fixed time-of-day rule",
        "description": "Charge 01:00-04:00, discharge 18:00-21:00, every day, ignoring price.",
        "summary": summarize_dispatch(fixed, config),
        "schedule": fixed,
    }

    ceiling = strategies["perfect_foresight"]["summary"]["profit"]
    baseline_profit = strategies["fixed_rule"]["summary"]["profit"]
    for strategy in strategies.values():
        profit = strategy["summary"]["profit"]
        strategy["summary"]["pct_of_perfect"] = profit / ceiling if ceiling else None
        strategy["summary"]["uplift_vs_fixed_rule"] = profit - baseline_profit

    # The demo week is the most volatile week in the window — a flat week shows
    # a battery doing nothing, which is true but not informative.
    daily_spread = (
        actual_prices.resample("D").max() - actual_prices.resample("D").min()
    ).rolling(7).mean()
    week_end = daily_spread.idxmax()
    demo_index = actual_prices.loc[:week_end].index[-DEMO_WEEK_HOURS:]

    payload = {
        "battery": {
            "capacity_mwh": config.capacity_mwh,
            "max_power_mw": config.max_power_mw,
            "round_trip_efficiency": config.round_trip_efficiency,
            "initial_soc_mwh": config.initial_soc_mwh,
            "duration_hours": config.capacity_mwh / config.max_power_mw,
        },
        "window": _span(actual_prices.index),
        "solver": "CBC (open source) via PuLP, 48h rolling horizon committed 24h at a time",
        "strategies": {
            key: {
                "label": s["label"],
                "description": s["description"],
                "summary": s["summary"],
            }
            for key, s in strategies.items()
        },
        "demo_week": {
            "timestamps": _timestamps(demo_index),
            "actual_price": actual_prices.loc[demo_index].tolist(),
            "forecast_price": forecast_prices.loc[demo_index].tolist(),
            "strategies": {
                key: {
                    "charge_mwh": s["schedule"].loc[demo_index, "charge_mwh"].tolist(),
                    "discharge_mwh": s["schedule"].loc[demo_index, "discharge_mwh"].tolist(),
                    "soc_mwh": s["schedule"].loc[demo_index, "soc_mwh"].tolist(),
                    "net_cost": s["schedule"].loc[demo_index, "net_cost"].tolist(),
                }
                for key, s in strategies.items()
            },
        },
        "daily_profit": {
            "dates": [d.date().isoformat() for d in perfect.resample("D").sum().index],
            "series": {
                key: (-s["schedule"]["net_cost"]).resample("D").sum().tolist()
                for key, s in strategies.items()
            },
        },
    }

    if run_sweep:
        sweep = []
        for label, cfg in SWEEP_CONFIGS:
            schedule = rolling_horizon_dispatch(actual_prices, cfg)
            summary = summarize_dispatch(schedule, cfg)
            sweep.append(
                {
                    "label": label,
                    "capacity_mwh": cfg.capacity_mwh,
                    "max_power_mw": cfg.max_power_mw,
                    "duration_hours": cfg.capacity_mwh / cfg.max_power_mw,
                    **summary,
                }
            )
        payload["duration_sweep"] = sweep

    return payload


def build_overview_artifact(df: pd.DataFrame) -> dict:
    """Headline dataset statistics, plus daily aggregates for the year chart."""
    price, demand = df["price_eur_mwh"], df["demand_mw"]
    daily = df.resample("D").agg(
        demand_mean=("demand_mw", "mean"),
        demand_peak=("demand_mw", "max"),
        price_mean=("price_eur_mwh", "mean"),
        price_min=("price_eur_mwh", "min"),
        price_max=("price_eur_mwh", "max"),
        temperature_mean=("temperature_c", "mean"),
        solar_mwh=("solar_mw", "sum"),
        wind_mwh=("wind_mw", "sum"),
    )
    profile = df.groupby(df.index.hour).agg(
        demand_mw=("demand_mw", "mean"), price_eur_mwh=("price_eur_mwh", "mean")
    )

    return {
        "span": _span(df.index),
        "columns": list(df.columns),
        "stats": {
            "demand_mw": {
                "min": demand.min(), "mean": demand.mean(), "max": demand.max(),
                "peak_timestamp": demand.idxmax(),
            },
            "price_eur_mwh": {
                "min": price.min(), "mean": price.mean(), "max": price.max(),
                "negative_hours": int((price < 0).sum()),
                "negative_share": float((price < 0).mean()),
                "hours_above_150": int((price > 150).sum()),
                "mean_daily_spread": float(
                    (price.resample("D").max() - price.resample("D").min()).mean()
                ),
            },
            "renewables": {
                "solar_share": float(df["solar_mw"].sum() / demand.sum()),
                "wind_share": float(df["wind_mw"].sum() / demand.sum()),
                "max_instant_share": float(
                    ((df["solar_mw"] + df["wind_mw"]) / demand).max()
                ),
            },
        },
        "daily": {
            "dates": [d.date().isoformat() for d in daily.index],
            **{col: daily[col].tolist() for col in daily.columns},
        },
        "hourly_profile": {
            "hour": profile.index.tolist(),
            **{col: profile[col].tolist() for col in profile.columns},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the battery duration sweep (4 extra year-long MIP solves)",
    )
    args = parser.parse_args()

    started = time.time()
    print("Loading sample dataset...")
    df = load_sample_dataset()
    print(f"  {len(df):,} hourly rows, {df.index.min()} -> {df.index.max()}")

    print("Building overview...")
    _write("overview.json", build_overview_artifact(df))

    forecasts = {}
    for target, unit in (("demand_mw", "MW"), ("price_eur_mwh", "EUR/MWh")):
        print(f"Forecasting {target}...")
        feature_matrix = build_feature_matrix(df, target)
        artifact, predicted = build_forecast_artifact(feature_matrix, target, unit)
        forecasts[target] = predicted
        print(
            f"  WMAPE {artifact['metrics']['wmape']:.4f}"
            f"  bias {artifact['metrics']['bias']:+.4f}"
            f"  skill vs naive {artifact['skill_vs_naive']:.1%}"
        )
        _write(f"forecast_{target}.json", artifact)

        print(f"Backtesting {target} across all regimes...")
        regimes = evaluate_all_regimes(feature_matrix, target, test_size=TEST_SIZE)
        _write(
            f"backtest_{target}.json",
            {"target": target, "unit": unit, "regimes": regimes.to_dict("records")},
        )

    print("Optimizing battery dispatch...")
    forecast_price = forecasts["price_eur_mwh"]
    actual_price = df["price_eur_mwh"].reindex(forecast_price.index)
    dispatch = build_dispatch_artifact(actual_price, forecast_price, run_sweep=not args.fast)
    for key, strategy in dispatch["strategies"].items():
        print(f"  {strategy['label']:<22} EUR {strategy['summary']['profit']:>12,.0f}")
    _write("dispatch.json", dispatch)

    _write(
        "metadata.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "data/sample/hourly_energy_sample.csv (synthetic, committed)",
            "rows": len(df),
            "test_size": TEST_SIZE,
            "model": "LightGBM regressor, calendar + lag + rolling + weather features",
            "build_seconds": round(time.time() - started, 1),
            "sweep_included": not args.fast,
        },
    )
    print(f"Done in {time.time() - started:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
