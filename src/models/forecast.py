"""
Demand/price forecasting model — Phase 1 (not yet built).

Planned: train a LightGBM model on demand_mw (or price_eur_mwh) using the
calendar + lag features from src/features/build_features.py, evaluate it
across the multi-regime backtesting harness in src/backtesting/harness.py,
and report WMAPE + bias per regime.

Once this exists, the price/demand series it outputs plugs directly into
optimize_dispatch() in src/optimization/battery_dispatch.py in place of the
historical/synthetic price series used today — the function signature
(a pd.Series of hourly prices) doesn't need to change.
"""

# TODO: train_lightgbm(df, target_col, feature_cols) -> model
# TODO: evaluate(model, df, target_col) -> {"wmape": ..., "bias": ...}
