# Energy Demand Forecasting & Battery Dispatch Optimization

A from-scratch, public-data project applying the same pattern used in
production revenue-management systems — demand forecasting, driver
decomposition, and constrained optimization under real-world limits — to
grid electricity demand and battery storage dispatch.

## What this project does (end state)

1. **Forecast** hourly electricity demand and/or price for a grid region
   with a LightGBM model, evaluated across multiple backtesting regimes so
   accuracy claims are robust, not a single lucky train/test split.
2. **Decompose** demand into weather, calendar, and trend effects.
3. **Optimize** battery charge/discharge dispatch against the (forecasted)
   price series to minimize cost / maximize arbitrage profit, subject to
   capacity, power-rate, and round-trip-efficiency constraints — solved as
   a mixed-integer program with the open-source CBC solver, so anyone can
   clone this and run it with no commercial license.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Synthetic data generator + battery dispatch optimizer core | ✅ done |
| 1 | Real demand/price forecasting (LightGBM) + 5-regime backtesting harness | 🔜 next |
| 2 | Driver decomposition (weather / calendar / trend) | planned |
| 3 | Wire the Phase 1 forecast into the dispatch optimizer; validate on real grid data | planned |

The optimizer was built first and runs today on synthetic (or any real)
hourly price data — it doesn't need the forecasting model to exist first,
since `optimize_dispatch()` just takes a `pd.Series` of hourly prices as
input. Phase 1's forecast output will plug into that same interface later,
replacing the synthetic/historical series with a forecasted one.

## Demo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -v
python examples/run_battery_demo.py
```

This runs the optimizer over 3 days of synthetic hourly prices and prints
the arbitrage profit plus a chart (`examples/battery_dispatch_demo.png`)
showing the battery charging in low-price hours and discharging into the
morning/evening demand peaks.

## Real data sources (no data ships in this repo)

- **Open Power System Data** — [data.open-power-system-data.org/time_series](https://data.open-power-system-data.org/time_series/)
  — hourly load and day-ahead price for several European countries, free,
  no account required, CC-BY licensed.
- **EIA hourly demand** — [api.eia.gov](https://api.eia.gov) — US grid
  balancing-authority demand, free API key required.

Download either, save it under `data/raw/`, and point `load_real_data()` in
`src/data/loader.py` at it.

## Project structure

```
src/
  data/loader.py            synthetic generator + real-data loader
  features/build_features.py   calendar features (Phase 1)
  models/forecast.py           LightGBM forecasting (Phase 1, TODO)
  backtesting/harness.py       5-regime backtest harness (Phase 1, TODO)
  optimization/battery_dispatch.py   battery dispatch MIP (done)
examples/run_battery_demo.py   standalone runnable demo
tests/                         pytest suite for the data generator and optimizer
```

## Why this exists

This is an independent, original project built entirely on public/synthetic
data. It mirrors the *pattern* of forecast → decompose → optimize used in
production revenue-management work — not any employer's codebase, data, or
proprietary methodology.
