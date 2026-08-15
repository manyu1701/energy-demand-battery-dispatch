# Energy Demand Forecasting & Battery Dispatch Optimization

A from-scratch, public-data project applying the same pattern used in
production revenue-management systems — demand forecasting, driver
decomposition, and constrained optimization under real-world limits — to
grid electricity demand and battery storage dispatch.

## What this project does

1. **Forecast** hourly electricity demand and price with a LightGBM model,
   evaluated across five backtesting regimes so accuracy claims are robust,
   not a single lucky train/test split.
2. **Optimize** battery charge/discharge dispatch against the forecast price
   series to maximize arbitrage profit, subject to capacity, power-rate and
   round-trip-efficiency constraints — a mixed-integer program solved with the
   open-source CBC solver, so anyone can clone this and run it with no
   commercial license.
3. **Measure what the forecast is worth**, in money. The schedule built on
   forecast prices is settled at the prices that actually occurred and
   compared against perfect foresight and against a no-model baseline.

That last step is the point. A forecast is only worth the better decision it
buys, so the headline number here is not WMAPE — it is the share of the
available arbitrage value the forecast-driven battery actually captured.

## Results on the bundled dataset

Held-out window: 1,719 hours the price model never saw. Battery: 20 MWh /
5 MW, 88% round-trip.

| Strategy | Profit | % of perfect |
|---|---:|---:|
| Perfect foresight (prices known in advance) | €60,518 | 100% |
| **Forecast-driven** (optimize on forecast, settle at actual) | **€47,526** | **78.5%** |
| Fixed time-of-day rule (no model) | €11,666 | 19.3% |

Forecast accuracy on the same chronological split:

| Target | WMAPE | Bias | Seasonal-naive WMAPE | Skill vs naive |
|---|---:|---:|---:|---:|
| Demand | 3.27% | +1.44% | 6.57% | +50.2% |
| Price | 13.45% | −5.88% | 18.78% | +28.4% |

## Quickstart

```bash
# Python pipeline — model, backtests, dispatch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -v
python scripts/build_dashboard_data.py     # ~15s, rebuilds every artifact

# Node dashboard
cd frontend && npm install && npm start     # http://localhost:3000
```

The dashboard reads committed artifacts, so `npm start` works on a fresh
clone with no Python environment at all. Re-run the pipeline only after
changing the model, the features, or the data.

## The dashboard

`frontend/` is a small Express app — one dependency, no build step, no CDN,
charts drawn as SVG from scratch. Four views:

- **Dataset** — the hourly series (demand, price, temperature, solar, wind,
  net load) at selectable ranges and resolutions, the average day, and the
  daily price spread that defines the arbitrage opportunity.
- **Forecast** — forecast against actual over the held-out window, error
  metrics, gain-based feature importance, and error broken down by hour of
  day (a model can look fine on average and still be worst at the evening
  peak, which is exactly when the battery decides whether to discharge).
- **Backtesting** — all five regimes side by side, each against a
  seasonal-naive baseline.
- **Battery dispatch** — the three strategies compared, the hour-by-hour
  schedule for the most volatile week, cumulative profit, and a storage
  duration sweep.

## Backtesting regimes

A single train/test split tells you almost nothing — pick a lucky window and
any model looks good. Each regime stresses a different failure mode:

| Regime | What it tests |
|---|---|
| `temporal` | Chronological split. Train on the past, predict the future — the honest default. |
| `random` | Random holdout across the full history. Optimistic by construction; the gap to `temporal` shows how much the model relies on interpolating within familiar conditions. |
| `recency_weighted` | Chronological, training rows exponentially down-weighted by age. Beating `temporal` here means the series has drifted. |
| `short_horizon` | Chronological, scored only on the next 24 hours — the horizon a day-ahead dispatch decision actually needs. |
| `feature_ablation` | Chronological, refit with one feature group removed at a time. Shows which features are load-bearing rather than decorative. |

Every regime is scored against a seasonal-naive baseline ("same hour, last
week"). A model that cannot beat that on hourly grid data is not adding
anything, so the comparison is reported everywhere the model is.

## Data

`data/sample/hourly_energy_sample.csv` — one year of hourly data, committed
so the project runs out of the box. It is **synthetic**, and deliberately not
smooth: annual and daily seasonality, holidays, a V-shaped temperature
response, AR(1) residuals (so lag features earn their place), solar and wind
generation, a fuel-cost random walk, scarcity spikes, and negative prices in
the ~2% of hours when renewables plus must-run generation over-serve load.

Regenerate it with `python scripts/generate_sample_data.py` — it is
deterministic, and CI checks the committed file still matches the code.

### Real data

Vetted public sources, with licences and caveats, are documented in
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). `scripts/fetch_real_data.py`
downloads from them into `data/raw/` and writes the canonical schema that
`load_real_data()` reads:

```bash
python scripts/fetch_real_data.py opsd --zone DE          # EU load + price, no account
python scripts/fetch_real_data.py open-meteo --lat 52.52 --lon 13.41 \
    --start 2015-01-01 --end 2020-09-30                   # ERA5 temperature, no account
python scripts/fetch_real_data.py merge \
    --merge data/raw/opsd_de_hourly.csv \
    --with  data/raw/open_meteo_52.52_13.41_hourly.csv
```

The recommended starting point is **Open Power System Data** (CC-BY 4.0, no
account) — one file carries hourly load and day-ahead price for 30+ European
zones on a single index, compiled from the ENTSO-E Transparency Platform.
**Fraunhofer ISE energy-charts** covers recent years, and **Open-Meteo ERA5**
supplies temperature. For US demand, **EIA Form EIA-930** needs a free key and
publishes no prices.

Nothing downstream changes: the feature builder, model and optimizer all take
a plain hourly `DataFrame`. What does change is the data quality —
`normalize_hourly_frame()` handles the missing hours, DST folds, duplicate
timestamps, sub-hourly feeds and zero-as-null outages that real feeds carry,
and refuses a series that is more than 5% missing rather than modelling a
gap-filled fiction.

No third-party data is committed: `data/raw/` is gitignored.

## Project structure

```
src/
  data/loader.py                     synthetic generators + CSV loaders
  features/build_features.py         calendar / lag / rolling / weather features
  models/forecast.py                 LightGBM training, WMAPE + bias, naive baseline
  backtesting/harness.py             five-regime backtest harness
  optimization/battery_dispatch.py   dispatch MIP, rolling horizon, settlement, baselines
scripts/
  generate_sample_data.py            rebuild the committed dataset
  build_dashboard_data.py            run the pipeline, export dashboard JSON
  fetch_real_data.py                 download real data from public sources
docs/DATA_SOURCES.md                 vetted sources, licences, caveats
frontend/
  server.js, lib/                    Express API over the CSV and the artifacts
  public/                            dashboard UI, hand-rolled SVG charts
  test/                              API tests (node --test)
data/
  sample/                            the committed dataset
  dashboard/                         generated artifacts the dashboard reads
examples/run_battery_demo.py         standalone optimizer demo
tests/                               pytest suite (95 tests)
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Synthetic data generator + battery dispatch optimizer core | ✅ done |
| 1 | Demand/price forecasting, 5-regime backtesting, forecast→dispatch loop, dashboard | ✅ done |
| 2 | Driver decomposition (weather / calendar / trend attribution) | planned |
| 3 | Validate the whole loop on real grid data (OPSD / EIA) | planned |
| 4 | Degradation cost in the objective; probabilistic forecasts driving risk-aware dispatch | planned |

## Known limitations

Worth stating plainly, since the numbers above look good and some of that is
the data's doing:

- **The dataset is synthetic.** Its price process is a merit-order model, not
  a market. Real prices are harder to forecast, so treat 13% price WMAPE as a
  property of this generator, not a claim about real markets.
- **Perfect price knowledge within the horizon.** The rolling-horizon
  optimizer sees 48 hours of forecast at a time and commits 24. Real
  operations also face intraday re-dispatch and imbalance settlement.
- **No degradation cost.** The objective maximizes arbitrage revenue only,
  which is why the duration sweep favours the highest-power configuration —
  it cycles hardest and pays nothing for the wear.
- **Single price node, no volume limits.** The battery is a price taker and
  its own dispatch never moves the market.

## Why this exists

An independent, original project built entirely on public/synthetic data. It
mirrors the *pattern* of forecast → decompose → optimize used in production
revenue-management work — not any employer's codebase, data, or proprietary
methodology.
