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
| L | Lift floor placement — same forecast-then-optimize pattern, applied to a 33-storey tower | ✅ done |

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

```bash
python examples/run_lift_placement_demo.py
```

This runs the lift placement model described below, end to end.

## Real data sources (no data ships in this repo)

- **Open Power System Data** — [data.open-power-system-data.org/time_series](https://data.open-power-system-data.org/time_series/)
  — hourly load and day-ahead price for several European countries, free,
  no account required, CC-BY licensed.
- **EIA hourly demand** — [api.eia.gov](https://api.eia.gov) — US grid
  balancing-authority demand, free API key required.

Download either, save it under `data/raw/`, and point `load_real_data()` in
`src/data/loader.py` at it.


## Lift floor placement (Verona, Hiranandani Gardens)

The same pattern — forecast demand, then optimize a decision under real
physical constraints — applied to a completely different problem: where
should a residential tower's lifts *wait* when they have nothing to do?

Verona is a 33-storey tower in Hiranandani Gardens, Powai, served by a group
of 3 lifts. The lobby-to-top-floor run alone takes 66 seconds. If all three
lifts idle in the lobby — the factory default in a great many residential
group controllers — a resident on the 30th floor pays most of that on every
call. If they idle high up, everyone walking in from the porch pays it
instead. And the right answer is not one answer: the tower's traffic inverts
across the day.

### What it does

1. **Forecast** — a LightGBM model with a Poisson objective predicts hall
   calls for each of the 33 floors, in each direction, for each hour of
   tomorrow, from calendar features and per-floor lags. Trained on a call log
   and scored on a chronological hold-out it never saw.
2. **Optimize** — a multi-period p-median MIP picks parking floors for all 3
   lifts across all 24 hours in one solve, minimizing total expected
   passenger waiting seconds, with a penalty per lift relocation so the plan
   doesn't reshuffle hourly for a two-second gain. Solved with CBC via PuLP,
   the same open-source solver as the battery model.
3. **Score** — the plan, built on the forecast alone, is scored against what
   actually happened, next to three baselines.

The wait model is not simply "distance to the nearest parked lift", because
the nearest lift is frequently already carrying someone. Each hour gets a
utilisation from its forecast call volume, and the call is answered by the
nearest parked lift with probability proportional to `(1 - rho)`, the second
nearest with `(1 - rho) * rho`, and so on. That one detail is what lets the
model see that a *second* lift parked on the same floor is worth something
when the group is busy — and it changes the answer:

| Time | Traffic | Where the solver parks the lifts |
|---|---|---|
| 00:00–05:00 | almost nothing | wherever they already are — moving costs more than it saves |
| 07:00–17:00 | residents leaving, spread across 33 floors | lobby + roughly floors 9 and 21 — fanned up the shaft |
| 18:00–21:00 | residents returning, half the calls in the lobby | all three collapse onto the lobby |
| 22:00–23:00 | late returns thinning out | lobby, lobby, and one mid-shaft |

Note that this is the opposite of an office block, which has its up-peak into
the lobby in the morning and its down-peak out of it in the evening.

### Results

One weekday (Thursday 24 April) from `run_lift_placement_demo.py`, scored on
that day's actual calls with the plan built from the forecast alone. The
Sunday in the same run comes out at −24.8% against the same baseline:

| Plan | Mean wait | vs lobby default |
|---|---|---|
| ML dynamic (this) | 13.5 s | **−30%** |
| Best possible fixed plan | 14.0 s | −28% |
| Even spread (hand-tuned zoning) | 19.0 s | −2% |
| All lifts in the lobby (default) | 19.4 s | — |

The best-fixed-plan row is the one that matters. Beating "everything in the
lobby" proves very little; beating the best plan that *doesn't change through
the day* is what shows that time-varying placement earns its keep — and the
margin there is a real but modest 2–3%, so most of the 30% is available from
a good static plan alone. That is worth knowing before anyone reprograms a
controller.

**These numbers come from synthetic traffic.** The generator in
`src/lifts/traffic.py` encodes a residential Mumbai tower's patterns — school
run at 7am, office commute spread by Powai travel times, domestic staff
arriving through the morning and leaving mid-afternoon, delivery traffic to
22:00, weekends shifted about two hours later — but they are modelled
patterns, not measured ones. Point the loader at a real controller export and
every number above should be recomputed.

### Using it on the real building

- **Get the call log.** Most modern group controllers (Otis, Kone,
  Schindler, Mitsubishi) can export a call or trip log through their service
  tool. Ask the facility manager or the AMC engineer on the next service
  visit. Save it as `data/raw/lift_calls.csv` with columns `timestamp,
  origin_floor, direction, destination_floor` and `load_real_call_log()`
  picks it up in place of the synthetic generator.
- **Check the config.** `BuildingConfig` in `src/lifts/building.py` holds the
  assumptions. Two are worth measuring rather than trusting: the lift's rated
  speed (on the inspection certificate inside the car) and
  `calls_per_lift_per_hour` in `PlacementConfig`, which is the group's
  handling capacity and the single most consequential number in the model —
  time a few round trips to set it.
- **Then what?** The output is an hour-by-hour table naming the floor each
  lift should idle on. Most group controllers expose exactly this as a
  parking/homing schedule, so it is something an AMC engineer can enter — but
  it is the building's lifts and the building's decision, so this is a case
  to make to the society and the lift vendor, not a change to make directly.

## Project structure

```
src/
  data/loader.py            synthetic generator + real-data loader
  features/build_features.py   calendar features (Phase 1)
  models/forecast.py           LightGBM forecasting (Phase 1, TODO)
  backtesting/harness.py       5-regime backtest harness (Phase 1, TODO)
  optimization/battery_dispatch.py   battery dispatch MIP (done)
  lifts/building.py            tower geometry + lift kinematics (Verona defaults)
  lifts/traffic.py             synthetic hall-call generator + real call-log loader
  lifts/features.py            calendar / spatial / lag features per floor
  lifts/model.py               LightGBM Poisson demand model + hold-out metrics
  lifts/placement.py           multi-period p-median parking MIP + baselines
  lifts/pipeline.py            the nightly job: forecast tomorrow, plan the day
examples/run_battery_demo.py         standalone runnable demo
examples/run_lift_placement_demo.py  end-to-end lift placement demo
tests/                         pytest suite for both applications
```

## Why this exists

This is an independent, original project built entirely on public/synthetic
data. It mirrors the *pattern* of forecast → decompose → optimize used in
production revenue-management work — not any employer's codebase, data, or
proprietary methodology.
