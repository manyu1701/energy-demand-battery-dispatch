"""Tests for the dispatch strategies added alongside the forecasting model."""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_hourly_dataset
from src.optimization.battery_dispatch import (
    BatteryConfig,
    fixed_rule_dispatch,
    optimize_dispatch,
    rolling_horizon_dispatch,
    settle_at_actual_prices,
    summarize_dispatch,
)

CONFIG = BatteryConfig(capacity_mwh=10, max_power_mw=2.5, round_trip_efficiency=0.9, initial_soc_mwh=5)


@pytest.fixture(scope="module")
def prices():
    return generate_hourly_dataset(n_days=10, seed=21)["price_eur_mwh"]


def _assert_physically_feasible(result, config):
    assert (result["charge_mwh"] <= config.max_power_mw + 1e-6).all()
    assert (result["discharge_mwh"] <= config.max_power_mw + 1e-6).all()
    assert (result["soc_mwh"] >= config.min_soc_mwh - 1e-6).all()
    assert (result["soc_mwh"] <= config.capacity_mwh + 1e-6).all()

    # State of charge must follow from the schedule, not float free of it.
    eff_leg = config.round_trip_efficiency**0.5
    previous = np.concatenate([[config.initial_soc_mwh], result["soc_mwh"].to_numpy()[:-1]])
    expected = previous + result["charge_mwh"] * eff_leg - result["discharge_mwh"] / eff_leg
    assert np.allclose(result["soc_mwh"], expected, atol=1e-6)


def test_rolling_horizon_covers_the_whole_series(prices):
    result = rolling_horizon_dispatch(prices, CONFIG, window_hours=48, step_hours=24)
    assert len(result) == len(prices)
    assert result.index.equals(prices.index)


def test_rolling_horizon_respects_every_constraint(prices):
    result = rolling_horizon_dispatch(prices, CONFIG, window_hours=48, step_hours=24)
    _assert_physically_feasible(result, CONFIG)


def test_rolling_horizon_never_charges_and_discharges_at_once(prices):
    result = rolling_horizon_dispatch(prices, CONFIG)
    assert not ((result["charge_mwh"] > 1e-6) & (result["discharge_mwh"] > 1e-6)).any()


def test_rolling_horizon_is_profitable(prices):
    result = rolling_horizon_dispatch(prices, CONFIG)
    assert summarize_dispatch(result, CONFIG)["profit"] > 0


def test_rolling_horizon_validates_its_windows(prices):
    with pytest.raises(ValueError):
        rolling_horizon_dispatch(prices, CONFIG, window_hours=12, step_hours=24)
    with pytest.raises(ValueError):
        rolling_horizon_dispatch(prices, CONFIG, window_hours=48, step_hours=0)


def test_full_horizon_beats_rolling_horizon(prices):
    """
    Perfect knowledge of the whole window cannot be worse than seeing 48h at a
    time. If this inverts, the rolling loop is leaking state between windows.
    """
    full = optimize_dispatch(prices, CONFIG)
    rolling = rolling_horizon_dispatch(prices, CONFIG)
    assert -full["net_cost"].sum() >= -rolling["net_cost"].sum() - 1e-6


def test_fixed_rule_respects_the_same_constraints(prices):
    result = fixed_rule_dispatch(prices, CONFIG)
    _assert_physically_feasible(result, CONFIG)


def test_fixed_rule_follows_the_clock_not_the_price(prices):
    result = fixed_rule_dispatch(prices, CONFIG, charge_hours=(2, 3), discharge_hours=(19, 20))
    charging = result[result["charge_mwh"] > 1e-6]
    discharging = result[result["discharge_mwh"] > 1e-6]
    assert set(charging.index.hour) <= {2, 3}
    assert set(discharging.index.hour) <= {19, 20}


def test_optimizer_beats_the_fixed_rule(prices):
    """The optimizer has to earn its complexity against the no-model baseline."""
    optimized = summarize_dispatch(rolling_horizon_dispatch(prices, CONFIG), CONFIG)
    fixed = summarize_dispatch(fixed_rule_dispatch(prices, CONFIG), CONFIG)
    assert optimized["profit"] > fixed["profit"]


def test_settling_reprices_without_touching_the_schedule(prices):
    forecast = prices * 1.05 + 3
    schedule = rolling_horizon_dispatch(forecast, CONFIG)
    settled = settle_at_actual_prices(schedule, prices)

    # Physical decisions are unchanged; only the money moves.
    assert settled["charge_mwh"].equals(schedule["charge_mwh"])
    assert settled["discharge_mwh"].equals(schedule["discharge_mwh"])
    assert settled["soc_mwh"].equals(schedule["soc_mwh"])
    assert np.allclose(settled["price"], prices.to_numpy())
    assert np.allclose(
        settled["net_cost"],
        prices.to_numpy() * (schedule["charge_mwh"] - schedule["discharge_mwh"]).to_numpy(),
    )


def test_settling_requires_full_price_coverage(prices):
    schedule = rolling_horizon_dispatch(prices, CONFIG)
    with pytest.raises(ValueError):
        settle_at_actual_prices(schedule, prices.iloc[:-5])


def test_perfect_foresight_bounds_a_forecast_driven_schedule(prices):
    """
    A schedule built on an imperfect forecast can never beat one built on the
    prices that actually occurred. This is the guarantee the dashboard's
    "% of perfect" column depends on.
    """
    rng = np.random.default_rng(3)
    noisy = prices + rng.normal(0, 25, len(prices))

    perfect = rolling_horizon_dispatch(prices, CONFIG)
    forecast_driven = settle_at_actual_prices(rolling_horizon_dispatch(noisy, CONFIG), prices)

    assert summarize_dispatch(perfect, CONFIG)["profit"] >= summarize_dispatch(
        forecast_driven, CONFIG
    )["profit"]


def test_summary_accounting_is_internally_consistent(prices):
    result = rolling_horizon_dispatch(prices, CONFIG)
    summary = summarize_dispatch(result, CONFIG)

    assert summary["hours"] == len(result)
    assert summary["energy_charged_mwh"] == pytest.approx(result["charge_mwh"].sum())
    assert summary["round_trip_loss_mwh"] == pytest.approx(
        summary["energy_charged_mwh"] - summary["energy_discharged_mwh"]
    )
    assert summary["equivalent_full_cycles"] == pytest.approx(
        summary["energy_discharged_mwh"] / CONFIG.capacity_mwh
    )
    # Round-trip losses mean a profitable battery always sells above its buy price.
    assert summary["avg_discharge_price"] > summary["avg_charge_price"]


def test_summary_handles_an_idle_battery():
    """A flat price series gives the battery nothing to do — that must not divide by zero."""
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    flat = pd.Series(50.0, index=index)
    summary = summarize_dispatch(fixed_rule_dispatch(flat, CONFIG, (), ()), CONFIG)

    assert summary["energy_charged_mwh"] == 0
    assert summary["avg_charge_price"] is None
    assert summary["avg_discharge_price"] is None
    assert summary["profit"] == 0
