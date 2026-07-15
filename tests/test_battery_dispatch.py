from src.data.loader import generate_synthetic_sample
from src.optimization.battery_dispatch import BatteryConfig, optimize_dispatch


def _sample_prices():
    df = generate_synthetic_sample(n_days=3, seed=7)
    return df["price_eur_mwh"]


def test_soc_within_bounds():
    prices = _sample_prices()
    config = BatteryConfig(capacity_mwh=10, max_power_mw=2.5, initial_soc_mwh=5)
    result = optimize_dispatch(prices, config)
    assert (result["soc_mwh"] >= config.min_soc_mwh - 1e-6).all()
    assert (result["soc_mwh"] <= config.capacity_mwh + 1e-6).all()


def test_power_rate_within_bounds():
    prices = _sample_prices()
    config = BatteryConfig(max_power_mw=2.5)
    result = optimize_dispatch(prices, config)
    assert (result["charge_mwh"] <= config.max_power_mw + 1e-6).all()
    assert (result["discharge_mwh"] <= config.max_power_mw + 1e-6).all()


def test_no_simultaneous_charge_and_discharge():
    prices = _sample_prices()
    result = optimize_dispatch(prices, BatteryConfig())
    both_active = (result["charge_mwh"] > 1e-6) & (result["discharge_mwh"] > 1e-6)
    assert not both_active.any()


def test_optimizer_finds_arbitrage_profit():
    prices = _sample_prices()
    config = BatteryConfig(capacity_mwh=10, max_power_mw=2.5, initial_soc_mwh=5)
    result = optimize_dispatch(prices, config)
    # With genuine price variation, the optimal schedule should not lose money net
    assert result["net_cost"].sum() <= 0
