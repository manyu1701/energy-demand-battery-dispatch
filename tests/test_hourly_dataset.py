"""
Tests for the realistic hourly generator behind the committed sample dataset.

These assert the *structure* the forecasting model depends on — seasonality,
persistence, price dynamics — rather than exact values, so the generator can be
retuned without rewriting the suite.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    MUST_RUN_FLOOR_MW,
    SAMPLE_DIR,
    SAMPLE_FILENAME,
    generate_hourly_dataset,
    load_sample_dataset,
)


@pytest.fixture(scope="module")
def year():
    return generate_hourly_dataset(n_days=365, seed=20240101)


def test_shape_and_columns(year):
    assert len(year) == 365 * 24
    assert isinstance(year.index, pd.DatetimeIndex)
    assert {
        "demand_mw",
        "price_eur_mwh",
        "temperature_c",
        "solar_mw",
        "wind_mw",
        "net_load_mw",
        "is_weekend",
        "is_holiday",
    } == set(year.columns)


def test_hours_are_contiguous(year):
    assert (year.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_reproducible_with_seed():
    assert generate_hourly_dataset(n_days=5, seed=3).equals(generate_hourly_dataset(n_days=5, seed=3))


def test_different_seeds_differ():
    assert not generate_hourly_dataset(n_days=5, seed=1).equals(generate_hourly_dataset(n_days=5, seed=2))


def test_demand_is_positive(year):
    assert (year["demand_mw"] > 0).all()


def test_weekends_and_holidays_are_quieter(year):
    off_day = year["is_weekend"] | year["is_holiday"]
    assert year.loc[~off_day, "demand_mw"].mean() > year.loc[off_day, "demand_mw"].mean()


def test_demand_has_a_double_daily_peak(year):
    profile = year.groupby(year.index.hour)["demand_mw"].mean()
    morning = profile.loc[6:11].max()
    midday_trough = profile.loc[12:16].min()
    evening = profile.loc[17:22].max()
    overnight = profile.loc[0:5].min()

    assert morning > midday_trough  # morning ramp, then a dip
    assert evening > midday_trough  # evening peak
    assert evening > overnight


def test_temperature_follows_an_annual_cycle(year):
    monthly = year.groupby(year.index.month)["temperature_c"].mean()
    assert monthly.idxmax() in (6, 7, 8)
    assert monthly.idxmin() in (12, 1, 2)


def test_residuals_are_autocorrelated(year):
    """
    Persistence is what makes lag and rolling features meaningful. Independent
    noise would flatter the model and make the whole feature set pointless.
    """
    assert year["demand_mw"].autocorr(1) > 0.8
    assert year["demand_mw"].autocorr(24) > 0.5


def test_solar_only_generates_in_daylight(year):
    by_hour = year.groupby(year.index.hour)["solar_mw"].max()
    assert (by_hour.loc[0:3] == 0).all()
    assert (by_hour.loc[22:23] == 0).all()
    assert by_hour.loc[11:14].min() > 0


def test_solar_is_stronger_in_summer(year):
    monthly = year.groupby(year.index.month)["solar_mw"].mean()
    assert monthly.loc[6:8].mean() > monthly.loc[[12, 1, 2]].mean()


def test_net_load_is_demand_minus_renewables(year):
    expected = year["demand_mw"] - year["solar_mw"] - year["wind_mw"]
    assert np.allclose(year["net_load_mw"], expected, atol=0.02)


def test_prices_go_negative_when_supply_is_long(year):
    negative = year["price_eur_mwh"] < 0
    assert negative.any(), "expected some negative-price hours"
    assert negative.mean() < 0.10, "negative prices should stay exceptional"
    # Every negative hour must be explained by the must-run floor, not by noise.
    assert (year.loc[negative, "net_load_mw"] < MUST_RUN_FLOOR_MW).all()


def test_scarcity_spikes_exist(year):
    price = year["price_eur_mwh"]
    assert price.max() > price.quantile(0.99) * 1.5


def test_price_tracks_net_load(year):
    """Merit order: tighter supply means a higher clearing price."""
    normal = year[year["price_eur_mwh"] > 0]
    assert normal["price_eur_mwh"].corr(normal["net_load_mw"]) > 0.5


def test_there_is_a_daily_spread_to_arbitrage(year):
    price = year["price_eur_mwh"]
    spread = price.resample("D").max() - price.resample("D").min()
    assert spread.mean() > 10
    assert (spread > 0).all()


def test_holidays_are_flagged(year):
    new_year = year.loc["2024-01-01"]
    assert new_year["is_holiday"].all()
    assert not year.loc["2024-01-15", "is_holiday"].any()


def test_committed_sample_matches_the_generator():
    """
    The checked-in CSV must be exactly what the script regenerates, or the
    dashboard artifacts describe data nobody else can reproduce.
    """
    if not (SAMPLE_DIR / SAMPLE_FILENAME).exists():
        pytest.skip("sample dataset not generated; run scripts/generate_sample_data.py")

    on_disk = load_sample_dataset()
    regenerated = generate_hourly_dataset()

    assert on_disk.shape == regenerated.shape
    assert list(on_disk.columns) == list(regenerated.columns)
    for column in regenerated.select_dtypes(include=[float]).columns:
        assert np.allclose(on_disk[column], regenerated[column], atol=1e-6), column
