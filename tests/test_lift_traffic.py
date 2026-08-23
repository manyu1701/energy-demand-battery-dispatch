import pandas as pd
import pytest

from src.lifts.building import VERONA
from src.lifts.traffic import (
    aggregate_to_slots,
    generate_synthetic_call_log,
    load_real_call_log,
    to_demand_matrix,
)


@pytest.fixture(scope="module")
def call_log():
    return generate_synthetic_call_log(VERONA, n_days=21, seed=5)


def test_log_schema_and_floor_range(call_log):
    assert list(call_log.columns) == ["timestamp", "origin_floor", "direction", "destination_floor"]
    assert call_log["origin_floor"].between(0, VERONA.n_floors - 1).all()
    assert call_log["destination_floor"].between(0, VERONA.n_floors - 1).all()
    assert call_log["timestamp"].is_monotonic_increasing


def test_residential_tower_peaks_the_opposite_way_to_an_office():
    """Down-calls from the flats peak in the morning; up-calls at the lobby peak
    in the evening. An office block does exactly the reverse."""
    log = generate_synthetic_call_log(VERONA, n_days=28, seed=6)
    weekday = log[log["timestamp"].dt.dayofweek < 5]
    hour = weekday["timestamp"].dt.hour

    down_from_flats = weekday[(weekday["direction"] == "down") & (weekday["origin_floor"] > 0)]
    up_from_lobby = weekday[(weekday["direction"] == "up") & (weekday["origin_floor"] == 0)]

    assert down_from_flats["timestamp"].dt.hour.between(7, 10).mean() > 0.3
    assert up_from_lobby["timestamp"].dt.hour.between(18, 22).mean() > 0.3
    assert hour.between(1, 5).mean() < 0.02  # the tower sleeps


def test_lobby_generates_a_large_share_of_all_calls(call_log):
    lobby_share = (call_log["origin_floor"] == 0).mean()
    assert 0.35 < lobby_share < 0.65


def test_weekend_mornings_start_later():
    log = generate_synthetic_call_log(VERONA, n_days=56, seed=8)
    early = log[log["timestamp"].dt.hour.between(6, 8)]
    weekend_rate = (early["timestamp"].dt.dayofweek >= 5).mean()
    assert weekend_rate < 2 / 7  # under-represented relative to a flat 2-in-7


def test_holidays_are_treated_as_rest_days():
    holiday = "2025-01-08"  # a Wednesday
    log = generate_synthetic_call_log(VERONA, n_days=21, seed=3, holidays=[holiday])
    on_holiday = log[log["timestamp"].dt.date == pd.Timestamp(holiday).date()]
    other_wednesdays = log[
        (log["timestamp"].dt.dayofweek == 2) & (log["timestamp"].dt.date != pd.Timestamp(holiday).date())
    ]
    n_other = other_wednesdays["timestamp"].dt.date.nunique()
    peak = lambda df: (df["timestamp"].dt.hour.between(7, 9)).sum()
    assert peak(on_holiday) < peak(other_wednesdays) / n_other


def test_slot_grid_is_dense_and_covers_whole_days(call_log):
    slots = aggregate_to_slots(call_log, VERONA)
    n_days = call_log["timestamp"].dt.date.nunique()
    assert len(slots) == n_days * 24 * VERONA.n_floors * 2
    assert not slots["calls"].isna().any()
    assert slots["calls"].sum() == len(call_log)


def test_demand_matrix_preserves_total_calls(call_log):
    slots = aggregate_to_slots(call_log, VERONA)
    index, demand = to_demand_matrix(slots, VERONA)
    assert demand.shape == (len(index), VERONA.n_floors)
    assert demand.sum() == len(call_log)


def test_reproducible_with_seed():
    first = generate_synthetic_call_log(VERONA, n_days=5, seed=99)
    second = generate_synthetic_call_log(VERONA, n_days=5, seed=99)
    assert first.equals(second)


def test_missing_real_log_points_at_the_fix():
    with pytest.raises(FileNotFoundError, match="lift controller"):
        load_real_call_log("definitely_not_here.csv")
