import numpy as np
import pandas as pd
import pytest

from src.lifts.building import BuildingConfig
from src.lifts.features import FEATURE_COLUMNS, build_features, build_future_frame
from src.lifts.traffic import aggregate_to_slots, generate_synthetic_call_log

SMALL = BuildingConfig(name="test tower", n_floors=8, n_lifts=3)


@pytest.fixture(scope="module")
def featured():
    log = generate_synthetic_call_log(SMALL, n_days=40, seed=4)
    return build_features(aggregate_to_slots(log, SMALL), SMALL)


def test_all_declared_features_are_present(featured):
    assert set(FEATURE_COLUMNS).issubset(featured.columns)
    assert featured[FEATURE_COLUMNS].dtypes.map(lambda d: np.issubdtype(d, np.number)).all()


def test_lag_features_line_up_with_the_actual_history(featured):
    row = featured[(featured["floor"] == 3) & (featured["direction"] == "up")].iloc[-1]
    same_series = featured[(featured["floor"] == 3) & (featured["direction"] == "up")]
    yesterday = same_series[same_series["slot_start"] == row["slot_start"] - pd.Timedelta(days=1)]
    assert row["lag_1d"] == yesterday["calls"].iloc[0]


def test_lag_features_never_look_forward(featured):
    """The rolling mean must be built only from days strictly before its own row —
    a leak here would flatter every accuracy number downstream."""
    series = featured[(featured["floor"] == 2) & (featured["direction"] == "down")]
    for slot_of_day in (8, 20):
        window = series[series["slot_of_day"] == slot_of_day].sort_values("slot_start")
        expected = window["calls"].shift(1).rolling(28, min_periods=3).mean()
        pd.testing.assert_series_equal(
            window["roll_mean_28d"], expected, check_names=False, check_dtype=False
        )


def test_cyclical_encoding_makes_midnight_and_23h_neighbours(featured):
    midnight = featured[featured["slot_of_day"] == 0].iloc[0]
    late = featured[featured["slot_of_day"] == 23].iloc[0]
    noon = featured[featured["slot_of_day"] == 12].iloc[0]
    distance = lambda a, b: np.hypot(a["slot_sin"] - b["slot_sin"], a["slot_cos"] - b["slot_cos"])
    assert distance(midnight, late) < distance(midnight, noon)


def test_future_frame_is_a_full_unlabelled_day_with_lags_filled(featured):
    history = featured[featured["slot_start"] < "2025-02-01"]
    future = build_future_frame(history, "2025-02-01", SMALL)
    assert len(future) == 24 * SMALL.n_floors * 2
    assert future["calls"].isna().all()  # tomorrow is unknown, by construction
    assert not future["lag_1d"].isna().any()  # but everything it leans on is known
    assert future["slot_start"].dt.date.nunique() == 1


def test_future_frame_marks_holidays():
    log = generate_synthetic_call_log(SMALL, n_days=30, seed=2)
    history = aggregate_to_slots(log, SMALL)
    future = build_future_frame(history, "2025-02-05", SMALL, holidays=("2025-02-05",))
    assert (future["is_holiday"] == 1).all()
