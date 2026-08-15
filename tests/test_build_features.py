import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic_sample
from src.features.build_features import (
    FEATURE_GROUPS,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
    feature_columns,
)


@pytest.fixture
def sample():
    return generate_synthetic_sample(n_days=30, seed=5)


def test_calendar_features_match_the_index(sample):
    out = add_calendar_features(sample)
    assert (out["hour"] == out.index.hour).all()
    assert (out["dayofweek"] == out.index.dayofweek).all()
    assert (out["is_weekend"] == (out.index.dayofweek >= 5)).all()


def test_cyclical_encodings_wrap(sample):
    out = add_calendar_features(sample)
    # Hours 23 and 0 must be adjacent in the encoded space, which is the whole
    # point of the sin/cos pair.
    at_23 = out.loc[out["hour"] == 23].iloc[0]
    at_00 = out.loc[out["hour"] == 0].iloc[0]
    distance = np.hypot(at_23["sin_hour"] - at_00["sin_hour"], at_23["cos_hour"] - at_00["cos_hour"])
    assert distance < 0.3


def test_lag_features_are_shifted_copies(sample):
    out = add_lag_features(sample, "demand_mw", lags=(24,))
    assert out["lag_24h"].iloc[:24].isna().all()
    assert out["lag_24h"].iloc[24] == sample["demand_mw"].iloc[0]


def test_rolling_features_exclude_the_current_hour(sample):
    """The rolling window must end at t-1, or the model trains on the answer."""
    out = add_rolling_features(sample, "demand_mw", windows=(24,))
    expected = sample["demand_mw"].iloc[:24].mean()
    assert out["roll_mean_24h"].iloc[24] == pytest.approx(expected)
    assert out["roll_mean_24h"].iloc[:24].isna().all()


def test_build_feature_matrix_drops_warmup_rows(sample):
    out = build_feature_matrix(sample, "demand_mw", lags=(24, 168), windows=(24, 168))
    # The 168h window is the binding constraint on how much history is needed.
    assert len(out) == len(sample) - 168
    assert not out.isna().any().any()


def test_build_feature_matrix_rejects_a_bad_index(sample):
    with pytest.raises(TypeError):
        build_feature_matrix(sample.reset_index(), "demand_mw")


def test_build_feature_matrix_rejects_a_missing_target(sample):
    with pytest.raises(KeyError):
        build_feature_matrix(sample, "not_a_column")


def test_feature_columns_never_include_the_target(sample):
    matrix = build_feature_matrix(sample, "demand_mw")
    assert "demand_mw" not in feature_columns(matrix, "demand_mw")


def test_feature_columns_can_be_restricted_to_groups(sample):
    matrix = build_feature_matrix(sample, "demand_mw")
    lag_only = feature_columns(matrix, "demand_mw", groups=["lag"])
    assert set(lag_only) == {"lag_24h", "lag_168h"}

    without_lag = feature_columns(matrix, "demand_mw", groups=[g for g in FEATURE_GROUPS if g != "lag"])
    assert not any(col.startswith("lag_") for col in without_lag)
    assert len(without_lag) < len(feature_columns(matrix, "demand_mw"))


def test_feature_columns_rejects_an_unknown_group(sample):
    matrix = build_feature_matrix(sample, "demand_mw")
    with pytest.raises(KeyError):
        feature_columns(matrix, "demand_mw", groups=["astrology"])


def test_weather_features_are_optional():
    """A dataset with no temperature column must still build."""
    index = pd.date_range("2024-01-01", periods=400, freq="h")
    frame = pd.DataFrame({"demand_mw": np.linspace(100, 200, 400)}, index=index)
    out = build_feature_matrix(frame, "demand_mw")
    assert "temp_heating_deg" not in out.columns
    assert "lag_24h" in out.columns
