import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_hourly_dataset
from src.features.build_features import build_feature_matrix, feature_columns
from src.models.forecast import (
    bias,
    evaluate,
    seasonal_naive_baseline,
    train_and_evaluate,
    train_lightgbm,
    wmape,
)


@pytest.fixture(scope="module")
def matrix():
    df = generate_hourly_dataset(n_days=60, seed=11)
    return build_feature_matrix(df, "demand_mw")


@pytest.fixture(scope="module")
def split(matrix):
    cut = int(len(matrix) * 0.8)
    return matrix.iloc[:cut], matrix.iloc[cut:]


def test_wmape_is_zero_for_a_perfect_forecast():
    actual = pd.Series([10.0, 20.0, 30.0])
    assert wmape(actual, actual) == 0.0


def test_wmape_scales_with_error():
    actual = pd.Series([100.0, 100.0])
    assert wmape(actual, pd.Series([110.0, 90.0])) == pytest.approx(0.1)


def test_bias_sign_follows_the_direction_of_error():
    actual = pd.Series([100.0, 100.0])
    assert bias(actual, pd.Series([110.0, 110.0])) == pytest.approx(0.1)
    assert bias(actual, pd.Series([90.0, 90.0])) == pytest.approx(-0.1)


def test_bias_cancels_out_for_symmetric_errors():
    actual = pd.Series([100.0, 100.0])
    assert bias(actual, pd.Series([110.0, 90.0])) == pytest.approx(0.0)


def test_metrics_handle_an_all_zero_actual():
    zeros = pd.Series([0.0, 0.0])
    assert np.isnan(wmape(zeros, pd.Series([1.0, 1.0])))
    assert np.isnan(bias(zeros, pd.Series([1.0, 1.0])))


def test_model_trains_and_predicts_aligned_output(split):
    train, test = split
    cols = feature_columns(train, "demand_mw")
    model = train_lightgbm(train, "demand_mw", cols)
    predicted = model.predict(test)

    assert len(predicted) == len(test)
    assert predicted.index.equals(test.index)
    assert predicted.notna().all()


def test_model_rejects_missing_features_at_predict_time(split):
    train, test = split
    cols = feature_columns(train, "demand_mw")
    model = train_lightgbm(train, "demand_mw", cols)
    with pytest.raises(KeyError):
        model.predict(test.drop(columns=[cols[0]]))


def test_training_requires_features(split):
    train, _ = split
    with pytest.raises(ValueError):
        train_lightgbm(train, "demand_mw", [])


def test_model_beats_the_seasonal_naive_baseline(split):
    """The whole point of the model — if this fails, the features are broken."""
    train, test = split
    cols = feature_columns(train, "demand_mw")
    _, metrics = train_and_evaluate(train, test, "demand_mw", cols)
    baseline = seasonal_naive_baseline(train, test, "demand_mw")
    assert metrics["wmape"] < baseline["wmape"]


def test_evaluate_reports_every_metric(split):
    train, test = split
    cols = feature_columns(train, "demand_mw")
    model = train_lightgbm(train, "demand_mw", cols)
    metrics = evaluate(model, test)

    assert set(metrics) == {"wmape", "bias", "mae", "rmse", "n_test"}
    assert metrics["n_test"] == len(test)
    assert metrics["rmse"] >= metrics["mae"]  # true for any error distribution


def test_sample_weights_change_the_fit(split):
    train, test = split
    cols = feature_columns(train, "demand_mw")
    weights = np.linspace(0.01, 1.0, len(train))

    _, plain = train_and_evaluate(train, test, "demand_mw", cols)
    _, weighted = train_and_evaluate(train, test, "demand_mw", cols, sample_weight=weights)
    assert plain["wmape"] != weighted["wmape"]


def test_feature_importance_shares_sum_to_one(split):
    train, _ = split
    cols = feature_columns(train, "demand_mw")
    model = train_lightgbm(train, "demand_mw", cols)
    importance = model.feature_importance()

    assert len(importance) == len(cols)
    assert importance["share"].sum() == pytest.approx(1.0)
    assert importance["gain"].is_monotonic_decreasing
