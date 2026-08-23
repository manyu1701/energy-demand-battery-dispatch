import numpy as np
import pytest

from src.lifts.building import BuildingConfig
from src.lifts.features import TARGET_COLUMN, build_features
from src.lifts.model import CallDemandForecaster, bias, temporal_split, wmape
from src.lifts.traffic import aggregate_to_slots, generate_synthetic_call_log

SMALL = BuildingConfig(name="test tower", n_floors=10, n_lifts=3)


@pytest.fixture(scope="module")
def fitted():
    log = generate_synthetic_call_log(SMALL, n_days=70, seed=12)
    featured = build_features(aggregate_to_slots(log, SMALL), SMALL)
    train, test = temporal_split(featured, test_days=14)
    return CallDemandForecaster(n_estimators=150).fit(train), train, test


def test_temporal_split_holds_out_the_future_only(fitted):
    _, train, test = fitted
    assert train["slot_start"].max() < test["slot_start"].min()
    assert test["slot_start"].dt.date.nunique() == 14


def test_predictions_are_non_negative(fitted):
    model, _, test = fitted
    assert (model.predict(test) >= 0).all()


def test_model_beats_a_flat_average_baseline(fitted):
    """The point of a per-floor model is to beat 'every floor is average'."""
    model, train, test = fitted
    actual = test[TARGET_COLUMN].to_numpy(dtype=float)
    flat = np.full_like(actual, train[TARGET_COLUMN].mean())
    assert wmape(actual, model.predict(test)) < 0.8 * wmape(actual, flat)


def test_forecast_is_not_badly_biased(fitted):
    model, _, test = fitted
    assert abs(bias(test[TARGET_COLUMN].to_numpy(dtype=float), model.predict(test))) < 0.1


def test_evaluate_reports_both_granularities(fitted):
    model, _, test = fitted
    metrics = model.evaluate(test)
    assert set(metrics) == {
        "wmape_floor_slot",
        "bias_floor_slot",
        "wmape_building_slot",
        "bias_building_slot",
        "n_rows",
    }
    # Errors on individual floor-slots partly cancel when summed over the
    # building, so the building-level number is always the kinder one
    assert metrics["wmape_building_slot"] < metrics["wmape_floor_slot"]


def test_demand_matrix_has_one_column_per_floor(fitted):
    model, _, test = fitted
    index, demand = model.predict_demand_matrix(test, SMALL)
    assert demand.shape == (len(index), SMALL.n_floors)
    assert (demand >= 0).all()
    assert index.is_monotonic_increasing


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        CallDemandForecaster().predict(None)


def test_wmape_and_bias_are_undefined_on_an_empty_target():
    zeros = np.zeros(5)
    assert np.isnan(wmape(zeros, np.ones(5)))
    assert np.isnan(bias(zeros, np.ones(5)))
