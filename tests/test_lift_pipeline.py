import numpy as np
import pandas as pd
import pytest

from src.lifts.building import BuildingConfig
from src.lifts.placement import PlacementConfig, all_at_entry_plan, evaluate_placement
from src.lifts.pipeline import plan_day, score_day_plan, train_forecaster
from src.lifts.traffic import aggregate_to_slots, generate_synthetic_call_log, to_demand_matrix

SMALL = BuildingConfig(name="test tower", n_floors=10, n_lifts=3)
PLAN_DATE = pd.Timestamp("2025-03-07")


@pytest.fixture(scope="module")
def planned():
    log = generate_synthetic_call_log(SMALL, n_days=70, seed=21)
    slots = aggregate_to_slots(log, SMALL)
    history = slots[slots["slot_start"] < PLAN_DATE]
    forecaster, metrics = train_forecaster(history, SMALL, test_days=7, n_estimators=150)
    plan = plan_day(forecaster, history, PLAN_DATE, SMALL)

    index, demand = to_demand_matrix(slots, SMALL)
    mask = (index >= PLAN_DATE) & (index < PLAN_DATE + pd.Timedelta(days=1))
    return plan, demand[mask], metrics


def test_plan_covers_every_hour_of_the_day(planned):
    plan, _, _ = planned
    assert len(plan.result.schedule) == 24
    assert plan.park_floors.shape == (24, SMALL.n_lifts)
    assert plan.index[0] == PLAN_DATE
    assert plan.forecast_demand.shape == (24, SMALL.n_floors)


def test_schedule_names_a_floor_for_each_lift_plus_what_it_expects(planned):
    plan, _, _ = planned
    schedule = plan.result.schedule
    assert list(schedule.columns) == ["lift_1_floor", "lift_2_floor", "lift_3_floor",
                                      "expected_wait_s", "calls"]
    assert (schedule["expected_wait_s"] >= 0).all()


def test_planning_a_day_the_model_has_already_seen_is_refused():
    log = generate_synthetic_call_log(SMALL, n_days=20, seed=1)
    slots = aggregate_to_slots(log, SMALL)
    forecaster, _ = train_forecaster(slots, SMALL, test_days=5, n_estimators=50)
    with pytest.raises(ValueError, match="not before"):
        plan_day(forecaster, slots, slots["slot_start"].max().normalize(), SMALL)


def test_training_reports_holdout_accuracy(planned):
    _, _, metrics = planned
    assert 0 < metrics["wmape_floor_slot"] < 1.5
    assert metrics["n_rows"] == 7 * 24 * SMALL.n_floors * 2


def test_plan_built_on_the_forecast_still_wins_on_the_actual_day(planned):
    """The plan only ever saw a forecast. Scored against what really happened,
    it still has to beat the lobby-parking default — otherwise the forecast
    error has eaten the whole gain."""
    plan, actual, _ = planned
    config = PlacementConfig()
    planned_wait = score_day_plan(plan, actual, SMALL, config)["mean_wait_s"]
    default_wait = evaluate_placement(actual, all_at_entry_plan(actual, SMALL), SMALL, config)[
        "mean_wait_s"
    ]
    assert planned_wait < default_wait
