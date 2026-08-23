import numpy as np
import pytest

from src.lifts.building import VERONA, BuildingConfig
from src.lifts.placement import (
    PlacementConfig,
    all_at_entry_plan,
    availability_weights,
    brute_force_slot_placement,
    compare_plans,
    count_relocations,
    evaluate_placement,
    even_spread_plan,
    expected_wait_by_slot,
    optimize_day_placement,
    static_optimal_plan,
)

SMALL = BuildingConfig(name="test tower", n_floors=10, n_lifts=3, entry_floors=(0,))
FREE = PlacementConfig(reposition_penalty_s=0.0, pin_lift_to_entry_above_share=None)


def _flip_flop_demand():
    """Four quiet slots with the calls high up, then four with them all in the
    lobby — a day whose best answer cannot be a single fixed plan."""
    demand = np.zeros((8, SMALL.n_floors))
    demand[:4, 7:10] = 6.0
    demand[4:, 0] = 30.0
    return demand


def test_availability_weights_sum_to_one_and_shift_with_load():
    demand = np.zeros((2, SMALL.n_floors))
    demand[0, 0] = 1.0  # near-idle group
    demand[1, 0] = 150.0  # saturated group
    weights = availability_weights(demand, SMALL, PlacementConfig())
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights[0, 0] > 0.98  # quiet: the nearest lift is almost always free
    assert weights[1, 0] < 0.6  # busy: the call often falls through to a farther one


def test_mip_matches_exhaustive_search_on_a_single_slot():
    """The optimizer is only worth trusting where it agrees with an exhaustive
    search over every possible placement."""
    demand = _flip_flop_demand()[:1]
    result = optimize_day_placement(demand, SMALL, FREE)
    weights = availability_weights(demand, SMALL, FREE)
    best_floors, best_cost = brute_force_slot_placement(demand[0], weights[0], SMALL)

    assert result.objective_passenger_seconds == pytest.approx(best_cost, rel=1e-6)
    assert sorted(result.schedule.filter(like="_floor").iloc[0]) == sorted(best_floors)


def test_mip_matches_exhaustive_search_slot_by_slot_when_moves_are_free():
    demand = _flip_flop_demand()
    result = optimize_day_placement(demand, SMALL, FREE)
    weights = availability_weights(demand, SMALL, FREE)
    plan = result.schedule.filter(like="_floor").to_numpy()
    for t in range(len(demand)):
        _, best_cost = brute_force_slot_placement(demand[t], weights[t], SMALL)
        actual = float(
            (demand[t] * (weights[t][:, None] * np.sort(SMALL.travel_time_matrix()[plan[t], :], axis=0)).sum(axis=0)).sum()
        )
        assert actual == pytest.approx(best_cost, rel=1e-6)


def test_every_slot_parks_exactly_the_whole_group_somewhere_real():
    result = optimize_day_placement(_flip_flop_demand(), SMALL, FREE)
    plan = result.schedule.filter(like="_floor").to_numpy()
    assert plan.shape == (8, SMALL.n_lifts)
    assert plan.min() >= 0 and plan.max() < SMALL.n_floors


def test_a_saturated_floor_pulls_the_whole_group_onto_it():
    """When one floor generates more calls than a single lift can handle, the
    second and third lift are worth parking there too — the model has to see
    that, or it can never justify stacking lifts in the lobby at 20:00."""
    demand = np.zeros((1, SMALL.n_floors))
    demand[0, 0] = 200.0
    result = optimize_day_placement(demand, SMALL, FREE)
    assert list(result.schedule.filter(like="_floor").iloc[0]) == [0, 0, 0]


def test_a_quiet_floor_does_not():
    """With the same floor lightly loaded, stacking buys nothing and the spare
    lifts should go cover the rest of the building."""
    demand = np.full((1, SMALL.n_floors), 0.05)
    demand[0, 0] = 2.0
    result = optimize_day_placement(demand, SMALL, FREE)
    assert len(set(result.schedule.filter(like="_floor").iloc[0])) == SMALL.n_lifts


def test_reposition_penalty_buys_a_calmer_plan():
    demand = _flip_flop_demand()
    restless = optimize_day_placement(demand, SMALL, FREE)
    calm = optimize_day_placement(
        demand, SMALL, PlacementConfig(reposition_penalty_s=5000.0, pin_lift_to_entry_above_share=None)
    )
    assert calm.n_relocations < restless.n_relocations
    assert calm.n_relocations == 0  # at that price, never move


def test_moving_through_the_day_beats_the_best_fixed_plan():
    """The whole premise: a plan that changes with the traffic must beat the
    best possible plan that doesn't."""
    demand = _flip_flop_demand()
    dynamic = optimize_day_placement(demand, SMALL, FREE).schedule.filter(like="_floor").to_numpy()
    static = static_optimal_plan(demand, SMALL, FREE)
    dynamic_wait = evaluate_placement(demand, dynamic, SMALL, FREE)["mean_wait_s"]
    static_wait = evaluate_placement(demand, static, SMALL, FREE)["mean_wait_s"]
    assert dynamic_wait < static_wait


def test_static_optimum_beats_the_naive_baselines():
    demand = _flip_flop_demand()
    scores = {
        name: evaluate_placement(demand, plan(demand, SMALL, FREE), SMALL, FREE)["mean_wait_s"]
        for name, plan in (
            ("static", static_optimal_plan),
            ("lobby", all_at_entry_plan),
            ("spread", even_spread_plan),
        )
    }
    assert scores["static"] <= min(scores["lobby"], scores["spread"])


def test_entry_pin_forces_a_lift_into_the_lobby():
    """Demand is entirely upstairs, so the objective alone would leave the lobby
    empty; the operational safeguard overrides it."""
    demand = np.zeros((1, SMALL.n_floors))
    demand[0, 9] = 10.0
    demand[0, 0] = 5.0  # a third of the slot's calls
    pinned = optimize_day_placement(
        demand, SMALL, PlacementConfig(reposition_penalty_s=0.0, pin_lift_to_entry_above_share=0.30)
    )
    assert 0 in list(pinned.schedule.filter(like="_floor").iloc[0])


def test_expected_wait_is_zero_where_there_are_no_calls():
    demand = np.zeros((2, SMALL.n_floors))
    demand[1, 5] = 4.0
    plan = np.tile([0, 4, 8], (2, 1))
    waits = expected_wait_by_slot(demand, plan, SMALL)
    assert waits[0] == 0.0
    assert waits[1] > 0.0


def test_relocation_count_ignores_which_lift_sits_where():
    swapped = np.array([[0, 4, 8], [8, 4, 0], [8, 4, 1]])
    assert count_relocations(swapped) == 1  # only the 0 -> 1 move is real


def test_comparison_table_is_ranked_and_scored_against_the_baseline():
    demand = _flip_flop_demand()
    table = compare_plans(
        demand,
        {
            "all_at_lobby": all_at_entry_plan(demand, SMALL),
            "best_static": static_optimal_plan(demand, SMALL, FREE),
        },
        SMALL,
        FREE,
    )
    assert table["mean_wait_s"].is_monotonic_increasing
    assert table.loc["all_at_lobby", "vs_baseline_pct"] == 0.0
    assert table.loc["best_static", "vs_baseline_pct"] <= 0.0


def test_demand_matrix_must_match_the_building():
    with pytest.raises(ValueError, match="floors"):
        optimize_day_placement(np.zeros((2, VERONA.n_floors)), SMALL, FREE)


def test_negative_demand_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        optimize_day_placement(-np.ones((1, SMALL.n_floors)), SMALL, FREE)


def test_evaluating_against_an_empty_day_is_an_error():
    with pytest.raises(ValueError, match="no calls"):
        evaluate_placement(np.zeros((2, SMALL.n_floors)), np.tile([0, 1, 2], (2, 1)), SMALL)
