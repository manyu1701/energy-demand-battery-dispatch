import pytest

from src.lifts.building import VERONA, BuildingConfig


def test_verona_defaults():
    assert VERONA.n_floors == 33
    assert VERONA.n_lifts == 3
    assert VERONA.lobby_floor in VERONA.entry_floors


def test_travel_time_matrix_is_symmetric_and_zero_on_the_diagonal_journey():
    travel = VERONA.travel_time_matrix()
    assert travel.shape == (33, 33)
    assert (travel == travel.T).all()
    # A lift already on your floor still costs the door cycle, but no flight time
    assert travel[7, 7] == pytest.approx(VERONA.start_delay_s + VERONA.door_open_s)


def test_travel_time_increases_with_distance_but_sublinearly_at_first():
    travel = VERONA.travel_time_matrix()
    row = travel[0]
    assert (row[1:] > row[:-1]).all()
    # Short hops never reach rated speed, so 4 floors costs less than twice 2 floors
    flight_2 = VERONA.flight_time_s(2)
    flight_4 = VERONA.flight_time_s(4)
    assert flight_4 < 2 * flight_2


def test_long_journey_matches_rated_speed():
    # 32 floors at 3m each, cruising at 1.6 m/s, plus one acceleration period
    expected = 96 / 1.6 + 1.6 / 0.9
    assert VERONA.flight_time_s(32) == pytest.approx(expected)


def test_population_excludes_entry_floors():
    population = VERONA.population()
    assert population[0] == 0
    assert (population[1:] > 0).all()


def test_rejects_more_lifts_than_floors():
    with pytest.raises(ValueError):
        BuildingConfig(n_floors=2, n_lifts=3)


def test_rejects_entry_floor_outside_the_building():
    with pytest.raises(ValueError):
        BuildingConfig(n_floors=10, entry_floors=(11,))
