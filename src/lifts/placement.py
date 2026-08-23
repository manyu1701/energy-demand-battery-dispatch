"""
Idle-lift floor placement optimization.

Where should each of the group's lifts wait when it has nothing to do?

Every hall call costs the resident the time for a free lift to reach their
floor, and that cost depends entirely on where the lifts were idling when the
button was pressed. If all three sit in the lobby, a resident on the 30th floor
waits over a minute. If they sit high, everyone walking in from the porch
waits. The right answer is not one answer: at 08:30 the calls come from
residential floors heading down, at 20:00 more than half come from the lobby
heading up, and at 03:00 there is nothing to serve at all.

Model
-----
For each time slot the optimizer picks where each lift parks, then charges
every floor's forecast calls the expected wait they incur, and minimizes the
day's total passenger waiting seconds.

The expected wait is not simply the distance to the nearest parked lift,
because the nearest lift is often already carrying someone. Each slot gets a
utilisation `rho` — its forecast calls over the group's handling capacity — and
a call is answered by the nearest parked lift with probability proportional to
(1 - rho), by the second nearest with (1 - rho) * rho, and so on:

    wait(floor) = sum_k  p_k * (travel time from the k-th nearest parked lift)

Two consequences fall out of that, and both match how a lift group actually
behaves. In a quiet slot `p_1` is near 1, the model reduces to a plain
p-median, and the lifts spread out to cover the building. In a saturated slot
the weight spreads across ranks, a *second* lift parked on the same floor
becomes valuable because it covers the first one being busy, and the group
clusters where the calls are — which is why the solved plan sends all three
lifts to the lobby for the evening return peak but fans them up the shaft for
the morning exodus.

Relocating an idle lift is not free — it costs energy, wear, and a plan that
reshuffles every hour is one no facility manager will leave switched on — so
each move is charged a penalty and the whole day is solved as a single
mixed-integer program rather than 24 independent ones.

Solved with the open-source CBC solver bundled with PuLP, the same solver the
battery dispatch model in this repo uses: no commercial licence needed.

Scope
-----
This optimizes the *parking* policy — where lifts idle — not the live dispatch
decision of which moving car to send to a new call. That is the right scope:
parking is the part a building can actually configure in its group controller,
and it governs response times outside the peaks, which is most of the day.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pulp

from src.lifts.building import BuildingConfig


@dataclass
class PlacementConfig:
    """Operational rules and tuning for the parking plan."""

    slot_minutes: int = 60

    reposition_penalty_s: float = 45.0
    """Passenger-seconds charged each time a lift is sent to a new parking floor.
    Raise it for a calmer plan; set it to 0 for the unconstrained per-slot
    optimum, which will churn."""

    calls_per_lift_per_hour: float = 55.0
    """How many hall calls one lift can handle per hour — the group's capacity,
    and the thing that decides whether lifts should spread out or cluster.
    55/hour is roughly a 65-second round trip with light loading in a tower this
    height. Time a few round trips in the actual building and set it from that;
    it is the single most consequential number in this config."""

    max_utilisation: float = 0.95
    """Cap on `rho`. Past this the group is saturated, waits are governed by
    queueing rather than by parking, and placement can no longer help."""

    pin_lift_to_entry_above_share: Optional[float] = 0.30
    """If the entry floors generate at least this share of a slot's calls, force
    a lift to park there. An operational safeguard against a forecast that
    under-reads lobby traffic. Set to None to let the objective decide alone."""

    initial_park_floors: Optional[Sequence[int]] = None
    """Where the lifts sit entering the first slot, so the first relocation is
    costed honestly. Defaults to all lifts on the first entry floor."""

    solver_time_limit_s: int = 300
    solver_msg: bool = False


@dataclass
class PlacementResult:
    schedule: pd.DataFrame
    """Indexed by slot: one column per lift, plus `expected_wait_s` and `calls`."""

    status: str
    objective_passenger_seconds: float
    n_relocations: int


def availability_weights(
    demand: np.ndarray, building: BuildingConfig, config: PlacementConfig
) -> np.ndarray:
    """
    Probability that a call is answered by the k-th nearest parked lift, per slot.

    Returns (n_slots, n_lifts). Row `t` sums to 1: with every lift busy the call
    still waits for whichever frees up first, so the weights are normalised over
    the group rather than leaking probability away.
    """
    capacity = config.calls_per_lift_per_hour * building.n_lifts * (config.slot_minutes / 60)
    rho = np.clip(demand.sum(axis=1) / capacity, 0.0, config.max_utilisation)
    ranks = np.arange(building.n_lifts)
    weights = (1 - rho)[:, None] * rho[:, None] ** ranks[None, :]
    return weights / weights.sum(axis=1, keepdims=True)


def _ranked_travel(park_floors_row: np.ndarray, travel: np.ndarray) -> np.ndarray:
    """(n_lifts, n_floors) travel times from the parked lifts to each floor,
    sorted nearest-first per floor. Duplicated parking floors stay duplicated —
    that is what makes a second lift on a busy floor worth something."""
    return np.sort(travel[np.asarray(park_floors_row), :], axis=0)


def expected_wait_by_slot(
    demand: np.ndarray,
    park_floors: np.ndarray,
    building: BuildingConfig,
    config: Optional[PlacementConfig] = None,
) -> np.ndarray:
    """
    Mean seconds a caller waits in each slot under a given parking plan.

    `demand` is (n_slots, n_floors); `park_floors` is (n_slots, n_lifts).
    Slots with no calls report 0.
    """
    config = config or PlacementConfig()
    travel = building.travel_time_matrix()
    p = availability_weights(demand, building, config)
    waits = np.stack(
        [(p[t][:, None] * _ranked_travel(park_floors[t], travel)).sum(axis=0) for t in range(len(demand))]
    )  # (n_slots, n_floors)
    weighted = (demand * waits).sum(axis=1)
    totals = demand.sum(axis=1)
    return np.divide(weighted, totals, out=np.zeros_like(weighted), where=totals > 0)


def count_relocations(park_floors: np.ndarray) -> int:
    """Lift moves across the day: how many cars have to change floor between one
    slot and the next, counted as the multiset difference so that swapping which
    lift sits where isn't charged as a move."""
    total = 0
    for previous, current in zip(park_floors[:-1], park_floors[1:]):
        shared = sum((Counter(previous.tolist()) & Counter(current.tolist())).values())
        total += len(current) - shared
    return int(total)


def evaluate_placement(
    demand: np.ndarray,
    park_floors: np.ndarray,
    building: BuildingConfig,
    config: Optional[PlacementConfig] = None,
) -> dict:
    """
    Score a parking plan against a demand matrix — normally the *actual* demand,
    while the plan itself was built on a forecast.

    `mean_wait_s` is call-weighted across the horizon: the average wait a
    resident actually experiences, so the busy peaks count for more than the
    empty small hours. That is the number worth optimizing; a plain per-slot
    average would let a plan look good by serving 03:00 well.
    """
    per_slot = expected_wait_by_slot(demand, park_floors, building, config)
    calls = demand.sum(axis=1)
    total_calls = calls.sum()
    if not total_calls:
        raise ValueError("demand matrix contains no calls to evaluate against")
    busy = calls >= np.quantile(calls, 0.9)
    return {
        "mean_wait_s": float((per_slot * calls).sum() / total_calls),
        "peak_slot_wait_s": float((per_slot[busy] * calls[busy]).sum() / calls[busy].sum()),
        "worst_slot_wait_s": float(per_slot[calls > 0].max()),
        "total_passenger_hours": float((per_slot * calls).sum() / 3600),
        "n_relocations": count_relocations(park_floors),
    }


def brute_force_slot_placement(
    demand_row: np.ndarray, weights_row: np.ndarray, building: BuildingConfig
) -> tuple[tuple[int, ...], float]:
    """
    Exact optimum for a single slot, by enumerating every way to place the lifts
    — 6,545 combinations for 3 lifts across 33 floors, including doubling two on
    one floor, which takes milliseconds.

    Kept because it is the ground truth the MIP is checked against in the tests:
    an optimizer nobody has verified against an exhaustive search is a guess.
    """
    travel = building.travel_time_matrix()
    best_floors, best_cost = None, np.inf
    for combo in combinations_with_replacement(range(building.n_floors), building.n_lifts):
        ranked = np.sort(travel[list(combo), :], axis=0)
        cost = float((demand_row * (weights_row[:, None] * ranked).sum(axis=0)).sum())
        if cost < best_cost:
            best_floors, best_cost = combo, cost
    return best_floors, best_cost


def optimize_day_placement(
    demand: np.ndarray,
    building: BuildingConfig,
    config: Optional[PlacementConfig] = None,
    index: Optional[pd.DatetimeIndex] = None,
) -> PlacementResult:
    """
    Solve the whole day's parking plan in one mixed-integer program.

    Parameters
    ----------
    demand : np.ndarray, shape (n_slots, n_floors)
        Forecast hall calls originating on each floor in each slot.
    building, config, index
        Building geometry, tuning, and slot timestamps for the schedule.

    Returns
    -------
    PlacementResult

    Formulation
    -----------
    `park[t, g]` counts the lifts idling on floor `g` in slot `t` and sums to
    the group size. `serve[t, f, g, k]` routes floor `f`'s calls to the lift on
    floor `g` sitting at rank `k` in the queue of nearest lifts; each rank must
    be filled exactly once, and a floor can supply as many ranks as it has lifts
    parked on it. Because the rank weights decrease, the solver has no incentive
    to mislabel a far lift as the nearest one, so the ranking needs no explicit
    ordering constraints.

    The day is solved jointly rather than slot by slot because the relocation
    penalty couples the slots: it is worth holding a slightly worse position
    through a quiet hour to avoid a move that never pays for itself.
    """
    config = config or PlacementConfig()
    n_slots, n_floors = demand.shape
    if n_floors != building.n_floors:
        raise ValueError(f"demand has {n_floors} floors, building has {building.n_floors}")
    if (demand < 0).any():
        raise ValueError("demand must be non-negative")

    n_lifts = building.n_lifts
    travel = building.travel_time_matrix()
    entry = list(building.entry_floors)
    initial = (
        list(config.initial_park_floors)
        if config.initial_park_floors is not None
        else [entry[0]] * n_lifts
    )
    p = availability_weights(demand, building, config)

    prob = pulp.LpProblem("lift_floor_placement", pulp.LpMinimize)
    park = pulp.LpVariable.dicts(
        "park", (range(n_slots), range(n_floors)), lowBound=0, upBound=n_lifts, cat="Integer"
    )
    moved = pulp.LpVariable.dicts(
        "moved", (range(n_slots), range(n_floors)), lowBound=0, upBound=n_lifts
    )

    serve: dict[tuple[int, int, int, int], pulp.LpVariable] = {}
    waiting_terms = []
    active_ranks = {}
    for t in range(n_slots):
        # Ranks whose weight rounds to nothing carry no information and would
        # only triple the size of the program in the quiet overnight slots.
        active_ranks[t] = [k for k in range(n_lifts) if p[t, k] > 1e-4]
        for f in range(n_floors):
            if demand[t, f] <= 0:
                continue
            for k in active_ranks[t]:
                for g in range(n_floors):
                    var = pulp.LpVariable(f"serve_{t}_{f}_{g}_{k}", lowBound=0, upBound=1)
                    serve[t, f, g, k] = var
                    waiting_terms.append(demand[t, f] * p[t, k] * travel[g, f] * var)

    prob += pulp.lpSum(waiting_terms) + config.reposition_penalty_s * pulp.lpSum(
        moved[t][g] for t in range(n_slots) for g in range(n_floors)
    )

    for t in range(n_slots):
        prob += pulp.lpSum(park[t][g] for g in range(n_floors)) == n_lifts

        for f in range(n_floors):
            if demand[t, f] <= 0:
                continue
            for k in active_ranks[t]:
                prob += pulp.lpSum(serve[t, f, g, k] for g in range(n_floors)) == 1
            for g in range(n_floors):
                prob += pulp.lpSum(serve[t, f, g, k] for k in active_ranks[t]) <= park[t][g]

        for g in range(n_floors):
            previous = initial.count(g) if t == 0 else park[t - 1][g]
            prob += moved[t][g] >= park[t][g] - previous

        if config.pin_lift_to_entry_above_share is not None:
            slot_total = demand[t].sum()
            entry_share = demand[t, entry].sum() / slot_total if slot_total > 0 else 0.0
            if entry_share >= config.pin_lift_to_entry_above_share:
                prob += pulp.lpSum(park[t][e] for e in entry) >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=config.solver_msg, timeLimit=config.solver_time_limit_s))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution: {status}")

    chosen_rows = [
        [g for g in range(n_floors) for _ in range(int(round(park[t][g].value() or 0)))]
        for t in range(n_slots)
    ]
    if any(len(row) != n_lifts for row in chosen_rows):
        raise RuntimeError(
            "solver returned a slot that does not park exactly the whole group; "
            "this should be impossible for an integral solution"
        )
    park_floors = _assign_lift_identities(np.array(chosen_rows), initial)

    schedule = pd.DataFrame(
        park_floors,
        columns=[f"lift_{i + 1}_floor" for i in range(n_lifts)],
        index=index if index is not None else pd.RangeIndex(n_slots, name="slot"),
    )
    schedule["expected_wait_s"] = expected_wait_by_slot(demand, park_floors, building, config).round(2)
    schedule["calls"] = demand.sum(axis=1).round(2)

    return PlacementResult(
        schedule=schedule,
        status=status,
        objective_passenger_seconds=float(pulp.value(prob.objective)),
        n_relocations=count_relocations(park_floors),
    )


def _assign_lift_identities(chosen: np.ndarray, initial: Sequence[int]) -> np.ndarray:
    """
    Turn a per-slot *set* of parking floors into per-lift instructions.

    The optimizer only decides which floors are occupied — to it the cars are
    interchangeable. A schedule a person can act on has to name the lift, so
    walk the day forward and at each slot pick the assignment that moves the
    cars the least; with three lifts that is six permutations to check.
    """
    current = np.array(initial, dtype=int)
    out = np.empty_like(chosen)
    for t, floors in enumerate(chosen):
        best = min(permutations(floors), key=lambda perm: np.abs(np.array(perm) - current).sum())
        out[t] = best
        current = np.array(best, dtype=int)
    return out


def all_at_entry_plan(
    demand: np.ndarray, building: BuildingConfig, config: Optional[PlacementConfig] = None
) -> np.ndarray:
    """Baseline: every lift returns to the lobby when idle — the factory default
    in a great many residential group controllers, including, most likely,
    whatever is running in your building right now."""
    entry = list(building.entry_floors)
    floors = [entry[i % len(entry)] for i in range(building.n_lifts)]
    return np.tile(floors, (len(demand), 1))


def even_spread_plan(
    demand: np.ndarray, building: BuildingConfig, config: Optional[PlacementConfig] = None
) -> np.ndarray:
    """Baseline: park at evenly spaced heights and ignore demand entirely — the
    usual hand-tuned "zoning" answer."""
    fractions = (np.arange(building.n_lifts) + 0.5) / building.n_lifts
    floors = np.round(fractions * (building.n_floors - 1)).astype(int)
    return np.tile(floors, (len(demand), 1))


def static_optimal_plan(
    demand: np.ndarray, building: BuildingConfig, config: Optional[PlacementConfig] = None
) -> np.ndarray:
    """
    Baseline: the single best *fixed* set of parking floors for the whole day,
    found exactly by enumeration.

    This is the honest bar for a time-varying plan to clear. Beating "all lifts
    in the lobby" proves very little. Beating the best fixed plan available is
    what shows that moving the lifts around as the day goes on is worth the
    trouble at all.
    """
    config = config or PlacementConfig()
    travel = building.travel_time_matrix()
    p = availability_weights(demand, building, config)
    rank_demand = demand.T @ p  # (n_floors, n_lifts): calls at each floor, weighted by rank

    best_floors, best_cost = None, np.inf
    for combo in combinations_with_replacement(range(building.n_floors), building.n_lifts):
        ranked = np.sort(travel[list(combo), :], axis=0)  # (n_lifts, n_floors)
        cost = float((rank_demand.T * ranked).sum())
        if cost < best_cost:
            best_floors, best_cost = combo, cost
    return np.tile(np.array(best_floors), (len(demand), 1))


def compare_plans(
    demand_actual: np.ndarray,
    plans: dict[str, np.ndarray],
    building: BuildingConfig,
    config: Optional[PlacementConfig] = None,
    baseline: str = "all_at_lobby",
) -> pd.DataFrame:
    """Score several parking plans on the same actual demand, best mean wait first."""
    rows = {
        name: evaluate_placement(demand_actual, plan, building, config)
        for name, plan in plans.items()
    }
    table = pd.DataFrame(rows).T.sort_values("mean_wait_s")
    if baseline in rows:
        reference = rows[baseline]["mean_wait_s"]
        table["vs_baseline_pct"] = (100 * (table["mean_wait_s"] / reference - 1)).round(1)
    return table
