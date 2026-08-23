"""
Physical model of the building and its lift group.

Defaults describe Verona, Hiranandani Gardens (Powai, Mumbai): a 33-storey
residential tower served by a group of 3 lifts. Everything is a dataclass
field, so if the real tower differs (a podium parking level that lifts stop
at, a different rated speed on the nameplate, more flats on the lower
floors), change the config rather than the code — see `VERONA` at the bottom
of this module for the defaults and where each number came from.

The only thing the optimizer needs out of this module is
`travel_time_matrix()`: the expected seconds a resident on floor `f` waits
for a lift that is idle-parked on floor `g`. That is what makes one parking
plan better than another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class BuildingConfig:
    """Geometry, lift kinematics, and occupancy of a single residential tower."""

    name: str = "Verona"
    n_floors: int = 33  # number of levels the lifts serve, indexed 0 .. n_floors - 1
    lobby_floor: int = 0
    entry_floors: Sequence[int] = (0,)  # levels people enter the building from
    floor_height_m: float = 3.0

    # Lift group and kinematics
    n_lifts: int = 3
    rated_speed_mps: float = 1.6
    acceleration_mps2: float = 0.9
    start_delay_s: float = 1.5  # call registration + doors closing before it can move
    door_open_s: float = 2.5  # doors opening once it arrives (part of the felt wait)

    # Occupancy — drives how much call demand each floor generates
    flats_per_floor: int = 4
    occupants_per_flat: float = 3.2
    floor_population: Optional[Sequence[float]] = field(default=None)

    def __post_init__(self) -> None:
        if self.n_lifts > self.n_floors:
            raise ValueError("n_lifts cannot exceed n_floors: each lift parks on its own floor")
        if not all(0 <= f < self.n_floors for f in self.entry_floors):
            raise ValueError(f"entry_floors must lie within 0..{self.n_floors - 1}")

    @property
    def floors(self) -> np.ndarray:
        return np.arange(self.n_floors)

    def population(self) -> np.ndarray:
        """
        Residents per floor. Entry floors hold no flats; every residential
        floor holds `flats_per_floor * occupants_per_flat` people unless an
        explicit `floor_population` override is supplied.
        """
        if self.floor_population is not None:
            pop = np.asarray(self.floor_population, dtype=float)
            if len(pop) != self.n_floors:
                raise ValueError(f"floor_population must have {self.n_floors} entries")
            return pop
        pop = np.full(self.n_floors, self.flats_per_floor * self.occupants_per_flat, dtype=float)
        pop[list(self.entry_floors)] = 0.0
        return pop

    def flight_time_s(self, n_floors_travelled: int) -> float:
        """
        Seconds a lift spends moving across `n_floors_travelled` floors, on a
        trapezoidal speed profile: accelerate to rated speed, cruise, brake.
        Over short hops the lift never reaches rated speed, so the profile is
        triangular and the time goes as sqrt(distance) instead of distance —
        which is exactly why parking a lift 2 floors away is not merely twice
        as good as parking it 4 floors away.
        """
        distance_m = abs(n_floors_travelled) * self.floor_height_m
        if distance_m == 0:
            return 0.0
        v, a = self.rated_speed_mps, self.acceleration_mps2
        if distance_m >= v**2 / a:  # long enough to reach rated speed
            return distance_m / v + v / a
        return 2 * (distance_m / a) ** 0.5  # triangular profile, never cruises

    def response_time_s(self, park_floor: int, call_floor: int) -> float:
        """Seconds between a resident on `call_floor` pressing the hall button
        and stepping into a lift idle-parked on `park_floor`."""
        return self.start_delay_s + self.flight_time_s(park_floor - call_floor) + self.door_open_s

    def travel_time_matrix(self) -> np.ndarray:
        """
        (n_floors, n_floors) matrix `T` where `T[g, f]` is the response time in
        seconds for a call from floor `f` answered by a lift parked on floor `g`.
        Symmetric — this model ignores the small up/down asymmetry from
        counterweight loading, which is well below the noise in call data.
        """
        deltas = np.abs(self.floors[:, None] - self.floors[None, :])
        flight = np.vectorize(self.flight_time_s)(deltas)
        return self.start_delay_s + flight + self.door_open_s


# Defaults for the tower this project was built around. The floor count and
# lift count are the tower's own; the kinematics are the standard nameplate
# figures for a mid-rise gearless traction group of this height, and the
# occupancy is a per-floor average. Measure and override the ones you can:
# `rated_speed_mps` is on the lift's inspection certificate in the car, and
# `flats_per_floor` you can simply count.
VERONA = BuildingConfig(
    name="Verona, Hiranandani Gardens",
    n_floors=33,
    n_lifts=3,
    lobby_floor=0,
    entry_floors=(0,),
    floor_height_m=3.0,
    rated_speed_mps=1.6,
    flats_per_floor=4,
)
