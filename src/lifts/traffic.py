"""
Hall-call demand for a residential tower.

`generate_synthetic_call_log()` produces an event-level log in the same shape
a lift controller exports — one row per hall call, with a timestamp, the floor
the button was pressed on, and the direction — so the forecasting model can be
built and validated before any real data exists. `load_real_call_log()` reads
the real thing once it's been exported (see README for how to get it).

The demand profile is a residential Mumbai tower, not an office block, and the
difference matters: an office has one sharp up-peak into the lobby in the
morning and one down-peak out of it at 18:00, whereas a tower like Verona has
a *down*-peak in the morning as residents leave and an *up*-peak from the lobby
in the evening as they return, plus two things an office never sees — domestic
staff arriving through the morning and leaving mid-afternoon, and a long
delivery tail into the late evening. Weekends shift the morning ~2 hours later
and thicken midday and late-evening traffic.

Streams modelled, each as a sum of Gaussian bumps over hour-of-day:

  leave_home    down-calls on residential floors  (school run, office commute)
  arrive_home   up-calls at the entry floor       (residents returning)
  service_in    up-calls at the entry floor       (staff, deliveries, visitors)
  service_out   down-calls on residential floors  (the same people leaving)
  interfloor    low-level two-way traffic between residential floors
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.lifts.building import BuildingConfig

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DIRECTIONS = ("up", "down")


def _gauss(hours: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((hours - mu) / sigma) ** 2)


def _leave_home_shape(hours: np.ndarray, is_rest_day: bool) -> np.ndarray:
    """Calls per resident per hour, down-direction, from residential floors."""
    if is_rest_day:
        return (
            0.055 * _gauss(hours, 10.0, 1.8)  # late, unhurried morning exit
            + 0.045 * _gauss(hours, 17.0, 2.2)  # afternoon/evening outing
            + 0.020 * _gauss(hours, 20.5, 1.5)
        )
    return (
        0.045 * _gauss(hours, 7.1, 0.45)  # school run — sharp, bus/van waits downstairs
        + 0.090 * _gauss(hours, 9.1, 1.35)  # office commute, spread by Mumbai travel times
        + 0.020 * _gauss(hours, 17.6, 1.6)  # evening walk / playground / errands
        + 0.012 * _gauss(hours, 20.5, 1.4)
    )


def _arrive_home_shape(hours: np.ndarray, is_rest_day: bool) -> np.ndarray:
    """Calls per resident per hour, up-direction, at the entry floor."""
    if is_rest_day:
        return (
            0.030 * _gauss(hours, 13.5, 2.0)
            + 0.060 * _gauss(hours, 20.0, 2.2)
            + 0.025 * _gauss(hours, 23.0, 1.2)  # late returns from Powai restaurants
        )
    return (
        0.030 * _gauss(hours, 14.2, 1.1)  # school pick-up returning
        + 0.100 * _gauss(hours, 19.8, 1.7)  # the day's dominant peak
        + 0.030 * _gauss(hours, 22.3, 1.3)
    )


def _service_in_shape(hours: np.ndarray, is_rest_day: bool) -> np.ndarray:
    """Up-calls at the entry floor from domestic staff, deliveries, visitors."""
    base = (
        0.055 * _gauss(hours, 8.6, 1.5)  # domestic staff arriving
        + 0.030 * _gauss(hours, 12.3, 1.3)  # lunch deliveries, grocery drops
        + 0.045 * _gauss(hours, 20.3, 1.8)  # dinner deliveries, evening visitors
    )
    return base * (1.25 if is_rest_day else 1.0)  # more visitors on rest days


def _service_out_shape(hours: np.ndarray, is_rest_day: bool) -> np.ndarray:
    """Down-calls on residential floors as staff and deliveries leave."""
    base = (
        0.020 * _gauss(hours, 9.3, 1.4)
        + 0.050 * _gauss(hours, 13.6, 1.6)  # staff finishing the day's work
        + 0.035 * _gauss(hours, 20.8, 1.8)
    )
    return base * (1.25 if is_rest_day else 1.0)


def _interfloor_shape(hours: np.ndarray, is_rest_day: bool) -> np.ndarray:
    """Low, flat two-way traffic between residential floors."""
    active = ((hours >= 8) & (hours <= 22.5)).astype(float)
    return (0.012 if is_rest_day else 0.008) * active


def _floor_activity(building: BuildingConfig, seed: int) -> np.ndarray:
    """
    A fixed per-floor multiplier on call generation. Real floors differ
    persistently — a floor with two young families generates far more trips
    than one with two retired couples — and because the effect is stable over
    time it is exactly the kind of structure the forecasting model can learn
    and the naive "spread the lifts evenly" policy cannot.
    """
    rng = np.random.default_rng(seed + 9973)
    return rng.lognormal(mean=0.0, sigma=0.18, size=building.n_floors)


def generate_synthetic_call_log(
    building: BuildingConfig,
    n_days: int = 90,
    seed: int = 42,
    start: str = "2025-01-06",  # a Monday
    holidays: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Generate an event-level hall-call log.

    Parameters
    ----------
    building : BuildingConfig
    n_days : int
        Days of history to generate.
    seed : int
    start : str
        First date of the log.
    holidays : iterable of date strings
        Public holidays (Diwali, Ganesh Chaturthi, ...). Treated as rest days
        with heavier visitor traffic.

    Returns
    -------
    pd.DataFrame with one row per hall call and columns:
        timestamp, origin_floor, direction, destination_floor
    """
    rng = np.random.default_rng(seed)
    holiday_dates = {pd.Timestamp(d).date() for d in holidays}
    pop = building.population()
    activity = _floor_activity(building, seed)
    entry = list(building.entry_floors)
    residential = np.array([f for f in range(building.n_floors) if f not in entry])
    total_pop = pop.sum()
    hours = np.arange(24, dtype=float) + 0.5  # evaluate each stream mid-hour

    # Weight for splitting entry-floor traffic across multiple entry levels,
    # and for choosing which residential floor a lobby passenger is heading to.
    entry_weights = np.ones(len(entry)) / len(entry)
    dest_weights = pop[residential] * activity[residential]
    dest_weights = dest_weights / dest_weights.sum()

    records = []
    for day in range(n_days):
        date = pd.Timestamp(start) + pd.Timedelta(days=day)
        is_rest_day = date.dayofweek >= 5 or date.date() in holiday_dates
        # One multiplier for the whole day: monsoon downpours, long weekends and
        # festivals move a tower's entire traffic level up or down together.
        day_factor = rng.lognormal(0.0, 0.10)

        down_res = np.outer(  # (24, n_residential) expected down-calls
            _leave_home_shape(hours, is_rest_day) + _service_out_shape(hours, is_rest_day),
            pop[residential] * activity[residential],
        )
        up_entry_total = (
            _arrive_home_shape(hours, is_rest_day) + _service_in_shape(hours, is_rest_day)
        ) * total_pop
        up_entry = np.outer(up_entry_total, entry_weights)  # (24, n_entry)
        inter = np.outer(
            _interfloor_shape(hours, is_rest_day), pop[residential] * activity[residential]
        )

        for source, floors, direction, matrix in (
            ("leave", residential, "down", down_res * day_factor),
            ("arrive", np.array(entry), "up", up_entry * day_factor),
            ("inter_up", residential, "up", inter * day_factor * 0.5),
            ("inter_down", residential, "down", inter * day_factor * 0.5),
        ):
            counts = rng.poisson(matrix)
            hour_idx, floor_idx = np.nonzero(counts)
            if len(hour_idx) == 0:
                continue
            reps = counts[hour_idx, floor_idx]
            call_hours = np.repeat(hour_idx, reps)
            call_floors = np.repeat(floors[floor_idx], reps)
            minutes = rng.uniform(0, 60, size=len(call_hours))
            timestamps = (
                date
                + pd.to_timedelta(call_hours, unit="h")
                + pd.to_timedelta(minutes, unit="m")
            )
            if source == "arrive":
                destinations = rng.choice(residential, size=len(call_floors), p=dest_weights)
            elif source == "leave":
                destinations = rng.choice(entry, size=len(call_floors))
            else:  # interfloor: another residential floor, usually a near one
                offsets = rng.integers(1, 6, size=len(call_floors)) * rng.choice(
                    [-1, 1], size=len(call_floors)
                )
                destinations = np.clip(call_floors + offsets, min(residential), max(residential))
            records.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "origin_floor": call_floors.astype(int),
                        "direction": direction,
                        "destination_floor": np.asarray(destinations).astype(int),
                    }
                )
            )

    log = pd.concat(records, ignore_index=True).sort_values("timestamp", ignore_index=True)
    log["direction"] = pd.Categorical(log["direction"], categories=DIRECTIONS)
    return log


def aggregate_to_slots(
    calls: pd.DataFrame,
    building: BuildingConfig,
    slot_minutes: int = 60,
    holidays: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Bin an event-level call log onto a dense (slot x floor x direction) grid.

    Dense matters: the hours and floors with *no* calls are as informative to
    the model as the busy ones, and the optimizer needs a value for every
    floor in every slot. Returns columns:
        slot_start, floor, direction, calls
    """
    if 1440 % slot_minutes:
        raise ValueError("slot_minutes must divide a 24-hour day evenly")
    binned = calls.assign(
        slot_start=calls["timestamp"].dt.floor(f"{slot_minutes}min"),
        direction=calls["direction"].astype(str),
    )
    counted = (
        binned.groupby(["slot_start", "origin_floor", "direction"], observed=True)
        .size()
        .rename("calls")
        .reset_index()
        .rename(columns={"origin_floor": "floor"})
    )

    # Whole days, so every date in the log carries a complete 24 hours of slots
    # even where the quiet pre-dawn hours saw no calls at all.
    slots = pd.date_range(
        calls["timestamp"].min().normalize(),
        calls["timestamp"].max().normalize() + pd.Timedelta(days=1),
        freq=f"{slot_minutes}min",
        inclusive="left",
    )
    grid = pd.MultiIndex.from_product(
        [slots, range(building.n_floors), DIRECTIONS], names=["slot_start", "floor", "direction"]
    ).to_frame(index=False)

    out = grid.merge(counted, on=["slot_start", "floor", "direction"], how="left")
    out["calls"] = out["calls"].fillna(0.0)
    holiday_dates = {pd.Timestamp(d).date() for d in holidays}
    out["is_holiday"] = out["slot_start"].dt.date.isin(holiday_dates)
    return out.sort_values(["slot_start", "floor", "direction"], ignore_index=True)


def to_demand_matrix(slots: pd.DataFrame, building: BuildingConfig) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Collapse a slot table to the (n_slots, n_floors) matrix of total hall calls
    the placement optimizer consumes. Direction is summed away here: a lift
    parked on a floor answers a call from it whichever way the resident is
    going, so placement only cares where the button is pressed.
    """
    wide = (
        slots.groupby(["slot_start", "floor"], observed=True)["calls"]
        .sum()
        .unstack("floor")
        .reindex(columns=range(building.n_floors), fill_value=0.0)
        .fillna(0.0)
        .sort_index()
    )
    return pd.DatetimeIndex(wide.index), wide.to_numpy(dtype=float)


def load_real_call_log(filename: str = "lift_calls.csv") -> pd.DataFrame:
    """
    Load a real hall-call log exported from the lift controller, from data/raw/.

    Most modern group controllers (Otis, Kone, Schindler, Mitsubishi) can export
    a call/trip log over their service tool or building-management interface;
    ask the facility manager or the AMC engineer on their next service visit.
    Expected columns after your own light renaming:
        timestamp, origin_floor, direction ("up"/"down"), destination_floor
    `destination_floor` is optional — placement only uses the origin — so fill
    it with the origin if your export lacks car-call data.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Export a call log from the lift controller (see docstring), "
            f"save it as data/raw/{filename}, or use generate_synthetic_call_log() to develop "
            "against synthetic traffic."
        )
    log = pd.read_csv(path, parse_dates=["timestamp"])
    missing = {"timestamp", "origin_floor", "direction"} - set(log.columns)
    if missing:
        raise ValueError(f"call log is missing required columns: {sorted(missing)}")
    if "destination_floor" not in log.columns:
        log["destination_floor"] = log["origin_floor"]
    return log.sort_values("timestamp", ignore_index=True)
