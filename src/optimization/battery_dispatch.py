"""
Battery storage dispatch optimization.

Given an hourly electricity price series (real historical prices, or later
a forecast from the Phase 1 demand/price model), decide how much to charge
and discharge a battery each hour to minimize net purchase cost (equivalently,
maximize arbitrage profit), subject to capacity, power-rate, and round-trip
efficiency constraints.

Formulated as a mixed-integer program and solved with the open-source CBC
solver bundled with PuLP — runs with no commercial license and no external
solver install, so it's reproducible by anyone who clones this repo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import pandas as pd
import pulp


@dataclass
class BatteryConfig:
    capacity_mwh: float = 10.0  # usable energy capacity
    max_power_mw: float = 2.5  # max charge/discharge rate
    round_trip_efficiency: float = 0.9  # AC-AC round-trip efficiency
    initial_soc_mwh: float = 5.0
    min_soc_mwh: float = 0.0
    final_soc_min_mwh: Optional[float] = None  # e.g. require ending at >= initial_soc


def optimize_dispatch(prices: pd.Series, config: BatteryConfig) -> pd.DataFrame:
    """
    Solve the hourly battery charge/discharge schedule that minimizes net
    purchase cost given a price series.

    Parameters
    ----------
    prices : pd.Series
        Hourly electricity price (currency per MWh), indexed by timestamp.
    config : BatteryConfig

    Returns
    -------
    pd.DataFrame indexed like `prices`, with columns:
        price, charge_mwh, discharge_mwh, soc_mwh, net_cost
    net_cost is negative where the battery earns money (discharges at a
    higher price than it charged at).
    """
    hours = list(prices.index)
    n = len(hours)
    eff_leg = config.round_trip_efficiency**0.5  # split round-trip loss across each leg

    prob = pulp.LpProblem("battery_dispatch", pulp.LpMinimize)

    charge = pulp.LpVariable.dicts("charge", range(n), lowBound=0, upBound=config.max_power_mw)
    discharge = pulp.LpVariable.dicts("discharge", range(n), lowBound=0, upBound=config.max_power_mw)
    soc = pulp.LpVariable.dicts("soc", range(n), lowBound=config.min_soc_mwh, upBound=config.capacity_mwh)
    is_charging = pulp.LpVariable.dicts("is_charging", range(n), cat="Binary")

    # Objective: minimize net cost of energy purchased from the grid to run the battery
    prob += pulp.lpSum(prices.iloc[t] * (charge[t] - discharge[t]) for t in range(n))

    for t in range(n):
        prev_soc = config.initial_soc_mwh if t == 0 else soc[t - 1]
        prob += soc[t] == prev_soc + charge[t] * eff_leg - discharge[t] / eff_leg

        # Prevent simultaneous charge and discharge in the same hour
        prob += charge[t] <= config.max_power_mw * is_charging[t]
        prob += discharge[t] <= config.max_power_mw * (1 - is_charging[t])

    if config.final_soc_min_mwh is not None:
        prob += soc[n - 1] >= config.final_soc_min_mwh

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution: {pulp.LpStatus[prob.status]}")

    result = pd.DataFrame(
        {
            "price": prices.values,
            "charge_mwh": [charge[t].value() for t in range(n)],
            "discharge_mwh": [discharge[t].value() for t in range(n)],
            "soc_mwh": [soc[t].value() for t in range(n)],
        },
        index=prices.index,
    )
    result["net_cost"] = result["price"] * (result["charge_mwh"] - result["discharge_mwh"])
    return result


def rolling_horizon_dispatch(
    prices: pd.Series,
    config: BatteryConfig,
    window_hours: int = 48,
    step_hours: int = 24,
) -> pd.DataFrame:
    """
    Dispatch a long price series by solving overlapping short windows.

    Solving a full year as one MIP is both slow and operationally unrealistic:
    an operator commits the next day knowing roughly two days of prices, not
    twelve months. This solves a `window_hours` look-ahead, keeps only the
    first `step_hours` of the schedule, carries the resulting state of charge
    into the next window, and repeats.

    The kept portion is a genuine dispatch decision — the extra look-ahead
    exists only to stop the battery from dumping its whole charge at the end
    of every window.
    """
    if step_hours <= 0 or window_hours < step_hours:
        raise ValueError("require 0 < step_hours <= window_hours")

    pieces: list[pd.DataFrame] = []
    soc = config.initial_soc_mwh

    for start in range(0, len(prices), step_hours):
        window = prices.iloc[start : start + window_hours]
        if window.empty:
            break

        window_config = replace(config, initial_soc_mwh=soc, final_soc_min_mwh=None)
        solved = optimize_dispatch(window, window_config)

        kept = solved.iloc[:step_hours]
        pieces.append(kept)
        soc = float(kept["soc_mwh"].iloc[-1])

    return pd.concat(pieces)


def settle_at_actual_prices(schedule: pd.DataFrame, actual_prices: pd.Series) -> pd.DataFrame:
    """
    Re-price a schedule that was optimized against forecast prices.

    The battery's physical constraints do not depend on price, so a schedule
    built from a forecast is always physically feasible — it is just not
    necessarily *profitable*. Settling it at the prices that actually occurred
    is the only honest way to score a forecast-driven strategy, and the gap to
    perfect foresight is the real cost of forecast error.
    """
    aligned = actual_prices.reindex(schedule.index)
    if aligned.isna().any():
        raise ValueError("actual_prices does not cover every timestamp in the schedule")

    settled = schedule.copy()
    settled["price"] = aligned
    settled["net_cost"] = aligned * (settled["charge_mwh"] - settled["discharge_mwh"])
    return settled


def fixed_rule_dispatch(
    prices: pd.Series,
    config: BatteryConfig,
    charge_hours: Sequence[int] = (1, 2, 3, 4),
    discharge_hours: Sequence[int] = (18, 19, 20, 21),
) -> pd.DataFrame:
    """
    The no-model baseline: charge overnight, discharge into the evening peak,
    on the same clock every day, ignoring prices entirely.

    This is what a site runs without a forecast, and it is the number the
    optimizer has to beat to be worth anything. It respects the same capacity
    and power limits, so the comparison is apples to apples.
    """
    charge_set, discharge_set = set(charge_hours), set(discharge_hours)
    eff_leg = config.round_trip_efficiency**0.5

    soc = config.initial_soc_mwh
    charges, discharges, socs = [], [], []

    for timestamp in prices.index:
        hour = timestamp.hour
        charge = discharge = 0.0

        if hour in charge_set:
            headroom = config.capacity_mwh - soc
            charge = min(config.max_power_mw, headroom / eff_leg if eff_leg else 0.0)
        elif hour in discharge_set:
            available = soc - config.min_soc_mwh
            discharge = min(config.max_power_mw, available * eff_leg)

        soc += charge * eff_leg - discharge / eff_leg
        charges.append(charge)
        discharges.append(discharge)
        socs.append(soc)

    result = pd.DataFrame(
        {
            "price": prices.values,
            "charge_mwh": charges,
            "discharge_mwh": discharges,
            "soc_mwh": socs,
        },
        index=prices.index,
    )
    result["net_cost"] = result["price"] * (result["charge_mwh"] - result["discharge_mwh"])
    return result


def summarize_dispatch(result: pd.DataFrame, config: BatteryConfig) -> dict:
    """
    Reduce a dispatch schedule to the numbers a stakeholder asks about.

    `profit` is the negated net cost: positive means the battery earned money.
    `equivalent_full_cycles` counts discharged energy against usable capacity —
    the standard way of expressing battery wear, and the reason a schedule that
    earns slightly more while cycling twice as hard is not automatically better.
    """
    charged = float(result["charge_mwh"].sum())
    discharged = float(result["discharge_mwh"].sum())
    profit = float(-result["net_cost"].sum())

    charge_cost = float((result["price"] * result["charge_mwh"]).sum())
    discharge_revenue = float((result["price"] * result["discharge_mwh"]).sum())

    return {
        "profit": profit,
        "energy_charged_mwh": charged,
        "energy_discharged_mwh": discharged,
        "round_trip_loss_mwh": charged - discharged,
        "avg_charge_price": charge_cost / charged if charged else None,
        "avg_discharge_price": discharge_revenue / discharged if discharged else None,
        "equivalent_full_cycles": discharged / config.capacity_mwh if config.capacity_mwh else None,
        "profit_per_mwh_capacity": profit / config.capacity_mwh if config.capacity_mwh else None,
        "hours": int(len(result)),
        "hours_charging": int((result["charge_mwh"] > 1e-6).sum()),
        "hours_discharging": int((result["discharge_mwh"] > 1e-6).sum()),
        "final_soc_mwh": float(result["soc_mwh"].iloc[-1]) if len(result) else None,
    }
