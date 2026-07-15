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

from dataclasses import dataclass
from typing import Optional

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
