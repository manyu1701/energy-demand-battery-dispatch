"""
End-to-end demo: forecast tomorrow's hall calls at Verona, then decide where
each of the three lifts should wait through the day.

Run from the repo root:
    python examples/run_lift_placement_demo.py

What it does, in the order a building would do it:

  1. Build 120 days of synthetic call history for the tower (stand-in for a
     controller export — see src/lifts/traffic.py to point it at real data).
  2. Train the per-floor demand model on everything up to a hold-out window and
     report its accuracy on days it has never seen.
  3. For one weekday and one Sunday in the hold-out, forecast the day's demand
     floor by floor and hour by hour, and solve the parking plan against that
     forecast alone.
  4. Score that plan against what actually happened, next to three baselines —
     including the best possible *fixed* plan, which is the bar that matters.

The plan is scored on demand it never saw, so the numbers printed are what the
building would really have got, not a fit to its own answer sheet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lifts.building import VERONA
from src.lifts.pipeline import plan_day, score_day_plan, train_forecaster
from src.lifts.placement import (
    PlacementConfig,
    all_at_entry_plan,
    compare_plans,
    even_spread_plan,
    static_optimal_plan,
)
from src.lifts.traffic import aggregate_to_slots, generate_synthetic_call_log, to_demand_matrix

HISTORY_DAYS = 120
PLAN_DATES = ["2025-04-24", "2025-04-27"]  # a Thursday and a Sunday


def main():
    building = VERONA
    config = PlacementConfig()

    print(f"Building: {building.name} — {building.n_floors} floors, {building.n_lifts} lifts")
    print(f"Lobby to top floor takes {building.response_time_s(0, building.n_floors - 1):.0f}s\n")

    calls = generate_synthetic_call_log(building, n_days=HISTORY_DAYS, seed=11)
    slots = aggregate_to_slots(calls, building)
    print(f"Call log: {len(calls):,} calls over {HISTORY_DAYS} days "
          f"({len(calls) / HISTORY_DAYS:.0f}/day)")

    print("\nTraining per-floor demand model...")
    history = slots[slots["slot_start"] < pd.Timestamp(PLAN_DATES[0])]
    forecaster, metrics = train_forecaster(history, building, test_days=14)
    print(f"  hold-out WMAPE per floor-slot : {metrics['wmape_floor_slot']:.1%}")
    print(f"  hold-out WMAPE per building-slot: {metrics['wmape_building_slot']:.1%}")
    print(f"  hold-out bias                 : {metrics['bias_floor_slot']:+.1%}")
    print("  top features: " + ", ".join(forecaster.feature_importance().head(5).index))

    day_index, actual_all = to_demand_matrix(slots, building)
    plans_for_plot = {}

    for date in PLAN_DATES:
        day = pd.Timestamp(date)
        print(f"\n{'=' * 70}\n{day:%A %d %B %Y}\n{'=' * 70}")
        past = slots[slots["slot_start"] < day]
        plan = plan_day(forecaster, past, day, building, config)

        mask = (day_index >= day) & (day_index < day + pd.Timedelta(days=1))
        actual = actual_all[mask]

        schedule = plan.result.schedule.copy()
        schedule["actual_calls"] = actual.sum(axis=1).round(0)
        schedule.index = schedule.index.strftime("%H:%M")
        print(schedule.to_string())
        print(f"\nRelocations across the day: {plan.result.n_relocations}")

        candidates = {
            "all_at_lobby": all_at_entry_plan(actual, building),
            "even_spread": even_spread_plan(actual, building),
            "best_static": static_optimal_plan(actual, building, config),
            "ml_dynamic": plan.park_floors,
        }
        table = compare_plans(actual, candidates, building, config)
        print("\nScored on the day's ACTUAL calls (the plan only ever saw the forecast):")
        print(table.round(2).to_string())

        best_static_wait = table.loc["best_static", "mean_wait_s"]
        ml_wait = table.loc["ml_dynamic", "mean_wait_s"]
        print(
            f"\n  vs the lobby default : {table.loc['ml_dynamic', 'vs_baseline_pct']:+.1f}% mean wait"
            f"\n  vs the best fixed plan: {100 * (ml_wait / best_static_wait - 1):+.1f}% mean wait"
        )
        plans_for_plot[date] = (plan, actual)

    _plot(plans_for_plot, building)


def _plot(plans_for_plot, building):
    fig, axes = plt.subplots(1, len(plans_for_plot), figsize=(13, 6), sharey=True)
    axes = np.atleast_1d(axes)

    # The lobby generates roughly half of all calls, so scaling the colours to it
    # flattens every residential floor into one shade. Scale to the busiest
    # residential floor instead and let the lobby row saturate.
    residential = [f for f in range(building.n_floors) if f not in building.entry_floors]
    vmax = max(actual[:, residential].max() for _, actual in plans_for_plot.values())

    for ax, (date, (plan, actual)) in zip(axes, plans_for_plot.items()):
        mesh = ax.pcolormesh(
            np.arange(actual.shape[0] + 1),
            np.arange(building.n_floors + 1),
            actual.T,
            cmap="magma_r",
            shading="flat",
            vmin=0,
            vmax=vmax,
        )
        colours = ["tab:cyan", "tab:green", "tab:orange"]
        hours = np.arange(actual.shape[0]) + 0.5
        for lift in range(building.n_lifts):
            floors = plan.park_floors[:, lift] + 0.5
            # Jitter the lines apart so lifts sharing a floor stay visible
            ax.step(
                hours,
                floors + (lift - 1) * 0.22,
                where="mid",
                color=colours[lift % len(colours)],
                linewidth=2,
                label=f"lift {lift + 1}",
            )
        ax.set_title(f"{pd.Timestamp(date):%A %d %b}")
        ax.set_xlabel("hour of day")
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xlim(0, actual.shape[0])

    axes[0].set_ylabel("floor")
    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.colorbar(mesh, ax=axes.tolist(), label="hall calls per hour (lobby row saturates)", pad=0.02, extend="max")
    fig.suptitle(
        f"{building.name}: where each lift should wait, against the calls it has to answer",
        fontsize=12,
    )
    out = Path(__file__).resolve().parent / "lift_placement_demo.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved chart to {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
