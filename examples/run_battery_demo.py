"""
Runs the battery dispatch optimizer on synthetic price data and prints/plots
the resulting schedule. Works standalone today, before the Phase 1
forecasting model exists — this is a proof of the optimization core, which
later takes a forecast price series as input instead of synthetic/historical
prices, with no change to its interface.

Run from the repo root:
    python examples/run_battery_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src.data.loader import generate_synthetic_sample
from src.optimization.battery_dispatch import BatteryConfig, optimize_dispatch


def main():
    df = generate_synthetic_sample(n_days=3, seed=7)
    config = BatteryConfig(capacity_mwh=10, max_power_mw=2.5, initial_soc_mwh=5)
    result = optimize_dispatch(df["price_eur_mwh"], config)

    profit = -result["net_cost"].sum()
    print(f"3-day arbitrage profit: EUR {profit:,.2f}")
    print(result.head(24).to_string())

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(result.index, result["price"], label="price (EUR/MWh)", color="tab:blue")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].legend()

    axes[1].plot(result.index, result["soc_mwh"], label="state of charge (MWh)", color="tab:green")
    axes[1].bar(result.index, result["charge_mwh"], width=0.03, label="charge", alpha=0.5, color="tab:orange")
    axes[1].bar(result.index, -result["discharge_mwh"], width=0.03, label="discharge", alpha=0.5, color="tab:red")
    axes[1].set_ylabel("MWh")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("examples/battery_dispatch_demo.png", dpi=120)
    print("Saved chart to examples/battery_dispatch_demo.png")


if __name__ == "__main__":
    main()
