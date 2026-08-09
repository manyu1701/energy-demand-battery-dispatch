"""
Regenerate the committed sample dataset at data/sample/hourly_energy_sample.csv.

The output is deterministic for a given seed, so re-running this on a clean
checkout reproduces the exact file that is checked in. The dataset is
synthetic — no real market data ships in this repository — but it is built to
have the structure a forecaster has to handle: annual seasonality, holidays,
autocorrelated residuals, renewable-driven negative prices and scarcity spikes.

Run from the repo root:
    python scripts/generate_sample_data.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import SAMPLE_DIR, SAMPLE_FILENAME, generate_hourly_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365, help="number of days to generate")
    parser.add_argument("--seed", type=int, default=20240101, help="RNG seed")
    parser.add_argument("--start", default="2024-01-01", help="first timestamp (YYYY-MM-DD)")
    parser.add_argument(
        "--out",
        type=Path,
        default=SAMPLE_DIR / SAMPLE_FILENAME,
        help="output CSV path",
    )
    args = parser.parse_args()

    df = generate_hourly_dataset(n_days=args.days, start=args.start, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)

    price = df["price_eur_mwh"]
    print(f"Wrote {len(df):,} hourly rows to {args.out}")
    print(f"  span         {df.index.min()} -> {df.index.max()}")
    print(f"  demand MW    min {df['demand_mw'].min():,.0f}  mean {df['demand_mw'].mean():,.0f}  max {df['demand_mw'].max():,.0f}")
    print(f"  price /MWh   min {price.min():,.1f}  mean {price.mean():,.1f}  max {price.max():,.1f}")
    print(f"  negative-price hours {int((price < 0).sum())} ({(price < 0).mean():.1%})")


if __name__ == "__main__":
    main()
