"""Baseline backtest: naive forecasts vs perfect foresight for one region.

Usage: uv run python scripts/run_baseline.py --region SA1 --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from nem_battery.backtest import run, summarise
from nem_battery.data import load_prices

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SA1")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    args = ap.parse_args()

    # Load 8 extra days of history so the first backtest day has its lookback window.
    history_start = (pd.Timestamp(args.start) - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    prices = load_prices(args.region, history_start, args.end)["rrp"]
    days = pd.date_range(args.start, pd.Timestamp(args.end) - pd.Timedelta(days=1), freq="D")

    t0 = time.perf_counter()
    results = run(prices, days)
    elapsed = time.perf_counter() - t0

    REPORTS.mkdir(exist_ok=True)
    tag = f"{args.region}_{args.start}_{args.end}"
    results.to_csv(REPORTS / f"backtest_{tag}.csv", index=False)
    summary = summarise(results)
    summary.to_csv(REPORTS / f"summary_{tag}.csv")

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(summary)
    print(f"\n{len(results)} optimisations in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
