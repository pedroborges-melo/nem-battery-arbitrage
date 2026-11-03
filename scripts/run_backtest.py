"""Day-ahead backtest for one region: naive baselines, MSTL and LightGBM vs perfect foresight.

Usage: uv run python scripts/run_backtest.py --region SA1 --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from nem_battery.backtest import FORECASTERS, run, summarise
from nem_battery.data import load_prices
from nem_battery.forecasters import GradientBoosting, SeasonalForecast

REPORTS = Path(__file__).resolve().parents[1] / "reports"
# LightGBM's first fit trains on this much history; the lag features need 7 more days.
TRAINING_HISTORY = pd.DateOffset(years=1, days=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SA1")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    args = ap.parse_args()

    history_start = (pd.Timestamp(args.start) - TRAINING_HISTORY).strftime("%Y-%m-%d")
    data = load_prices(args.region, history_start, args.end)
    days = pd.date_range(args.start, pd.Timestamp(args.end) - pd.Timedelta(days=1), freq="D")
    forecasters = {**FORECASTERS, "mstl": SeasonalForecast(), "lightgbm": GradientBoosting(data)}

    t0 = time.perf_counter()
    results = run(data["rrp"], days, forecasters)
    elapsed = time.perf_counter() - t0

    REPORTS.mkdir(exist_ok=True)
    tag = f"{args.region}_{args.start}_{args.end}"
    results.to_csv(REPORTS / f"backtest_{tag}.csv", index=False)
    summary = summarise(results)
    summary.to_csv(REPORTS / f"summary_{tag}.csv")

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(summary)
    print(f"\n{len(results)} forecasts and optimisations in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
