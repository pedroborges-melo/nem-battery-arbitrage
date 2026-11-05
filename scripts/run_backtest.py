"""Day-ahead backtest for one region: naive baselines, MSTL and LightGBM vs perfect foresight.

The forecasts are slow to make (MSTL refits every day) and do not depend on the battery,
so they are saved to data/processed/ and reused with --reuse-forecasts.

Usage: uv run python scripts/run_backtest.py --region SA1 --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from nem_battery.backtest import FORECASTERS, evaluate, forecast, load_forecasts, summarise
from nem_battery.data import PROCESSED_DIR, load_prices
from nem_battery.forecasters import GradientBoosting, SeasonalForecast

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
# LightGBM's first fit trains on this much history; the lag features need 7 more days.
TRAINING_HISTORY = pd.DateOffset(years=1, days=8)


def make_forecasts(data: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """All forecasts, one month at a time so the run reports its progress."""
    forecasters = {**FORECASTERS, "mstl": SeasonalForecast(), "lightgbm": GradientBoosting(data)}
    months = []
    t0 = time.perf_counter()
    for month, month_days in pd.Series(days, index=days).groupby(days.to_period("M")):
        months.append(forecast(data["rrp"], pd.DatetimeIndex(month_days), forecasters))
        print(f"forecast {month} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return pd.concat(months)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SA1")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--reuse-forecasts", action="store_true", help="evaluate the forecasts saved by an earlier run")
    args = ap.parse_args()

    tag = f"{args.region}_{args.start}_{args.end}"
    forecasts_path = PROCESSED_DIR / f"forecasts_{tag}.csv"
    history_start = (pd.Timestamp(args.start) - TRAINING_HISTORY).strftime("%Y-%m-%d")
    data = load_prices(args.region, history_start, args.end)
    days = pd.date_range(args.start, pd.Timestamp(args.end) - pd.Timedelta(days=1), freq="D")

    if args.reuse_forecasts:
        forecasts = load_forecasts(forecasts_path)
    else:
        forecasts = make_forecasts(data, days)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(forecasts_path)

    t0 = time.perf_counter()
    results = evaluate(forecasts, data["rrp"])
    print(f"{len(results)} optimisations in {time.perf_counter() - t0:.0f}s\n")

    REPORTS.mkdir(exist_ok=True)
    results.to_csv(REPORTS / f"backtest_{tag}.csv", index=False)
    summary = summarise(results)
    summary.to_csv(REPORTS / f"summary_{tag}.csv")
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(summary)


if __name__ == "__main__":
    main()
