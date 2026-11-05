"""Which forecast score tracks revenue? Compares MAE with scores of what the battery needs.

Reads the backtest written by run_backtest.py; the checks are described in
nem_battery/agreement.py. Picking a method at random would make it the month's top
earner in 12 / 5 = 2.4 months on average.

Usage: uv run python scripts/compare_metrics.py --region SA1 --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nem_battery.agreement import SCORES, agreement, methods_only

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SA1")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    args = ap.parse_args()

    tag = f"{args.region}_{args.start}_{args.end}"
    results = pd.read_csv(REPORTS / f"backtest_{tag}.csv", parse_dates=["day"])
    table = agreement(results)
    table.to_csv(REPORTS / f"score_agreement_{tag}.csv")

    means = {s: (s, "mean") for s in SCORES}
    by_method = methods_only(results).groupby("forecaster").agg(revenue_aud=("revenue_aud", "sum"), **means)
    by_method.to_csv(REPORTS / f"scores_by_method_{tag}.csv")

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 200)
    print(by_method.sort_values("revenue_aud", ascending=False), "\n")
    print(table.sort_values("daily_agreement", ascending=False))


if __name__ == "__main__":
    main()
