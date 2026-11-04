"""Which forecast score tracks revenue? Compares MAE with scores of what the battery needs.

For each score, two checks on the backtest written by run_backtest.py, across the five
forecasting methods (perfect foresight left out):
- day by day: the rank correlation between the score and the revenue, averaged over days,
  with a 95% interval and its difference from MAE's (paired by day, so the day's own
  difficulty cancels out);
- month by month: in how many months the method with the best score earned the most.
  Picking a method at random would hit 12 / 5 = 2.4 months on average.

Usage: uv run python scripts/compare_metrics.py --region SA1 --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
# +1: higher is better; -1: lower is better.
SCORES = {"mae": -1, "spread_error": -1, "rank_corr": 1, "trough_hits": 1, "peak_hits": 1, "extreme_hits": 1}


def methods_only(results: pd.DataFrame) -> pd.DataFrame:
    """The forecasting methods' rows (perfect foresight left out), plus the mean of both hit rates."""
    results = results[results["forecaster"] != "perfect_foresight"]
    return results.assign(extreme_hits=(results["trough_hits"] + results["peak_hits"]) / 2)


def oriented(results: pd.DataFrame) -> pd.DataFrame:
    """Scores flipped so that higher is always better."""
    return results.assign(**{s: results[s] * sign for s, sign in SCORES.items()})


def daily_agreement(results: pd.DataFrame) -> pd.DataFrame:
    """Per score: the mean over days of its rank correlation with revenue across methods.

    A day on which every method gets the same score has no correlation and is left out;
    `days` counts the days used. The interval and the gap to MAE use the normal
    approximation over days.
    """

    def one_day(day: pd.DataFrame) -> pd.Series:
        return day[list(SCORES)].rank().corrwith(day["revenue_aud"].rank())

    with np.errstate(invalid="ignore", divide="ignore"):  # tied days give NaN, as intended
        daily = results.groupby("day").apply(one_day)
    gap = daily.sub(daily["mae"], axis=0)

    def half_width(x: pd.DataFrame) -> pd.Series:
        return 1.96 * x.std() / np.sqrt(x.count())

    return pd.DataFrame(
        {
            "daily_agreement": daily.mean(),
            "ci95": half_width(daily),
            "days": daily.count(),
            "gap_to_mae": gap.mean(),
            "gap_ci95": half_width(gap),
        }
    )


def monthly_winners(results: pd.DataFrame) -> pd.Series:
    """Months in which the method with the best mean score also earned the most."""
    months = results.assign(month=results["day"].dt.to_period("M"))
    agg = months.groupby(["month", "forecaster"]).agg(
        revenue_aud=("revenue_aud", "sum"), **{s: (s, "mean") for s in SCORES}
    )
    top_earner = agg["revenue_aud"].groupby("month").idxmax()
    return pd.Series({s: int((agg[s].groupby("month").idxmax() == top_earner).sum()) for s in SCORES})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SA1")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    args = ap.parse_args()

    tag = f"{args.region}_{args.start}_{args.end}"
    results = methods_only(pd.read_csv(REPORTS / f"backtest_{tag}.csv", parse_dates=["day"]))
    scored = oriented(results)
    table = daily_agreement(scored).assign(months_top_earner=monthly_winners(scored))
    table.index.name = "score"
    table.to_csv(REPORTS / f"score_agreement_{tag}.csv")

    means = {s: (s, "mean") for s in SCORES}
    by_method = results.groupby("forecaster").agg(revenue_aud=("revenue_aud", "sum"), **means)
    by_method.to_csv(REPORTS / f"scores_by_method_{tag}.csv")

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 200)
    print(by_method.sort_values("revenue_aud", ascending=False), "\n")
    print(table.sort_values("daily_agreement", ascending=False))


if __name__ == "__main__":
    main()
