"""Run the analysis from the command line, in three steps:

nem-battery forecast --region SA1   # slow, about 50 minutes per region; run once for each
nem-battery evaluate                # every scenario, about 6 minutes
nem-battery figures                 # the README's figures, from the reports
"""

from __future__ import annotations

import argparse

import pandas as pd

from .data import REGIONS
from .figures import draw_figures
from .pipeline import evaluate_scenarios, make_forecasts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="nem-battery", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    year = argparse.ArgumentParser(add_help=False)
    year.add_argument("--year", type=int, default=2024, help="calendar year to backtest (default: 2024)")
    commands = parser.add_subparsers(dest="command", required=True)
    forecast = commands.add_parser("forecast", parents=[year], help="forecast every day with every method and save")
    forecast.add_argument("--region", choices=REGIONS, default="SA1")
    commands.add_parser("evaluate", parents=[year], help="optimise and settle the saved forecasts in every scenario")
    commands.add_parser("figures", parents=[year], help="draw the README's figures from the reports")
    args = parser.parse_args(argv)

    if args.command == "forecast":
        make_forecasts(args.region, args.year)
    elif args.command == "evaluate":
        capture, agree = evaluate_scenarios(args.year)
        pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
        pd.set_option("display.width", 200)
        print("\nCapture rate\n", capture.pivot_table("capture_rate", "scenario", "forecaster", sort=False))
        print("\nDaily agreement with revenue\n", agree.pivot_table("daily_agreement", "scenario", "score", sort=False))
    else:
        draw_figures(args.year)


if __name__ == "__main__":
    main()
