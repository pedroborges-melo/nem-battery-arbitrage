"""Do the results hold for other batteries and other regions?

Re-evaluates the forecasts saved by run_backtest.py under each scenario below, so run that
first for every region listed (scenarios whose forecasts are missing are skipped). Writes,
per scenario, each method's capture rate and each score's agreement with revenue.

Usage: uv run python scripts/run_sensitivity.py --start 2024-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from nem_battery.agreement import agreement
from nem_battery.backtest import evaluate, load_forecasts, summarise
from nem_battery.battery import Battery
from nem_battery.data import PROCESSED_DIR, load_prices

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

BASE = Battery()
SCENARIOS: dict[str, tuple[str, Battery]] = {
    "South Australia (base)": ("SA1", BASE),
    "SA, 1-hour battery": ("SA1", Battery(energy_mwh=100)),
    "SA, 4-hour battery": ("SA1", Battery(energy_mwh=400)),
    "SA, no wear cost": ("SA1", Battery(degradation_aud_per_mwh=0)),
    "SA, wear cost AUD 30/MWh": ("SA1", Battery(degradation_aud_per_mwh=30)),
    "New South Wales": ("NSW1", BASE),
    "Victoria": ("VIC1", BASE),
    "Queensland": ("QLD1", BASE),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    args = ap.parse_args()

    capture, agree = [], []
    for name, (region, battery) in SCENARIOS.items():
        path = PROCESSED_DIR / f"forecasts_{region}_{args.start}_{args.end}.csv"
        if not path.exists():
            print(f"skipped {name}: run run_backtest.py --region {region} first")
            continue
        t0 = time.perf_counter()
        results = evaluate(load_forecasts(path), load_prices(region, args.start, args.end)["rrp"], battery)
        labels = {"scenario": name, "region": region, **asdict(battery)}
        capture.append(summarise(results).reset_index().assign(**labels))
        agree.append(agreement(results).reset_index().assign(**labels))
        print(f"{name} ({time.perf_counter() - t0:.0f}s)", flush=True)

    capture_table, agree_table = pd.concat(capture), pd.concat(agree)
    capture_table.to_csv(REPORTS / f"sensitivity_capture_{args.start}_{args.end}.csv", index=False)
    agree_table.to_csv(REPORTS / f"sensitivity_agreement_{args.start}_{args.end}.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 200)
    for title, table, column, value in (
        ("Capture rate", capture_table, "forecaster", "capture_rate"),
        ("Daily agreement with revenue", agree_table, "score", "daily_agreement"),
    ):
        print(f"\n{title}\n", table.pivot_table(index="scenario", columns=column, values=value, sort=False))


if __name__ == "__main__":
    main()
