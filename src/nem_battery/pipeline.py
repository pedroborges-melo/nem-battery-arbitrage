"""The analysis in two steps, plus where each step keeps its files.

1. `make_forecasts` runs every method over one region and year and saves the forecasts.
   It is slow (about 50 minutes per region, mostly MSTL) and runs once.
2. `evaluate_scenarios` optimises and settles the saved forecasts for every scenario
   below and writes the reports. It takes about 40 seconds per scenario.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .agreement import agreement
from .backtest import evaluate, forecast, load_forecasts, summarise
from .battery import Battery
from .data import PROCESSED_DIR, REPORTS_DIR, load_prices
from .forecasters import all_forecasters

BASE = Battery()
BASE_SCENARIO = "South Australia (base)"
SCENARIOS: dict[str, tuple[str, Battery]] = {
    BASE_SCENARIO: ("SA1", BASE),
    "SA, 1-hour battery": ("SA1", Battery(energy_mwh=100)),
    "SA, 4-hour battery": ("SA1", Battery(energy_mwh=400)),
    "SA, no wear cost": ("SA1", Battery(degradation_aud_per_mwh=0)),
    "SA, wear cost AUD 30/MWh": ("SA1", Battery(degradation_aud_per_mwh=30)),
    "New South Wales": ("NSW1", BASE),
    "Victoria": ("VIC1", BASE),
    "Queensland": ("QLD1", BASE),
}

# LightGBM's first fit trains on a year of history, and its lag features need 7 more days.
TRAINING_HISTORY = pd.DateOffset(years=1, days=8)


def forecasts_path(region: str, year: int) -> Path:
    return PROCESSED_DIR / f"forecasts_{region}_{year}.csv"


def daily_path(region: str, year: int) -> Path:
    """Day-by-day results for the base battery."""
    return REPORTS_DIR / f"daily_{region}_{year}.csv"


def capture_path(year: int) -> Path:
    """Each method's capture rate in each scenario."""
    return REPORTS_DIR / f"capture_{year}.csv"


def agreement_path(year: int) -> Path:
    """Each score's agreement with revenue in each scenario."""
    return REPORTS_DIR / f"agreement_{year}.csv"


def year_days(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")


def make_forecasts(region: str, year: int) -> None:
    """Forecast every day of `year` with every method and save the result, reporting progress monthly."""
    days = year_days(year)
    data = load_prices(region, days[0] - TRAINING_HISTORY, days[-1] + pd.Timedelta(days=1))
    forecasters = all_forecasters(data)
    months = days.to_period("M")
    started = time.perf_counter()
    forecasts = []
    for month in months.unique():
        forecasts.append(forecast(data["rrp"], days[months == month], forecasters))
        print(f"{region} {month} forecast ({time.perf_counter() - started:.0f}s)", flush=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(forecasts).to_csv(forecasts_path(region, year))


def evaluate_scenarios(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every scenario whose forecasts exist and write the reports.

    Returns the capture-rate and agreement tables, one block of rows per scenario.
    """
    days = year_days(year)
    REPORTS_DIR.mkdir(exist_ok=True)
    capture, agree = [], []
    for name, (region, battery) in SCENARIOS.items():
        path = forecasts_path(region, year)
        if not path.exists():
            print(f"skipped {name}: run `nem-battery forecast --region {region} --year {year}` first")
            continue
        started = time.perf_counter()
        prices = load_prices(region, days[0], days[-1] + pd.Timedelta(days=1))["rrp"]
        results = evaluate(load_forecasts(path), prices, battery)
        if battery == BASE:
            results.to_csv(daily_path(region, year), index=False)
        labels = {"scenario": name, "region": region, **asdict(battery)}
        capture.append(summarise(results).reset_index().assign(**labels))
        agree.append(agreement(results).reset_index().assign(**labels))
        print(f"{name} evaluated ({time.perf_counter() - started:.0f}s)", flush=True)
    if not capture:
        raise SystemExit("no forecasts found: run `nem-battery forecast` first")

    capture_table, agree_table = pd.concat(capture), pd.concat(agree)
    capture_table.to_csv(capture_path(year), index=False)
    agree_table.to_csv(agreement_path(year), index=False)
    return capture_table, agree_table
