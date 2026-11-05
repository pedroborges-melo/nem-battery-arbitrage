"""AEMO's public price and demand data: download, tidy, and slice into trading days.

AEMO publishes one CSV per region and month. SETTLEMENTDATE marks the *end* of each
5-minute interval, in NEM time (UTC+10, no daylight saving). Everything downstream
works on half-hours labelled the same way, by their end, so a trading day runs from
the interval ending at 00:30 to the one ending at midnight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

AEMO_URL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
REGIONS = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"  # AEMO's files, downloaded on demand
PROCESSED_DIR = ROOT / "data" / "processed"  # saved forecasts
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"

INTERVAL_HOURS = 0.5
INTERVALS = 48  # half-hours in a trading day


def download_month(region: str, year: int, month: int) -> Path:
    """Download one month of 5-minute data, skipping files already cached."""
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    path = RAW_DIR / f"PRICE_AND_DEMAND_{year}{month:02d}_{region}.csv"
    if not path.exists():
        response = requests.get(
            AEMO_URL.format(yyyymm=f"{year}{month:02d}", region=region),
            headers={"User-Agent": "Mozilla/5.0 (research; nem-battery-arbitrage)"},
            timeout=60,
        )
        response.raise_for_status()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path


def load_prices(region: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Half-hourly price (`rrp`, AUD/MWh) and demand (MW) for the trading days from `start` to `end` (exclusive).

    The 5-minute values are averaged into half-hours labelled by their end. The index
    is a regular half-hourly grid: a half-hour with no data stays in it as NaN.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    months = pd.period_range(start, end - pd.Timedelta("1min"), freq="M")
    raw = pd.concat([pd.read_csv(download_month(region, m.year, m.month)) for m in months], ignore_index=True)
    raw.index = pd.to_datetime(raw["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    five_min = raw.rename(columns={"RRP": "rrp", "TOTALDEMAND": "demand"})[["rrp", "demand"]].sort_index()
    five_min = five_min[~five_min.index.duplicated()]
    half_hours = five_min.resample("30min", label="right", closed="right").mean()
    return half_hours[(half_hours.index > start) & (half_hours.index <= end)]


def trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The trading day of each interval: the one ending at midnight belongs to the day before."""
    return (index - pd.Timedelta("1min")).normalize()


def prices_on(prices: pd.Series, day: pd.Timestamp) -> np.ndarray:
    """The day's half-hourly prices: intervals ending after `day` 00:00, up to and including the next midnight."""
    return prices[(prices.index > day) & (prices.index <= day + pd.Timedelta(days=1))].to_numpy()


def is_complete(values: np.ndarray) -> bool:
    """True for a full trading day of values with none missing."""
    return len(values) == INTERVALS and not np.isnan(values).any()
