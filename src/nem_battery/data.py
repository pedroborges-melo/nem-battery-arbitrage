"""Download and tidy AEMO's public 5-minute price and demand data.

AEMO publishes one CSV per region and month. SETTLEMENTDATE marks the *end* of
each 5-minute interval, in NEM time (UTC+10, no daylight saving).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

AEMO_URL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
REGIONS = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_month(region: str, year: int, month: int, raw_dir: Path = RAW_DIR) -> Path:
    """Download one month of 5-minute data, skipping files already cached."""
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    yyyymm = f"{year}{month:02d}"
    path = raw_dir / f"PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
    if path.exists():
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        AEMO_URL.format(yyyymm=yyyymm, region=region),
        headers={"User-Agent": "Mozilla/5.0 (research; nem-battery-arbitrage)"},
        timeout=60,
    )
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def load_prices(region: str, start: str, end: str, freq: str = "30min") -> pd.DataFrame:
    """Return regional reference price (RRP, AUD/MWh) and demand (MW) between two dates.

    Prices are averaged to `freq` intervals, labelled by interval end like AEMO's own
    timestamps. `end` is exclusive.
    """
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta("1min"), freq="M")
    frames = [pd.read_csv(download_month(region, p.year, p.month)) for p in months]
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    df = (
        df.rename(columns={"RRP": "rrp", "TOTALDEMAND": "demand"})
        .set_index("ts")[["rrp", "demand"]]
        .sort_index()
    )
    df = df[~df.index.duplicated()]
    out = df.resample(freq, label="right", closed="right").mean()
    return out.loc[(out.index > pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
