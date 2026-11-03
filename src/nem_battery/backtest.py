"""Rolling day-ahead backtest: forecast tomorrow, optimise on the forecast, settle on actuals.

Simplification to state in the write-up: the plan is fixed a day ahead. Real NEM
batteries rebid every 5 minutes, so these numbers understate what they capture.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .battery import Battery, schedule, settle

DEFAULT_BATTERY = Battery()

# A forecaster sees prices strictly before `day` and returns one value per interval of `day`.
Forecaster = Callable[[pd.Series, pd.Timestamp, int], np.ndarray]


def perfect_foresight(prices: pd.Series, day: pd.Timestamp, n: int) -> np.ndarray:
    """Upper bound: the actual prices. Not a forecast; the yardstick."""
    return _day(prices, day).to_numpy()


def naive_yesterday(prices: pd.Series, day: pd.Timestamp, n: int) -> np.ndarray:
    return _day(prices, day - pd.Timedelta(days=1)).to_numpy()


def naive_last_week(prices: pd.Series, day: pd.Timestamp, n: int) -> np.ndarray:
    return _day(prices, day - pd.Timedelta(days=7)).to_numpy()


def profile_7d_median(prices: pd.Series, day: pd.Timestamp, n: int) -> np.ndarray:
    """Median of each interval over the previous 7 days: robust to one-off price spikes."""
    past = np.stack([_day(prices, day - pd.Timedelta(days=k)).to_numpy() for k in range(1, 8)])
    return np.median(past, axis=0)


FORECASTERS: dict[str, Forecaster] = {
    "perfect_foresight": perfect_foresight,
    "naive_yesterday": naive_yesterday,
    "naive_last_week": naive_last_week,
    "profile_7d_median": profile_7d_median,
}


def _day(prices: pd.Series, day: pd.Timestamp) -> pd.Series:
    """Intervals ending in (day 00:00, day+1 00:00], i.e. one trading day of interval-end labels."""
    return prices.loc[(prices.index > day) & (prices.index <= day + pd.Timedelta(days=1))]


def run(
    prices: pd.Series,
    days: pd.DatetimeIndex,
    forecasters: dict[str, Forecaster] = FORECASTERS,
    battery: Battery = DEFAULT_BATTERY,
    dt_hours: float = 0.5,
) -> pd.DataFrame:
    """One row per (day, forecaster): forecast error and settled revenue."""
    rows = []
    for day in days:
        actual = _day(prices, day).to_numpy()
        n = len(actual)
        if n != round(24 / dt_hours):
            continue  # incomplete day in the source data
        for name, forecaster in forecasters.items():
            fc = forecaster(prices, day, n)
            if len(fc) != n or np.isnan(fc).any():
                continue
            plan = schedule(fc, battery, dt_hours)
            rows.append(
                {
                    "day": day,
                    "forecaster": name,
                    "mae": float(np.abs(fc - actual).mean()),
                    "revenue_aud": settle(plan, actual, battery, dt_hours),
                    "mwh_discharged": float(plan["discharge_mw"].sum() * dt_hours),
                }
            )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Total revenue and capture rate (share of the perfect-foresight revenue) per forecaster."""
    out = results.groupby("forecaster").agg(
        days=("day", "nunique"),
        mae=("mae", "mean"),
        revenue_aud=("revenue_aud", "sum"),
        mwh_discharged=("mwh_discharged", "sum"),
    )
    out["capture_rate"] = out["revenue_aud"] / out.loc["perfect_foresight", "revenue_aud"]
    return out.sort_values("revenue_aud", ascending=False)
