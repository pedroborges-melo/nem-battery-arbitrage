"""Rolling day-ahead backtest: forecast tomorrow, optimise on the forecast, settle on actuals.

Simplification to state in the write-up: the plan is fixed a day ahead. Real NEM
batteries rebid every 5 minutes, so these numbers understate what they capture.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .battery import Battery, schedule, settle
from .metrics import scores

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


def forecast(
    prices: pd.Series, days: pd.DatetimeIndex, forecasters: dict[str, Forecaster] = FORECASTERS, n: int = 48
) -> pd.DataFrame:
    """Every forecaster's forecast for every complete day: one row per (day, forecaster), one column per interval.

    Days with gaps in the source data, and forecasts a method could not make, are left out.
    """
    rows = {}
    for day in days:
        if len(_day(prices, day)) != n:
            continue
        for name, forecaster in forecasters.items():
            fc = forecaster(prices, day, n)
            if len(fc) == n and not np.isnan(fc).any():
                rows[(day, name)] = fc
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=["day", "forecaster"])
    return out


def load_forecasts(path: Path) -> pd.DataFrame:
    """Forecasts saved with `forecast(...).to_csv(path)`, read back exactly."""
    out = pd.read_csv(path, index_col=["day", "forecaster"], parse_dates=["day"], float_precision="round_trip")
    out.columns = out.columns.astype(int)
    return out


def evaluate(
    forecasts: pd.DataFrame, prices: pd.Series, battery: Battery = DEFAULT_BATTERY, dt_hours: float = 0.5
) -> pd.DataFrame:
    """One row per (day, forecaster): the forecast's scores and the revenue of the plan made on it."""
    k = round(battery.energy_mwh / battery.power_mw / dt_hours)  # intervals in a full discharge
    rows = []
    for (day, name), fc in forecasts.iterrows():
        actual = _day(prices, day).to_numpy()
        plan = schedule(fc.to_numpy(), battery, dt_hours)
        rows.append(
            {
                "day": day,
                "forecaster": name,
                **scores(fc.to_numpy(), actual, k),
                "revenue_aud": settle(plan, actual, battery, dt_hours),
                "mwh_discharged": float(plan["discharge_mw"].sum() * dt_hours),
            }
        )
    return pd.DataFrame(rows)


def run(
    prices: pd.Series,
    days: pd.DatetimeIndex,
    forecasters: dict[str, Forecaster] = FORECASTERS,
    battery: Battery = DEFAULT_BATTERY,
    dt_hours: float = 0.5,
) -> pd.DataFrame:
    """Forecast, then evaluate: one row per (day, forecaster)."""
    n = round(24 / dt_hours)
    return evaluate(forecast(prices, days, forecasters, n), prices, battery, dt_hours)


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
