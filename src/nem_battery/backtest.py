"""Rolling day-ahead backtest: forecast each day, optimise on the forecast, settle at the actual prices.

It runs in two steps because they cost very different amounts. Forecasting is slow
(MSTL refits every day) and does not depend on the battery, so it runs once per region
and is saved. Evaluating, which optimises and settles every forecast, takes about a
minute and reruns for each battery.

Simplification to state in the write-up: the plan is fixed a day ahead. Real NEM
batteries rebid every 5 minutes, so these numbers understate what they capture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .battery import Battery, schedule, settle
from .data import INTERVAL_HOURS, is_complete, prices_on
from .forecasters import PERFECT, Forecaster
from .metrics import scores


def forecast(prices: pd.Series, days: pd.DatetimeIndex, forecasters: dict[str, Forecaster]) -> pd.DataFrame:
    """Every method's forecast for every day: one row per (day, forecaster), one column per half-hour.

    Days with missing prices are left out, and so is any forecast a method could not make.
    """
    rows = {}
    for day in days:
        if not is_complete(prices_on(prices, day)):
            continue
        for name, forecaster in forecasters.items():
            values = forecaster(prices, day)
            if is_complete(values):
                rows[(day, name)] = values
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=["day", "forecaster"])
    return out


def load_forecasts(path: Path) -> pd.DataFrame:
    """Forecasts saved with `forecast(...).to_csv(path)`, read back exactly."""
    out = pd.read_csv(path, index_col=["day", "forecaster"], parse_dates=["day"], float_precision="round_trip")
    out.columns = out.columns.astype(int)
    return out


def evaluate(forecasts: pd.DataFrame, prices: pd.Series, battery: Battery) -> pd.DataFrame:
    """One row per (day, forecaster): the forecast's scores and the revenue of the plan made on it."""
    # The scores look at the day's k dearest and cheapest half-hours: the battery's duration.
    k = round(battery.energy_mwh / battery.power_mw / INTERVAL_HOURS)
    rows = []
    for (day, name), row in forecasts.iterrows():
        predicted, actual = row.to_numpy(), prices_on(prices, day)
        plan = schedule(predicted, battery, INTERVAL_HOURS)
        rows.append(
            {
                "day": day,
                "forecaster": name,
                **scores(predicted, actual, k),
                "revenue_aud": settle(plan, actual, battery, INTERVAL_HOURS),
                "mwh_discharged": plan["discharge_mw"].sum() * INTERVAL_HOURS,
            }
        )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Per method: days, mean MAE, total revenue, energy discharged and capture rate.

    The capture rate is the share of the perfect-foresight revenue a method earned.
    """
    out = results.groupby("forecaster").agg(
        days=("day", "nunique"),
        mae=("mae", "mean"),
        revenue_aud=("revenue_aud", "sum"),
        mwh_discharged=("mwh_discharged", "sum"),
    )
    out["capture_rate"] = out["revenue_aud"] / out.loc[PERFECT, "revenue_aud"]
    return out.sort_values("revenue_aud", ascending=False)
