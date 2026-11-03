"""Model-based day-ahead price forecasts.

Both forecasters see only data up to the end of the day before the one they
forecast, and both follow the backtest's Forecaster signature, so they slot in
next to the naive baselines. South Australian prices have extreme spikes, so
both are fitted robustly: MSTL with robust smoothing, LightGBM with an
absolute-error loss, which targets the median.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

INTERVALS = 48  # half-hours per day


def trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Each interval's trading day: AEMO labels intervals by their end, so 00:00 belongs to the day before."""
    return (index - pd.Timedelta("1min")).normalize()


class SeasonalForecast:
    """MSTL decomposition with a daily and a weekly cycle on the last `weeks` weeks.

    Tomorrow is the trend's last value, plus the last day of the daily component,
    plus the weekly component's mean level on the same weekday a week earlier. Only
    the level: the weekly component's half-hourly shape rests on `weeks` points per
    slot and is mostly noise, which the optimiser trades on (June 2024, SA1: the
    full weekly shape cut revenue to almost nothing for a 10 AUD/MWh worse MAE).
    """

    def __init__(self, weeks: int = 8):
        self.weeks = weeks

    def __call__(self, prices: pd.Series, day: pd.Timestamp, n: int) -> np.ndarray:
        history = prices.loc[(prices.index > day - pd.Timedelta(weeks=self.weeks)) & (prices.index <= day)]
        if len(history) != self.weeks * 7 * n:
            return np.full(n, np.nan)  # gap in the history: skip the day
        fit = MSTL(history.to_numpy(), periods=(n, 7 * n), stl_kwargs={"robust": True}).fit()
        daily, weekly = fit.seasonal[:, 0], fit.seasonal[:, 1]
        return fit.trend[-1] + daily[-n:] + weekly[-7 * n : -6 * n].mean()


class GradientBoosting:
    """LightGBM on calendar features and the previous days' prices and demand,
    refitted at the start of each month on every earlier day.

    Demand enters only through lags: the day's own demand is not known a day ahead.
    """

    PARAMS = {
        "objective": "l1",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "seed": 0,
        "deterministic": True,
        "num_threads": 4,
        "verbose": -1,
    }
    ROUNDS = 400

    def __init__(self, data: pd.DataFrame):
        self.features = self._features(data)
        self.models: dict[pd.Period, lgb.Booster] = {}

    @staticmethod
    def _features(data: pd.DataFrame) -> pd.DataFrame:
        day = trading_day(data.index)
        daily = data["rrp"].groupby(day).agg(["mean", "max"]).shift(1)  # the previous day's level and spike
        out = pd.DataFrame(
            {
                "price": data["rrp"],
                **{f"price_lag{d}d": data["rrp"].shift(d * INTERVALS) for d in (1, 2, 7)},
                **{f"demand_lag{d}d": data["demand"].shift(d * INTERVALS) for d in (1, 7)},
                "prev_day_mean": daily["mean"].reindex(day).to_numpy(),
                "prev_day_max": daily["max"].reindex(day).to_numpy(),
                "interval": (data.index - day).total_seconds() // 1800 - 1,
                "weekday": day.dayofweek,
                "month": day.month,
                "day": day,
            },
            index=data.index,
        )
        return out.dropna()

    def _model(self, month: pd.Period) -> lgb.Booster:
        if month not in self.models:
            train = self.features[self.features["day"] < month.start_time]
            dataset = lgb.Dataset(train.drop(columns=["price", "day"]), train["price"])
            self.models[month] = lgb.train(self.PARAMS, dataset, num_boost_round=self.ROUNDS)
        return self.models[month]

    def __call__(self, prices: pd.Series, day: pd.Timestamp, n: int) -> np.ndarray:
        rows = self.features[self.features["day"] == day]
        if len(rows) != n:
            return np.full(n, np.nan)
        return np.asarray(self._model(day.to_period("M")).predict(rows.drop(columns=["price", "day"])))
