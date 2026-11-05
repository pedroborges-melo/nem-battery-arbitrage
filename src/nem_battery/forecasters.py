"""Day-ahead price forecasts: three naive baselines, two models and the perfect-foresight yardstick.

A forecaster takes the price series and a trading day and returns that day's 48
half-hourly prices, using only data up to the end of the day before (perfect
foresight aside). When it cannot forecast a day, for example because its history
has a gap, it returns NaNs and the backtest skips that day for that method.

South Australian prices have extreme spikes, so both models are fitted robustly:
MSTL with robust smoothing, LightGBM with an absolute-error loss, which targets the
median.
"""

from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from .data import INTERVALS, is_complete, prices_on, trading_day

Forecaster = Callable[[pd.Series, pd.Timestamp], np.ndarray]
PERFECT = "perfect_foresight"
NOT_AVAILABLE = np.full(INTERVALS, np.nan)


def perfect_foresight(prices: pd.Series, day: pd.Timestamp) -> np.ndarray:
    """The actual prices: not a forecast, but the upper bound every forecast is measured against."""
    return prices_on(prices, day)


def days_before(prices: pd.Series, day: pd.Timestamp, k: int) -> np.ndarray:
    """The prices of the day `k` days before `day`."""
    return prices_on(prices, day - pd.Timedelta(days=k))


def naive_yesterday(prices: pd.Series, day: pd.Timestamp) -> np.ndarray:
    return days_before(prices, day, 1)


def naive_last_week(prices: pd.Series, day: pd.Timestamp) -> np.ndarray:
    return days_before(prices, day, 7)


def profile_7d_median(prices: pd.Series, day: pd.Timestamp) -> np.ndarray:
    """Each half-hour's median over the previous 7 days: robust to one-off price spikes."""
    week = [days_before(prices, day, k) for k in range(1, 8)]
    return np.median(week, axis=0) if all(map(is_complete, week)) else NOT_AVAILABLE


class SeasonalForecast:
    """MSTL decomposition of the last `weeks` weeks into a trend, a daily cycle and a weekly cycle.

    Tomorrow is the trend's last value, plus the last day of the daily cycle, plus the
    weekly cycle's mean level on the same weekday a week earlier. Only that level: the
    weekly cycle's half-hourly shape rests on `weeks` points per slot and is mostly
    noise, which the optimiser trades on (June 2024, SA1: the full weekly shape cut
    revenue to almost nothing for a 10 AUD/MWh worse MAE).
    """

    def __init__(self, weeks: int = 8):
        self.weeks = weeks

    def __call__(self, prices: pd.Series, day: pd.Timestamp) -> np.ndarray:
        history = prices[(prices.index > day - pd.Timedelta(weeks=self.weeks)) & (prices.index <= day)]
        if len(history) != self.weeks * 7 * INTERVALS or history.isna().any():
            return NOT_AVAILABLE
        fit = MSTL(history.to_numpy(), periods=(INTERVALS, 7 * INTERVALS), stl_kwargs={"robust": True}).fit()
        daily, weekly = fit.seasonal[:, 0], fit.seasonal[:, 1]
        same_weekday_last_week = weekly[-7 * INTERVALS : -6 * INTERVALS]
        return fit.trend[-1] + daily[-INTERVALS:] + same_weekday_last_week.mean()


class GradientBoosting:
    """LightGBM on calendar features and the previous days' prices and demand.

    It is built once on the whole data set, so the features are computed once, and it
    refits at the start of each month on every earlier day. Each day's features are
    lags, known by the end of the day before: the day's own demand is not known a day
    ahead. The model reads prices from the data it was built with, not from the
    `prices` argument, which it takes only to match the other forecasters.
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
        self.features = self.make_features(data)
        self.models: dict[pd.Period, lgb.Booster] = {}

    @staticmethod
    def make_features(data: pd.DataFrame) -> pd.DataFrame:
        """One row per half-hour: the target `price`, its features and its trading `day`."""
        day = trading_day(data.index)
        previous_day = data["rrp"].groupby(day).agg(["mean", "max"]).shift(1)  # yesterday's level and spike
        lagged = {f"price_lag{d}d": data["rrp"].shift(d * INTERVALS) for d in (1, 2, 7)}
        lagged |= {f"demand_lag{d}d": data["demand"].shift(d * INTERVALS) for d in (1, 7)}
        features = pd.DataFrame(
            {
                "price": data["rrp"],
                **lagged,
                "prev_day_mean": previous_day["mean"].reindex(day).to_numpy(),
                "prev_day_max": previous_day["max"].reindex(day).to_numpy(),
                "interval": (data.index - day).total_seconds() // 1800 - 1,
                "weekday": day.dayofweek,
                "month": day.month,
                "day": day,
            },
            index=data.index,
        )
        return features.dropna()

    def model_for(self, month: pd.Period) -> lgb.Booster:
        """The model used throughout `month`, trained on every day before it."""
        if month not in self.models:
            train = self.features[self.features["day"] < month.start_time]
            dataset = lgb.Dataset(train.drop(columns=["price", "day"]), train["price"])
            self.models[month] = lgb.train(self.PARAMS, dataset, num_boost_round=self.ROUNDS)
        return self.models[month]

    def __call__(self, prices: pd.Series, day: pd.Timestamp) -> np.ndarray:
        rows = self.features[self.features["day"] == day]
        if len(rows) != INTERVALS:
            return NOT_AVAILABLE
        return np.asarray(self.model_for(day.to_period("M")).predict(rows.drop(columns=["price", "day"])))


def all_forecasters(data: pd.DataFrame) -> dict[str, Forecaster]:
    """The six methods in the backtest, keyed by the names used in the reports."""
    return {
        PERFECT: perfect_foresight,
        "naive_yesterday": naive_yesterday,
        "naive_last_week": naive_last_week,
        "profile_7d_median": profile_7d_median,
        "mstl": SeasonalForecast(),
        "lightgbm": GradientBoosting(data),
    }
