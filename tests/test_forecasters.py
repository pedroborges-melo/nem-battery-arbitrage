from __future__ import annotations

import numpy as np
import pandas as pd

from nem_battery.forecasters import GradientBoosting, SeasonalForecast, trading_day

DAY = pd.Timestamp("2024-03-15")


def test_midnight_belongs_to_the_day_before():
    index = pd.DatetimeIndex(["2024-03-15 00:00", "2024-03-15 00:30", "2024-03-15 23:30"])
    assert list(trading_day(index)) == [pd.Timestamp("2024-03-14")] + [pd.Timestamp("2024-03-15")] * 2


def scrambled(market: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The same data with `day` and everything after it replaced by noise."""
    out = market.copy()
    later = trading_day(out.index) >= day
    out.loc[later] = np.random.default_rng(2).normal(1000, 500, (later.sum(), 2))
    return out


def test_seasonal_forecast_ignores_the_day_it_forecasts(market):
    forecast = SeasonalForecast(weeks=4)
    expected = forecast(market["rrp"], DAY, 48)
    assert expected.shape == (48,) and np.isfinite(expected).all()
    np.testing.assert_allclose(forecast(scrambled(market, DAY)["rrp"], DAY, 48), expected)


def test_seasonal_forecast_skips_days_with_a_gap_in_the_history(market):
    prices = market["rrp"].drop(market.index[(market.index > "2024-03-10") & (market.index <= "2024-03-10 03:00")])
    assert np.isnan(SeasonalForecast(weeks=4)(prices, DAY, 48)).all()


def test_seasonal_forecast_finds_the_evening_peak(market):
    forecast = SeasonalForecast(weeks=4)(market["rrp"], DAY, 48)
    assert forecast[36:42].mean() > forecast[:36].mean() + 50


def test_gradient_boosting_ignores_the_day_it_forecasts(market):
    expected = GradientBoosting(market)(market["rrp"], DAY, 48)
    assert expected.shape == (48,) and np.isfinite(expected).all()
    np.testing.assert_allclose(GradientBoosting(scrambled(market, DAY))(market["rrp"], DAY, 48), expected)


def test_gradient_boosting_trains_only_on_months_before_the_forecast(market):
    model = GradientBoosting(market)
    model(market["rrp"], DAY, 48)
    (month,) = model.models
    assert month == DAY.to_period("M")
    assert model.models[month].num_trees() > 0
