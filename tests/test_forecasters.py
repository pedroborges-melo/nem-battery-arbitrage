from __future__ import annotations

import numpy as np
import pandas as pd

from nem_battery.data import prices_on, trading_day
from nem_battery.forecasters import GradientBoosting, SeasonalForecast, naive_yesterday, profile_7d_median

DAY = pd.Timestamp("2024-03-15")


def scrambled(market: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The same data with `day` and everything after it replaced by noise."""
    out = market.copy()
    later = trading_day(out.index) >= day
    out.loc[later] = np.random.default_rng(2).normal(1000, 500, (later.sum(), 2))
    return out


def test_naive_yesterday_is_the_previous_day(market):
    np.testing.assert_array_equal(
        naive_yesterday(market["rrp"], DAY), prices_on(market["rrp"], DAY - pd.Timedelta(days=1))
    )


def test_median_profile_skips_a_week_with_a_missing_half_hour(market):
    prices = market["rrp"].copy()
    prices["2024-03-10 12:00"] = np.nan
    assert np.isnan(profile_7d_median(prices, DAY)).all()
    assert np.isfinite(profile_7d_median(market["rrp"], DAY)).all()


def test_seasonal_forecast_ignores_the_day_it_forecasts(market):
    forecast = SeasonalForecast(weeks=4)
    expected = forecast(market["rrp"], DAY)
    assert expected.shape == (48,) and np.isfinite(expected).all()
    np.testing.assert_allclose(forecast(scrambled(market, DAY)["rrp"], DAY), expected)


def test_seasonal_forecast_skips_days_with_a_gap_in_the_history(market):
    prices = market["rrp"].copy()
    prices["2024-03-10 12:00"] = np.nan
    assert np.isnan(SeasonalForecast(weeks=4)(prices, DAY)).all()


def test_seasonal_forecast_finds_the_evening_peak(market):
    forecast = SeasonalForecast(weeks=4)(market["rrp"], DAY)
    assert forecast[36:42].mean() > forecast[:36].mean() + 50


def test_gradient_boosting_ignores_the_day_it_forecasts(market):
    expected = GradientBoosting(market)(market["rrp"], DAY)
    assert expected.shape == (48,) and np.isfinite(expected).all()
    np.testing.assert_allclose(GradientBoosting(scrambled(market, DAY))(market["rrp"], DAY), expected)


def test_gradient_boosting_trains_only_on_months_before_the_forecast(market):
    model = GradientBoosting(market)
    model(market["rrp"], DAY)
    (month,) = model.models
    assert month == DAY.to_period("M")
    assert model.models[month].num_trees() > 0
